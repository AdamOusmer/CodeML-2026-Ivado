from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.graph import Graph, Node
from src.policy import committee_features

from .rules import (
    CATEGORIES, ID_PATTERN, MAX_REPORTED_ISSUES, NUMERIC_BOUNDS, POSTAL_PATTERN,
    DataValidationError, binary_columns, required_columns,
)

FRAME_WORKERS = 4
ID_COLUMN = "id_candidat"
POSTAL_COLUMN = "code_postal_3"
REGION_COLUMN = "region_administrative"


@dataclass(frozen=True)
class Findings:
    issues: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameReport:
    history_rows: int
    batch_rows: int
    warnings: tuple[str, ...] = ()


def _frames(history: pd.DataFrame, batch: pd.DataFrame):
    return (("history", history, True), ("batch", batch, False))


def _rows(mask) -> str:
    positions = np.flatnonzero(np.asarray(mask))[:3] + 1
    return "rows " + ", ".join(str(position) for position in positions)


def _blocked(*upstream: Findings) -> bool:
    return any(findings.issues for findings in upstream)


def _pattern_failures(series: pd.Series, pattern) -> np.ndarray:
    compiled = re.compile(pattern)
    present = series.notna()
    matches = pd.Series(False, index=series.index)
    matches[present.to_numpy()] = (
        series[present].astype(str).map(lambda value: compiled.fullmatch(value) is not None).to_numpy()
    )
    return (present & ~matches).to_numpy()


def columns(history: pd.DataFrame, batch: pd.DataFrame) -> None:
    for frame, df, labeled in _frames(history, batch):
        if df.columns.duplicated().any():
            raise DataValidationError(f"{frame}: duplicate column names")
        expected = set(required_columns(labeled))
        missing_columns = sorted(expected - set(df.columns))
        extra_columns = sorted(set(df.columns) - expected)
        details = []
        if missing_columns:
            details.append("missing columns: " + ", ".join(missing_columns))
        if extra_columns:
            details.append("unexpected columns: " + ", ".join(extra_columns))
        if details:
            raise DataValidationError(f"{frame}: {'; '.join(details)}")
        if len(df) == 0:
            raise DataValidationError(f"{frame}: no applicant rows")


def missing(history: pd.DataFrame, batch: pd.DataFrame) -> Findings:
    issues = []
    for frame, df, _ in _frames(history, batch):
        for column, count in df.isna().sum().items():
            if count:
                issues.append(f"{frame}: {column} has {count:,} missing value(s)")
    return Findings(tuple(issues))


def ids(history: pd.DataFrame, batch: pd.DataFrame) -> Findings:
    issues = []
    seen = {}
    for frame, df, _ in _frames(history, batch):
        series = df[ID_COLUMN]
        bad = _pattern_failures(series, ID_PATTERN)
        if bad.any():
            issues.append(f"{frame}: {ID_COLUMN} does not match {re.compile(ID_PATTERN).pattern} "
                          f"in {bad.sum():,} row(s) ({_rows(bad)})")
        duplicated = (series.notna() & series.duplicated()).to_numpy()
        if duplicated.any():
            issues.append(f"{frame}: duplicate {ID_COLUMN} in {duplicated.sum():,} row(s) ({_rows(duplicated)})")
        seen[frame] = set(series.dropna().astype(str))
    overlap = seen["history"] & seen["batch"]
    if overlap:
        issues.append(f"history and batch share {len(overlap):,} applicant IDs")
    return Findings(tuple(issues))


def categories(history: pd.DataFrame, batch: pd.DataFrame) -> Findings:
    issues = []
    for frame, df, _ in _frames(history, batch):
        for column, allowed in CATEGORIES.items():
            series = df[column]
            bad = (series.notna() & ~series.isin(allowed)).to_numpy()
            if bad.any():
                issues.append(f"{frame}: unknown {column} in {bad.sum():,} row(s) "
                              f"(e.g. {series[bad].iloc[0]!r}; {_rows(bad)})")
    return Findings(tuple(issues))


def postal(history: pd.DataFrame, batch: pd.DataFrame) -> Findings:
    issues = []
    for frame, df, _ in _frames(history, batch):
        bad = _pattern_failures(df[POSTAL_COLUMN], POSTAL_PATTERN)
        if bad.any():
            issues.append(f"{frame}: {POSTAL_COLUMN} does not match {re.compile(POSTAL_PATTERN).pattern} "
                          f"in {bad.sum():,} row(s) ({_rows(bad)})")
    return Findings(tuple(issues))


def binary(history: pd.DataFrame, batch: pd.DataFrame) -> Findings:
    issues = []
    for frame, df, labeled in _frames(history, batch):
        for column in binary_columns(labeled):
            series = df[column]
            bad = (series.notna() & ~series.isin([0, 1])).to_numpy()
            if bad.any():
                issues.append(f"{frame}: {column} must be 0 or 1 in {bad.sum():,} row(s) ({_rows(bad)})")
    return Findings(tuple(issues))


def ranges(history: pd.DataFrame, batch: pd.DataFrame) -> Findings:
    issues = []
    for frame, df, _ in _frames(history, batch):
        for column, bound in NUMERIC_BOUNDS.items():
            series = df[column]
            if not pd.api.types.is_numeric_dtype(series):
                issues.append(f"{frame}: {column} is not numeric")
                continue
            bad = series.notna().to_numpy() & ~np.asarray(bound.accepts(series.to_numpy(dtype=float)))
            if bad.any():
                issues.append(f"{frame}: {column} must be a finite number ({bound.describe()}) "
                              f"in {bad.sum():,} row(s) ({_rows(bad)})")
    return Findings(tuple(issues))


def finite_features(history: pd.DataFrame, batch: pd.DataFrame, missing: Findings, categories: Findings,
                    ranges: Findings, binary: Findings) -> Findings:
    if _blocked(missing, categories, ranges, binary):
        return Findings()
    issues = []
    with np.errstate(all="ignore"):
        for frame, df, _ in _frames(history, batch):
            values = committee_features(df).to_numpy(dtype=float)
            bad = ~np.isfinite(values).all(axis=1)
            if bad.any():
                issues.append(f"{frame}: {bad.sum():,} row(s) with non-finite features after transforms "
                              f"({_rows(bad)})")
    return Findings(tuple(issues))


def postal_regions(history: pd.DataFrame, batch: pd.DataFrame, missing: Findings, postal: Findings,
                   categories: Findings) -> Findings:
    if _blocked(missing, postal, categories):
        return Findings()
    warnings = []
    pairs = history[[POSTAL_COLUMN, REGION_COLUMN]]
    sizes = pairs.groupby([POSTAL_COLUMN, REGION_COLUMN]).size().reset_index(name="rows")
    sizes = sizes.sort_values(["rows", REGION_COLUMN], ascending=[False, True])
    mapping = sizes.drop_duplicates(POSTAL_COLUMN).set_index(POSTAL_COLUMN)[REGION_COLUMN]
    region_counts = pairs.groupby(POSTAL_COLUMN)[REGION_COLUMN].nunique()
    split = sorted(region_counts[region_counts > 1].index)
    if split:
        warnings.append(f"history: {len(split):,} code_postal_3 prefix(es) seen in more than one region "
                        f"(e.g. {split[0]})")
    expected = batch[POSTAL_COLUMN].map(mapping)
    mismatch = (expected.notna() & (expected != batch[REGION_COLUMN])).to_numpy()
    if mismatch.any():
        warnings.append(f"batch: {mismatch.sum():,} row(s) whose code_postal_3 maps to another region "
                        f"in history (e.g. {batch[POSTAL_COLUMN][mismatch].iloc[0]})")
    return Findings(warnings=tuple(warnings))


def range_shift(history: pd.DataFrame, batch: pd.DataFrame, missing: Findings, ranges: Findings) -> Findings:
    if _blocked(missing, ranges):
        return Findings()
    warnings = []
    for column in NUMERIC_BOUNDS:
        lo, hi = history[column].min(), history[column].max()
        values = batch[column]
        outside = int(((values < lo) | (values > hi)).sum())
        if outside:
            warnings.append(f"batch: {column} outside history range {lo:g} to {hi:g} in {outside:,} row(s) "
                            f"(batch {values.min():g} to {values.max():g})")
    return Findings(warnings=tuple(warnings))


def consolidate(history: pd.DataFrame, batch: pd.DataFrame, *results) -> FrameReport:
    findings = [result for result in results if isinstance(result, Findings)]
    issues = [issue for result in findings for issue in result.issues]
    if issues:
        shown = issues[:MAX_REPORTED_ISSUES]
        suffix = f"; {len(issues) - len(shown)} more" if len(issues) > len(shown) else ""
        raise DataValidationError(f"{len(issues)} issue(s): {'; '.join(shown)}{suffix}")
    warnings = tuple(warning for result in findings for warning in result.warnings)
    return FrameReport(len(history), len(batch), warnings)


def _check(fn, *upstream: str) -> Node:
    return Node(fn.__name__, fn, ("history", "batch", *upstream), after=("columns",))


FRAME_CHECK_NODES: tuple[Node, ...] = (
    Node("columns", columns, ("history", "batch")),
    _check(missing),
    _check(ids),
    _check(categories),
    _check(postal),
    _check(binary),
    _check(ranges),
    _check(finite_features, "missing", "categories", "ranges", "binary"),
    _check(postal_regions, "missing", "postal", "categories"),
    _check(range_shift, "missing", "ranges"),
)


def frame_graph(nodes: tuple[Node, ...] = FRAME_CHECK_NODES) -> Graph:
    names = [node.name for node in nodes]
    return Graph([*nodes, Node("report", consolidate, ("history", "batch", *names))])


FRAME_CHECKS = frame_graph()


def validate_frames(history: pd.DataFrame, batch: pd.DataFrame, *, graph: Graph = FRAME_CHECKS,
                    workers: int = FRAME_WORKERS) -> FrameReport:
    return graph.run({"history": history, "batch": batch}, targets=["report"], workers=workers)["report"]

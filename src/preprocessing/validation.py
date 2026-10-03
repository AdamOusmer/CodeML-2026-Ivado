from __future__ import annotations

import csv
import math
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.common.logging import RunContext

FEATURE_COLUMNS = (
    "id_candidat", "cote_r_equivalent", "programme_etudes", "region_administrative",
    "code_postal_3", "revenu_familial_estime", "heures_travail_semaine",
    "distance_domicile_campus_km", "premiere_generation_universitaire",
)
LABEL_COLUMN = "decision_octroi"
REGIONS = (
    "Montreal", "Capitale-Nationale", "Bas-Saint-Laurent", "Cote-Nord",
    "Gaspesie-Iles-de-la-Madeleine",
)
PROGRAMS = {"Genie", "Arts et lettres", "Sciences", "Sante", "Sciences sociales"}
NUMERIC_RANGES = {
    "cote_r_equivalent": (15, 40),
    "revenu_familial_estime": (0, None),
    "heures_travail_semaine": (0, 168),
    "distance_domicile_campus_km": (0, None),
}
MAX_REPORTED_ISSUES = 8


class DataValidationError(ValueError):
    pass


@dataclass
class DatasetReport:
    path: Path
    rows: int
    ids: set[str] = field(repr=False)
    regions: Counter[str]
    grants: int | None

    def to_dict(self) -> dict:
        return {"path": str(self.path), "rows": self.rows, "unique_ids": len(self.ids),
                "regions": dict(self.regions), "historical_grants": self.grants,
                "historical_grant_rate": self.grants / self.rows if self.grants is not None else None}


def _check_header(path: Path, header: list[str], labeled: bool) -> None:
    if len(set(header)) != len(header):
        raise DataValidationError(f"{path.name}: duplicate column names")
    expected = set(FEATURE_COLUMNS) | ({LABEL_COLUMN} if labeled else set())
    missing, extra = expected - set(header), set(header) - expected
    details = []
    if missing:
        details.append("missing columns: " + ", ".join(sorted(missing)))
    if extra:
        details.append("unexpected columns: " + ", ".join(sorted(extra)))
    if details:
        raise DataValidationError(f"{path.name}: {'; '.join(details)}")


def _in_range(text: str, minimum: float, maximum: float | None) -> bool:
    try:
        value = float(text)
    except ValueError:
        return False
    return math.isfinite(value) and value >= minimum and (maximum is None or value <= maximum)


def _row_issues(row: dict, header: list[str], labeled: bool, seen_ids: set[str]) -> Iterator[str]:
    if None in row or any(row[key] is None for key in header):
        yield "wrong number of fields"
        return
    empty = {key for key in header if not row[key].strip()}
    for key in sorted(empty):
        yield f"{key} is empty"

    candidate_id = row["id_candidat"]
    if "id_candidat" not in empty:
        if re.fullmatch(r"C\d{6}", candidate_id) is None:
            yield "id_candidat must match C000000"
        if candidate_id in seen_ids:
            yield "duplicate id_candidat"
        seen_ids.add(candidate_id)

    for key, (minimum, maximum) in NUMERIC_RANGES.items():
        if key not in empty and not _in_range(row[key], minimum, maximum):
            bounds = f">= {minimum}" if maximum is None else f"{minimum} to {maximum}"
            yield f"{key} must be a finite number ({bounds})"

    for key, allowed in (("programme_etudes", PROGRAMS), ("region_administrative", set(REGIONS))):
        if key not in empty and row[key] not in allowed:
            yield f"unknown {key}"

    if "code_postal_3" not in empty and re.fullmatch(r"[A-Z]\d[A-Z]", row["code_postal_3"]) is None:
        yield "code_postal_3 must match A1A"

    binary_columns = ("premiere_generation_universitaire", LABEL_COLUMN) if labeled else ("premiere_generation_universitaire",)
    for key in binary_columns:
        if key not in empty and row[key] not in {"0", "1"}:
            yield f"{key} must be 0 or 1"


def validate_csv(path: Path, labeled: bool, expected_rows: int | None) -> DatasetReport:
    ids: set[str] = set()
    regions: Counter[str] = Counter()
    grants = 0 if labeled else None
    row_count = 0
    issues: list[str] = []

    try:
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source, strict=True)
            header = reader.fieldnames or []
            _check_header(path, header, labeled)
            for row_count, row in enumerate(reader, start=1):
                row_issues = list(_row_issues(row, header, labeled, ids))
                issues += [f"row {row_count}: {message}" for message in row_issues]
                if row_issues != ["wrong number of fields"]:
                    regions[row["region_administrative"]] += 1
                    if labeled and row[LABEL_COLUMN] == "1":
                        grants += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataValidationError(f"Cannot read {path}: {exc}") from exc

    if row_count == 0:
        issues.append("no applicant rows")
    if expected_rows is not None and row_count != expected_rows:
        issues.append(f"expected {expected_rows:,} rows, found {row_count:,}")
    if issues:
        shown = issues[:MAX_REPORTED_ISSUES]
        suffix = f"; {len(issues) - len(shown)} more" if len(issues) > len(shown) else ""
        raise DataValidationError(f"{path.name}: {len(issues)} issue(s): {'; '.join(shown)}{suffix}")
    return DatasetReport(path, row_count, ids, regions, grants)


def _collect(future, progress, run: RunContext) -> DatasetReport:
    report = future.result()
    progress.advance()
    run.logger.debug("Validated %s: %s rows, %s unique IDs", report.path.name, report.rows, len(report.ids))
    return report


def validate_datasets(train_path: Path, evaluation_path: Path, *, run: RunContext,
                      train_rows: int | None = 10_000,
                      evaluation_rows: int | None = 4_000) -> tuple[DatasetReport, DatasetReport]:
    jobs = [(train_path, True, train_rows), (evaluation_path, False, evaluation_rows)]
    with run.stage("Checking application data"), ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for path, labeled, rows in jobs:
            run.logger.debug("Reading %s; expected rows=%s, labeled=%s", path, rows, labeled)
        futures = [pool.submit(validate_csv, *job) for job in jobs]
        with run.progress("Validate CSV files", total=len(futures)) as progress:
            historical, evaluation = (_collect(future, progress, run) for future in futures)
        overlap = historical.ids & evaluation.ids
        if overlap:
            raise DataValidationError(f"Historical and evaluation data share {len(overlap):,} applicant IDs")
    return historical, evaluation

from __future__ import annotations

import numpy as np
import pandas as pd

from src.policy import BUDGET_BOUNDS


def has_missing(values: np.ndarray) -> bool:
    return bool(np.asarray(pd.isna(values)).any())


def id_issues(batch_ids: np.ndarray, ids: np.ndarray) -> list[str]:
    if ids.ndim != 1:
        return ["ids must be one-dimensional"]
    issues = []
    if len(ids) != len(batch_ids):
        issues.append(f"ids length {len(ids)} differs from batch length {len(batch_ids)}")
    if has_missing(ids):
        return issues
    if len(ids) == len(batch_ids) and not np.array_equal(ids, batch_ids):
        issues.append("ids differ from batch id_candidat in order")
    if pd.Series(ids).duplicated().any():
        issues.append("ids are not unique")
    return issues


def is_binary(decisions: np.ndarray) -> bool:
    return decisions.dtype.kind in "iu" and bool(np.isin(decisions, (0, 1)).all())


def decision_issues(n: int, decisions: np.ndarray, k: int) -> list[str]:
    if decisions.ndim != 1:
        return ["decisions must be one-dimensional"]
    issues = []
    if len(decisions) != n:
        issues.append(f"decisions length {len(decisions)} differs from batch length {n}")
    if not is_binary(decisions):
        return issues + ["decisions are not integers in {0, 1}"]
    granted = int(decisions.sum())
    if granted != k:
        issues.append(f"{granted:,} grants, expected {k:,}")
    low, high = BUDGET_BOUNDS
    if len(decisions) == n and n > 0 and not low <= granted / n <= high:
        issues.append(f"grant rate {granted / n:.1%} outside budget {low:.0%}-{high:.0%}")
    return issues


def submission_issues(batch: pd.DataFrame, ids, decisions, k: int) -> list[str]:
    batch_ids = batch["id_candidat"].to_numpy(dtype=object)
    ids = np.asarray(ids, dtype=object)
    decisions = np.asarray(decisions)
    issues = []
    if has_missing(ids) or has_missing(decisions):
        issues.append("output contains missing values")
    issues += id_issues(batch_ids, ids)
    issues += decision_issues(len(batch_ids), decisions, k)
    return issues

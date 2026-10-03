from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.policy import BUDGET_BOUNDS, eo_gap, is_remote, legitimate_features, reference_labels

DP_GAP_WARN = 0.10
IMPACT_RATIO_WARN = 0.80
EO_GAP_WARN, EO_GAP_ALERT = 0.03, 0.05
INTERSECTION_GAP_WARN = 0.15
PROXY_AUC_DRIFT_ALERT = 0.05
PSI_WARN, PSI_ALERT = 0.10, 0.25
MAX_CATEGORIES = 5
MIN_CLASS_COUNT = 3
MONITORED_FEATURES = ["cote_r", "log_revenu", "heures_travail", "premiere_generation"]
STATUS_ORDER = {"OK": 0, "WARN": 1, "ALERT": 2}


@dataclass
class Check:
    name: str
    value: float
    threshold: str
    status: str
    detail: str = ""
    correctable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def graded(name, value, warn, alert=None, *, higher_is_worse=True, threshold="", detail="", correctable=False):
    if np.isnan(value):
        return Check(name, float("nan"), threshold, "ALERT", f"not computable: {detail}" if detail else "not computable")
    worse = (lambda limit: value > limit) if higher_is_worse else (lambda limit: value < limit)
    status = "ALERT" if alert is not None and worse(alert) else "WARN" if worse(warn) else "OK"
    return Check(name, float(value), threshold, status, detail, correctable)


def group_rate(decisions, mask):
    return decisions[mask].mean() if mask.any() else np.nan


def share_gap(decisions, mask_a, mask_b):
    return abs(group_rate(decisions, mask_a) - group_rate(decisions, mask_b))


def population_stability_index(expected, actual, bins=10):
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    categories = np.unique(np.concatenate([expected, actual]))
    if len(categories) <= MAX_CATEGORIES:
        expected_share = np.array([(expected == c).mean() for c in categories]) + 1e-6
        actual_share = np.array([(actual == c).mean() for c in categories]) + 1e-6
        return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([-np.inf, np.median(expected), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    expected_share = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    actual_share = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def proxy_auc(df):
    remote = is_remote(df)
    if min(remote.sum(), len(remote) - remote.sum()) < MIN_CLASS_COUNT:
        return np.nan
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(model, legitimate_features(df), remote, cv=3, scoring="roc_auc").mean()


def eo_gap_check(name, decisions, y_ref, remote, detail, alert):
    return graded(name, eo_gap(decisions, y_ref, remote), EO_GAP_WARN, alert,
                  threshold=f"warn > {EO_GAP_WARN}, alert > {alert}", detail=detail, correctable=True)


def run_checks(history: pd.DataFrame, batch: pd.DataFrame, decisions: np.ndarray,
               reviewed: pd.Series | None = None, eo_gap_alert: float = EO_GAP_ALERT) -> list[Check]:
    remote = is_remote(batch)
    share = decisions.mean()
    low, high = BUDGET_BOUNDS
    checks = [Check("grant rate within budget", share, f"{low:.0%}-{high:.0%}",
                    "OK" if low <= share <= high else "ALERT", f"{int(decisions.sum()):,} of {len(decisions):,}")]

    centre_rate, remote_rate = group_rate(decisions, remote == 0), group_rate(decisions, remote == 1)
    checks.append(graded("demographic parity gap (centre vs remote)", abs(centre_rate - remote_rate), DP_GAP_WARN,
                         threshold=f"warn > {DP_GAP_WARN}", detail=f"centre {centre_rate:.1%}, remote {remote_rate:.1%}"))

    region_rates = pd.Series(decisions).groupby(batch["region_administrative"].to_numpy()).mean()
    checks.append(graded("impact ratio, lowest/highest region", region_rates.min() / region_rates.max(), IMPACT_RATIO_WARN,
                         higher_is_worse=False, threshold=f"warn < {IMPACT_RATIO_WARN}",
                         detail=f"{region_rates.idxmin()} {region_rates.min():.1%} vs {region_rates.idxmax()} {region_rates.max():.1%}"))

    references = reference_labels(history, batch, share)
    checks.append(eo_gap_check("opportunity gap vs merit reference", decisions, references["merit"], remote,
                               "deserving = top R scores at the same budget", eo_gap_alert))
    checks.append(eo_gap_check("opportunity gap vs corrected committee", decisions, references["corrected"], remote,
                               "deserving = committee rule without regional penalty", eo_gap_alert))
    if reviewed is not None:
        known = reviewed.notna().to_numpy()
        checks.append(eo_gap_check("opportunity gap vs human-reviewed sample", decisions[known],
                                   reviewed[known].astype(int).to_numpy(), remote[known],
                                   f"{known.sum():,} blind-reviewed applications", eo_gap_alert))

    first_gen = batch["premiere_generation_universitaire"].to_numpy() == 1
    income_tercile = pd.qcut(batch["revenu_familial_estime"], 3, labels=False, duplicates="drop").to_numpy()
    groups = {"first-generation": first_gen,
              **{f"income tercile {int(t) + 1}": income_tercile == t for t in np.unique(income_tercile[~np.isnan(income_tercile)])}}
    intersection_gaps = {name: share_gap(decisions, members & (remote == 0), members & (remote == 1))
                         for name, members in groups.items()}
    computable = {name: gap for name, gap in intersection_gaps.items() if not np.isnan(gap)}
    worst_group = max(computable, key=computable.get, default=None)
    checks.append(graded("largest intersectional gap (centre vs remote)", computable.get(worst_group, np.nan),
                         INTERSECTION_GAP_WARN, threshold=f"warn > {INTERSECTION_GAP_WARN}",
                         detail=worst_group or "no subgroup has both centre and remote applicants"))

    baseline_auc, batch_auc = proxy_auc(history), proxy_auc(batch)
    checks.append(graded("proxy drift: region predictability (AUC change)", batch_auc - baseline_auc, PROXY_AUC_DRIFT_ALERT,
                         PROXY_AUC_DRIFT_ALERT, threshold=f"alert > +{PROXY_AUC_DRIFT_ALERT}",
                         detail=f"history {baseline_auc:.3f}, batch {batch_auc:.3f}"))

    history_features, batch_features = legitimate_features(history), legitimate_features(batch)
    history_remote = is_remote(history)
    drift = {
        f"{feature} ({group})": population_stability_index(history_features[feature][history_remote == flag],
                                                           batch_features[feature][remote == flag])
        for feature in MONITORED_FEATURES for group, flag in [("centre", 0), ("remote", 1)]
    }
    computable_drift = {name: value for name, value in drift.items() if not np.isnan(value)}
    worst_feature = max(computable_drift, key=computable_drift.get, default=None)
    checks.append(graded("feature drift, max PSI", computable_drift.get(worst_feature, np.nan), PSI_WARN, PSI_ALERT,
                         threshold=f"warn > {PSI_WARN}, alert > {PSI_ALERT}", detail=worst_feature or "no feature has both history and batch rows"))
    return checks


def overall_status(checks: list[Check]) -> str:
    return max((check.status for check in checks), key=STATUS_ORDER.get)


@dataclass
class Verdict:
    status: str
    checks: list[Check]

    @property
    def correctable(self) -> bool:
        alerts = [check for check in self.checks if check.status == "ALERT"]
        return self.status == "ALERT" and all(check.correctable for check in alerts)


def verdict(checks: list[Check]) -> Verdict:
    return Verdict(overall_status(checks), checks)

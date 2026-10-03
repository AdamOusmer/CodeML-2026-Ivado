from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.policy.core import BUDGET_BOUNDS, eo_gap, legitimate_features
from src.policy.models import reference_labels
from src.policy.regions import is_remote

DP_GAP_WARN = 0.10
IMPACT_RATIO_WARN = 0.80
EO_GAP_WARN, EO_GAP_ALERT = 0.03, 0.05
INTERSECTION_GAP_WARN = 0.15
PROXY_AUC_DRIFT_ALERT = 0.05
PSI_WARN, PSI_ALERT = 0.10, 0.25
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
    worse = (lambda limit: value > limit) if higher_is_worse else (lambda limit: value < limit)
    status = "ALERT" if alert is not None and worse(alert) else "WARN" if worse(warn) else "OK"
    return Check(name, float(value), threshold, status, detail, correctable)


def population_stability_index(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([-np.inf, np.median(expected), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    expected_share = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    actual_share = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def proxy_auc(df):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return cross_val_score(model, legitimate_features(df), is_remote(df), cv=3, scoring="roc_auc").mean()


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

    centre_rate, remote_rate = decisions[remote == 0].mean(), decisions[remote == 1].mean()
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
    income_tercile = pd.qcut(batch["revenu_familial_estime"], 3, labels=False).to_numpy()
    intersection_gaps = {
        "first-generation": abs(decisions[first_gen & (remote == 0)].mean() - decisions[first_gen & (remote == 1)].mean()),
        **{f"income tercile {t + 1}": abs(decisions[(income_tercile == t) & (remote == 0)].mean()
                                          - decisions[(income_tercile == t) & (remote == 1)].mean()) for t in range(3)},
    }
    worst_group = max(intersection_gaps, key=intersection_gaps.get)
    checks.append(graded("largest intersectional gap (centre vs remote)", intersection_gaps[worst_group],
                         INTERSECTION_GAP_WARN, threshold=f"warn > {INTERSECTION_GAP_WARN}", detail=worst_group))

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
    worst_feature = max(drift, key=drift.get)
    checks.append(graded("feature drift, max PSI", drift[worst_feature], PSI_WARN, PSI_ALERT,
                         threshold=f"warn > {PSI_WARN}, alert > {PSI_ALERT}", detail=worst_feature))
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

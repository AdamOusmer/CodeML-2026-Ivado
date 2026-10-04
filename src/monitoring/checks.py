from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.graph import Graph, Node
from src.policy import (BUDGET_BOUNDS, PROGRAMMES, REGIONS, SCORING_FEATURES, eo_gap, is_remote, reference_labels,
                        scoring_features, signed_eo_gap, LIMITS, warn_level)

DP_GAP_WARN = 0.10
IMPACT_RATIO_WARN = 0.80
EO_GAP_WARN, EO_GAP_ALERT = 0.03, LIMITS["max_eo_gap_vs_merit"]
INTERSECTION_GAP_WARN = 0.15
PROXY_AUC_DRIFT_ALERT = 0.05
PSI_WARN, PSI_ALERT = 0.10, 0.25
MIN_CLASS_COUNT = 3
MONITORED_FEATURES = SCORING_FEATURES
CATEGORICAL_FEATURES = {"programme_etudes": PROGRAMMES, "premiere_generation_universitaire": (0, 1)}
MONITOR_WORKERS = 4
STATUS_ORDER = {"OK": 0, "WARN": 1, "ALERT": 2}


@dataclass
class Check:
    name: str
    value: float
    threshold: str
    status: str
    detail: str = ""
    correctable: bool = False
    excess: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def graded(name, value, warn, alert=None, *, higher_is_worse=True, threshold="", detail="", correctable=False):
    if np.isnan(value):
        return Check(name, float("nan"), threshold, "ALERT", f"not computable: {detail}" if detail else "not computable",
                     excess=float("inf"))
    worse = (lambda limit: value > limit) if higher_is_worse else (lambda limit: value < limit)
    status = "ALERT" if alert is not None and worse(alert) else "WARN" if worse(warn) else "OK"
    excess = 0.0 if alert is None else float(value - alert if higher_is_worse else alert - value)
    return Check(name, float(value), threshold, status, detail, correctable, excess)


def group_rate(decisions, mask):
    return decisions[mask].mean() if mask.any() else np.nan


def share_gap(decisions, mask_a, mask_b):
    return abs(group_rate(decisions, mask_a) - group_rate(decisions, mask_b))


def categorical_psi(expected, actual, categories):
    expected_share = np.array([(expected == c).mean() for c in categories]) + 1e-6
    actual_share = np.array([(actual == c).mean() for c in categories]) + 1e-6
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def histogram_psi(expected, actual, bins=10):
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
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
    return cross_val_score(model, scoring_features(df), remote, cv=3, scoring="roc_auc").mean()


def eo_gap_check(name, decisions, y_ref, remote, detail, alert):
    threshold = f"warn > {EO_GAP_WARN}" if alert is None else f"warn > {EO_GAP_WARN}, alert > {alert}"
    return graded(name, eo_gap(decisions, y_ref, remote), EO_GAP_WARN, alert, threshold=threshold, detail=detail,
                  correctable=alert is not None)


def budget_check(decisions):
    share = decisions.mean()
    low, high = BUDGET_BOUNDS
    return Check("grant rate within budget", share, f"{low:.0%}-{high:.0%}",
                 "OK" if low <= share <= high else "ALERT", f"{int(decisions.sum()):,} of {len(decisions):,}")


def parity_check(decisions, remote):
    centre_rate, remote_rate = group_rate(decisions, remote == 0), group_rate(decisions, remote == 1)
    return graded("demographic parity gap (centre vs remote)", abs(centre_rate - remote_rate), DP_GAP_WARN,
                  threshold=f"warn > {DP_GAP_WARN}", detail=f"centre {centre_rate:.1%}, remote {remote_rate:.1%}")


def impact_check(decisions, batch):
    region_rates = pd.Series(decisions).groupby(batch["region_administrative"].to_numpy()).mean()
    return graded("impact ratio, lowest/highest region", region_rates.min() / region_rates.max(), IMPACT_RATIO_WARN,
                  higher_is_worse=False, threshold=f"warn < {IMPACT_RATIO_WARN}",
                  detail=f"{region_rates.idxmin()} {region_rates.min():.1%} vs {region_rates.idxmax()} {region_rates.max():.1%}")


def eo_merit_check(decisions, references, remote, eo_gap_alert):
    name = "opportunity gap vs merit reference"
    gap = signed_eo_gap(decisions, references["merit"], remote)
    own = signed_eo_gap(references["consensus"], references["merit"], remote)
    gap_warn = warn_level(own, eo_gap_alert)
    overshoot_alert = LIMITS["max_eo_overshoot_vs_merit"]
    overshoot_warn = warn_level(-own, overshoot_alert)
    disadvantage = graded(name, gap, gap_warn, eo_gap_alert, correctable=True)
    overshoot = graded(name, -gap, overshoot_warn, overshoot_alert, correctable=True)
    worst = max((disadvantage, overshoot), key=lambda check: (STATUS_ORDER[check.status], check.excess))
    threshold = (f"ok in [{-overshoot_warn:.4f}, {gap_warn:.4f}], alert outside [{-overshoot_alert}, {eo_gap_alert}]")
    detail = f"deserving = top R scores at the same budget; centre minus remote {gap:+.4f}, absolute {abs(gap):.4f}"
    if np.isnan(gap):
        return replace(worst, threshold=threshold)
    return replace(worst, value=float(gap), threshold=threshold, detail=detail,
                   excess=max(disadvantage.excess, overshoot.excess))


def eo_corrected_check(decisions, references, remote, eo_gap_alert, corrected_enforced):
    return eo_gap_check("opportunity gap vs corrected committee", decisions, references["corrected"], remote,
                        "deserving = committee rule without regional penalty",
                        eo_gap_alert if corrected_enforced else None)


def opportunity_checks(decisions, references, remote, eo_gap_alert, corrected_enforced=True):
    return [eo_merit_check(decisions, references, remote, eo_gap_alert),
            eo_corrected_check(decisions, references, remote, eo_gap_alert, corrected_enforced)]


def eo_reviewed_check(decisions, reviewed, remote, eo_gap_alert):
    if reviewed is None:
        return None
    known = reviewed.notna().to_numpy()
    return eo_gap_check("opportunity gap vs human-reviewed sample", decisions[known],
                        reviewed[known].astype(int).to_numpy(), remote[known],
                        f"{known.sum():,} blind-reviewed applications", eo_gap_alert)


def intersection_check(decisions, batch, remote):
    first_gen = batch["premiere_generation_universitaire"].to_numpy() == 1
    income_tercile = pd.qcut(batch["revenu_familial_estime"], 3, labels=False, duplicates="drop").to_numpy()
    groups = {"first-generation": first_gen,
              **{f"income tercile {int(t) + 1}": income_tercile == t for t in np.unique(income_tercile[~np.isnan(income_tercile)])}}
    intersection_gaps = {name: share_gap(decisions, members & (remote == 0), members & (remote == 1))
                         for name, members in groups.items()}
    computable = {name: gap for name, gap in intersection_gaps.items() if not np.isnan(gap)}
    worst_group = max(computable, key=computable.get, default=None)
    return graded("largest intersectional gap (centre vs remote)", computable.get(worst_group, np.nan),
                  INTERSECTION_GAP_WARN, threshold=f"warn > {INTERSECTION_GAP_WARN}",
                  detail=worst_group or "no subgroup has both centre and remote applicants")


def proxy_drift_check(history_auc, batch_auc):
    return graded("proxy drift: region predictability (AUC change)", batch_auc - history_auc, PROXY_AUC_DRIFT_ALERT,
                  PROXY_AUC_DRIFT_ALERT, threshold=f"alert > +{PROXY_AUC_DRIFT_ALERT}",
                  detail=f"history {history_auc:.3f}, batch {batch_auc:.3f}")


def feature_drift_check(history_features, batch_features, history_remote, remote):
    drift = {
        f"{feature} ({group})": histogram_psi(history_features[feature][history_remote == flag],
                                              batch_features[feature][remote == flag])
        for feature in MONITORED_FEATURES for group, flag in [("centre", 0), ("remote", 1)]
    }
    computable_drift = {name: value for name, value in drift.items() if not np.isnan(value)}
    worst_feature = max(computable_drift, key=computable_drift.get, default=None)
    return graded("feature drift, max PSI", computable_drift.get(worst_feature, np.nan), PSI_WARN, PSI_ALERT,
                  threshold=f"warn > {PSI_WARN}, alert > {PSI_ALERT}", detail=worst_feature or "no feature has both history and batch rows")


def categorical_drift_check(history, batch):
    history_regions, batch_regions = history["region_administrative"].to_numpy(), batch["region_administrative"].to_numpy()
    worst = None
    for region in REGIONS:
        for column, categories in CATEGORICAL_FEATURES.items():
            expected, actual = history[column][history_regions == region], batch[column][batch_regions == region]
            if len(expected) == 0 or len(actual) == 0:
                continue
            value = categorical_psi(expected, actual, categories)
            if worst is None or value > worst[0]:
                worst = (value, column, region, expected, actual, categories)
    if worst is None:
        return graded("categorical drift, max PSI", np.nan, PSI_WARN, PSI_ALERT,
                      threshold=f"warn > {PSI_WARN}, alert > {PSI_ALERT}",
                      detail="no region has both history and batch rows")
    value, column, region, expected, actual, categories = worst
    shifts = {category: 100 * ((actual == category).mean() - (expected == category).mean()) for category in categories}
    category = max(shifts, key=lambda c: abs(shifts[c]))
    return graded("categorical drift, max PSI", value, PSI_WARN, PSI_ALERT,
                  threshold=f"warn > {PSI_WARN}, alert > {PSI_ALERT}",
                  detail=f"{column} ({region}), largest shift {category} {shifts[category]:+.2f} points")


def collect(*values):
    return [value for value in values if isinstance(value, Check)]


CHECK_NODES = (
    Node("remote", is_remote, ("batch",)),
    Node("history_remote", is_remote, ("history",)),
    Node("share", lambda decisions: decisions.mean(), ("decisions",)),
    Node("references", reference_labels, ("history", "batch", "share")),
    Node("history_auc", proxy_auc, ("history",)),
    Node("batch_auc", proxy_auc, ("batch",)),
    Node("history_features", scoring_features, ("history",)),
    Node("batch_features", scoring_features, ("batch",)),
    Node("budget", budget_check, ("decisions",)),
    Node("parity", parity_check, ("decisions", "remote")),
    Node("impact", impact_check, ("decisions", "batch")),
    Node("eo_merit", eo_merit_check, ("decisions", "references", "remote", "eo_gap_alert")),
    Node("eo_corrected", eo_corrected_check,
         ("decisions", "references", "remote", "eo_gap_alert", "corrected_enforced")),
    Node("eo_reviewed", eo_reviewed_check, ("decisions", "reviewed", "remote", "eo_gap_alert")),
    Node("intersection", intersection_check, ("decisions", "batch", "remote")),
    Node("proxy_drift", proxy_drift_check, ("history_auc", "batch_auc")),
    Node("feature_drift", feature_drift_check, ("history_features", "batch_features", "history_remote", "remote")),
    Node("categorical_drift", categorical_drift_check, ("history", "batch")),
)


def monitoring_graph(nodes=CHECK_NODES) -> Graph:
    return Graph(nodes).add(Node("checks", collect, tuple(node.name for node in nodes)))


MONITORING = monitoring_graph()


def run_checks(history: pd.DataFrame, batch: pd.DataFrame, decisions: np.ndarray,
               reviewed: pd.Series | None = None, eo_gap_alert: float = EO_GAP_ALERT, *,
               corrected_enforced: bool = True, graph: Graph = MONITORING,
               workers: int = MONITOR_WORKERS) -> list[Check]:
    seeds = {"history": history, "batch": batch, "decisions": decisions, "reviewed": reviewed,
             "eo_gap_alert": eo_gap_alert, "corrected_enforced": corrected_enforced}
    return graph.run(seeds, targets=["checks"], workers=workers)["checks"]


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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from .core import allocate
from .regions import is_remote
from .schema import COMMITTEE_FEATURES, committee_features

NO_PENALTY_WARNING = "no regional penalty found"
PENALTY_NOT_DISTINGUISHABLE_WARNING = "penalty not distinguishable from zero"
GROUP_RULES_WARNING = "group-specific committee rules; single-penalty correction may be incomplete"

SLOPE_TEST_SIGNIFICANCE = 0.05


class CommitteeModel:
    def fit(self, df, y):
        X = committee_features(df)
        self.scaler_ = StandardScaler().fit(X)
        Z = np.column_stack([self.scaler_.transform(X), is_remote(df)])
        self.lr_ = LogisticRegression(max_iter=3000).fit(Z, y)
        self.remote_penalty_ = self.lr_.coef_[0][-1]
        return self

    def corrected_logit(self, df, removal=1.0, discounts=()):
        Z = self.scaler_.transform(committee_features(df))
        kept = np.array([1.0 - dict(discounts).get(name, 0.0) for name in COMMITTEE_FEATURES])
        base = Z @ (self.lr_.coef_[0][:-1] * kept) + self.lr_.intercept_[0]
        return base + (1.0 - removal) * self.remote_penalty_ * is_remote(df)

    def rule_weights(self, income_weight):
        coefficients = dict(zip(COMMITTEE_FEATURES, np.abs(self.lr_.coef_[0][:-1])))
        weights = {"cote_r": 1.0, "heures_travail": coefficients["heures_travail"] / coefficients["cote_r"],
                   "log_revenu": income_weight}
        return np.array([weights.get(name, 0.0) for name in COMMITTEE_FEATURES])

    def rule_score(self, df, income_weight):
        return self.scaler_.transform(committee_features(df)) @ self.rule_weights(income_weight)


@dataclass(frozen=True)
class LabelCorrection:
    committee: CommitteeModel
    labels: np.ndarray
    penalty: float
    k: int
    flipped_in: np.ndarray
    flipped_out: np.ndarray


@dataclass(frozen=True)
class CorrectionReport:
    penalty: float
    penalty_ci: tuple | None
    slope_test_statistic: float
    slope_test_df: int
    slope_test_p: float
    flipped_in: int
    flipped_out: int
    flipped_in_remote: int
    flipped_out_centre: int


def effective_removal(penalty, removal):
    return removal if penalty < 0 else 0.0


def correct_labels(history, share, removal=1.0, discounts=()):
    historical = history["decision_octroi"].to_numpy()
    committee = CommitteeModel().fit(history, historical)
    penalty = float(committee.remote_penalty_)
    labels = allocate(committee.corrected_logit(history, effective_removal(penalty, removal), discounts), share)
    flipped_in = np.flatnonzero((historical == 0) & (labels == 1))
    flipped_out = np.flatnonzero((historical == 1) & (labels == 0))
    if len(flipped_in) != len(flipped_out):
        raise ValueError(f"Label correction flipped {len(flipped_in)} in but {len(flipped_out)} out")
    return LabelCorrection(committee, labels, penalty, int(round(share * len(history))), flipped_in, flipped_out)


def report_warnings(report):
    warnings = []
    if report.penalty >= 0:
        warnings.append(NO_PENALTY_WARNING)
    if report.penalty_ci is None or report.penalty_ci[0] <= 0 <= report.penalty_ci[1]:
        warnings.append(PENALTY_NOT_DISTINGUISHABLE_WARNING)
    if report.slope_test_p < SLOPE_TEST_SIGNIFICANCE:
        warnings.append(GROUP_RULES_WARNING)
    return warnings


def _bootstrap_penalty(history, rows):
    sample = history.iloc[rows]
    return CommitteeModel().fit(sample, sample["decision_octroi"].to_numpy()).remote_penalty_


def _penalty_interval(history, n_boot, seed, workers):
    rng = np.random.default_rng(seed)
    outcomes = history["decision_octroi"].to_numpy()
    samples = [rng.integers(0, len(history), len(history)) for _ in range(n_boot)]
    samples = [rows for rows in samples if 0 < outcomes[rows].sum() < len(rows)]
    if not samples:
        return None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        penalties = list(pool.map(lambda rows: _bootstrap_penalty(history, rows), samples))
    low, high = np.percentile(penalties, [2.5, 97.5])
    return float(low), float(high)


def _unpenalized_log_loss(X, y):
    model = LogisticRegression(C=np.inf, max_iter=10000).fit(X, y)
    return log_loss(y, model.predict_proba(X), normalize=False)


def _slope_equality_test(history):
    y = history["decision_octroi"].to_numpy()
    features = StandardScaler().fit_transform(committee_features(history))
    remote = is_remote(history)[:, None]
    base = np.hstack([features, remote])
    full = np.hstack([base, remote * features])
    statistic = 2 * (_unpenalized_log_loss(base, y) - _unpenalized_log_loss(full, y))
    df = features.shape[1]
    return float(statistic), df, float(chi2.sf(statistic, df))


def correction_report(history, share, n_boot=200, seed=0, workers=4):
    correction = correct_labels(history, share)
    remote = is_remote(history)
    statistic, df, p_value = _slope_equality_test(history)
    return CorrectionReport(
        penalty=correction.penalty,
        penalty_ci=_penalty_interval(history, n_boot, seed, workers),
        slope_test_statistic=statistic,
        slope_test_df=df,
        slope_test_p=p_value,
        flipped_in=len(correction.flipped_in),
        flipped_out=len(correction.flipped_out),
        flipped_in_remote=int(remote[correction.flipped_in].sum()),
        flipped_out_centre=int((remote[correction.flipped_out] == 0).sum()),
    )

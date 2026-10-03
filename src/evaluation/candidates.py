from dataclasses import dataclass

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier

from src.policy import (
    DECLARED_CONFIG,
    CommitteeModel,
    FairPipeline,
    allocate,
    is_remote,
    legitimate_features,
    logistic_regression,
    production_features,
)

from .core import Candidate, run, summarize
from .pareto import pareto_mask, plot
from .tuner import SEARCH_SPACE

SEED = 42
EXPGRAD_BOUNDS = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
REMOVALS = np.linspace(0, 1, 6)
PANELS = [("eo_gap_corrected", "acc_historical"), ("eo_gap_merit", "acc_historical")]
REPORT_COLUMNS = ["method", "grant_rate", "budget_ok", "rate_centre", "rate_remote", "eo_gap_corrected",
                  "eo_gap_corrected_std", "eo_gap_merit", "eo_gap_merit_std", "agree_merit", "acc_historical", "pareto"]


def labels(train):
    return train["decision_octroi"].to_numpy()


def production_forest(train, test):
    X_train, X_test = production_features(train, test)
    forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=SEED, n_jobs=1)
    return forest.fit(X_train, labels(train)), X_test


def production_natural(train, test, share):
    forest, X_test = production_forest(train, test)
    return forest.predict(X_test)


def production_at_budget(train, test, share):
    forest, X_test = production_forest(train, test)
    return allocate(forest.predict_proba(X_test)[:, 1], share)


def drop_proxies(train, test, share):
    model = logistic_regression().fit(legitimate_features(train), labels(train))
    return allocate(model.predict_proba(legitimate_features(test))[:, 1], share)


def threshold_optimizer(train, test, share):
    model = logistic_regression().fit(legitimate_features(train), labels(train))
    optimizer = ThresholdOptimizer(estimator=model, constraints="true_positive_rate_parity",
                                   objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
    optimizer.fit(legitimate_features(train), labels(train), sensitive_features=is_remote(train))
    return optimizer.predict(legitimate_features(test), sensitive_features=is_remote(test), random_state=SEED)


def expgrad(bound):
    def decide(train, test, share):
        model = ExponentiatedGradient(logistic_regression(), constraints=TruePositiveRateParity(difference_bound=bound),
                                      sample_weight_name="logisticregression__sample_weight")
        model.fit(legitimate_features(train), labels(train), sensitive_features=is_remote(train))
        return allocate(model._pmf_predict(legitimate_features(test))[:, 1], share)
    return decide


def committee_removal(removal):
    def decide(train, test, share):
        return allocate(CommitteeModel().fit(train, labels(train)).corrected_logit(test, removal), share)
    return decide


def pipeline(config):
    def decide(train, test, share):
        return FairPipeline(config, share).fit(train).predict(test)
    return decide


def default_candidates() -> list[Candidate]:
    return [
        Candidate("production RF, natural threshold (anchor)", "baseline", production_natural),
        Candidate("production RF at budget", "baseline", production_at_budget),
        Candidate("drop proxies only", "ablation", drop_proxies),
        Candidate("ThresholdOptimizer, natural rate", "post-processing", threshold_optimizer),
        *[Candidate(f"ExpGrad eps={bound}", "ExpGrad sweep", expgrad(bound), bound) for bound in EXPGRAD_BOUNDS],
        *[Candidate(f"committee, penalty removal={removal:.1f}", "penalty removal sweep", committee_removal(removal), removal)
          for removal in REMOVALS],
        *[Candidate(f"jury, {config.name}", "jury merit weight sweep", pipeline(config),
                    config.jury_weights.get("merit", 0.0)) for config in SEARCH_SPACE],
        Candidate(f"declared: {DECLARED_CONFIG.name}", "full pipeline", pipeline(DECLARED_CONFIG)),
    ]


@dataclass(frozen=True)
class ParetoReport:
    summary: pd.DataFrame
    table: pd.DataFrame
    figure: Figure


def pareto_report(history: pd.DataFrame, share: float, splits: int, workers: int) -> ParetoReport:
    summary = summarize(run(history, default_candidates(), share, splits, workers))
    summary["pareto"] = pareto_mask(summary, *PANELS[1])
    return ParetoReport(summary, summary[REPORT_COLUMNS], plot(summary, PANELS, splits))

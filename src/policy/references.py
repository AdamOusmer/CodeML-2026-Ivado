from .core import allocate
from .schema import COMMITTEE_FEATURES

DECLARED_INCOME_WEIGHT = 0.025
REVIEWER_INCOME_WEIGHTS = {"merit": 0.0, "need": -0.05, "legal": 0.0, "data_process": 0.19, "regional": 0.0}
REVIEWER_NAMES = tuple(REVIEWER_INCOME_WEIGHTS)
REFERENCE_INCOME_WEIGHTS = {"consensus": DECLARED_INCOME_WEIGHT, **REVIEWER_INCOME_WEIGHTS}
REFERENCE_JURORS = tuple(f"ref_{name}" for name in REVIEWER_NAMES)


def reference_scores(committee, df):
    return {name: committee.rule_score(df, weight) for name, weight in REFERENCE_INCOME_WEIGHTS.items()}


def reference_decisions(committee, df, share):
    return {name: allocate(score, share) for name, score in reference_scores(committee, df).items()}


def consensus_labels(committee, df, share):
    return allocate(committee.rule_score(df, DECLARED_INCOME_WEIGHT), share)


def reference_weights(committee):
    return {name: dict(zip(COMMITTEE_FEATURES, committee.rule_weights(weight)))
            for name, weight in REFERENCE_INCOME_WEIGHTS.items()}

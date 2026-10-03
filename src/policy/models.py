from dataclasses import dataclass

import pandas as pd

from .core import allocate, logistic_regression
from .jury import JurySettings, JuryOutcome, validate
from .label_correction import CommitteeModel, correct_labels, effective_removal
from .regions import is_remote
from .schema import scoring_features


@dataclass(frozen=True)
class Config:
    name: str
    removal: float = 1.0
    jury: JurySettings = JurySettings()


DECLARED_CONFIG = Config("validator jury, merit", jury=JurySettings(jurors=("merit",)))

OFFSET_BOUND = 0.10


def check_offset(offset):
    if not abs(offset) <= OFFSET_BOUND:
        raise ValueError(f"Offset {offset} exceeds the bound {OFFSET_BOUND}")


class FairPipeline:
    def __init__(self, config, share):
        self.config = config
        self.share = share

    def fit(self, history):
        self.label_correction_ = correct_labels(history, history["decision_octroi"].mean(), self.config.removal)
        self.main_model_ = logistic_regression().fit(scoring_features(history), self.label_correction_.labels)
        self.programme_stats_ = history.groupby("programme_etudes")["cote_r_equivalent"].agg(["mean", "std"])
        return self

    def model_probability(self, df):
        return self.main_model_.predict_proba(scoring_features(df))[:, 1]

    def juror_scores(self, df):
        merit = df["cote_r_equivalent"].to_numpy(dtype=float)
        stats = self.programme_stats_.reindex(df["programme_etudes"])
        mean = stats["mean"].to_numpy(dtype=float)
        std = stats["std"].to_numpy(dtype=float)
        return {"merit": merit, "programme_merit": (merit - mean) / std}

    def score(self, df, offset=0.0):
        check_offset(offset)
        p = self.model_probability(df)
        return p if offset == 0 else p + offset * is_remote(df)

    def decide(self, df, offset=0.0) -> JuryOutcome:
        check_offset(offset)
        probability = self.model_probability(df)
        ranking = probability if offset == 0 else probability + offset * is_remote(df)
        return validate(probability, self.juror_scores(df), int(round(self.share * len(df))),
                        self.config.jury, ranking=ranking)

    def predict(self, df, offset=0.0):
        return self.decide(df, offset).decisions

    def contributions(self, df):
        scaler, lr = self.main_model_[0], self.main_model_[-1]
        features = scoring_features(df)
        return pd.DataFrame(scaler.transform(features) * lr.coef_[0], columns=features.columns, index=df.index)


def reference_labels(history, target, share):
    committee = CommitteeModel().fit(history, history["decision_octroi"].to_numpy())
    references = {
        "corrected": allocate(committee.corrected_logit(target, effective_removal(committee.remote_penalty_, 1.0)), share),
        "merit": allocate(target["cote_r_equivalent"].to_numpy(), share),
    }
    if "decision_octroi" in target:
        references["historical"] = target["decision_octroi"].to_numpy()
    return references

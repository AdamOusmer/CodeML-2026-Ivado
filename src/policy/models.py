from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .core import allocate, logistic_regression
from .jury import JurySettings, JuryOutcome, validate
from .regions import is_remote
from .schema import committee_features, scoring_features


@dataclass(frozen=True)
class Config:
    name: str
    removal: float = 1.0
    jury: JurySettings = JurySettings()


DECLARED_CONFIG = Config("validator jury, merit", jury=JurySettings(jurors=("merit",)))


class CommitteeModel:
    def fit(self, df, y):
        X = committee_features(df)
        self.scaler_ = StandardScaler().fit(X)
        Z = np.column_stack([self.scaler_.transform(X), is_remote(df)])
        self.lr_ = LogisticRegression(max_iter=3000).fit(Z, y)
        self.remote_penalty_ = self.lr_.coef_[0][-1]
        return self

    def corrected_logit(self, df, removal=1.0):
        Z = self.scaler_.transform(committee_features(df))
        base = Z @ self.lr_.coef_[0][:-1] + self.lr_.intercept_[0]
        return base + (1.0 - removal) * self.remote_penalty_ * is_remote(df)


class FairPipeline:
    def __init__(self, config, share):
        self.config = config
        self.share = share

    def fit(self, history):
        self.committee_ = CommitteeModel().fit(history, history["decision_octroi"].to_numpy())
        corrected = allocate(self.committee_.corrected_logit(history, self.config.removal), self.share)
        self.main_model_ = logistic_regression().fit(scoring_features(history), corrected)
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
        p = self.model_probability(df)
        return p if offset == 0 else p + offset * is_remote(df)

    def decide(self, df, offset=0.0) -> JuryOutcome:
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
        "corrected": allocate(committee.corrected_logit(target), share),
        "merit": allocate(target["cote_r_equivalent"].to_numpy(), share),
    }
    if "decision_octroi" in target:
        references["historical"] = target["decision_octroi"].to_numpy()
    return references

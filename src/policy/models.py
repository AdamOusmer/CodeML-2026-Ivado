from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .core import allocate, legitimate_features, logistic_regression, percentile
from .regions import is_remote


@dataclass(frozen=True)
class Config:
    name: str
    removal: float = 1.0
    jury_weights: dict = field(default_factory=lambda: {"main_model": 0.5, "merit": 0.5})


DECLARED_CONFIG = Config("jury 50/50")


class CommitteeModel:
    def fit(self, df, y):
        X = legitimate_features(df)
        self.scaler_ = StandardScaler().fit(X)
        Z = np.column_stack([self.scaler_.transform(X), is_remote(df)])
        self.lr_ = LogisticRegression(max_iter=3000).fit(Z, y)
        self.remote_penalty_ = self.lr_.coef_[0][-1]
        return self

    def corrected_logit(self, df, removal=1.0):
        Z = self.scaler_.transform(legitimate_features(df))
        base = Z @ self.lr_.coef_[0][:-1] + self.lr_.intercept_[0]
        return base + (1.0 - removal) * self.remote_penalty_ * is_remote(df)


class FairPipeline:
    def __init__(self, config, share):
        self.config = config
        self.share = share

    def fit(self, history):
        self.committee_ = CommitteeModel().fit(history, history["decision_octroi"].to_numpy())
        corrected = allocate(self.committee_.corrected_logit(history, self.config.removal), self.share)
        self.main_model_ = logistic_regression().fit(legitimate_features(history), corrected)
        return self

    def model_probability(self, df):
        return self.main_model_.predict_proba(legitimate_features(df))[:, 1]

    def jury_score(self, df):
        votes = {
            "main_model": percentile(self.model_probability(df)),
            "merit": percentile(df["cote_r_equivalent"].to_numpy()),
        }
        return sum(weight * votes[juror] for juror, weight in self.config.jury_weights.items())

    def score(self, df, offset=0.0):
        jury = self.jury_score(df)
        return jury if offset == 0 else jury + offset * is_remote(df)

    def predict(self, df, offset=0.0):
        return allocate(self.score(df, offset), self.share)

    def contributions(self, df):
        scaler, lr = self.main_model_[0], self.main_model_[-1]
        features = legitimate_features(df)
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

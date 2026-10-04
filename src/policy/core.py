import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BUDGET_BOUNDS = (0.36, 0.44)


def production_features(train_df, target_df):
    categorical = ["programme_etudes", "region_administrative", "code_postal_3"]
    drop = ["id_candidat", "decision_octroi"]
    X = pd.get_dummies(train_df.drop(columns=drop, errors="ignore"), columns=categorical)
    Xt = pd.get_dummies(target_df.drop(columns=drop, errors="ignore"), columns=categorical)
    return X, Xt.reindex(columns=X.columns, fill_value=0)


def budget_share(history):
    share = history["decision_octroi"].mean()
    low, high = BUDGET_BOUNDS
    if not low <= share <= high:
        raise ValueError(f"Historical grant rate {share:.1%} is outside the allowed budget {low:.0%}-{high:.0%}")
    return share


def allocate(scores, share):
    k = int(round(share * len(scores)))
    decisions = np.zeros(len(scores), dtype=int)
    decisions[np.argsort(-scores, kind="stable")[:k]] = 1
    return decisions


def percentile(values):
    return rankdata(values) / len(values)


def auc(truth, scores):
    truth = np.asarray(truth)
    if truth.min() == truth.max():
        return np.nan
    return float(roc_auc_score(truth, scores))


def logistic_regression(c=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def signed_eo_gap(decisions, y_ref, remote):
    deserving = y_ref == 1
    centre, far = deserving & (remote == 0), deserving & (remote == 1)
    if not (centre.any() and far.any()):
        return np.nan
    return decisions[centre].mean() - decisions[far].mean()


def eo_gap(decisions, y_ref, remote):
    return abs(signed_eo_gap(decisions, y_ref, remote))

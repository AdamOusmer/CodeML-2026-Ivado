import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import auc
from .schema import COMMITTEE_FEATURES, SCORING_FEATURES, feature_frame

HOLDOUT_SHARE = 0.20
HOLDOUT_SEED = 0


MODEL_JURORS = {
    "lr_all": (COMMITTEE_FEATURES, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
    "boosting": (COMMITTEE_FEATURES, lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)),
    "forest": (COMMITTEE_FEATURES, lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=20, max_features=0.5, n_jobs=2, random_state=0)),
    "neural_net": (COMMITTEE_FEATURES, lambda: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=400, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=20, random_state=0))),
    "neighbours": (SCORING_FEATURES, lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(
        n_neighbors=75, weights="distance"))),
}


def fit_model_jurors(names, history, labels):
    models = {}
    for name in names:
        schema, build = MODEL_JURORS[name]
        models[name] = build().fit(feature_frame(history, schema), labels)
    return models


def holdout_auc(names, history, labels):
    labels = np.asarray(labels)
    fit_rows, holdout_rows = train_test_split(np.arange(len(history)), test_size=HOLDOUT_SHARE, stratify=labels,
                                              random_state=HOLDOUT_SEED)
    quality = {}
    for name in names:
        schema, build = MODEL_JURORS[name]
        features = feature_frame(history, schema)
        model = build().fit(features.iloc[fit_rows], labels[fit_rows])
        quality[name] = auc(labels[holdout_rows], model.predict_proba(features.iloc[holdout_rows])[:, 1])
    return quality


def model_juror_scores(models, df):
    return {name: model.predict_proba(feature_frame(df, MODEL_JURORS[name][0]))[:, 1] for name, model in models.items()}

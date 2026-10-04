import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .core import logistic_regression

PROBABILITY_FLOOR = 1e-9


def fit_main_model(features, labels):
    return logistic_regression().fit(features, labels)


def logit(probability):
    probability = np.clip(probability, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    return np.log(probability / (1 - probability))


def contributions(model: Pipeline, features):
    values = model[:-1].transform(features) * model[-1].coef_[0]
    return pd.DataFrame(values, columns=list(features.columns), index=features.index)

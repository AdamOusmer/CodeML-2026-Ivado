import numpy as np
import pandas as pd

PROGRAMMES = ["Arts et lettres", "Genie", "Sante", "Sciences", "Sciences sociales"]


def _programme_dummy(programme):
    def transform(df):
        return (df["programme_etudes"] == programme).astype(float)

    return transform


TRANSFORMS = {
    "cote_r": lambda df: df["cote_r_equivalent"],
    "log_revenu": lambda df: np.log(df["revenu_familial_estime"]),
    "heures_travail": lambda df: df["heures_travail_semaine"],
    "premiere_generation": lambda df: df["premiere_generation_universitaire"],
    **{f"prog_{programme}": _programme_dummy(programme) for programme in PROGRAMMES[1:]},
}

COMMITTEE_FEATURES = list(TRANSFORMS)
SCORING_FEATURES = ["cote_r", "log_revenu", "heures_travail"]


def feature_frame(df, schema):
    return pd.DataFrame({name: TRANSFORMS[name](df) for name in schema}, index=df.index)


def committee_features(df):
    return feature_frame(df, COMMITTEE_FEATURES)


def scoring_features(df):
    return feature_frame(df, SCORING_FEATURES)

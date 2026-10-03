from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.policy import PROGRAMMES, REGIONS

FEATURE_COLUMNS = (
    "id_candidat", "cote_r_equivalent", "programme_etudes", "region_administrative",
    "code_postal_3", "revenu_familial_estime", "heures_travail_semaine",
    "distance_domicile_campus_km", "premiere_generation_universitaire",
)
LABEL_COLUMN = "decision_octroi"
ID_PATTERN = r"C[0-9]{6}"
POSTAL_PATTERN = r"[A-Z][0-9][A-Z]"
CATEGORIES = {
    "programme_etudes": frozenset(PROGRAMMES),
    "region_administrative": frozenset(REGIONS),
}
BINARY_COLUMNS = ("premiere_generation_universitaire",)
MAX_REPORTED_ISSUES = 8


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Bound:
    low: float
    high: float | None = None
    strict_low: bool = False

    def accepts(self, values):
        values = np.asarray(values, dtype=float)
        above = values > self.low if self.strict_low else values >= self.low
        below = values <= self.high if self.high is not None else True
        return np.isfinite(values) & above & below

    def describe(self) -> str:
        if self.high is None:
            return f"> {self.low:g}" if self.strict_low else f">= {self.low:g}"
        return f"{self.low:g} to {self.high:g}"


NUMERIC_BOUNDS = {
    "cote_r_equivalent": Bound(15, 40),
    "revenu_familial_estime": Bound(0, strict_low=True),
    "heures_travail_semaine": Bound(0, 168),
    "distance_domicile_campus_km": Bound(0),
}


def required_columns(labeled: bool) -> set[str]:
    return set(FEATURE_COLUMNS) | ({LABEL_COLUMN} if labeled else set())


def binary_columns(labeled: bool) -> tuple[str, ...]:
    return BINARY_COLUMNS + ((LABEL_COLUMN,) if labeled else ())

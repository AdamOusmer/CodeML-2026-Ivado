from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.policy import BUDGET_BOUNDS, eo_gap, is_remote, reference_labels

Decide = Callable[[pd.DataFrame, pd.DataFrame, float], np.ndarray]
METRICS = ["grant_rate", "rate_centre", "rate_remote", "dp_gap", "eo_gap_corrected", "eo_gap_merit",
           "eo_gap_historical", "agree_corrected", "agree_merit", "acc_historical"]


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    decide: Decide
    setting: float = np.nan


def scaled_utility(decisions, reference, share):
    random_agreement = share**2 + (1 - share) ** 2
    return ((decisions == reference).mean() - random_agreement) / (1 - random_agreement)


def evaluate(name, family, decisions, references, remote, setting=np.nan):
    low, high = BUDGET_BOUNDS
    centre_rate, remote_rate = decisions[remote == 0].mean(), decisions[remote == 1].mean()
    return {
        "method": name,
        "family": family,
        "setting": setting,
        "grant_rate": decisions.mean(),
        "budget_ok": low <= decisions.mean() <= high,
        "rate_centre": centre_rate,
        "rate_remote": remote_rate,
        "dp_gap": abs(centre_rate - remote_rate),
        "eo_gap_corrected": eo_gap(decisions, references["corrected"], remote),
        "eo_gap_merit": eo_gap(decisions, references["merit"], remote),
        "eo_gap_historical": eo_gap(decisions, references["historical"], remote),
        "agree_corrected": (decisions == references["corrected"]).mean(),
        "agree_merit": (decisions == references["merit"]).mean(),
        "acc_historical": (decisions == references["historical"]).mean(),
    }


def evaluate_split(history: pd.DataFrame, share: float, seed: int, candidate: Candidate) -> dict:
    train, test = train_test_split(history, test_size=0.3, random_state=seed, stratify=history["decision_octroi"])
    decisions = candidate.decide(train, test, share)
    references = reference_labels(train, test, share)
    row = evaluate(candidate.name, candidate.family, decisions, references, is_remote(test), candidate.setting)
    return {**row, "seed": seed}


def run(history: pd.DataFrame, candidates: list[Candidate], share: float, splits: int, workers: int) -> pd.DataFrame:
    jobs = list(product(range(splits), candidates))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = pool.map(lambda job: evaluate_split(history, share, *job), jobs)
        return pd.DataFrame(list(rows))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["method", "family", "setting"], dropna=False, sort=False)
    summary = grouped[METRICS].mean()
    spread = grouped[METRICS].std().add_suffix("_std")
    budget_ok = grouped["budget_ok"].all()
    return pd.concat([summary, spread, budget_ok], axis=1).reset_index()

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .core import scaled_utility
from src.policy import CommitteeModel, Config, FairPipeline, JurySettings, allocate, eo_gap, is_remote, reference_labels

REFERENCES = ("corrected", "merit")
EQUITY_WEIGHT, UTILITY_WEIGHT = 20, 15

JURY_BANDS = (0.0125, 0.025, 0.05)
QUORUMS = (0.5, 1.0)
JUROR_SETS = (("merit",), ("merit", "programme_merit"))

SEARCH_SPACE = [
    Config(
        f"band {band}, quorum {quorum}, {'+'.join(jurors)}",
        removal=1.0,
        jury=JurySettings(band=band, quorum=quorum, jurors=jurors),
    )
    for band, quorum, jurors in product(JURY_BANDS, QUORUMS, JUROR_SETS)
    if not (quorum == 0.5 and len(jurors) == 1)
]


def score_split(history, share, config, seed):
    train, test = train_test_split(history, test_size=0.3, random_state=seed, stratify=history["decision_octroi"])
    remote = is_remote(test)
    references = reference_labels(train, test, share)
    committee = CommitteeModel().fit(train, train["decision_octroi"].to_numpy())
    committee_decisions = allocate(committee.corrected_logit(test, removal=0.0), share)
    decisions = FairPipeline(config, share).fit(train).predict(test)
    row = {"config": config.name}
    for name in REFERENCES:
        reference = references[name]
        gap = eo_gap(decisions, reference, remote)
        equity = 1 - gap / eo_gap(committee_decisions, reference, remote)
        utility = scaled_utility(decisions, reference, share)
        row[f"eo_gap_{name}"] = gap
        row[f"equity_{name}"] = equity
        row[f"utility_{name}"] = utility
        row[f"score_{name}"] = (EQUITY_WEIGHT * equity + UTILITY_WEIGHT * utility) / (EQUITY_WEIGHT + UTILITY_WEIGHT)
    return row


def tune(history: pd.DataFrame, share: float, configs: list[Config], splits: int = 5, workers: int = 4) -> pd.DataFrame:
    jobs = list(product(configs, range(splits)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda job: score_split(history, share, *job), jobs))
    table = pd.DataFrame(rows).groupby("config", sort=False).mean().reset_index()
    table["worst_case"] = np.minimum(table["score_corrected"], table["score_merit"])
    return table.sort_values("worst_case", ascending=False, ignore_index=True)

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .core import REVIEWER_REFERENCES, scaled_utility
from src.policy import (AUDIT_PANEL_CONFIG, CONSENSUS_PANEL_CONFIG, MODEL_JURORS, CommitteeModel, Config, FairPipeline,
                        JurySettings, allocate, eo_gap, is_remote, reference_labels)

REFERENCES = ("corrected", "merit", "consensus")
EQUITY_WEIGHT, UTILITY_WEIGHT = 20, 15

MODELS = tuple(MODEL_JURORS)
JURY_BANDS = (0.025, 0.05, 0.1)
QUORUMS = (0.5, 0.67, 0.8, 1.0)
JUROR_SETS = (("merit", "programme_merit"), ("merit", *MODELS), MODELS, ("hours_credit", *MODELS),
              ("merit", "hours_credit"))
BLENDS = ((), ("boosting", "neural_net"))

SEARCH_SPACE = [
    Config(
        f"band {band}, quorum {quorum}, {'+'.join(jurors)}, blend {'+'.join(blend) or 'none'}",
        removal=1.0,
        jury=JurySettings(band=band, quorum=quorum, jurors=jurors),
        blend=blend,
    )
    for band, quorum, jurors, blend in product(JURY_BANDS, QUORUMS, JUROR_SETS, BLENDS)
    if not (quorum < 1.0 and len(jurors) == 1)
]

SEARCH_SPACE += [CONSENSUS_PANEL_CONFIG, AUDIT_PANEL_CONFIG]


def superset_config(configs):
    names = dict.fromkeys(name for config in configs for name in (*config.jury.jurors, *config.blend))
    first = configs[0]
    return replace(first, jury=replace(first.jury, jurors=tuple(names)), blend=tuple(names))


def by_fit(configs):
    groups = {}
    for config in configs:
        key = (config.target, config.discounts, config.removal)
        groups.setdefault(key, []).append(config)
    return groups.values()


def score_split(history, share, configs, seed):
    train, test = train_test_split(history, test_size=0.3, random_state=seed, stratify=history["decision_octroi"])
    remote = is_remote(test)
    references = reference_labels(train, test, share)
    committee = CommitteeModel().fit(train, train["decision_octroi"].to_numpy())
    committee_decisions = allocate(committee.corrected_logit(test, removal=0.0), share)
    rows = []
    for group in by_fit(configs):
        pipeline = FairPipeline(superset_config(group), share).fit(train)
        for config in group:
            pipeline.config = config
            decisions = pipeline.predict(test)
            rows.append(score_decisions(config, decisions, committee_decisions, references, remote, share))
    return rows


def score_decisions(config, decisions, committee_decisions, references, remote, share):
    row = {"config": config.name,
           "eo_gap_reviewers_mean": float(np.mean([eo_gap(decisions, references[key], remote)
                                                   for key in REVIEWER_REFERENCES])),
           "agree_reviewers_mean": float(np.mean([(decisions == references[key]).mean()
                                                  for key in REVIEWER_REFERENCES]))}
    for name in REFERENCES:
        reference = references[name]
        gap = eo_gap(decisions, reference, remote)
        baseline_gap = eo_gap(committee_decisions, reference, remote)
        if not baseline_gap > 0:
            raise ValueError("no regional penalty: equity undefined")
        equity = 1 - gap / baseline_gap
        utility = scaled_utility(decisions, reference, share)
        row[f"eo_gap_{name}"] = gap
        row[f"equity_{name}"] = equity
        row[f"utility_{name}"] = utility
        row[f"score_{name}"] = (EQUITY_WEIGHT * equity + UTILITY_WEIGHT * utility) / (EQUITY_WEIGHT + UTILITY_WEIGHT)
    return row


def tune(history: pd.DataFrame, share: float, configs: list[Config], splits: int = 5, workers: int = 4) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_split = pool.map(lambda seed: score_split(history, share, configs, seed), range(splits))
        rows = [row for split_rows in per_split for row in split_rows]
    table = pd.DataFrame(rows).groupby("config", sort=False).mean().reset_index()
    table["worst_case"] = table[[f"score_{name}" for name in REFERENCES]].min(axis=1)
    return table.sort_values("worst_case", ascending=False, ignore_index=True)

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring import Verdict, graded, verdict
from src.policy import COMMITTEE_FEATURES, DECLARED_INCOME_WEIGHT, allocate, eo_gap, is_remote

from .consensus import consensus_guard, group_rate, reference_decisions

SUBGROUPS = ("programme_etudes", "premiere_generation_universitaire")
MIN_IMPACT_RATIO = 0.90
MAX_INCOME_COST_SHARE = 0.05
TOLERANCE = 0.012


def region_impact(decisions, batch):
    rates = pd.Series(decisions).groupby(batch["region_administrative"].to_numpy()).mean()
    return float(rates.min() / rates.max())


def worst_subgroup_gap(decisions, batch, remote):
    gaps = [abs(group_rate(decisions, (batch[column] == value).to_numpy() & ~remote)
                - group_rate(decisions, (batch[column] == value).to_numpy() & remote))
            for column in SUBGROUPS for value in batch[column].unique()]
    return float(np.nanmax(gaps))


def five_region_merit_gap(decisions, batch, merit):
    regions = batch["region_administrative"].to_numpy()
    rates = [decisions[merit & (regions == region)].mean() for region in np.unique(regions)]
    return float(max(rates) - min(rates))


def income_cost_share(committee, history, batch, remote):
    income = np.log(batch["revenu_familial_estime"].to_numpy(dtype=float))
    reference = np.log(history["revenu_familial_estime"].to_numpy(dtype=float))
    z = (income - reference.mean()) / reference.std()
    gap = abs(z[~remote].mean() - z[remote].mean())
    merit_weight = abs(committee.lr_.coef_[0][COMMITTEE_FEATURES.index("cote_r")])
    return float(DECLARED_INCOME_WEIGHT * merit_weight * gap / abs(committee.remote_penalty_))


def strong_measurer(pipeline, history, batch, share):
    committee = pipeline.committee_
    remote = is_remote(batch).astype(bool)
    references = reference_decisions(committee, batch, share)
    references["corrected"] = allocate(committee.corrected_logit(batch), share)
    rule = references["consensus"]
    merit = allocate(batch["cote_r_equivalent"].to_numpy(dtype=float), share) == 1
    rule_gaps = {name: eo_gap(rule, reference, remote.astype(int)) for name, reference in references.items()}
    rule_values = (region_impact(rule, batch), worst_subgroup_gap(rule, batch, remote),
                   five_region_merit_gap(rule, batch, merit))
    income_cost = income_cost_share(committee, history, batch, remote)

    def measure(decisions) -> Verdict:
        decisions = np.asarray(decisions)
        guard = consensus_guard(committee, history, batch, share, decisions)
        not_ok = [check.name for check in guard.checks if check.status != "OK"]
        excess = {name: eo_gap(decisions, reference, remote.astype(int)) - rule_gaps[name]
                  for name, reference in references.items()}
        worst_reference = max(excess, key=excess.get)
        impact, subgroup, five_region = (region_impact(decisions, batch), worst_subgroup_gap(decisions, batch, remote),
                                         five_region_merit_gap(decisions, batch, merit))
        impact_floor = max(MIN_IMPACT_RATIO, rule_values[0] - TOLERANCE)
        return verdict([
            graded("strong: consensus guard checks not OK", float(len(not_ok)), 0, 0, threshold="all OK",
                   detail=", ".join(not_ok)),
            graded("strong: region impact ratio", impact, impact_floor, impact_floor, higher_is_worse=False,
                   threshold=f">= max({MIN_IMPACT_RATIO}, consensus rule {rule_values[0]:.4f} - {TOLERANCE})"),
            graded("strong: worst programme / first-generation subgroup gap", subgroup, rule_values[1] + TOLERANCE,
                   rule_values[1] + TOLERANCE, threshold=f"<= consensus rule {rule_values[1]:.4f} + {TOLERANCE}"),
            graded("strong: five-region merit gap", five_region, rule_values[2] + TOLERANCE,
                   rule_values[2] + TOLERANCE, threshold=f"<= consensus rule {rule_values[2]:.4f} + {TOLERANCE}"),
            graded("strong: worst reference gap beyond the consensus rule", excess[worst_reference], TOLERANCE, TOLERANCE,
                   threshold=f"<= {TOLERANCE}", detail=worst_reference),
            graded("strong: income cost share of the removed penalty", income_cost, MAX_INCOME_COST_SHARE,
                   MAX_INCOME_COST_SHARE, threshold=f"<= {MAX_INCOME_COST_SHARE}"),
        ])

    return measure


def strong_guard(pipeline, history, batch, share, decisions) -> Verdict:
    if not pipeline.config.strong_guard:
        return Verdict("OK", [])
    return strong_measurer(pipeline, history, batch, share)(decisions)

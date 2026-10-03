from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.explain import explain
from src.monitoring import EO_GAP_ALERT, Verdict, run_checks, verdict
from src.policy import DECLARED_CONFIG, Config, FairPipeline, budget_share, eo_gap, is_remote, reference_labels

from .record import Action, ActionKind, DecisionRecord, InputError

OFFSET_GRID = np.linspace(-0.10, 0.10, 41).round(3)
NO_OFFSET_REASON = "no offset in the allowed grid lowers the gap"
MERIT_ONLY_SUGGESTION = "review data drift, then consider a merit-only policy"
JURY_REASONS = ("near_cutoff", "low_confidence", "disagreement")

logger = get_logger("harness")


def jury_counts(outcome) -> dict:
    return {
        "triggered": int(outcome.triggered.sum()),
        "overturned_out": len(outcome.overturned_out),
        "overturned_in": len(outcome.overturned_in),
        "reasons": {name: sum(name in reasons for reasons in outcome.reasons) for name in JURY_REASONS},
    }


def audit(history, batch, decisions, eo_gap_alert) -> Verdict:
    result = verdict(run_checks(history, batch, decisions, eo_gap_alert=eo_gap_alert))
    logger.info("Audit: %s", result.status)
    return result


def fit_offset(pipeline: FairPipeline, history, batch, share) -> float:
    references = reference_labels(history, batch, share)
    remote = is_remote(batch)

    def worst_gap(offset):
        decisions = pipeline.predict(batch, offset)
        gaps = [eo_gap(decisions, references[name], remote) for name in ("corrected", "merit")]
        return np.inf if np.isnan(gaps).any() else round(max(gaps), 3)

    return float(min(OFFSET_GRID, key=lambda offset: (worst_gap(offset), abs(offset), -offset)))


def alerting(result: Verdict) -> list[str]:
    return [check.name for check in result.checks if check.status == "ALERT"]


def check_ids(batch: pd.DataFrame) -> None:
    ids = batch["id_candidat"]
    if ids.isna().any():
        raise InputError("batch has missing id_candidat values")
    if ids.duplicated().any():
        raise InputError(f"batch has {int(ids.duplicated().sum()):,} duplicated id_candidat values")


def decide(history: pd.DataFrame, batch: pd.DataFrame, input_hashes: dict[str, str],
           config: Config = DECLARED_CONFIG, eo_gap_alert: float = EO_GAP_ALERT) -> DecisionRecord:
    check_ids(batch)
    try:
        share = budget_share(history)
    except ValueError as error:
        raise InputError(str(error)) from error
    actions = [Action(ActionKind.SELECT_CONFIG, "declared configuration", {"config": config.name})]
    pipeline = FairPipeline(config, share).fit(history)
    offset = 0.0
    outcome = pipeline.decide(batch)
    decisions = outcome.decisions
    verdicts = [audit(history, batch, decisions, eo_gap_alert)]
    moved_ids: list[str] = []

    block_reason = "alert remains after allowed corrections"
    if verdicts[-1].correctable:
        offset = fit_offset(pipeline, history, batch, share)
        if offset == 0.0:
            block_reason = NO_OFFSET_REASON
        else:
            adjusted = pipeline.decide(batch, offset)
            moved_ids = batch["id_candidat"][adjusted.decisions != decisions].tolist()
            actions.append(Action(ActionKind.ADJUST_OFFSET, "opportunity gap alert",
                                  {"offset": offset, "moved": len(moved_ids), "alerts": alerting(verdicts[-1])}))
            outcome = adjusted
            decisions = adjusted.decisions
            verdicts.append(audit(history, batch, decisions, eo_gap_alert))

    status = "blocked" if verdicts[-1].status == "ALERT" else "published"
    if status == "blocked":
        actions.append(Action(ActionKind.BLOCK, block_reason,
                              {"checks": alerting(verdicts[-1]), "suggestion": MERIT_ONLY_SUGGESTION}))

    scores = pipeline.score(batch, offset)
    region_rates = pd.Series(decisions).groupby(batch["region_administrative"].to_numpy()).mean().round(4).to_dict()
    jury = jury_counts(outcome)
    logger.info("Jury: %s", jury)
    return DecisionRecord(config, share, status, batch["id_candidat"], decisions, scores, offset, verdicts, actions,
                          moved_ids, explain(pipeline, batch, outcome, scores, offset), region_rates, input_hashes,
                          jury)

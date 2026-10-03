from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.explain import explain
from src.monitoring import EO_GAP_ALERT, Verdict, run_checks, verdict
from src.policy import DECLARED_CONFIG, Config, FairPipeline, budget_share, eo_gap, is_remote, reference_labels

from .record import Action, ActionKind, DecisionRecord

OFFSET_GRID = np.linspace(-0.10, 0.10, 41)
NO_OFFSET_REASON = "no offset in the allowed grid lowers the gap"
MERIT_ONLY_SUGGESTION = "review data drift, then consider the merit-only policy (jury weight merit = 1)"

logger = get_logger("harness")


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
        raise ValueError("batch has missing id_candidat values")
    if ids.duplicated().any():
        raise ValueError(f"batch has {int(ids.duplicated().sum()):,} duplicated id_candidat values")


def decide(history: pd.DataFrame, batch: pd.DataFrame, input_hashes: dict[str, str],
           config: Config = DECLARED_CONFIG, eo_gap_alert: float = EO_GAP_ALERT) -> DecisionRecord:
    check_ids(batch)
    share = budget_share(history)
    actions = [Action(ActionKind.SELECT_CONFIG, "declared configuration", {"config": config.name})]
    pipeline = FairPipeline(config, share).fit(history)
    offset = 0.0
    decisions = pipeline.predict(batch)
    verdicts = [audit(history, batch, decisions, eo_gap_alert)]
    moved_ids: list[str] = []

    block_reason = "alert remains after allowed corrections"
    if verdicts[-1].correctable:
        offset = fit_offset(pipeline, history, batch, share)
        if offset == 0.0:
            block_reason = NO_OFFSET_REASON
        else:
            adjusted = pipeline.predict(batch, offset)
            moved_ids = batch["id_candidat"][adjusted != decisions].tolist()
            actions.append(Action(ActionKind.ADJUST_OFFSET, "opportunity gap alert",
                                  {"offset": offset, "moved": len(moved_ids), "alerts": alerting(verdicts[-1])}))
            decisions = adjusted
            verdicts.append(audit(history, batch, decisions, eo_gap_alert))

    status = "blocked" if verdicts[-1].status == "ALERT" else "published"
    if status == "blocked":
        actions.append(Action(ActionKind.BLOCK, block_reason,
                              {"checks": alerting(verdicts[-1]), "suggestion": MERIT_ONLY_SUGGESTION}))

    scores = pipeline.score(batch, offset)
    region_rates = pd.Series(decisions).groupby(batch["region_administrative"].to_numpy()).mean().round(4).to_dict()
    return DecisionRecord(config, share, status, batch["id_candidat"], decisions, scores, offset, verdicts, actions,
                          moved_ids, explain(pipeline, batch, decisions, scores, offset), region_rates, input_hashes)

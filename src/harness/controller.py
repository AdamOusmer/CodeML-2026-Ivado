from __future__ import annotations

import pandas as pd

from src.common.logging import get_logger
from src.explain import explain
from src.monitoring import EO_GAP_ALERT
from src.policy import DECLARED_CONFIG, NO_PENALTY_WARNING, Config, FairPipeline, budget_share

from .postprocessing import POSTPROCESSING_NODES, alerting, postprocessing_graph
from .record import Action, ActionKind, DecisionRecord, InputError

SUBMISSION_REASON = "submission guard rejected the output"
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


def check_ids(batch: pd.DataFrame) -> None:
    ids = batch["id_candidat"]
    if ids.isna().any():
        raise InputError("batch has missing id_candidat values")
    if ids.duplicated().any():
        raise InputError(f"batch has {int(ids.duplicated().sum()):,} duplicated id_candidat values")


def block_action(first_verdict, final_verdict, offset, issues) -> Action:
    checks = alerting(final_verdict)
    if not checks:
        return Action(ActionKind.BLOCK, SUBMISSION_REASON, {"checks": ["submission"], "issues": issues})
    reason = NO_OFFSET_REASON if first_verdict.correctable and offset == 0 else "alert remains after allowed corrections"
    params = {"checks": checks, "suggestion": MERIT_ONLY_SUGGESTION}
    if issues:
        params = {"checks": [*checks, "submission"], "issues": issues, "suggestion": MERIT_ONLY_SUGGESTION}
    return Action(ActionKind.BLOCK, reason, params)


def decide(history: pd.DataFrame, batch: pd.DataFrame, input_hashes: dict[str, str],
           config: Config = DECLARED_CONFIG, eo_gap_alert: float = EO_GAP_ALERT,
           nodes=POSTPROCESSING_NODES) -> DecisionRecord:
    check_ids(batch)
    try:
        share = budget_share(history)
    except ValueError as error:
        raise InputError(str(error)) from error
    actions = [Action(ActionKind.SELECT_CONFIG, "declared configuration", {"config": config.name})]
    pipeline = FairPipeline(config, share).fit(history)
    seeds = {"pipeline": pipeline, "history": history, "batch": batch, "share": share, "eo_gap_alert": eo_gap_alert}
    artifacts = postprocessing_graph(nodes).run(seeds, workers=1)
    outcome, final_outcome = artifacts["outcome"], artifacts["final_outcome"]
    offset, issues = artifacts["offset"], artifacts["submission"]
    first_verdict, final_verdict = artifacts["verdict"], artifacts["final_verdict"]
    ids = batch["id_candidat"]
    verdicts = [first_verdict] if offset == 0 else [first_verdict, final_verdict]
    offset_moved_ids = ids[final_outcome.proposed != outcome.proposed].tolist()
    jury_moved_ids = ids[final_outcome.decisions != final_outcome.proposed].tolist()

    if offset != 0:
        actions.append(Action(ActionKind.ADJUST_OFFSET, "opportunity gap alert",
                              {"offset": offset, "moved": int((final_outcome.decisions != outcome.decisions).sum()),
                               "offset_moved": len(offset_moved_ids), "alerts": alerting(first_verdict)}))
    if artifacts["status"] == "blocked":
        actions.append(block_action(first_verdict, final_verdict, offset, issues))

    scores = pipeline.score(batch, offset)
    region_rates = (pd.Series(final_outcome.decisions).groupby(batch["region_administrative"].to_numpy())
                    .mean().round(4).to_dict())
    jury = jury_counts(final_outcome)
    logger.info("Jury: %s", jury)
    return DecisionRecord(config, share, artifacts["status"], ids, final_outcome.decisions, final_outcome.proposed,
                          scores, offset, verdicts, actions, issues, offset_moved_ids, jury_moved_ids,
                          explain(pipeline, batch, final_outcome, scores, offset), region_rates, input_hashes, jury,
                          [NO_PENALTY_WARNING] if pipeline.label_correction_.penalty >= 0 else [])

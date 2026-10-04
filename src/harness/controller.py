from __future__ import annotations

import pandas as pd

from src.common.logging import get_logger
from src.explain import explain
from src.monitoring import EO_GAP_ALERT
from src.policy import DECLARED_CONFIG, NO_PENALTY_WARNING, Config, FairPipeline, budget_share, reference_weights

from .postprocessing import POSTPROCESSING_NODES, alerting, postprocessing_graph
from .reasoning_gate import reasoning_gate
from .record import Action, ActionKind, DecisionRecord, InputError
from .strong_guard import strong_measurer

OUTPUT_REASON = "output guard rejected the output"
JURY_GUARD_REASON = "jury guard alert, jury outcome not published"
CONSENSUS_REASON = "consensus guard rejected the output"
STRONG_REASON = "strong fairness guard rejected the output"
NO_OFFSET_REASON ="no offset in the allowed grid lowers the gap"
MERIT_ONLY_SUGGESTION = "review data drift, then consider a merit-only policy"
JURY_REASONS =("near_cutoff", "low_confidence", "disagreement")

logger = get_logger("harness")


def jury_counts(outcome) -> dict:
    return {
        "triggered": int(outcome.triggered.sum()),
        "overturned_out": len(outcome.overturned_out),
        "overturned_in": len(outcome.overturned_in),
        "applied": not outcome.audit_only,
        "reasons": {name: sum(name in reasons for reasons in outcome.reasons) for name in JURY_REASONS},
    }


def check_ids(batch: pd.DataFrame) -> None:
    ids = batch["id_candidat"]
    if ids.isna().any():
        raise InputError("batch has missing id_candidat values")
    if ids.duplicated().any():
        raise InputError(f"batch has {int(ids.duplicated().sum()):,} duplicated id_candidat values")


def training_report(pipeline, history, batch) -> dict:
    historical = history["decision_octroi"].to_numpy()
    labels = pipeline.training_labels_
    config = pipeline.config
    report = {"target": config.target, "grants": int(labels.sum()),
              "differs_from_committee": int((labels != historical).sum()),
              "differs_from_corrected": int((labels != pipeline.label_correction_.labels).sum()),
              "reference_weights": reference_weights(pipeline.committee_),
              "committee_hours_over_r": pipeline.committee_.hours_over_r()}
    if config.strong_guard:
        report["learned_ratios"] = pipeline.learned_ratios(batch)
    return report


def strong_values(measure, decisions) -> dict:
    return {check.name.removeprefix("strong: "): round(check.value, 4) for check in measure(decisions).checks}


def deliberation_report(pipeline, history, batch, share, outcome) -> dict | None:
    deliberation = outcome.deliberation
    if deliberation is None:
        return None
    report = dict(deliberation.counts)
    if pipeline.config.strong_guard and pipeline.config.reasoning.apply_moves:
        measure = strong_measurer(pipeline, history, batch, share)
        report |= {"acceptance_rule": "moves accepted under the strong-guard repair rule: no monitoring or consensus check "
                                      "worse, and strong guard OK or strictly less total excess without a new failure",
                   "strong_before": strong_values(measure, deliberation.before),
                   "strong_after": strong_values(measure, deliberation.decisions)}
    return report


def block_action(correctable, final_verdict, offset, issues, consensus_verdict, strong_verdict) -> Action:
    audit_alerts, consensus_alerts = alerting(final_verdict), alerting(consensus_verdict)
    strong_alerts = alerting(strong_verdict)
    checks = [*audit_alerts, *consensus_alerts, *strong_alerts]
    if not checks:
        return Action(ActionKind.BLOCK, OUTPUT_REASON, {"checks": ["output"], "issues": issues})
    if not audit_alerts:
        reason = CONSENSUS_REASON if consensus_alerts else STRONG_REASON
    else:
        reason = NO_OFFSET_REASON if correctable and offset == 0 else "alert remains after allowed corrections"
    params = {"checks": checks, "suggestion": MERIT_ONLY_SUGGESTION}
    if issues:
        params = {"checks": [*checks, "output"], "issues": issues, "suggestion": MERIT_ONLY_SUGGESTION}
    return Action(ActionKind.BLOCK, reason, params)


def decide(history: pd.DataFrame, batch: pd.DataFrame, input_hashes: dict[str, str],
           config: Config = DECLARED_CONFIG, residual: pd.Series | None = None, eo_gap_alert: float = EO_GAP_ALERT,
           nodes=POSTPROCESSING_NODES) -> DecisionRecord:
    check_ids(batch)
    try:
        share = budget_share(history)
    except ValueError as error:
        raise InputError(str(error)) from error
    actions = [Action(ActionKind.SELECT_CONFIG, "declared configuration", {"config": config.name})]
    pipeline = FairPipeline(config, share, residual).fit(history)
    if config.reasoning and config.reasoning.apply_moves:
        pipeline.guards = reasoning_gate(pipeline, history, batch, share, eo_gap_alert)
    seeds = {"pipeline": pipeline, "history": history, "batch": batch, "share": share, "eo_gap_alert": eo_gap_alert,
             "corrected_enforced": config.target == "corrected"}
    artifacts = postprocessing_graph(nodes).run(seeds, workers=1)
    outcome, shifted_outcome, final_outcome = artifacts["outcome"], artifacts["final_outcome"], artifacts["released"]
    first_offset, offset = artifacts["offset"], artifacts["released_offset"]
    issues, guard = artifacts["output"], artifacts["guard"]
    first_verdict, final_verdict = artifacts["verdict"], artifacts["released_verdict"]
    reverted = guard.status == "ALERT"
    ids = batch["id_candidat"]
    verdicts = [first_verdict] if first_offset == 0 else [first_verdict, artifacts["final_verdict"]]
    if reverted:
        verdicts.append(final_verdict)
    offset_moved_ids = ids[final_outcome.proposed != outcome.proposed].tolist()
    jury_moved_ids = ids[final_outcome.decisions != final_outcome.proposed].tolist()

    if first_offset != 0:
        alerts = [*alerting(first_verdict), *alerting(artifacts["first_consensus"])]
        actions.append(Action(ActionKind.ADJUST_OFFSET, "opportunity gap alert",
                              {"offset": first_offset,
                               "moved": int((shifted_outcome.decisions != outcome.decisions).sum()),
                               "offset_moved": len(offset_moved_ids), "alerts": alerts}))
    if reverted:
        actions.append(Action(ActionKind.REVERT_JURY, JURY_GUARD_REASON,
                              {"checks": alerting(guard), "offset": offset}))
    if artifacts["status"] == "blocked":
        actions.append(block_action(artifacts["correctable"], final_verdict, offset, issues, artifacts["consensus"],
                                    artifacts["strong"]))

    scores = pipeline.score(batch, offset)
    region_rates = (pd.Series(final_outcome.decisions).groupby(batch["region_administrative"].to_numpy())
                    .mean().round(4).to_dict())
    jury = jury_counts(final_outcome)
    logger.info("Jury: %s", jury)
    return DecisionRecord(config, share, artifacts["status"], ids, final_outcome.decisions, final_outcome.proposed,
                          scores, offset, verdicts, actions, issues, offset_moved_ids, jury_moved_ids,
                          explain(pipeline, batch, final_outcome, scores, offset), region_rates, input_hashes, jury,
                          guard, artifacts["consensus"], [NO_PENALTY_WARNING] if pipeline.label_correction_.penalty >= 0 else [],
                          training_labels=training_report(pipeline, history, batch),
                          strong_guard=artifacts["strong"] if config.strong_guard else None,
                          deliberation=deliberation_report(pipeline, history, batch, share, final_outcome))

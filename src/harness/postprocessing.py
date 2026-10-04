from __future__ import annotations

import numpy as np

from src.common.graph import Graph, Node
from src.common.logging import get_logger
from src.monitoring import EO_GAP_ALERT, Verdict, opportunity_checks, run_checks, verdict as summarise
from src.policy import OFFSET_BOUND, FairPipeline, is_remote, reference_labels

from .consensus import consensus_guard
from .jury_guard import jury_guard
from .strong_guard import strong_guard
from .output_guard import output_issues

OFFSET_GRID = np.linspace(-OFFSET_BOUND, OFFSET_BOUND, 41).round(3)

logger = get_logger("harness")


def audit(history, batch, decisions, eo_gap_alert, corrected_enforced) -> Verdict:
    result = summarise(run_checks(history, batch, decisions, eo_gap_alert=eo_gap_alert,
                                  corrected_enforced=corrected_enforced))
    logger.info("Audit: %s", result.status)
    return result


def fit_offset(pipeline: FairPipeline, history, batch, share, eo_gap_alert=EO_GAP_ALERT, jury=True) -> float:
    references = reference_labels(history, batch, share)
    remote = is_remote(batch)

    def violation(offset):
        decisions = pipeline.decide(batch, offset, jury=jury).decisions
        checks = [*opportunity_checks(decisions, references, remote, eo_gap_alert,
                                      pipeline.config.target == "corrected"),
                  *consensus_guard(pipeline.committee_, history, batch, share, decisions).checks]
        return max((check.excess for check in checks if check.correctable), default=0.0)

    return float(min(OFFSET_GRID, key=lambda offset: (max(violation(offset), 0.0), abs(offset), -offset)))


def alerting(result: Verdict) -> list[str]:
    return [check.name for check in result.checks if check.status == "ALERT"]


def initial_outcome(pipeline, batch):
    return pipeline.decide(batch)


def initial_verdict(history, batch, outcome, eo_gap_alert, corrected_enforced):
    return audit(history, batch, outcome.decisions, eo_gap_alert, corrected_enforced)


def initial_consensus(pipeline, history, batch, share, outcome):
    return consensus_guard(pipeline.committee_, history, batch, share, outcome.decisions)


def correctable_alerts(first_verdict, first_consensus):
    alerts = [check for result in (first_verdict, first_consensus) for check in result.checks if check.status == "ALERT"]
    return bool(alerts) and all(check.correctable for check in alerts)


def choose_offset(pipeline, history, batch, share, correctable, eo_gap_alert):
    return fit_offset(pipeline, history, batch, share, eo_gap_alert) if correctable else 0.0


def shifted_outcome(pipeline, batch, offset, outcome):
    return outcome if offset == 0 else pipeline.decide(batch, offset)


def shifted_verdict(history, batch, final_outcome, offset, first_verdict, eo_gap_alert, corrected_enforced):
    if offset == 0:
        return first_verdict
    return audit(history, batch, final_outcome.decisions, eo_gap_alert, corrected_enforced)


def guard_jury(pipeline, history, batch, share, final_outcome):
    result = jury_guard(pipeline, history, batch, share, final_outcome)
    logger.info("Jury guard: %s", result.status)
    return result


def released_offset(pipeline, history, batch, share, offset, guard, eo_gap_alert):
    if guard.status != "ALERT":
        return offset
    return fit_offset(pipeline, history, batch, share, eo_gap_alert, jury=False)


def released_outcome(pipeline, batch, released_shift, final_outcome, guard):
    return pipeline.decide(batch, released_shift, jury=False) if guard.status == "ALERT" else final_outcome


def released_verdict(history, batch, released, final_outcome, final_verdict, eo_gap_alert, corrected_enforced):
    if released is final_outcome:
        return final_verdict
    return audit(history, batch, released.decisions, eo_gap_alert, corrected_enforced)


def guard_consensus(pipeline, history, batch, share, released):
    result = consensus_guard(pipeline.committee_, history, batch, share, released.decisions)
    logger.info("Consensus guard: %s", result.status)
    return result


def guard_strong(pipeline, history, batch, share, released):
    result = strong_guard(pipeline, history, batch, share, released.decisions)
    logger.info("Strong guard: %s", result.status)
    return result


def check_output(batch, released, share):
    k = int(round(share * len(batch)))
    return output_issues(batch, batch["id_candidat"], released.decisions, k)


def publication_status(final, issues, consensus, strong):
    alerting_layers = (final.status, consensus.status, strong.status)
    return "blocked" if "ALERT" in alerting_layers or issues else "published"


POSTPROCESSING_NODES = (
    Node("outcome", initial_outcome, ("pipeline", "batch")),
    Node("verdict", initial_verdict, ("history", "batch", "outcome", "eo_gap_alert", "corrected_enforced")),
    Node("first_consensus", initial_consensus, ("pipeline", "history", "batch", "share", "outcome")),
    Node("correctable", correctable_alerts, ("verdict", "first_consensus")),
    Node("offset", choose_offset, ("pipeline", "history", "batch", "share", "correctable", "eo_gap_alert")),
    Node("final_outcome", shifted_outcome, ("pipeline", "batch", "offset", "outcome")),
    Node("final_verdict", shifted_verdict,
         ("history", "batch", "final_outcome", "offset", "verdict", "eo_gap_alert", "corrected_enforced")),
    Node("guard", guard_jury, ("pipeline", "history", "batch", "share", "final_outcome")),
    Node("released_offset", released_offset,
         ("pipeline", "history", "batch", "share", "offset", "guard", "eo_gap_alert")),
    Node("released", released_outcome, ("pipeline", "batch", "released_offset", "final_outcome", "guard")),
    Node("released_verdict", released_verdict,
         ("history", "batch", "released", "final_outcome", "final_verdict", "eo_gap_alert", "corrected_enforced")),
    Node("consensus", guard_consensus, ("pipeline", "history", "batch", "share", "released")),
    Node("strong", guard_strong, ("pipeline", "history", "batch", "share", "released")),
    Node("output", check_output, ("batch", "released", "share"),
         after=("released_verdict", "consensus", "strong")),
    Node("status", publication_status, ("released_verdict", "output", "consensus", "strong")),
)


def postprocessing_graph(nodes=POSTPROCESSING_NODES) -> Graph:
    return Graph(nodes)

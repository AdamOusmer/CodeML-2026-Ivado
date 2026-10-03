from __future__ import annotations

import numpy as np

from src.common.graph import Graph, Node
from src.common.logging import get_logger
from src.monitoring import Verdict, run_checks, verdict as summarise
from src.policy import OFFSET_BOUND, FairPipeline, eo_gap, is_remote, reference_labels

from .submission import submission_issues

OFFSET_GRID = np.linspace(-OFFSET_BOUND, OFFSET_BOUND, 41).round(3)

logger = get_logger("harness")


def audit(history, batch, decisions, eo_gap_alert) -> Verdict:
    result = summarise(run_checks(history, batch, decisions, eo_gap_alert=eo_gap_alert))
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


def initial_outcome(pipeline, batch):
    return pipeline.decide(batch)


def initial_verdict(history, batch, outcome, eo_gap_alert):
    return audit(history, batch, outcome.decisions, eo_gap_alert)


def choose_offset(pipeline, history, batch, share, first_verdict):
    return fit_offset(pipeline, history, batch, share) if first_verdict.correctable else 0.0


def shifted_outcome(pipeline, batch, offset, outcome):
    return outcome if offset == 0 else pipeline.decide(batch, offset)


def shifted_verdict(history, batch, final_outcome, offset, first_verdict, eo_gap_alert):
    if offset == 0:
        return first_verdict
    return audit(history, batch, final_outcome.decisions, eo_gap_alert)


def check_submission(batch, final_outcome, share):
    k = int(round(share * len(batch)))
    return submission_issues(batch, batch["id_candidat"], final_outcome.decisions, k)


def publication_status(final_verdict, issues):
    return "blocked" if final_verdict.status == "ALERT" or issues else "published"


POSTPROCESSING_NODES = (
    Node("outcome", initial_outcome, ("pipeline", "batch")),
    Node("verdict", initial_verdict, ("history", "batch", "outcome", "eo_gap_alert")),
    Node("offset", choose_offset, ("pipeline", "history", "batch", "share", "verdict")),
    Node("final_outcome", shifted_outcome, ("pipeline", "batch", "offset", "outcome")),
    Node("final_verdict", shifted_verdict,
         ("history", "batch", "final_outcome", "offset", "verdict", "eo_gap_alert")),
    Node("submission", check_submission, ("batch", "final_outcome", "share"), after=("final_verdict",)),
    Node("status", publication_status, ("final_verdict", "submission")),
)


def postprocessing_graph(nodes=POSTPROCESSING_NODES) -> Graph:
    return Graph(nodes)

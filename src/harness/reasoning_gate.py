from __future__ import annotations

from src.monitoring import EO_GAP_ALERT, opportunity_checks
from src.policy import is_remote, reference_labels

from .consensus import consensus_guard
from .strong_guard import strong_measurer

STATUS_RANK = {"OK": 0, "WARN": 1, "ALERT": 2}
EXCESS_TOLERANCE = 1e-9


def guard_checks(pipeline, history, batch, share, eo_gap_alert=EO_GAP_ALERT):
    references = reference_labels(history, batch, share)
    remote = is_remote(batch)
    corrected_enforced = pipeline.config.target == "corrected"

    def checks(decisions):
        return [*opportunity_checks(decisions, references, remote, eo_gap_alert, corrected_enforced),
                *consensus_guard(pipeline.committee_, history, batch, share, decisions).checks]

    return checks


def worse(after, before):
    return STATUS_RANK[after.status] > STATUS_RANK[before.status] or after.excess > before.excess + EXCESS_TOLERANCE


def strong_failures(result):
    return {check.name: check.excess for check in result.checks if check.status == "ALERT"}


class ReasoningGate:
    def __init__(self, checks, strong, before):
        self.checks = checks
        self.strong = strong
        self.baseline = {check.name: check for check in checks(before)}
        self.failures = {} if strong is None else strong_failures(strong(before))

    def admits(self, candidate, current):
        if any(worse(check, self.baseline[check.name]) for check in self.checks(candidate)
               if check.name in self.baseline):
            return False
        if self.strong is None:
            return True
        failures = strong_failures(self.strong(candidate))
        if not failures or (set(failures) <= set(self.failures) and sum(failures.values()) < sum(self.failures.values())):
            self.failures = failures
            return True
        return False

    def settled(self, current):
        return not self.failures


def reasoning_gate(pipeline, history, batch, share, eo_gap_alert=EO_GAP_ALERT):
    checks = guard_checks(pipeline, history, batch, share, eo_gap_alert)
    strong = strong_measurer(pipeline, history, batch, share) if pipeline.config.strong_guard else None
    return lambda before: ReasoningGate(checks, strong, before)

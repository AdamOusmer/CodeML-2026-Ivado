from __future__ import annotations

from src.monitoring import EO_GAP_ALERT

from .reasoning_gate import guard_checks, worse
from .strong_guard import strong_measurer


class SwapGate:
    def __init__(self, measure, before):
        self.measure = measure
        self.baseline = {check.name: check for check in measure(before)}

    def admits(self, candidate):
        return not any(worse(check, self.baseline[check.name]) for check in self.measure(candidate)
                       if check.name in self.baseline)


def swap_gate(pipeline, history, batch, share, eo_gap_alert=EO_GAP_ALERT):
    checks = guard_checks(pipeline, history, batch, share, eo_gap_alert)
    strong = strong_measurer(pipeline, history, batch, share) if pipeline.config.strong_guard else None
    measure = checks if strong is None else (lambda decisions: [*checks(decisions), *strong(decisions).checks])
    return lambda before: SwapGate(measure, before)

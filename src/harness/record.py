from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from src.monitoring import Verdict
from src.policy import Config


class InputError(ValueError):
    pass


class ActionKind(str, Enum):
    SELECT_CONFIG = "SELECT_CONFIG"
    ADJUST_OFFSET = "ADJUST_OFFSET"
    REVERT_JURY = "REVERT_JURY"
    BLOCK = "BLOCK"


@dataclass
class Action:
    kind: ActionKind
    reason: str
    params: dict = field(default_factory=dict)


@dataclass
class DecisionRecord:
    config: Config
    share: float
    status: str
    ids: pd.Series
    decisions: np.ndarray
    proposed: np.ndarray
    scores: np.ndarray
    offset: float
    verdicts: list[Verdict]
    actions: list[Action]
    output_issues: list[str]
    offset_moved_ids: list[str]
    jury_moved_ids: list[str]
    explanations: pd.DataFrame
    region_rates: dict[str, float]
    input_hashes: dict[str, str]
    jury: dict
    jury_guard: Verdict
    consensus_guard: Verdict
    warnings: list[str] = field(default_factory=list)
    label_correction: dict | None = None
    training_labels: dict | None = None
    strong_guard: Verdict | None = None
    deliberation: dict | None = None

    @property
    def published(self) -> bool:
        return self.status == "published"

    def summary(self) -> dict:
        return {
            "status": self.status,
            "config": asdict(self.config),
            "share": self.share,
            "grants": int(self.decisions.sum()),
            "applicants": len(self.decisions),
            "offset": self.offset,
            "actions": [{"kind": a.kind.value, "reason": a.reason, "params": a.params} for a in self.actions],
            "verdicts": [{"status": v.status, "checks": [c.to_dict() for c in v.checks]} for v in self.verdicts],
            "proposed_grants": int(self.proposed.sum()),
            "output_issues": self.output_issues,
            "offset_moved_ids": self.offset_moved_ids,
            "jury_moved_ids": self.jury_moved_ids,
            "warnings": self.warnings,
            "label_correction": self.label_correction,
            "training_labels": self.training_labels,
            "jury": self.jury,
            **({} if self.deliberation is None else {"deliberation": self.deliberation}),
            **({} if self.strong_guard is None else {
                "strong_guard": {"status": self.strong_guard.status,
                                 "checks": [c.to_dict() for c in self.strong_guard.checks]}}),
            "jury_guard": {"status": self.jury_guard.status, "checks": [c.to_dict() for c in self.jury_guard.checks]},
            "consensus_guard": {"status": self.consensus_guard.status,
                                "checks": [c.to_dict() for c in self.consensus_guard.checks]},
            "region_rates": self.region_rates,
            "input_hashes": self.input_hashes,
        }

    def predictions(self) -> pd.DataFrame:
        return pd.DataFrame({"id_candidat": self.ids, "decision_octroi": self.decisions})

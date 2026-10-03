from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from src.monitoring import Verdict
from src.policy import Config


class ActionKind(str, Enum):
    SELECT_CONFIG = "SELECT_CONFIG"
    ADJUST_OFFSET = "ADJUST_OFFSET"
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
    scores: np.ndarray
    offset: float
    verdicts: list[Verdict]
    actions: list[Action]
    moved_ids: list[str]
    explanations: pd.DataFrame
    region_rates: dict[str, float]
    input_hashes: dict[str, str]

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
            "moved_ids": self.moved_ids,
            "region_rates": self.region_rates,
            "input_hashes": self.input_hashes,
        }

    def predictions(self) -> pd.DataFrame:
        return pd.DataFrame({"id_candidat": self.ids, "decision_octroi": self.decisions})

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring.checks import Verdict
from src.policy.models import Config


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

    def write(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        record_path, explanations_path = out_dir / "decision_record.json", out_dir / "explanations.csv"
        record_path.write_text(json.dumps(self.summary(), indent=2, default=float))
        self.explanations.to_csv(explanations_path, index=False)
        written = [record_path, explanations_path]
        if self.published:
            predictions_path = out_dir / "predictions.csv"
            pd.DataFrame({"id_candidat": self.ids, "decision_octroi": self.decisions}).to_csv(predictions_path, index=False)
            written.append(predictions_path)
        return written

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.harness import DecisionRecord


@dataclass(frozen=True)
class Inputs:
    history: pd.DataFrame
    batch: pd.DataFrame
    hashes: dict[str, str]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_inputs(history_path: Path, batch_path: Path) -> Inputs:
    return Inputs(read_table(history_path), read_table(batch_path),
                  {path.name: sha256(path) for path in (history_path, batch_path)})


def read_decisions(path: Path) -> pd.Series:
    return read_table(path).set_index("id_candidat")["decision_octroi"]


def write_table(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def write_decision(record: DecisionRecord, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "decision_record.json"
    record_path.write_text(json.dumps(record.summary(), indent=2, default=float))
    written = [record_path, write_table(record.explanations, out_dir / "explanations.csv")]
    if record.published:
        written.append(write_table(record.predictions(), out_dir / "predictions.csv"))
    return written


def save_figure(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path

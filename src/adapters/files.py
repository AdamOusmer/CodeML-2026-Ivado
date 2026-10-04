from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.harness import DecisionRecord, InputError


@dataclass(frozen=True)
class Inputs:
    history: pd.DataFrame
    batch: pd.DataFrame
    hashes: dict[str, str]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    index = table.index
    if not (isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1):
        raise InputError(f"{path.name}: rows have more fields than the header")
    return table


def read_inputs(history_path: Path, batch_path: Path) -> Inputs:
    return Inputs(read_table(history_path), read_table(batch_path),
                  {path.name: sha256(path) for path in (history_path, batch_path)})


RESIDUAL_FILE = "residuals.csv"
RESIDUAL_MANIFEST = "manifest.json"


def read_residual(directory: Path, history_path: Path, batch_path: Path, batch_ids: pd.Series) -> pd.Series:
    residuals_path, manifest_path = directory / RESIDUAL_FILE, directory / RESIDUAL_MANIFEST
    for path in (residuals_path, manifest_path):
        if not path.is_file():
            raise InputError(f"TabM residual artefact incomplete: {path} not found")
    manifest = json.loads(manifest_path.read_text())
    expected = {RESIDUAL_FILE: (manifest.get("residuals_sha256"), sha256(residuals_path)),
                history_path.name: (manifest.get("input_hashes", {}).get("history"), sha256(history_path)),
                batch_path.name: (manifest.get("input_hashes", {}).get("candidates"), sha256(batch_path))}
    for name, (declared, actual) in expected.items():
        if declared != actual:
            raise InputError(f"TabM residual artefact does not match {name}: manifest {declared}, found {actual}")
    residual = read_table(residuals_path).set_index("id_candidat")["residual_rsd"]
    if not residual.index.is_unique or set(residual.index) != set(batch_ids):
        raise InputError("TabM residual ids differ from the batch ids")
    if not np.isfinite(residual).all():
        raise InputError("TabM residual contains non-finite values")
    return residual


JUROR_FILE = "residuals.csv"


def read_juror_residuals(directory: Path, history_path: Path, batch_path: Path, batch_ids: pd.Series,
                         jurors: tuple[str, ...]) -> pd.DataFrame:
    residuals_path, manifest_path = directory / JUROR_FILE, directory / RESIDUAL_MANIFEST
    for path in (residuals_path, manifest_path):
        if not path.is_file():
            raise InputError(f"TabM juror artefact incomplete: {path} not found")
    manifest = json.loads(manifest_path.read_text())
    expected = {JUROR_FILE: (manifest.get("residuals_sha256"), sha256(residuals_path)),
                history_path.name: (manifest.get("input_hashes", {}).get("history"), sha256(history_path)),
                batch_path.name: (manifest.get("input_hashes", {}).get("candidates"), sha256(batch_path))}
    for name, (declared, actual) in expected.items():
        if declared != actual:
            raise InputError(f"TabM juror artefact does not match {name}: manifest {declared}, found {actual}")
    if [member["cfg_id"] for member in manifest.get("members", [])] != list(jurors):
        raise InputError("TabM juror artefact members differ from the declared jurors")
    table = read_table(residuals_path).set_index("id_candidat")
    if list(table.columns) != list(jurors):
        raise InputError("TabM juror residual columns differ from the declared jurors")
    if not table.index.is_unique or set(table.index) != set(batch_ids):
        raise InputError("TabM juror residual ids differ from the batch ids")
    if not np.isfinite(table.to_numpy(dtype=float)).all():
        raise InputError("TabM juror residuals contain non-finite values")
    return table


def read_decisions(path: Path) -> pd.Series:
    return read_table(path).set_index("id_candidat")["decision_octroi"]


def write_table(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def finite(value):
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def json_text(payload) -> str:
    return json.dumps(finite(payload), indent=2, default=float, allow_nan=False)


def write_decision(record: DecisionRecord, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "decision_record.json"
    record_path.write_text(json_text(record.summary()))
    written = [record_path, write_table(record.explanations, out_dir / "explanations.csv")]
    if record.published:
        written.append(write_table(record.predictions(), out_dir / "predictions.csv"))
    return written


def save_figure(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return path

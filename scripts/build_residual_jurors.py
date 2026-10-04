import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policy import DECLARED_RESIDUAL_BLEND, JUROR_ARTEFACT, TABM_JURORS

HISTORY = ROOT / "data/donnees_demandes.csv"
CANDIDATES = ROOT / "data/candidats_evaluation.csv"
DEFAULT_SWEEP = Path.home() / "Downloads/equialgo_tabm_sweep_history_base"
OOF_BLEND = 1.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def member(sweep: Path, summary: pd.DataFrame, cfg_id: str) -> tuple[dict, pd.Series]:
    rows = summary[(summary["cfg_id"] == cfg_id) & summary["penalty_selected"].astype(str).eq("True")]
    row = rows[rows["blend"] == OOF_BLEND].iloc[0]
    penalty = float(row["penalty"])
    source = sweep / "audit_arrays" / cfg_id / f"pen{penalty:g}_candidate_residuals.npz"
    arrays = np.load(source, allow_pickle=True)
    residual = pd.Series(arrays["mean_residual"], index=arrays["id_candidat"], name=cfg_id)
    base_r, base_hours = arrays["base_coefficients"][:2]
    gradient = arrays["mean_gradient"]
    return ({
        "cfg_id": cfg_id, "selected_penalty": penalty, "source_file": str(source.relative_to(sweep)),
        "source_sha256": sha256(source), "oof_logloss": float(row["oof_logloss"]),
        "oof_logloss_improvement": float(row["logloss_improvement"]),
        "oof_logloss_improvement_se": float(row["logloss_improvement_se"]),
        "oof_auc_improvement": float(row["auc_improvement"]), "oof_brier_improvement": float(row["brier_improvement"]),
        "monotonicity": {
            "grid_status": "GRIDPASS" if bool(row["mono_pass"]) and int(row["mono_violations"]) == 0 else "GRIDFAIL",
            "grid_violations": int(row["mono_violations"]),
            "min_R_derivative_at_declared_blend": float(base_r + DECLARED_RESIDUAL_BLEND * gradient[:, 0].min()),
            "min_hours_derivative_at_declared_blend": float(base_hours + DECLARED_RESIDUAL_BLEND * gradient[:, 1].min()),
        },
    }, residual)


def build(sweep: Path, out: Path) -> None:
    summary = pd.read_csv(sweep / "summary.csv")
    members, columns = [], []
    for cfg_id in TABM_JURORS:
        info, residual = member(sweep, summary, cfg_id)
        members.append(info)
        columns.append(residual)
    table = pd.concat(columns, axis=1)
    table.index.name = "id_candidat"
    ids = pd.read_csv(CANDIDATES)["id_candidat"]
    table = table.reindex(ids)
    assert np.isfinite(table.to_numpy()).all(), "missing juror residual"
    out.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(out / "residuals.csv", index=False)
    manifest = {
        "status": "complete", "model": "tabm.TabM",
        "kind": "one TabM residual model per juror; juror score = declared base + blend x juror residual",
        "residual_units": "full-history standardized R equivalent",
        "penalty_selection": "per configuration, penalty marked selected in the sweep summary",
        "declared_blend": DECLARED_RESIDUAL_BLEND,
        "oof_note": "OOF gains are the sweep's per-configuration historical estimates; monotonicity derivatives are "
                    "recomputed on the sweep grid at the declared blend.",
        "label_warning": "Historical decision_octroi is committee output, not independent merit; diagnostics only.",
        "members": members,
        "input_hashes": {"history": sha256(HISTORY), "candidates": sha256(CANDIDATES)},
        "source_hashes": {m["cfg_id"]: m["source_sha256"] for m in members},
        "residuals_sha256": sha256(out / "residuals.csv"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / JUROR_ARTEFACT)
    arguments = parser.parse_args()
    build(arguments.sweep, arguments.out)

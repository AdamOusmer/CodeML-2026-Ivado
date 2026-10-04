from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from src.adapters import (RESIDUAL_FILE, read_inputs, read_juror_residuals, read_residual, save_figure, sha256,
                          write_decision, write_table)
from src.common.graph import Graph, Node
from src.common.logging import get_logger
from src.evaluation import pareto_report
from src.harness import DecisionRecord, decide
from src.policy import DECLARED_CONFIG, TABM_JURORS, correction_report, report_warnings
from src.preprocessing import validate_frames

PIPELINE_WORKERS = 2
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

logger = get_logger("pipelines")


def history_of(inputs):
    return inputs.history


def batch_of(inputs):
    return inputs.batch


def log_warnings(report) -> tuple[str, ...]:
    warnings = tuple(report.warnings)
    for warning in warnings:
        logger.warning("%s", warning)
    return warnings


def residual_dir_of(config, residual_override):
    return Path(residual_override) if residual_override else MODELS_DIR / config.residual_artefact


def juror_dir_of(config, juror_override):
    if not config.juror_artefact:
        return None
    return Path(juror_override) if juror_override else MODELS_DIR / config.juror_artefact


def hashes_of(inputs, config, residual_dir, juror_dir):
    if config.residual_blend is None:
        return inputs.hashes
    hashes = {**inputs.hashes, f"{config.residual_artefact}/{RESIDUAL_FILE}": sha256(Path(residual_dir) / RESIDUAL_FILE)}
    if juror_dir is not None:
        hashes[f"{config.juror_artefact}/{RESIDUAL_FILE}"] = sha256(Path(juror_dir) / RESIDUAL_FILE)
    return hashes


def load_juror_residuals(juror_dir, history_path, batch_path, batch):
    if juror_dir is None:
        return None
    return read_juror_residuals(Path(juror_dir), history_path, batch_path, batch["id_candidat"], TABM_JURORS)


def load_residual(config, residual_dir, history_path, batch_path, batch):
    if config.residual_blend is None:
        return None
    return read_residual(Path(residual_dir), history_path, batch_path, batch["id_candidat"])


def share_of(record):
    return record.share


def attach_correction(record: DecisionRecord, report) -> DecisionRecord:
    added = [warning for warning in report_warnings(report) if warning not in record.warnings]
    for warning in added:
        logger.warning("%s", warning)
    return replace(record, label_correction=asdict(report), warnings=[*record.warnings, *added])


def write_pareto(report, out_dir: Path) -> list[Path]:
    table_path = Path(out_dir) / "resultats_pareto.csv"
    figure_path = Path(out_dir) / "pareto_front.png"
    write_table(report.summary, table_path)
    save_figure(report.figure, figure_path)
    return [table_path, figure_path]


DECISION_NODES = (
    Node("inputs", read_inputs, ("history_path", "batch_path")),
    Node("history", history_of, ("inputs",)),
    Node("batch", batch_of, ("inputs",)),
    Node("frame_report", validate_frames, ("history", "batch")),
    Node("frame_warnings", log_warnings, ("frame_report",)),
    Node("residual_dir", residual_dir_of, ("config", "residual_override")),
    Node("juror_dir", juror_dir_of, ("config", "juror_override")),
    Node("hashes", hashes_of, ("inputs", "config", "residual_dir", "juror_dir"), after=("residual", "juror_residuals")),
    Node("juror_residuals", load_juror_residuals, ("juror_dir", "history_path", "batch_path", "batch")),
    Node("residual", load_residual, ("config", "residual_dir", "history_path", "batch_path", "batch")),
    Node("decision", decide, ("history", "batch", "hashes", "config", "residual", "juror_residuals"), after=("frame_report",)),
    Node("share", share_of, ("decision",)),
    Node("correction_report", correction_report, ("history", "share"), after=("frame_report",)),
    Node("record", attach_correction, ("decision", "correction_report")),
    Node("written", write_decision, ("record", "out_dir")),
)

PARETO_NODES = (
    Node("pareto", pareto_report, ("history", "share", "splits", "pareto_workers")),
    Node("pareto_written", write_pareto, ("pareto", "out_dir")),
)

DECISION = Graph(DECISION_NODES)
FULL = Graph(DECISION_NODES + PARETO_NODES)


def run_decision(history_path: Path, batch_path: Path, out_dir: Path, *, config=DECLARED_CONFIG,
                 residual_dir: Path | None = None, juror_dir: Path | None = None, graph: Graph = DECISION,
                 workers: int = PIPELINE_WORKERS) -> dict:
    seeds = {"history_path": history_path, "batch_path": batch_path, "out_dir": out_dir, "config": config,
             "residual_override": residual_dir,
             "juror_override": juror_dir}
    return graph.run(seeds, workers=workers)


def run_full(history_path: Path, batch_path: Path, out_dir: Path, splits: int, pareto_workers: int, *,
             config=DECLARED_CONFIG, residual_dir: Path | None = None, juror_dir: Path | None = None, graph: Graph = FULL,
             workers: int = PIPELINE_WORKERS) -> dict:
    seeds = {"history_path": history_path, "batch_path": batch_path, "out_dir": out_dir, "config": config,
             "residual_override": residual_dir,
             "juror_override": juror_dir, "splits": splits, "pareto_workers": pareto_workers}
    return graph.run(seeds, workers=workers)

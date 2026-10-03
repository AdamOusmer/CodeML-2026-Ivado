from __future__ import annotations

from pathlib import Path

from src.adapters import read_inputs, save_figure, write_decision, write_table
from src.common.graph import Graph, Node
from src.common.logging import get_logger
from src.evaluation import pareto_report
from src.harness import decide
from src.preprocessing import validate_frames

PIPELINE_WORKERS = 2

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


def hashes_of(inputs):
    return inputs.hashes


def share_of(record):
    return record.share


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
    Node("hashes", hashes_of, ("inputs",)),
    Node("record", decide, ("history", "batch", "hashes"), after=("frame_report",)),
    Node("written", write_decision, ("record", "out_dir")),
)

PARETO_NODES = (
    Node("share", share_of, ("record",)),
    Node("pareto", pareto_report, ("history", "share", "splits", "pareto_workers")),
    Node("pareto_written", write_pareto, ("pareto", "out_dir")),
)

DECISION = Graph(DECISION_NODES)
FULL = Graph(DECISION_NODES + PARETO_NODES)


def run_decision(history_path: Path, batch_path: Path, out_dir: Path, *,
                 graph: Graph = DECISION, workers: int = PIPELINE_WORKERS) -> dict:
    seeds = {"history_path": history_path, "batch_path": batch_path, "out_dir": out_dir}
    return graph.run(seeds, workers=workers)


def run_full(history_path: Path, batch_path: Path, out_dir: Path, splits: int, pareto_workers: int, *,
             graph: Graph = FULL, workers: int = PIPELINE_WORKERS) -> dict:
    seeds = {"history_path": history_path, "batch_path": batch_path, "out_dir": out_dir,
             "splits": splits, "pareto_workers": pareto_workers}
    return graph.run(seeds, workers=workers)

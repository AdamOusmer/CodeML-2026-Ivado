import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

import pandas as pd

from src.harness import InputError
from src.pipelines import run_full
from src.policy import is_remote
from src.preprocessing import DataValidationError

ROOT = Path(__file__).parent
HISTORY = ROOT / "data" / "donnees_demandes.csv"
CANDIDATES = ROOT / "data" / "candidats_evaluation.csv"
SPLITS = 10
WORKERS = 4


def main():
    try:
        artifacts = run_full(HISTORY, CANDIDATES, ROOT, SPLITS, WORKERS)
    except (DataValidationError, InputError, FileNotFoundError, KeyError) as exc:
        print(f"Invalid input data: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Budget: historical grant rate {artifacts['share']:.2%}")

    pd.set_option("display.width", 250)
    print(artifacts["pareto"].table.round(3).to_string(index=False))

    sweep = artifacts["declared_sweep"]
    columns = ["method", "family", "setting", "rate_centre", "rate_remote", "eo_gap_reviewers_mean",
               "agree_reviewers_mean", "g_merit", "eo_gap_corrected", "dp_gap", "pareto"]
    print("\nFront de Pareto de la chaîne déclarée (4 000 candidats, pareto_front.png) :")
    print(sweep.summary[columns].round(4).to_string(index=False))
    print(f"Point déclaré identique aux décisions publiées : {sweep.matches_published}")

    record = artifacts["record"]
    remote = is_remote(artifacts["inputs"].batch)
    print(f"\nHarness: {record.status}, actions {[action.kind.value for action in record.actions]}, offset {record.offset:+.3f}")
    print(f"predictions: {int(record.decisions.sum())} of {len(record.decisions)} granted "
          f"(centre {record.decisions[remote == 0].mean():.1%}, remote {record.decisions[remote == 1].mean():.1%})")
    if not record.published:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

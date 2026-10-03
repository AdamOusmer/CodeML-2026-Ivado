"""ÉquiAlgo mitigation: Pareto front over fairness settings, then automated decisions for the candidates.

1. Preprocessing   : validate both frames; the committee is rebuilt on 8 legitimate features, the main model scores on R, log income and hours; region is kept for auditing, never for default scoring.
2. Fairness (pre)  : remove the committee's remote-region penalty from the training labels.
3. Main model      : region-blind logistic regression trained on the corrected labels.
4. Jury            : rank vote between fairness policies (main model, academic merit).
5. Allocation      : grant the historical share of applicants, the institution's budget.
6. Harness         : audit the decisions, apply a bounded correction on a fairness alert, or block.
"""

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

    record = artifacts["record"]
    remote = is_remote(artifacts["inputs"].batch)
    print(f"\nHarness: {record.status}, actions {[action.kind.value for action in record.actions]}, offset {record.offset:+.3f}")
    print(f"predictions: {int(record.decisions.sum())} of {len(record.decisions)} granted "
          f"(centre {record.decisions[remote == 0].mean():.1%}, remote {record.decisions[remote == 1].mean():.1%})")
    if not record.published:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

"""ÉquiAlgo mitigation: Pareto front over fairness settings, then automated decisions for the candidates.

1. Preprocessing   : legitimate features only; region is kept for auditing, never for default scoring.
2. Fairness (pre)  : remove the committee's remote-region penalty from the training labels.
3. Main model      : region-blind logistic regression trained on the corrected labels.
4. Jury            : rank vote between fairness policies (main model, academic merit).
5. Allocation      : grant the historical share of applicants, the institution's budget.
6. Harness         : audit the decisions, apply a bounded correction on a fairness alert, or block.
"""

import hashlib
from pathlib import Path

import pandas as pd

from src.evaluation.candidates import pareto_report
from src.harness.controller import decide
from src.policy.core import budget_share
from src.policy.regions import is_remote

ROOT = Path(__file__).parent
HISTORY = ROOT / "data" / "donnees_demandes.csv"
CANDIDATES = ROOT / "data" / "candidats_evaluation.csv"
SPLITS = 10
WORKERS = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    history, candidates = pd.read_csv(HISTORY), pd.read_csv(CANDIDATES)
    share = budget_share(history)
    print(f"Budget: historical grant rate {share:.2%}")

    pd.set_option("display.width", 250)
    print(pareto_report(history, share, SPLITS, WORKERS, ROOT).round(3).to_string(index=False))

    record = decide(history, candidates, {path.name: sha256(path) for path in (HISTORY, CANDIDATES)})
    record.write(ROOT)
    remote = is_remote(candidates)
    print(f"\nHarness: {record.status}, actions {[action.kind.value for action in record.actions]}, offset {record.offset:+.3f}")
    print(f"predictions: {int(record.decisions.sum())} of {len(record.decisions)} granted "
          f"(centre {record.decisions[remote == 0].mean():.1%}, remote {record.decisions[remote == 1].mean():.1%})")
    if not record.published:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

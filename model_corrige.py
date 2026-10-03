"""ÉquiAlgo mitigation: Pareto front over fairness settings, then automated decisions for the candidates.

1. Preprocessing   : legitimate features only; region is kept for auditing, never for default scoring.
2. Fairness (pre)  : remove the committee's remote-region penalty from the training labels.
3. Main model      : region-blind logistic regression trained on the corrected labels.
4. Jury            : rank vote between fairness policies (main model, academic merit).
5. Allocation      : grant the historical share of applicants, the institution's budget.
6. Harness         : audit the decisions, apply a bounded correction on a fairness alert, or block.
"""

from pathlib import Path

import pandas as pd

from src.adapters import read_inputs, save_figure, write_decision, write_table
from src.evaluation import pareto_report
from src.harness import decide
from src.policy import budget_share, is_remote

ROOT = Path(__file__).parent
HISTORY = ROOT / "data" / "donnees_demandes.csv"
CANDIDATES = ROOT / "data" / "candidats_evaluation.csv"
SPLITS = 10
WORKERS = 4


def main():
    inputs = read_inputs(HISTORY, CANDIDATES)
    share = budget_share(inputs.history)
    print(f"Budget: historical grant rate {share:.2%}")

    report = pareto_report(inputs.history, share, SPLITS, WORKERS)
    write_table(report.summary, ROOT / "resultats_pareto.csv")
    save_figure(report.figure, ROOT / "pareto_front.png")
    pd.set_option("display.width", 250)
    print(report.table.round(3).to_string(index=False))

    record = decide(inputs.history, inputs.batch, inputs.hashes)
    write_decision(record, ROOT)
    remote = is_remote(inputs.batch)
    print(f"\nHarness: {record.status}, actions {[action.kind.value for action in record.actions]}, offset {record.offset:+.3f}")
    print(f"predictions: {int(record.decisions.sum())} of {len(record.decisions)} granted "
          f"(centre {record.decisions[remote == 0].mean():.1%}, remote {record.decisions[remote == 1].mean():.1%})")
    if not record.published:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

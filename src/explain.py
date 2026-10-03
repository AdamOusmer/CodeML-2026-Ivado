import numpy as np
import pandas as pd

from src.policy import is_remote, percentile


def explain(pipeline, df, outcome, scores, offset, top=3):
    contributions = pipeline.contributions(df)
    values = contributions.to_numpy()
    order = np.argsort(-np.abs(values), axis=1, kind="stable")
    triggered = np.asarray(outcome.triggered, dtype=bool)
    jury_outcome = np.full(len(df), "not_reviewed", dtype=object)
    jury_outcome[triggered] = "confirmed"
    jury_outcome[np.asarray(outcome.overturned, dtype=bool)] = "overturn_unpaired"
    jury_outcome[outcome.overturned_out] = "overturned_out"
    jury_outcome[outcome.overturned_in] = "overturned_in"
    ranking = np.argsort(-np.asarray(scores), kind="stable")
    final_rank = np.empty(len(df), dtype=int)
    final_rank[ranking] = np.arange(1, len(df) + 1)
    result = pd.DataFrame(
        {
            "id_candidat": df["id_candidat"].to_numpy(),
            "decision": np.asarray(outcome.decisions),
            "proposed_decision": np.asarray(outcome.proposed),
            "score": np.asarray(scores),
            "final_rank": final_rank,
            "merit_vote": percentile(df["cote_r_equivalent"]),
            "model_vote": percentile(pipeline.model_probability(df)),
            "offset": offset * is_remote(df),
            "validated": triggered,
            "trigger_reasons": [";".join(reasons) for reasons in outcome.reasons],
            "juror_votes": [
                ";".join(
                    f"{name}:{'GRANT' if vote[i] else 'REFUSE'}"
                    for name, vote in outcome.votes.items()
                )
                if is_validated
                else ""
                for i, is_validated in enumerate(triggered)
            ],
            "jury_outcome": jury_outcome,
        },
        index=df.index,
    )
    for i in range(top):
        result[f"factor_{i + 1}"] = [
            f"{contributions.columns[j]} {row[j]:+.2f}"
            for row, j in zip(values, order[:, i])
        ]
    return result

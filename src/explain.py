import numpy as np
import pandas as pd

from src.policy import is_remote, percentile


def explain(pipeline, df, outcome, scores, offset, top=3):
    contributions = pipeline.contributions(df)
    values = contributions.to_numpy()
    order = np.argsort(-np.abs(values), axis=1, kind="stable")
    triggered = np.asarray(outcome.triggered, dtype=bool)
    prefix = "contested" if outcome.audit_only else "overturned"
    jury_outcome = np.full(len(df), "not_reviewed", dtype=object)
    jury_outcome[triggered] = "confirmed"
    jury_outcome[np.asarray(outcome.overturned, dtype=bool)] = f"{prefix}_unpaired"
    jury_outcome[outcome.overturned_out] = f"{prefix}_out"
    jury_outcome[outcome.overturned_in] = f"{prefix}_in"
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
    if outcome.deliberation is not None:
        result["reasoning_outcome"] = outcome.deliberation.outcomes
        result["reasoning_trace"] = outcome.deliberation.traces
    for i in range(top):
        result[f"factor_{i + 1}"] = [
            f"{contributions.columns[j]} {row[j]:+.2f}"
            for row, j in zip(values, order[:, i])
        ] if i < values.shape[1] else ""
    return result

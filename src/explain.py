import numpy as np
import pandas as pd

from src.policy import is_remote, percentile


def explain(pipeline, df, decisions, scores, offset, top=3):
    contributions = pipeline.contributions(df)
    values = contributions.to_numpy()
    order = np.argsort(-np.abs(values), axis=1, kind="stable")
    result = pd.DataFrame(
        {
            "id_candidat": df["id_candidat"].to_numpy(),
            "decision": np.asarray(decisions),
            "score": np.asarray(scores),
            "merit_vote": percentile(df["cote_r_equivalent"]),
            "model_vote": percentile(pipeline.model_probability(df)),
            "offset": offset * is_remote(df),
        },
        index=df.index,
    )
    for i in range(top):
        result[f"factor_{i + 1}"] = [
            f"{contributions.columns[j]} {row[j]:+.2f}"
            for row, j in zip(values, order[:, i])
        ]
    return result

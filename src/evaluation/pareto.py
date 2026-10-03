import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

LABELS = {
    "eo_gap_corrected": "Equal-opportunity gap vs corrected labels (committee minus regional penalty)",
    "eo_gap_merit": "Equal-opportunity gap vs merit-only labels (top R scores)",
    "acc_historical": "Agreement with historical committee",
    "agree_corrected": "Agreement with corrected labels",
    "agree_merit": "Agreement with merit-only labels",
}
CURVE_STYLES = ["o-", "s--", "d-.", "v:"]
POINT_STYLES = {"baseline": "kX", "ablation": "m^", "post-processing": "cD", "full pipeline": "r*"}


def pareto_mask(summary: pd.DataFrame, x: str, y: str) -> np.ndarray:
    eligible = summary["budget_ok"].to_numpy()
    xs, ys = summary[x].to_numpy(), summary[y].to_numpy()
    dominated = [
        any(eligible[j] and xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]) for j in range(len(xs)))
        for i in range(len(xs))
    ]
    return eligible & ~np.array(dominated)


def plot(summary: pd.DataFrame, panels: list[tuple[str, str]], splits: int) -> Figure:
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5), squeeze=False)
    curves = [family for family in summary["family"].unique() if family not in POINT_STYLES]
    for ax, (x, y) in zip(axes[0], panels):
        for family, style in zip(curves, CURVE_STYLES):
            part = summary[summary.family == family].sort_values("setting")
            ax.errorbar(part[x], part[y], xerr=part[f"{x}_std"], fmt=style, capsize=2, alpha=0.85, label=family)
        for family, style in POINT_STYLES.items():
            part = summary[summary.family == family]
            ax.errorbar(part[x], part[y], xerr=part[f"{x}_std"], fmt=style, markersize=11, capsize=2, label=family)
        front = summary[pareto_mask(summary, x, y)].sort_values(x)
        ax.plot(front[x], front[y], color="grey", linewidth=4, alpha=0.25, label="Pareto front")
        ax.set_xlabel(LABELS.get(x, x))
        ax.set_ylabel(LABELS.get(y, y))
        ax.grid(alpha=0.3)
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Fairness vs utility, mean ± std over {splits} splits, every method at the same budget except the anchor")
    fig.tight_layout()
    return fig

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

LABELS = {
    "eo_gap_reviewers_mean": "Écart d'égalité des chances moyen vs les 5 références des évaluateurs (↓)",
    "agree_reviewers_mean": "Accord moyen avec les 5 références des évaluateurs (↑)",
    "eo_gap_corrected": "Écart d'égalité des chances vs comité corrigé (↓)",
    "eo_gap_merit": "Écart d'égalité des chances vs mérite seul (↓)",
    "acc_historical": "Accord avec le comité historique (↑)",
    "agree_corrected": "Accord avec le comité corrigé (↑)",
    "agree_merit": "Accord avec le mérite seul (↑)",
}
FAMILY_LABELS = {
    "baseline": "référence de production", "ablation": "retrait des proxys", "post-processing": "post-traitement",
    "ExpGrad sweep": "ExpGrad (balayage)", "penalty removal sweep": "retrait de la pénalité (balayage)",
    "jury band sweep": "jury, largeur de bande (balayage)", "full pipeline": "pipelines complets",
    "declared base removal sweep": "base déclarée, part de la pénalité retirée (balayage)",
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


def plot(summary: pd.DataFrame, panels: list[tuple[str, str]], splits: int, declared: str | None = None,
         annotated: dict[str, str] | None = None) -> Figure:
    fig = Figure(figsize=(6.5 * len(panels), 5))
    axes = fig.subplots(1, len(panels), squeeze=False)
    curves = [family for family in summary["family"].unique() if family not in POINT_STYLES]
    for ax, (x, y) in zip(axes[0], panels):
        for family, style in zip(curves, CURVE_STYLES):
            part = summary[summary.family == family].sort_values("setting")
            ax.errorbar(part[x], part[y], xerr=part[f"{x}_std"], fmt=style, capsize=2, alpha=0.85, label=FAMILY_LABELS.get(family, family))
        for family, style in POINT_STYLES.items():
            part = summary[summary.family == family]
            ax.errorbar(part[x], part[y], xerr=part[f"{x}_std"], fmt=style, markersize=11, capsize=2, label=FAMILY_LABELS.get(family, family))
        marked = summary[summary.method == declared]
        ax.scatter(marked[x], marked[y], s=420, facecolors="none", edgecolors="red", linewidths=2.5, zorder=5,
                   label="point déclaré (base de consensus, sans résidu)")
        for method, text in (annotated or {}).items():
            row = summary[summary.method == method]
            for xv, yv in zip(row[x], row[y]):
                ax.annotate(text, (xv, yv), textcoords="offset points", xytext=(8, -14), fontsize=8,
                            arrowprops={"arrowstyle": "-", "alpha": 0.5})
        front = summary[pareto_mask(summary, x, y)].sort_values(x)
        ax.plot(front[x], front[y], color="grey", linewidth=4, alpha=0.25, label="front de Pareto")
        ax.set_xlabel(LABELS.get(x, x))
        ax.set_ylabel(LABELS.get(y, y))
        ax.grid(alpha=0.3)
    axes[0][0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Équité et utilité : moyenne ± écart-type sur {splits} partitions, même budget pour toutes les méthodes sauf l'ancre")
    fig.tight_layout()
    return fig

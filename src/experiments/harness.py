from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import model_corrige as policy

Decide = Callable[[pd.DataFrame, pd.DataFrame, float], np.ndarray]
METRICS = ["grant_rate", "rate_centre", "rate_remote", "dp_gap", "eo_gap_corrected", "eo_gap_merit",
           "eo_gap_historical", "agree_corrected", "agree_merit", "acc_historical"]
LABELS = {
    "eo_gap_corrected": "Equal-opportunity gap vs corrected labels (committee minus regional penalty)",
    "eo_gap_merit": "Equal-opportunity gap vs merit-only labels (top R scores)",
    "acc_historical": "Agreement with historical committee",
    "agree_corrected": "Agreement with corrected labels",
    "agree_merit": "Agreement with merit-only labels",
}
CURVE_STYLES = ["o-", "s--", "d-.", "v:"]
POINT_STYLES = {"baseline": "kX", "ablation": "m^", "post-processing": "cD", "full pipeline": "r*"}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    decide: Decide
    setting: float = np.nan


def evaluate_split(history: pd.DataFrame, share: float, seed: int, candidate: Candidate) -> dict:
    train, test = train_test_split(history, test_size=0.3, random_state=seed, stratify=history["decision_octroi"])
    decisions = candidate.decide(train, test, share)
    references = policy.reference_labels(train, test, share)
    row = policy.evaluate(candidate.name, candidate.family, decisions, references, policy.is_remote(test), candidate.setting)
    return {**row, "seed": seed}


def run(history: pd.DataFrame, candidates: list[Candidate], share: float, splits: int, workers: int) -> pd.DataFrame:
    jobs = list(product(range(splits), candidates))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = pool.map(lambda job: evaluate_split(history, share, *job), jobs)
        return pd.DataFrame(list(rows))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["method", "family", "setting"], dropna=False, sort=False)
    summary = grouped[METRICS].mean()
    spread = grouped[METRICS].std().add_suffix("_std")
    budget_ok = grouped["budget_ok"].all()
    return pd.concat([summary, spread, budget_ok], axis=1).reset_index()


def pareto_mask(summary: pd.DataFrame, x: str, y: str) -> np.ndarray:
    eligible = summary["budget_ok"].to_numpy()
    xs, ys = summary[x].to_numpy(), summary[y].to_numpy()
    dominated = [
        any(eligible[j] and xs[j] <= xs[i] and ys[j] >= ys[i] and (xs[j] < xs[i] or ys[j] > ys[i]) for j in range(len(xs)))
        for i in range(len(xs))
    ]
    return eligible & ~np.array(dominated)


def plot(summary: pd.DataFrame, panels: list[tuple[str, str]], path: Path, splits: int) -> None:
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
    fig.savefig(path, dpi=150)
    plt.close(fig)

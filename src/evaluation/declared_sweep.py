from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier

from src.policy import (DECLARED_HOURS_WEIGHT, DECLARED_INCOME_WEIGHT, DECLARED_RESIDUAL_BLEND, CommitteeModel,
                        allocate, is_remote, production_features, reference_labels)

from .core import evaluate
from .pareto import pareto_mask

SEED = 42
REMOVALS = tuple(np.round(np.linspace(0, 1, 11), 2))
BLENDS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
INCOME_WEIGHTS = (-0.05, 0.0, 0.025, 0.05, 0.10, 0.19)
REMOVAL_FAMILY = "contrainte d'équité : part de la pénalité régionale retirée"
BLEND_FAMILY = "mélange du résidu TabM"
INCOME_FAMILY = "poids du revenu"
ANCHOR_FAMILY = "modèle de production"
DECLARED_METHOD = "déclaré"
AXES = ("eo_gap_reviewers_mean", "agree_reviewers_mean")
STYLES = {REMOVAL_FAMILY: ("o-", "tab:blue"), BLEND_FAMILY: ("s--", "tab:green"), INCOME_FAMILY: ("d-.", "tab:purple")}


def declared_score(committee: CommitteeModel, df: pd.DataFrame, residual: np.ndarray | None = None,
                   removal: float = 1.0, income: float = DECLARED_INCOME_WEIGHT,
                   blend: float = DECLARED_RESIDUAL_BLEND) -> np.ndarray:
    score = committee.rule_score(df, income, DECLARED_HOURS_WEIGHT)
    if residual is not None:
        score = score + blend * residual
    return score + (1.0 - removal) * committee.penalty_in_rule_units() * is_remote(df)


def settings() -> list[tuple[str, str, float, dict]]:
    rows = [(f"retrait {removal:.0%}", REMOVAL_FAMILY, removal, {"removal": removal}) for removal in REMOVALS]
    rows += [(f"mélange {blend:g}", BLEND_FAMILY, blend, {"blend": blend}) for blend in BLENDS]
    rows += [(f"revenu {weight:+.3f}", INCOME_FAMILY, weight, {"income": weight}) for weight in INCOME_WEIGHTS]
    return rows


def production_at_budget(history: pd.DataFrame, batch: pd.DataFrame, share: float) -> np.ndarray:
    X_train, X_batch = production_features(history, batch)
    forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=SEED, n_jobs=1)
    forest.fit(X_train, history["decision_octroi"].to_numpy())
    return allocate(forest.predict_proba(X_batch)[:, 1], share)


@dataclass(frozen=True)
class DeclaredSweep:
    summary: pd.DataFrame
    figure: Figure
    matches_published: bool


def declared_pipeline_sweep(history: pd.DataFrame, batch: pd.DataFrame, residual: pd.Series, share: float,
                   published_ids: pd.Series, published: np.ndarray) -> DeclaredSweep:
    committee = CommitteeModel().fit(history, history["decision_octroi"].to_numpy())
    aligned = residual.reindex(batch["id_candidat"]).to_numpy(dtype=float)
    references = reference_labels(history, batch, share)
    remote = is_remote(batch)
    rows = [evaluate("forêt aléatoire de production, au budget", ANCHOR_FAMILY,
                     production_at_budget(history, batch, share), references, remote)]
    for name, family, setting, knobs in settings():
        decisions = allocate(declared_score(committee, batch, aligned, **knobs), share)
        rows.append(evaluate(name, family, decisions, references, remote, setting))
    declared = allocate(declared_score(committee, batch, aligned), share)
    rows.append(evaluate(DECLARED_METHOD, "déclaré", declared, references, remote))
    summary = pd.DataFrame(rows).drop(columns=["eo_gap_historical", "acc_historical"])
    summary["pareto"] = pareto_mask(summary, *AXES)
    published_aligned = pd.Series(published, index=published_ids.to_numpy()).reindex(batch["id_candidat"]).to_numpy()
    return DeclaredSweep(summary, plot(summary), bool((declared == published_aligned).all()))


def draw_front(ax, summary: pd.DataFrame, labelled: bool) -> None:
    x, y = AXES
    for family, (style, color) in STYLES.items():
        part = summary[summary.family == family].sort_values("setting")
        ax.plot(part[x], part[y], style, color=color, alpha=0.85, label=family)
        if labelled and family != REMOVAL_FAMILY:
            for _, row in part.iterrows():
                ax.annotate(f"{row.setting:g}", (row[x], row[y]), textcoords="offset points", xytext=(5, 4),
                            fontsize=7, color=color)
    removal = summary[summary.family == REMOVAL_FAMILY]
    for _, row in (removal if labelled else removal.iloc[::2]).iterrows():
        ax.annotate(f"{row.setting:.0%}", (row[x], row[y]), textcoords="offset points", xytext=(6, -12),
                    fontsize=8, color="tab:blue")
    anchor = summary[summary.family == ANCHOR_FAMILY]
    ax.plot(anchor[x], anchor[y], "kX", markersize=11, label="forêt aléatoire de production (au budget)")
    marked = summary[summary.method == DECLARED_METHOD]
    ax.scatter(marked[x], marked[y], s=380, facecolors="none", edgecolors="red", linewidths=2.5, zorder=5,
               label="point déclaré (retrait 100 %, mélange 2,5, revenu +0,025)")
    front = summary[summary.pareto].sort_values(x)
    ax.plot(front[x], front[y], color="grey", linewidth=5, alpha=0.3, label="front de Pareto (non dominés)")
    ax.set_xlabel("Écart d'égalité des chances moyen vs les 5 références des examinateurs (↓)")
    ax.set_ylabel("Accord moyen avec les 5 références des examinateurs (↑)")
    ax.grid(alpha=0.3)


def plot(summary: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(21, 6.2))
    front_ax, zoom_ax, knob_ax = fig.subplots(1, 3)
    draw_front(front_ax, summary, labelled=False)
    front_ax.set_title("(a) Balayage de la contrainte d'équité et des réglages")
    front_ax.legend(loc="lower left", fontsize=8)
    draw_front(zoom_ax, summary, labelled=True)
    zoom_ax.set_xlim(0.0, 0.065)
    zoom_ax.set_ylim(0.945, 0.984)
    zoom_ax.set_title("(b) Agrandissement : mélange du résidu et poids du revenu (retrait 100 %)")

    sweep = summary[summary.family == REMOVAL_FAMILY].sort_values("setting")
    knob_ax.plot(sweep.setting, sweep.eo_gap_reviewers_mean, "o-", color="tab:blue",
                 label="écart EO moyen vs les 5 références")
    knob_ax.plot(sweep.setting, sweep.g_merit, "s--", color="tab:orange", label="écart EO signé vs mérite (R seul)")
    knob_ax.plot(sweep.setting, sweep.dp_gap, "d:", color="tab:grey", label="écart de parité des taux")
    knob_ax.axhspan(-0.09, 0.05, color="tab:green", alpha=0.08, label="plage admise pour l'EO signé [−0,09 ; +0,05]")
    knob_ax.axhline(0, color="black", linewidth=0.6)
    knob_ax.axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="réglage déclaré")
    knob_ax.set_xlabel("Contrainte d'équité : part de la pénalité régionale du comité retirée")
    knob_ax.set_ylabel("Écart (centre − éloigné)")
    knob_ax.set_title("(c) Effet de la contrainte (mélange 2,5, revenu +0,025)")
    knob_ax.grid(alpha=0.3)
    knob_ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Front de Pareto de la chaîne déclarée sur les 4 000 candidats : même budget (1 598 octrois) "
                 "pour chaque point ; références = règles des cinq examinateurs, construites sur l'historique")
    fig.tight_layout()
    return fig

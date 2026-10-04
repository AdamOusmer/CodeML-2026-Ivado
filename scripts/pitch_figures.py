import json
import os
import sys
import warnings
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapters import read_residual
from src.evaluation.pareto import pareto_mask
from src.harness.controller import decide
from src.policy import DECLARED_CONFIG, auc, is_remote

HISTORY = ROOT / "data/donnees_demandes.csv"
CANDIDATES = ROOT / "data/candidats_evaluation.csv"
RESIDUAL_DIR = ROOT / "models" / DECLARED_CONFIG.residual_artefact
OUT = ROOT / "docs/pitch_figures"
DECLARED_METHOD = "declared base: consensus panel, audit mode"

INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CENTRE, REMOTE = "#2a78d6", "#eb6834"
COMMITTEE, OLD, FINAL = "#9b9a94", "#eda100", "#1baf7a"
DECLARED, ALERT = "#4a3aa7", "#e34948"
REGIONS = [("Montreal", "Montréal"), ("Capitale-Nationale", "Capitale-\nNationale"),
           ("Bas-Saint-Laurent", "Bas-Saint-\nLaurent"), ("Cote-Nord", "Côte-Nord"),
           ("Gaspesie-Iles-de-la-Madeleine", "Gaspésie–Îles-\nde-la-Madeleine")]

plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 22, "axes.titleweight": "bold", "axes.labelsize": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15, "legend.frameon": False,
    "axes.edgecolor": INK_SOFT, "axes.labelcolor": INK, "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def pct(value):
    return f"{value * 100:.1f} %".replace(".", ",")


def num(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def french_ticks(ax, axis="both", digits=None):
    def formatter(value, _):
        text = f"{value:.{digits}f}" if digits is not None else f"{value:g}"
        return text.replace(".", ",").replace("-", "−")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def inputs():
    history, batch = pd.read_csv(HISTORY), pd.read_csv(CANDIDATES)
    residual = read_residual(RESIDUAL_DIR, HISTORY, CANDIDATES, batch["id_candidat"])
    return history, batch, residual


def region_rates_figure(history, final_rates):
    committee = history.groupby("region_administrative")["decision_octroi"].mean()
    stages = [("Comité historique (10 000 demandes)", committee, COMMITTEE),
              ("Solution finale (4 000 candidats)", final_rates, FINAL)]
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(REGIONS))
    width = 0.38
    for i, (label, rates, color) in enumerate(stages):
        values = [rates[key] for key, _ in REGIONS]
        bars = ax.bar(x + (i - 0.5) * width, values, width - 0.03, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value * 100:.0f} %", ha="center",
                    fontsize=15, color=INK_SOFT)
    ax.axvspan(1.5, 4.5, color=REMOTE, alpha=0.06, zorder=0)
    ax.text(3, 0.60, "Régions éloignées", ha="center", color=REMOTE, fontsize=16, fontweight="bold")
    ax.text(0.5, 0.60, "Grands centres", ha="center", color=CENTRE, fontsize=16, fontweight="bold")
    ax.set_xticks(x, [label for _, label in REGIONS])
    ax.set_ylim(0, 0.65)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f} %")
    ax.set_ylabel("Taux d'octroi")
    ax.set_title("Taux d'octroi par région : avant / après")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=15)
    return save(fig, "fig1_taux_par_region.png")


def same_r_figure(history):
    remote = is_remote(history)
    bins = np.arange(22, 36)
    labels = (bins[:-1] + bins[1:]) / 2
    cut = pd.cut(history["cote_r_equivalent"], bins)
    table = history.groupby([cut, remote], observed=False)["decision_octroi"].agg(["mean", "size"]).unstack()
    band = history[(history["cote_r_equivalent"] > 28) & (history["cote_r_equivalent"] <= 30)]
    band_rates = band.groupby(is_remote(band))["decision_octroi"].mean()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axvspan(28, 30, color=GRID, alpha=0.7, zorder=0)
    for group, name, color in ((0, "Grands centres", CENTRE), (1, "Régions éloignées", REMOTE)):
        rates = table["mean"][group].to_numpy()
        keep = table["size"][group].to_numpy() >= 30
        ax.plot(labels[keep], rates[keep], color=color, linewidth=3, marker="o", markersize=9, label=name)
    ax.annotate(f"Cote R 28–30\ncentres {pct(band_rates[0])}\néloignées {pct(band_rates[1])}",
                xy=(29, 0.55), xytext=(23.2, 0.62), fontsize=17, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": INK_SOFT, "lw": 1.5})
    ax.set_xlim(22, 35)
    ax.set_ylim(-0.02, 1.04)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f} %")
    ax.set_xlabel("Cote R")
    ax.set_ylabel("Taux d'octroi du comité")
    ax.set_title("À cote R égale, la région change la décision (historique)")
    ax.legend(loc="lower right")
    return save(fig, "fig2_meme_cote_r.png"), band_rates


def proxy_figure(history):
    remote = is_remote(history)
    half = np.random.default_rng(0).random(len(history)) < 0.5
    postal_rate = pd.Series(remote[half]).groupby(history["code_postal_3"][half].to_numpy()).mean()
    postal_score = history["code_postal_3"][~half].map(postal_rate).fillna(0.5).to_numpy()
    columns = {"Code postal (3 car.)": None, "Distance au campus": "distance_domicile_campus_km",
               "Heures de travail": "heures_travail_semaine", "Revenu familial": "revenu_familial_estime",
               "Cote R": "cote_r_equivalent"}
    scores = {}
    for label, column in columns.items():
        score = postal_score if column is None else history[column].to_numpy()
        truth = remote[~half] if column is None else remote
        value = auc(truth, score)
        scores[label] = max(value, 1 - value)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    names = list(scores)[::-1]
    values = [scores[name] for name in names]
    bars = ax.barh(names, [v - 0.5 for v in values], left=0.5, height=0.6, color=CENTRE)
    for bar, value in zip(bars, values):
        ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, "1,0" if value > 0.9995 else num(value),
                va="center", fontsize=17, fontweight="bold")
    ax.axvline(0.5, color=INK_SOFT, linewidth=1.5)
    ax.text(0.505, -0.75, "0,5 = hasard", color=INK_SOFT, fontsize=14)
    ax.set_xlim(0.5, 1.06)
    ax.set_ylim(-0.9, 4.5)
    french_ticks(ax, "x", 1)
    ax.set_xlabel("AUC pour prédire « région éloignée » (1 = la variable révèle la région)")
    ax.set_title("Retirer la région ne suffit pas : les proxys la portent")
    ax.grid(axis="y", visible=False)
    return save(fig, "fig3_proxys_auc.png"), scores


def box(ax, x, y, w, h, text, color, size=15, weight="normal", style="round,pad=0.02,rounding_size=0.04",
        dashed=False):
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, linewidth=2.2, edgecolor=color,
                           facecolor=color + "1f", linestyle="--" if dashed else "-")
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=size, fontweight=weight, color=INK, linespacing=1.35)


def arrow(ax, start, end, color=INK_SOFT, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=22, linewidth=2, color=color,
                                 linestyle="--" if dashed else "-"))


def pipeline_figure(record):
    blend = record["config"]["residual_blend"]
    weights = record["training_labels"]["reference_weights"]["consensus"]
    hours, income = weights["heures_travail"], weights["log_revenu"]
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Un modèle entraîné sur l'historique seulement, sous garde-fous automatiques", pad=12)
    y = 5.2
    box(ax, 1.6, y, 2.8, 1.7, "Historique\n10 000 demandes\n(seule source)", COMMITTEE, weight="bold")
    box(ax, 5.2, y, 3.4, 1.7, f"Base de consensus\nR + {num(hours)} × heures\n+ {num(income)} × revenu",
        CENTRE, weight="bold")
    box(ax, 9.05, y, 3.75, 1.7, f"+ {num(blend, 1)} × résidu TabM\n4 réseaux\n(R, heures ; GPU Kaggle)",
        DECLARED, weight="bold")
    box(ax, 12.4, y, 2.6, 1.7, f"k meilleurs\n{pct(record['share']).replace(' %', '')} %\ndu lot", FINAL, weight="bold")
    box(ax, 14.9, y, 1.9, 1.7, "PUBLIER\nou\nBLOQUER", ALERT, weight="bold")
    for a, b in ((3.0, 3.5), (6.9, 7.17), (10.93, 11.1), (13.7, 13.95)):
        arrow(ax, (a, y), (b, y))
    box(ax, 6.2, 7.25, 7.0, 0.9, "Diagnostic : modèle du comité (pénalité, rapport heures / R)", COMMITTEE, size=14)
    arrow(ax, (1.6, 6.05), (2.7, 7.25), dashed=True)
    arrow(ax, (5.2, 6.8), (5.2, 6.05), dashed=True)
    box(ax, 5.4, 2.1, 7.0, 1.6, "Jury des 5 règles + raisonnement (MODE AUDIT)\nvotes et traces pour chaque cas limite"
        "\n→ explications Loi 25, aucune décision déplacée", OLD, size=14, dashed=True)
    arrow(ax, (12.0, 4.35), (8.9, 2.9), dashed=True)
    box(ax, 12.9, 2.1, 5.6, 1.6, "4 couches de garde + garde de sortie\nsurveillance · jury · consensus · forte"
        "\ndécalage |δ| ≤ 0,10 · REVERT_JURY · BLOCK", ALERT, size=14)
    arrow(ax, (14.9, 4.35), (14.4, 2.9), color=ALERT)
    ax.text(0.2, 0.55, "Le score ne lit jamais région, code postal ni distance. Les décisions ne servent jamais "
            "à l'entraînement.", fontsize=14, color=INK_SOFT)
    return save(fig, "fig4_pipeline.png")


def pareto_figure():
    summary = pd.read_csv(ROOT / "resultats_pareto.csv")
    x, y = "eo_gap_reviewers_mean", "agree_reviewers_mean"
    fig, ax = plt.subplots(figsize=(14, 7.5))
    sweeps = [("penalty removal sweep", "Comité : retrait de la pénalité 0 → 1", CENTRE, "o"),
              ("ExpGrad sweep", "ExponentiatedGradient (ε)", REMOTE, "s"),
              ("declared base removal sweep", "Base déclarée : retrait de la pénalité 0 → 1", DECLARED, "d")]
    sweeps = [sweep for sweep in sweeps if (summary.family == sweep[0]).any()]
    for family, label, color, marker in sweeps:
        part = summary[summary.family == family].sort_values("setting")
        ax.plot(part[x], part[y], color=color, marker=marker, markersize=10, linewidth=2.5, label=label, alpha=0.85)
    others = summary[~summary.family.isin([f for f, *_ in sweeps]) & (summary.method != DECLARED_METHOD)]
    feasible = others[others.budget_ok]
    ax.scatter(feasible[x], feasible[y], s=90, color=COMMITTEE, zorder=3, label="Autres méthodes (même budget)")
    outside = others[~others.budget_ok]
    ax.scatter(outside[x], outside[y], s=90, facecolors="none", edgecolors=COMMITTEE, linewidths=2, zorder=3,
               label="Hors budget")
    front = summary[pareto_mask(summary, x, y)].sort_values(x)
    declared = summary[summary.method == DECLARED_METHOD].iloc[0]
    ax.scatter([declared[x]], [declared[y]], s=700, marker="*", color=DECLARED, zorder=5, label="Point déclaré (base)")
    labels = {"production RF at budget": ("Production (RF)", (-60, -30)),
              "drop proxies only": ("Retrait des proxys", (10, -6)),
              "ThresholdOptimizer, natural rate": ("ThresholdOptimizer", (-175, -6)),
              "income-blind validator jury": ("Jury sans revenu", (12, -22))}
    for method, (text, offset) in labels.items():
        row = summary[summary.method == method].iloc[0]
        ax.annotate(text, (row[x], row[y]), textcoords="offset points", xytext=offset, fontsize=13, color=INK_SOFT)
    ax.annotate(f"Déclaré : écart {num(declared[x])}\naccord {num(declared[y])}", (declared[x], declared[y]),
                textcoords="offset points", xytext=(30, -30), fontsize=16, fontweight="bold", color=DECLARED)
    french_ticks(ax)
    ax.set_xlabel("Écart d'égalité des chances moyen vs 5 références (↓ mieux)")
    ax.set_ylabel("Accord moyen avec 5 références (↑ mieux)")
    ax.set_title("Front de Pareto : 10 partitions de l'historique")
    ax.legend(loc="lower left", fontsize=13)
    return save(fig, "fig5_pareto.png"), declared, front.method.tolist()


def active_jury(history, batch, residual):
    config = replace(DECLARED_CONFIG, name="active jury", jury=replace(DECLARED_CONFIG.jury, audit_only=False))
    record = decide(history, batch, {}, config, residual)
    checks = {check.name: check for check in record.jury_guard.checks}
    effect = checks["jury fairness effect (opportunity gap, worst reference)"]
    tokens = effect.detail.split("; ")[1].split()
    effects = {tokens[i]: float(tokens[i + 1]) for i in range(0, len(tokens), 2)}
    swaps = int(checks["jury swap volume"].detail.split()[0].replace(",", ""))
    reverted = any(action.kind.name == "REVERT_JURY" for action in record.actions)
    return effects, swaps, reverted, record.status


def jury_figure(effects, swaps, reverted):
    names = {"merit": "Mérite (R seul)", "consensus": "Règle de consensus", "corrected": "Comité corrigé"}
    fig, ax = plt.subplots(figsize=(14, 7))
    keys = [k for k in ("merit", "consensus", "corrected") if k in effects]
    values = [effects[k] for k in keys]
    bars = ax.bar([names[k] for k in keys], values, width=0.55, color=[ALERT if v > 0.01 else OLD for v in values])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0006, "+" + num(value), ha="center", fontsize=18,
                fontweight="bold")
    ax.axhline(0.01, color=ALERT, linestyle="--", linewidth=2, label="Seuil d'ALERTE (0,010)")
    ax.axhline(0.005, color=OLD, linestyle=":", linewidth=2.5, label="Seuil WARN (0,005)")
    ax.legend(loc="upper right")
    french_ticks(ax, "y", 3)
    ax.set_ylim(0, max(values) * 1.3)
    ax.set_ylabel("Hausse de l'écart d'égalité des chances")
    ax.set_title(f"Jury actif : {swaps} échanges auraient aggravé l'équité")
    verdict = "REVERT_JURY : jury retiré, décision du modèle publiée" if reverted else "jury conservé"
    ax.text(0.5, -0.17, f"{swaps} échanges proposés → garde du jury en ALERTE → {verdict}",
            transform=ax.transAxes, ha="center", fontsize=16, fontweight="bold", color=DECLARED)
    ax.grid(axis="x", visible=False)
    return save(fig, "fig6_jury_revert.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record = json.loads((ROOT / "decision_record.json").read_text())
    history, batch, residual = inputs()
    paths = [region_rates_figure(history, record["region_rates"])]
    path, band = same_r_figure(history)
    paths.append(path)
    path, proxies = proxy_figure(history)
    paths.append(path)
    paths.append(pipeline_figure(record))
    path, declared, front = pareto_figure()
    paths.append(path)
    effects, swaps, reverted, status = active_jury(history, batch, residual)
    paths.append(jury_figure(effects, swaps, reverted))
    history_rates = history.groupby(is_remote(history))["decision_octroi"].mean()
    print(f"historique centre {history_rates[0]:.4f} éloigné {history_rates[1]:.4f}")
    print(f"R 28-30 centre {band[0]:.4f} éloigné {band[1]:.4f}")
    print("proxys " + ", ".join(f"{k} {v:.4f}" for k, v in proxies.items()))
    print(f"final {record['region_rates']}")
    print(f"pareto déclaré écart {declared['eo_gap_reviewers_mean']:.4f} accord {declared['agree_reviewers_mean']:.4f}; front {front}")
    print(f"jury actif : {swaps} échanges, effets {effects}, REVERT_JURY {reverted}, statut {status}")
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()

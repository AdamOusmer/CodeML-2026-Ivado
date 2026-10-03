"""ÉquiAlgo mitigation pipeline: ablation, Pareto front, stability, predictions.csv.

1. Preprocessing   : legitimate features only; region is kept for auditing, never for decisions.
2. Fairness (pre)  : remove the committee's remote-region penalty from the training labels.
3. Main model      : region-blind logistic regression trained on the corrected labels.
4. Jury            : rank vote between fairness policies (main model, academic merit).
5. Fairness (post) : remote score offset that equalizes opportunity, zero when no gap remains.
6. Allocation      : grant the historical share of applicants, the institution's budget.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.policy.core import (
    BUDGET_BOUNDS,
    allocate,
    budget_share,
    eo_gap,
    legitimate_features,
    logistic_regression,
    production_features,
)
from src.policy.models import DECLARED_CONFIG, CommitteeModel, Config, FairPipeline, reference_labels
from src.policy.regions import is_remote

ROOT = Path(__file__).parent
EXPGRAD_BOUNDS = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
STABILITY_SPLITS = 30
SEED = 42
MAX_WORKERS = 4


def evaluate(name, family, decisions, references, remote, setting=np.nan):
    low, high = BUDGET_BOUNDS
    centre_rate, remote_rate = decisions[remote == 0].mean(), decisions[remote == 1].mean()
    return {
        "method": name,
        "family": family,
        "setting": setting,
        "grant_rate": decisions.mean(),
        "budget_ok": low <= decisions.mean() <= high,
        "rate_centre": centre_rate,
        "rate_remote": remote_rate,
        "dp_gap": abs(centre_rate - remote_rate),
        "eo_gap_corrected": eo_gap(decisions, references["corrected"], remote),
        "eo_gap_merit": eo_gap(decisions, references["merit"], remote),
        "eo_gap_historical": eo_gap(decisions, references["historical"], remote),
        "agree_corrected": (decisions == references["corrected"]).mean(),
        "agree_merit": (decisions == references["merit"]).mean(),
        "acc_historical": (decisions == references["historical"]).mean(),
    }


def compare_methods(train, test, share):
    y_train, remote_train, remote_test = train["decision_octroi"].to_numpy(), is_remote(train), is_remote(test)
    references = reference_labels(train, test, share)

    def row(name, family, decisions, setting=np.nan):
        return evaluate(name, family, decisions, references, remote_test, setting)

    X_prod_train, X_prod_test = production_features(train, test)
    production = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=SEED, n_jobs=MAX_WORKERS)
    production.fit(X_prod_train, y_train)
    rows = [
        row("production RF, natural threshold (official anchor)", "baseline", production.predict(X_prod_test)),
        row("production RF at budget", "baseline", allocate(production.predict_proba(X_prod_test)[:, 1], share)),
    ]

    region_blind = logistic_regression().fit(legitimate_features(train), y_train)
    region_blind_scores = region_blind.predict_proba(legitimate_features(test))[:, 1]
    rows.append(row("drop proxies only (region-blind LR) at budget", "ablation", allocate(region_blind_scores, share)))

    def expgrad_decisions(bound):
        model = ExponentiatedGradient(
            logistic_regression(),
            constraints=TruePositiveRateParity(difference_bound=bound),
            sample_weight_name="logisticregression__sample_weight",
        )
        model.fit(legitimate_features(train), y_train, sensitive_features=remote_train)
        return allocate(model._pmf_predict(legitimate_features(test))[:, 1], share)

    def pipeline_decisions(config):
        return FairPipeline(config, share).fit(train).predict(test)

    sweep = [Config(f"removal {removal:.1f}", removal=removal) for removal in np.linspace(0, 1, 11)]
    ablations = {
        "pipeline without jury (main model only)": Config("main model only", jury_weights={"main_model": 1.0}),
        "merit only (R score, no income or hours)": Config("merit only", jury_weights={"merit": 1.0}),
        "FULL pipeline": DECLARED_CONFIG,
    }
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        expgrad = list(pool.map(expgrad_decisions, EXPGRAD_BOUNDS))
        swept = list(pool.map(pipeline_decisions, sweep))
        ablated = list(pool.map(pipeline_decisions, ablations.values()))

    rows += [row(f"ExpGrad eps={bound} at budget", "ExpGrad sweep", d, bound) for bound, d in zip(EXPGRAD_BOUNDS, expgrad)]

    threshold_optimizer = ThresholdOptimizer(
        estimator=region_blind, constraints="true_positive_rate_parity",
        objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba",
    )
    threshold_optimizer.fit(legitimate_features(train), y_train, sensitive_features=remote_train)
    rows.append(row("ThresholdOptimizer, natural rate (cannot fix budget)", "post-processing",
                    threshold_optimizer.predict(legitimate_features(test), sensitive_features=remote_test, random_state=SEED)))

    rows += [row(f"pipeline, penalty removal={c.removal:.1f}", "pipeline sweep", d, c.removal) for c, d in zip(sweep, swept)]
    rows += [row(name, "full pipeline" if name == "FULL pipeline" else "ablation", d) for name, d in zip(ablations, ablated)]
    return pd.DataFrame(rows)


def stability(history, share):
    def one_split(seed):
        train, test = train_test_split(history, test_size=0.3, random_state=seed, stratify=history["decision_octroi"])
        references = reference_labels(train, test, share)
        remote = is_remote(test)
        decisions = FairPipeline(DECLARED_CONFIG, share).fit(train).predict(test)
        committee_decisions = allocate(CommitteeModel().fit(train, train["decision_octroi"].to_numpy()).corrected_logit(test, 0.0), share)
        return {
            "eo_gap_merit": eo_gap(decisions, references["merit"], remote),
            "eo_gap_corrected": eo_gap(decisions, references["corrected"], remote),
            "closure_merit": 1 - eo_gap(decisions, references["merit"], remote) / eo_gap(committee_decisions, references["merit"], remote),
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        return pd.DataFrame(pool.map(one_split, range(STABILITY_SPLITS)))


def plot_pareto(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, gap, title in [
        (axes[0], "eo_gap_corrected", "vs corrected labels (committee minus regional penalty)"),
        (axes[1], "eo_gap_merit", "vs merit-only labels (top R scores)"),
    ]:
        for family, marker in [("pipeline sweep", "o-"), ("ExpGrad sweep", "s--")]:
            part = results[results.family == family].sort_values(gap)
            ax.plot(part[gap], part["acc_historical"], marker, label=family)
        for family, marker in [("baseline", "kX"), ("ablation", "m^"), ("post-processing", "cD"), ("full pipeline", "r*")]:
            part = results[results.family == family]
            ax.plot(part[gap], part["acc_historical"], marker, markersize=12, label=family, linestyle="none")
        ax.set_xlabel(f"Equal-opportunity gap {title}")
        ax.set_ylabel("Agreement with historical committee")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Pareto front: fairness vs fidelity to the audited committee")
    fig.tight_layout()
    fig.savefig(ROOT / "pareto_front.png", dpi=150)


def main():
    history = pd.read_csv(ROOT / "data" / "donnees_demandes.csv")
    candidates = pd.read_csv(ROOT / "data" / "candidats_evaluation.csv")
    share = budget_share(history)
    print(f"Budget: historical grant rate {share:.2%}")

    train, test = train_test_split(history, test_size=0.3, random_state=SEED, stratify=history["decision_octroi"])
    committee = CommitteeModel().fit(train, train["decision_octroi"].to_numpy())
    print(f"Fairness (pre): remote penalty {committee.remote_penalty_:.2f} log-odds "
          f"(odds x {np.exp(committee.remote_penalty_):.2f})")

    results = compare_methods(train, test, share)
    pd.set_option("display.width", 250)
    columns = ["method", "grant_rate", "budget_ok", "rate_centre", "rate_remote", "dp_gap",
               "eo_gap_corrected", "eo_gap_merit", "eo_gap_historical", "agree_merit", "acc_historical"]
    print(results[columns].round(3).to_string(index=False))
    results.to_csv(ROOT / "resultats_pareto.csv", index=False)
    plot_pareto(results)

    splits = stability(history, share)
    splits.to_csv(ROOT / "resultats_stabilite.csv", index=False)
    print(f"\nStability over {STABILITY_SPLITS} splits (mean ± std, 95% range):")
    for column in splits.columns:
        low, high = splits[column].quantile([0.025, 0.975])
        print(f"  {column:<17} {splits[column].mean():.3f} ± {splits[column].std():.3f}   [{low:.3f}, {high:.3f}]")

    final = FairPipeline(DECLARED_CONFIG, share).fit(history)
    decisions = final.predict(candidates)
    submission = pd.DataFrame({"id_candidat": candidates["id_candidat"], "decision_octroi": decisions})
    submission.to_csv(ROOT / "predictions.csv", index=False)

    remote = is_remote(candidates)
    print(f"predictions.csv: {len(submission)} rows, grant rate {decisions.mean():.2%} "
          f"(centre {decisions[remote == 0].mean():.1%}, remote {decisions[remote == 1].mean():.1%})")
    low, high = BUDGET_BOUNDS
    if len(submission) != len(candidates) or not low <= decisions.mean() <= high:
        raise ValueError("predictions.csv violates the submission format or budget")


if __name__ == "__main__":
    main()

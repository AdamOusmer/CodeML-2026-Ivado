"""EquiAlgo mitigation: three-stage pipeline, ablation, Pareto front, predictions.csv.

Stage 1 (pre-processing)  : fit the committee's rule with an explicit remote-region term,
                            then remove that term to obtain corrected labels.
Stage 2 (in-processing)   : deployment model trained on corrected labels, without region,
                            postal code or distance.
Stage 3 (post-processing) : budget lock, grant exactly the top BUDGET share.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
REMOTE_REGIONS = ["Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"]
PROGRAMMES = ["Arts et lettres", "Genie", "Sante", "Sciences", "Sciences sociales"]
BUDGET = 0.40
SEED = 42
EXPGRAD_BOUNDS = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
MAX_WORKERS = 4


def is_remote(df):
    return df["region_administrative"].isin(REMOTE_REGIONS).to_numpy().astype(int)


def legitimate_features(df):
    X = pd.DataFrame(
        {
            "cote_r": df["cote_r_equivalent"],
            "log_revenu": np.log(df["revenu_familial_estime"]),
            "heures_travail": df["heures_travail_semaine"],
            "premiere_generation": df["premiere_generation_universitaire"],
        }
    )
    for programme in PROGRAMMES[1:]:
        X[f"prog_{programme}"] = (df["programme_etudes"] == programme).astype(float)
    return X


def production_features(train_df, target_df):
    categorical = ["programme_etudes", "region_administrative", "code_postal_3"]
    drop = ["id_candidat", "decision_octroi"]
    X = pd.get_dummies(train_df.drop(columns=drop, errors="ignore"), columns=categorical)
    Xt = pd.get_dummies(target_df.drop(columns=drop, errors="ignore"), columns=categorical)
    return X, Xt.reindex(columns=X.columns, fill_value=0)


def top_share(scores, share=BUDGET):
    k = int(round(share * len(scores)))
    decisions = np.zeros(len(scores), dtype=int)
    decisions[np.argsort(-scores, kind="stable")[:k]] = 1
    return decisions


class CommitteeModel:
    def fit(self, df, y):
        X = legitimate_features(df)
        self.scaler_ = StandardScaler().fit(X)
        Z = np.column_stack([self.scaler_.transform(X), is_remote(df)])
        self.lr_ = LogisticRegression(max_iter=3000).fit(Z, y)
        self.remote_penalty_ = self.lr_.coef_[0][-1]
        return self

    def corrected_logit(self, df, removal=1.0):
        Z = self.scaler_.transform(legitimate_features(df))
        base = Z @ self.lr_.coef_[0][:-1] + self.lr_.intercept_[0]
        return base + (1.0 - removal) * self.remote_penalty_ * is_remote(df)


def tpr(y_ref, decisions, mask):
    positives = (y_ref == 1) & mask
    return decisions[positives].mean() if positives.any() else np.nan


def evaluate(name, decisions, y_hist, y_corrected, y_merit, remote, family, setting=np.nan):
    remote_mask, centre_mask = remote == 1, remote == 0

    def eo_gap(y_ref):
        return abs(tpr(y_ref, decisions, centre_mask) - tpr(y_ref, decisions, remote_mask))

    return {
        "method": name,
        "family": family,
        "setting": setting,
        "grant_rate": decisions.mean(),
        "rate_centre": decisions[centre_mask].mean(),
        "rate_remote": decisions[remote_mask].mean(),
        "dp_gap": abs(decisions[centre_mask].mean() - decisions[remote_mask].mean()),
        "eo_gap_corrected": eo_gap(y_corrected),
        "eo_gap_merit": eo_gap(y_merit),
        "eo_gap_historical": eo_gap(y_hist),
        "agree_corrected": (decisions == y_corrected).mean(),
        "acc_historical": (decisions == y_hist).mean(),
    }


def main():
    history = pd.read_csv(ROOT / "data" / "donnees_demandes.csv")
    candidates = pd.read_csv(ROOT / "data" / "candidats_evaluation.csv")

    train, test = train_test_split(
        history, test_size=0.3, random_state=SEED, stratify=history["decision_octroi"]
    )
    y_train = train["decision_octroi"].to_numpy()
    y_test = test["decision_octroi"].to_numpy()
    remote_train, remote_test = is_remote(train), is_remote(test)

    committee = CommitteeModel().fit(train, y_train)
    print(f"Stage 1: remote penalty {committee.remote_penalty_:.2f} log-odds "
          f"(odds x {np.exp(committee.remote_penalty_):.2f})")

    y_corrected_test = top_share(committee.corrected_logit(test))
    y_merit_test = top_share(test["cote_r_equivalent"].to_numpy())

    def score(name, decisions, family, setting=np.nan):
        return evaluate(name, decisions, y_test, y_corrected_test, y_merit_test, remote_test, family, setting)

    rows = []

    X_prod_train, X_prod_test = production_features(train, test)
    production = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=SEED, n_jobs=MAX_WORKERS)
    production.fit(X_prod_train, y_train)
    rows.append(score("production RF (baseline)", production.predict(X_prod_test), "baseline"))

    raw_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    raw_lr.fit(legitimate_features(train), y_train)
    raw_scores = raw_lr.predict_proba(legitimate_features(test))[:, 1]
    rows.append(score("stage 3 only (LR, no region, budget lock)", top_share(raw_scores), "ablation"))

    for removal in np.linspace(0, 1, 11):
        decisions = top_share(committee.corrected_logit(test, removal))
        rows.append(score(f"stage 1 removal={removal:.1f} + stage 3", decisions, "stage 1 sweep", removal))

    def expgrad_decisions(bound):
        eg = ExponentiatedGradient(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
            constraints=TruePositiveRateParity(difference_bound=bound),
            sample_weight_name="logisticregression__sample_weight",
        )
        eg.fit(legitimate_features(train), y_train, sensitive_features=remote_train)
        return top_share(eg._pmf_predict(legitimate_features(test))[:, 1])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for eps, decisions in zip(EXPGRAD_BOUNDS, pool.map(expgrad_decisions, EXPGRAD_BOUNDS)):
            rows.append(score(f"stage 2 only (ExpGrad eps={eps}) + stage 3", decisions, "ExpGrad sweep", eps))

    to = ThresholdOptimizer(
        estimator=raw_lr, constraints="true_positive_rate_parity",
        objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba",
    )
    to.fit(legitimate_features(train), y_train, sensitive_features=remote_train)
    decisions = to.predict(legitimate_features(test), sensitive_features=remote_test, random_state=SEED)
    rows.append(score("ThresholdOptimizer (TPR parity, no budget lock)", decisions, "post-processing"))

    y_corrected_train = top_share(committee.corrected_logit(train))
    deployment = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    deployment.fit(legitimate_features(train), y_corrected_train)
    full_scores = deployment.predict_proba(legitimate_features(test))[:, 1]
    rows.append(score("FULL: stage 1 + stage 2 + stage 3", top_share(full_scores), "full pipeline"))

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    columns = ["method", "grant_rate", "rate_centre", "rate_remote", "dp_gap",
               "eo_gap_corrected", "eo_gap_merit", "eo_gap_historical", "agree_corrected", "acc_historical"]
    print(results[columns].round(3).to_string(index=False))
    results.to_csv(ROOT / "resultats_pareto.csv", index=False)

    plot_pareto(results)

    final_committee = CommitteeModel().fit(history, history["decision_octroi"].to_numpy())
    final_deployment = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    final_deployment.fit(legitimate_features(history), top_share(final_committee.corrected_logit(history)))
    final_decisions = top_share(final_deployment.predict_proba(legitimate_features(candidates))[:, 1])

    submission = pd.DataFrame({"id_candidat": candidates["id_candidat"], "decision_octroi": final_decisions})
    submission.to_csv(ROOT / "predictions.csv", index=False)
    remote_candidates = is_remote(candidates)
    print(f"\npredictions.csv: {len(submission)} rows, grant rate {final_decisions.mean():.1%} "
          f"(centre {final_decisions[remote_candidates == 0].mean():.1%}, "
          f"remote {final_decisions[remote_candidates == 1].mean():.1%})")
    assert len(submission) == 4000 and 0.36 <= final_decisions.mean() <= 0.44


def plot_pareto(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, gap, title in [
        (axes[0], "eo_gap_corrected", "vs corrected labels (stage 1 reference)"),
        (axes[1], "eo_gap_merit", "vs merit-only labels (R score top 40%)"),
    ]:
        for family, marker in [("stage 1 sweep", "o-"), ("ExpGrad sweep", "s--")]:
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


if __name__ == "__main__":
    main()

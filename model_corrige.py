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
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
REMOTE_REGIONS = ["Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"]
PROGRAMMES = ["Arts et lettres", "Genie", "Sante", "Sciences", "Sciences sociales"]
BUDGET_BOUNDS = (0.36, 0.44)
JURY_WEIGHTS = {"main_model": 0.5, "merit": 0.5}
POST_OFFSETS = np.linspace(-0.10, 0.10, 41)
EXPGRAD_BOUNDS = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
STABILITY_SPLITS = 30
SEED = 42
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


def budget_share(history):
    share = history["decision_octroi"].mean()
    low, high = BUDGET_BOUNDS
    if not low <= share <= high:
        raise ValueError(f"Historical grant rate {share:.1%} is outside the allowed budget {low:.0%}-{high:.0%}")
    return share


def allocate(scores, share):
    k = int(round(share * len(scores)))
    decisions = np.zeros(len(scores), dtype=int)
    decisions[np.argsort(-scores, kind="stable")[:k]] = 1
    return decisions


def percentile(values):
    return rankdata(values) / len(values)


def logistic_regression():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def eo_gap(decisions, y_ref, remote):
    deserving = y_ref == 1
    return abs(decisions[deserving & (remote == 0)].mean() - decisions[deserving & (remote == 1)].mean())


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


class FairPipeline:
    def __init__(self, share, removal=1.0, jury_weights=JURY_WEIGHTS, post_processing=True):
        self.share = share
        self.removal = removal
        self.jury_weights = jury_weights
        self.post_processing = post_processing

    def fit(self, train):
        self.committee_ = CommitteeModel().fit(train, train["decision_octroi"].to_numpy())
        corrected = allocate(self.committee_.corrected_logit(train, self.removal), self.share)
        self.main_model_ = logistic_regression().fit(legitimate_features(train), corrected)
        self.remote_offset_ = 0.0
        if self.post_processing:
            references = [corrected, allocate(train["cote_r_equivalent"].to_numpy(), self.share)]
            self.remote_offset_ = self._fit_offset(train, references)
        return self

    def jury_score(self, df):
        votes = {
            "main_model": percentile(self.main_model_.predict_proba(legitimate_features(df))[:, 1]),
            "merit": percentile(df["cote_r_equivalent"].to_numpy()),
        }
        return sum(weight * votes[juror] for juror, weight in self.jury_weights.items())

    def score(self, df):
        return self.jury_score(df) + self.remote_offset_ * is_remote(df)

    def predict(self, df):
        return allocate(self.score(df), self.share)

    def _fit_offset(self, train, references):
        jury, remote = self.jury_score(train), is_remote(train)

        def mean_gap(offset):
            decisions = allocate(jury + offset * remote, self.share)
            return np.mean([eo_gap(decisions, y_ref, remote) for y_ref in references])

        return min(POST_OFFSETS, key=lambda offset: (round(mean_gap(offset), 3), abs(offset)))


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


def held_out_references(train, test, share):
    committee = CommitteeModel().fit(train, train["decision_octroi"].to_numpy())
    return {
        "historical": test["decision_octroi"].to_numpy(),
        "corrected": allocate(committee.corrected_logit(test), share),
        "merit": allocate(test["cote_r_equivalent"].to_numpy(), share),
    }


def compare_methods(train, test, share):
    y_train, remote_train, remote_test = train["decision_octroi"].to_numpy(), is_remote(train), is_remote(test)
    references = held_out_references(train, test, share)

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

    def pipeline_decisions(settings):
        return FairPipeline(share, **settings).fit(train).predict(test)

    sweep = [{"removal": removal, "post_processing": False} for removal in np.linspace(0, 1, 11)]
    ablations = {
        "pipeline without jury (main model only)": {"jury_weights": {"main_model": 1.0}},
        "pipeline without post-processing": {"post_processing": False},
        "merit only (R score, no income or hours)": {"jury_weights": {"merit": 1.0}, "post_processing": False},
        "FULL pipeline": {},
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

    rows += [row(f"pipeline (no post), penalty removal={s['removal']:.1f}", "pipeline sweep", d, s["removal"]) for s, d in zip(sweep, swept)]
    rows += [row(name, "full pipeline" if name == "FULL pipeline" else "ablation", d) for name, d in zip(ablations, ablated)]
    return pd.DataFrame(rows)


def stability(history, share):
    def one_split(seed):
        train, test = train_test_split(history, test_size=0.3, random_state=seed, stratify=history["decision_octroi"])
        references = held_out_references(train, test, share)
        remote = is_remote(test)
        decisions = FairPipeline(share).fit(train).predict(test)
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

    final = FairPipeline(share).fit(history)
    decisions = final.predict(candidates)
    submission = pd.DataFrame({"id_candidat": candidates["id_candidat"], "decision_octroi": decisions})
    submission.to_csv(ROOT / "predictions.csv", index=False)

    remote = is_remote(candidates)
    print(f"\nFairness (post): remote offset {final.remote_offset_:+.3f}")
    print(f"predictions.csv: {len(submission)} rows, grant rate {decisions.mean():.2%} "
          f"(centre {decisions[remote == 0].mean():.1%}, remote {decisions[remote == 1].mean():.1%})")
    low, high = BUDGET_BOUNDS
    if len(submission) != len(candidates) or not low <= decisions.mean() <= high:
        raise ValueError("predictions.csv violates the submission format or budget")


if __name__ == "__main__":
    main()

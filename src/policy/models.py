from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .core import allocate, logistic_regression
from .jurors import MODEL_JURORS, fit_model_jurors, holdout_auc, model_juror_scores
from .jury import JurySettings, JuryOutcome, validate
from .label_correction import CommitteeModel, correct_labels, effective_removal
from .main_models import contributions, fit_main_model, logit
from .reasoning import ReasoningSettings, deliberate
from .references import (DECLARED_HOURS_WEIGHT, DECLARED_INCOME_WEIGHT, REFERENCE_JURORS, REVIEWER_NAMES,
                         consensus_labels, reference_decisions, reference_scores)
from .regions import is_remote
from .schema import COMMITTEE_FEATURES, SCORING_FEATURES, committee_features, feature_frame

TARGETS = ("corrected", "consensus")
RATIO_STEP = 0.5
DECLARED_RESIDUAL_BLEND = 2.5
SINGLE_RESIDUAL_BLEND = 1.0
SINGLE_RESIDUAL_ARTEFACT = "tabm_residual"
ENSEMBLE_RESIDUAL_ARTEFACT = "tabm_residual_ensemble_rh"
RESIDUAL_COLUMN = "residual_rsd"


@dataclass(frozen=True)
class Config:
    name: str
    removal: float = 1.0
    jury: JurySettings = JurySettings()
    discounts: tuple[tuple[str, float], ...] = ()
    blend: tuple[str, ...] = ()
    target: str = "corrected"
    reasoning: ReasoningSettings | None = None
    strong_guard: bool = False
    residual_blend: float | None = None
    residual_artefact: str = SINGLE_RESIDUAL_ARTEFACT

    def __post_init__(self):
        if self.target not in TARGETS:
            raise ValueError(f"Unknown target: {self.target}")
        if self.residual_blend is not None and self.target != "consensus":
            raise ValueError("A residual requires the consensus target")

    @property
    def features(self):
        dropped = {name for name, discount in self.discounts if discount >= 1.0}
        return [name for name in SCORING_FEATURES if name not in dropped]


VALIDATOR_JURY_CONFIG = Config("validator jury")
INCOME_BLIND_CONFIG = Config("income-blind validator jury", discounts=(("log_revenu", 1.0),))
INCOME_BLIND_NO_JURY_CONFIG = Config("income-blind, no jury", discounts=(("log_revenu", 1.0),), jury=JurySettings(jurors=()))
MODEL_JURY = ("merit", "lr_all", "boosting", "forest", "neural_net", "neighbours")
MODEL_JURY_CONFIG = Config("model jury", jury=JurySettings(quorum=0.8, jurors=MODEL_JURY))
CONSENSUS_PANEL_CONFIG = Config("consensus panel", target="consensus", jury=JurySettings(quorum=0.8, jurors=REFERENCE_JURORS))
AUDIT_PANEL_CONFIG = replace(CONSENSUS_PANEL_CONFIG, name="consensus panel, audit mode",
                             jury=replace(CONSENSUS_PANEL_CONFIG.jury, audit_only=True),
                             reasoning=ReasoningSettings(), strong_guard=True)
SINGLE_RESIDUAL_CONFIG = replace(AUDIT_PANEL_CONFIG, name="declared: base + TabM residual",
                                 residual_blend=SINGLE_RESIDUAL_BLEND)
DECLARED_CONFIG = replace(AUDIT_PANEL_CONFIG, name="declared: base + TabM ensemble residual",
                          residual_blend=DECLARED_RESIDUAL_BLEND, residual_artefact=ENSEMBLE_RESIDUAL_ARTEFACT)
CONFIGS = {config.name: config for config in (VALIDATOR_JURY_CONFIG, INCOME_BLIND_CONFIG, INCOME_BLIND_NO_JURY_CONFIG,
                                               MODEL_JURY_CONFIG,
                                               CONSENSUS_PANEL_CONFIG, AUDIT_PANEL_CONFIG, SINGLE_RESIDUAL_CONFIG,
                                               DECLARED_CONFIG)}

OFFSET_BOUND = 0.10


def check_offset(offset):
    if not abs(offset) <= OFFSET_BOUND:
        raise ValueError(f"Offset {offset} exceeds the bound {OFFSET_BOUND}")


class FairPipeline:
    def __init__(self, config, share, residual=None):
        self.config = config
        self.share = share
        self.residual = residual
        self.guards = None
        if config.residual_blend is not None and residual is None:
            raise ValueError("The configuration needs the TabM residual artefact")

    def fit(self, history):
        history_share = history["decision_octroi"].mean()
        self.label_correction_ = correct_labels(history, history_share, self.config.removal, self.config.discounts)
        self.committee_ = self.label_correction_.committee
        self.training_labels_ = (consensus_labels(self.committee_, history, history_share)
                                 if self.config.target == "consensus" else self.label_correction_.labels)
        if self.config.residual_blend is None:
            self.main_model_ = fit_main_model(self.features(history), self.training_labels_)
        else:
            self.calibration_ = logistic_regression().fit(self.base_score(history)[:, None], self.training_labels_)
        self.scales_ = {"cote_r_equivalent": history["cote_r_equivalent"].std(),
                        "heures_travail_semaine": history["heures_travail_semaine"].std(),
                        "log_revenu": np.log(history["revenu_familial_estime"]).std()}
        self.programme_stats_ = history.groupby("programme_etudes")["cote_r_equivalent"].agg(["mean", "std"])
        models = list(dict.fromkeys(name for name in (*self.config.jury.jurors, *self.config.blend)
                                    if name in MODEL_JURORS))
        self.juror_quality_ = holdout_auc(models, history, self.training_labels_)
        self.model_jurors_ = fit_model_jurors(models, history, self.training_labels_)
        return self

    def features(self, df):
        return feature_frame(df, self.config.features)

    def base_score(self, df):
        return self.committee_.rule_score(df, DECLARED_INCOME_WEIGHT, DECLARED_HOURS_WEIGHT)

    def declared_score(self, df):
        residual = self.residual.reindex(df["id_candidat"]).to_numpy(dtype=float)
        if np.isnan(residual).any():
            raise ValueError(f"No TabM residual for {int(np.isnan(residual).sum()):,} applicants")
        return self.base_score(df) + self.config.residual_blend * residual

    def model_probability(self, df):
        if self.config.residual_blend is not None:
            probability = self.calibration_.predict_proba(self.declared_score(df)[:, None])[:, 1]
        else:
            probability = self.main_model_.predict_proba(self.features(df))[:, 1]
        if not self.config.blend:
            return probability
        blended = model_juror_scores({name: self.model_jurors_[name] for name in self.config.blend}, df)
        return (probability + sum(blended.values())) / (1 + len(blended))

    def juror_scores(self, df):
        merit = df["cote_r_equivalent"].to_numpy(dtype=float)
        stats = self.programme_stats_.reindex(df["programme_etudes"])
        mean = stats["mean"].to_numpy(dtype=float)
        std = stats["std"].to_numpy(dtype=float)
        rules = reference_scores(self.committee_, df)
        return {"merit": merit, "programme_merit": (merit - mean) / std, "hours_credit": rules["consensus"],
                **{f"ref_{name}": rules[name] for name in REVIEWER_NAMES},
                **model_juror_scores(self.model_jurors_, df)}

    def score(self, df, offset=0.0):
        check_offset(offset)
        p = self.model_probability(df)
        return p if offset == 0 else p + offset * is_remote(df)

    def evidence(self, df):
        return {"cote_r": df["cote_r_equivalent"].to_numpy(dtype=float),
                "heures_travail": df["heures_travail_semaine"].to_numpy(dtype=float),
                "consensus": reference_scores(self.committee_, df)["consensus"]}

    def reasoned(self, outcome, df) -> JuryOutcome:
        settings = self.config.reasoning
        k = int(round(self.share * len(df)))
        examined = outcome.triggered if settings.scope == "reviewed" else None
        gate = None if self.guards is None else self.guards(outcome.decisions)
        deliberation = deliberate(outcome.decisions, self.evidence(df), k, settings, examined, gate)
        return replace(outcome, decisions=deliberation.decisions, deliberation=deliberation)

    def decide(self, df, offset=0.0, jury=True) -> JuryOutcome:
        outcome = self.jury_outcome(df, offset, jury)
        return self.reasoned(outcome, df) if self.config.reasoning and jury else outcome

    def learned_ratios(self, df):
        def slope(change):
            shifted = lambda sign: logit(self.model_probability(change(df.copy(), sign * RATIO_STEP)))
            return float((shifted(1) - shifted(-1)).mean() / (2 * RATIO_STEP))

        def add(column):
            def change(frame, step):
                frame[column] = frame[column] + step * self.scales_[column]
                return frame
            return change

        def scale_income(frame, step):
            frame["revenu_familial_estime"] = frame["revenu_familial_estime"] * np.exp(step * self.scales_["log_revenu"])
            return frame

        merit = slope(add("cote_r_equivalent"))
        return {"hours_over_r": slope(add("heures_travail_semaine")) / merit, "income_over_r": slope(scale_income) / merit}

    def jury_outcome(self, df, offset=0.0, jury=True) -> JuryOutcome:
        check_offset(offset)
        probability = self.model_probability(df)
        ranking = probability if offset == 0 else probability + offset * is_remote(df)
        k = int(round(self.share * len(df)))
        settings = self.config.jury if jury else replace(self.config.jury, jurors=())
        scores = self.juror_scores(df) if jury else {}
        return validate(probability, scores, k, settings, ranking=ranking)

    def predict(self, df, offset=0.0):
        return self.decide(df, offset).decisions

    def contributions(self, df):
        if self.config.residual_blend is None:
            return contributions(self.main_model_, self.features(df))
        terms = self.committee_.scaler_.transform(committee_features(df)) * self.committee_.rule_weights(
            DECLARED_INCOME_WEIGHT, DECLARED_HOURS_WEIGHT)
        frame = pd.DataFrame(terms, columns=COMMITTEE_FEATURES, index=df.index)[SCORING_FEATURES]
        frame[RESIDUAL_COLUMN] = self.declared_score(df) - self.base_score(df)
        return frame


def reference_labels(history, target, share):
    committee = CommitteeModel().fit(history, history["decision_octroi"].to_numpy())
    references = {
        "corrected": allocate(committee.corrected_logit(target, effective_removal(committee.remote_penalty_, 1.0)), share),
        "merit": allocate(target["cote_r_equivalent"].to_numpy(), share),
        "consensus": consensus_labels(committee, target, share),
    }
    panel = reference_decisions(committee, target, share)
    references.update({f"reviewer_{name}": panel[name] for name in REVIEWER_NAMES})
    if "decision_octroi" in target:
        references["historical"] = target["decision_octroi"].to_numpy()
    return references

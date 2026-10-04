from importlib import import_module

_EXPORTS = {
    "regions": ("REGIONS", "REMOTE_REGIONS", "is_remote"),
    "core": ("BUDGET_BOUNDS", "budget_share", "allocate", "percentile", "logistic_regression", "eo_gap",
             "signed_eo_gap", "production_features", "auc"),
    "references": ("DECLARED_INCOME_WEIGHT", "DECLARED_HOURS_WEIGHT", "REFERENCE_HOURS_WEIGHTS", "REVIEWER_INCOME_WEIGHTS", "REFERENCE_INCOME_WEIGHTS", "REVIEWER_NAMES",
                   "REFERENCE_JURORS", "reference_scores", "reference_decisions", "consensus_labels",
                   "reference_weights"),
    "thresholds": ("LIMITS", "STRICT_LIMITS", "GRANT_RATE_BANDS", "GRANT_RATE_STRICT_BANDS", "WARN_FRACTION",
                   "MIN_AGREEMENT_WARN", "MIN_AGREEMENT_ALERT", "JUMP_WARN_RATIO", "JUMP_ALERT_RATIO", "DROP_SLACK",
                   "DROP_ALERT_RATIO", "warn_level"),
    "schema": ("PROGRAMMES", "COMMITTEE_FEATURES", "SCORING_FEATURES", "feature_frame", "committee_features",
               "scoring_features"),
    "jury": ("JurySettings", "JuryOutcome", "validate"),
    "reasoning": ("ReasoningSettings", "ReasoningOutcome", "Deliberation", "reason", "deliberate"),
    "jurors": ("MODEL_JURORS",),
    "label_correction": ("CommitteeModel", "LabelCorrection", "correct_labels", "CorrectionReport",
                         "correction_report", "report_warnings", "NO_PENALTY_WARNING"),
    "models": ("Config", "DECLARED_CONFIG", "DECLARED_RESIDUAL_BLEND", "SINGLE_RESIDUAL_BLEND", "SINGLE_RESIDUAL_CONFIG", "VALIDATOR_JURY_CONFIG",
               "INCOME_BLIND_CONFIG", "INCOME_BLIND_NO_JURY_CONFIG", "MODEL_JURY", "MODEL_JURY_CONFIG",
               "CONSENSUS_PANEL_CONFIG", "AUDIT_PANEL_CONFIG", "ACTIVE_BOTH_CONFIG", "ACTIVE_JURY_CONFIG", "ACTIVE_REASONING_CONFIG", "CONFIGS", "OFFSET_BOUND", "FairPipeline",
               "reference_labels"),
}
_MODULE_OF = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = list(_MODULE_OF)


def __getattr__(name):
    if name not in _MODULE_OF:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{_MODULE_OF[name]}", __name__), name)


def __dir__():
    return sorted([*globals(), *__all__])

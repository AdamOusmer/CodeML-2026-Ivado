from importlib import import_module

_EXPORTS = {
    "regions": ("REGIONS", "REMOTE_REGIONS", "is_remote"),
    "core": ("BUDGET_BOUNDS", "budget_share", "allocate", "percentile", "logistic_regression", "eo_gap",
             "production_features"),
    "schema": ("PROGRAMMES", "COMMITTEE_FEATURES", "SCORING_FEATURES", "feature_frame", "committee_features",
               "scoring_features"),
    "models": ("Config", "DECLARED_CONFIG", "CommitteeModel", "FairPipeline", "reference_labels"),
}
_MODULE_OF = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = list(_MODULE_OF)


def __getattr__(name):
    if name not in _MODULE_OF:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{_MODULE_OF[name]}", __name__), name)


def __dir__():
    return sorted([*globals(), *__all__])

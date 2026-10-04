from __future__ import annotations

import numpy as np

from src.monitoring import Check, Verdict, graded, verdict
from src.policy import MODEL_JURORS, auc, eo_gap, is_remote, reference_labels

JUROR_AUC_WARN, JUROR_AUC_ALERT = 0.90, 0.85
LEAKAGE_WARN, LEAKAGE_ALERT = 0.03, 0.05
FAIRNESS_EFFECT_WARN, FAIRNESS_EFFECT_ALERT = 0.005, 0.01
SWAP_SHARE_WARN, SWAP_SHARE_ALERT = 0.03, 0.05
SPLIT_SHARE_WARN = 0.5
EFFECT_REFERENCES = ("merit", "corrected", "consensus")


def model_jurors(pipeline):
    return [name for name in pipeline.config.jury.jurors if name in MODEL_JURORS]


def quality_checks(pipeline):
    return [graded(f"juror quality: {name}", pipeline.juror_quality_[name], JUROR_AUC_WARN, JUROR_AUC_ALERT,
                   higher_is_worse=False, threshold=f"warn < {JUROR_AUC_WARN}, alert < {JUROR_AUC_ALERT}",
                   detail="holdout AUC vs training labels")
            for name in model_jurors(pipeline)]


def region_auc(remote, scores):
    value = auc(remote, scores)
    return max(value, 1 - value)


def leakage_checks(pipeline, batch):
    remote = is_remote(batch)
    main_auc = region_auc(remote, pipeline.model_probability(batch))
    scores = pipeline.juror_scores(batch)
    checks = []
    for name in model_jurors(pipeline):
        juror_auc = region_auc(remote, scores[name])
        checks.append(graded(f"juror region leakage: {name}", juror_auc - main_auc, LEAKAGE_WARN, LEAKAGE_ALERT,
                             threshold=f"warn > {LEAKAGE_WARN}, alert > {LEAKAGE_ALERT}",
                             detail=f"orientation-free region AUC juror {juror_auc:.3f} vs main model {main_auc:.3f}; "
                                    f"absolute leakage {juror_auc - 0.5:.3f}"))
    return checks


def fairness_effect_check(outcome, references, remote):
    effects = {name: eo_gap(outcome.decisions, references[name], remote) - eo_gap(outcome.proposed, references[name], remote)
               for name in EFFECT_REFERENCES if name in references}
    worst = max(effects, key=lambda name: effects[name] if not np.isnan(effects[name]) else np.inf)
    return graded("jury fairness effect (opportunity gap, worst reference)", effects[worst], FAIRNESS_EFFECT_WARN,
                  FAIRNESS_EFFECT_ALERT, threshold=f"warn > {FAIRNESS_EFFECT_WARN}, alert > {FAIRNESS_EFFECT_ALERT}",
                  detail=f"worst {worst}; " + " ".join(f"{name} {effect:+.4f}" for name, effect in effects.items()))


def swap_volume_check(outcome):
    swaps = int((outcome.decisions != outcome.proposed).sum()) // 2
    return graded("jury swap volume", swaps / len(outcome.decisions), SWAP_SHARE_WARN, SWAP_SHARE_ALERT,
                  threshold=f"warn > {SWAP_SHARE_WARN}, alert > {SWAP_SHARE_ALERT}", detail=f"{swaps:,} swaps")


def agreement_check(pipeline, outcome):
    voters = model_jurors(pipeline) or list(pipeline.config.jury.jurors)
    triggered = np.asarray(outcome.triggered, dtype=bool)
    if not triggered.any():
        return None
    votes = np.array([outcome.votes[name] for name in voters])
    split = votes.any(axis=0) & ~votes.all(axis=0)
    return graded("juror agreement: split share of triggered cases", split[triggered].mean(), SPLIT_SHARE_WARN,
                  threshold=f"warn > {SPLIT_SHARE_WARN}", detail=f"{int(triggered.sum()):,} triggered cases")


def jury_guard(pipeline, history, batch, share, outcome) -> Verdict:
    if not pipeline.config.jury.jurors:
        return Verdict("OK", [])
    checks = []
    if model_jurors(pipeline):
        checks += [*quality_checks(pipeline), *leakage_checks(pipeline, batch)]
    references = reference_labels(history, batch, share)
    checks += [fairness_effect_check(outcome, references, is_remote(batch)), swap_volume_check(outcome),
               agreement_check(pipeline, outcome)]
    return verdict([check for check in checks if isinstance(check, Check)])

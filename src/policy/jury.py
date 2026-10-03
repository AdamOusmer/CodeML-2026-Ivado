from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JurySettings:
    band: float = 0.025
    conf: float = 0.15
    disagree: float = 0.30
    quorum: float = 1.0
    jurors: tuple[str, ...] = ("merit", "programme_merit")


@dataclass(frozen=True)
class JuryOutcome:
    decisions: np.ndarray
    proposed: np.ndarray
    triggered: np.ndarray
    reasons: tuple[tuple[str, ...], ...]
    votes: dict[str, np.ndarray]
    overturned: np.ndarray
    overturned_out: np.ndarray
    overturned_in: np.ndarray


def percentile(values):
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    result = np.empty(n, dtype=float)
    result[order] = np.repeat((starts + ends + 1) / 2, ends - starts) / n
    return result


def strongest(indices, strength, rank, count, descending_rank=False):
    tie_break = -rank[indices] if descending_rank else rank[indices]
    return indices[np.lexsort((tie_break, -strength[indices]))[:count]]


def validate(main_probability, juror_scores, k, settings, ranking=None) -> JuryOutcome:
    main_probability = np.asarray(main_probability, dtype=float)
    ranking = main_probability if ranking is None else np.asarray(ranking, dtype=float)
    n = len(main_probability)
    if not 0 <= k <= n:
        raise ValueError("k must be between zero and the applicant count")
    for juror in settings.jurors:
        if juror not in juror_scores:
            raise ValueError(f"Missing juror scores: {juror}")
    if len(ranking) != n or any(len(scores) != n for scores in juror_scores.values()):
        raise ValueError("Juror score lengths must match the applicant count")
    if not np.isfinite(main_probability).all():
        raise ValueError("Main probabilities are not finite")
    if not np.isfinite(ranking).all():
        raise ValueError("Ranking scores are not finite")
    selected = {}
    for juror in settings.jurors:
        selected[juror] = np.asarray(juror_scores[juror], dtype=float)
        if not np.isfinite(selected[juror]).all():
            raise ValueError(f"Juror scores for {juror} are not finite")

    order = np.argsort(-ranking, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    proposed = (rank < k).astype(int)
    cutoff = 1 - k / n if n else 1.0
    main_percentile = percentile(main_probability)
    juror_percentiles = np.array([percentile(selected[juror]) for juror in settings.jurors])
    juror_votes = juror_percentiles > cutoff
    votes = dict(zip(settings.jurors, juror_votes))

    near_cutoff = np.abs(rank - (k - 0.5)) <= settings.band * n
    low_confidence = (0.5 - settings.conf < main_probability) & (main_probability < 0.5 + settings.conf)
    disagreement = np.zeros(n, dtype=bool)
    if settings.jurors:
        disagreement = np.max(np.abs(juror_percentiles - main_percentile), axis=0) > settings.disagree
    triggered = near_cutoff | low_confidence | disagreement
    triggers = (("near_cutoff", near_cutoff), ("low_confidence", low_confidence), ("disagreement", disagreement))
    reasons = tuple(tuple(name for name, mask in triggers if mask[index]) for index in range(n))

    overturned = np.zeros(n, dtype=bool)
    overturned_out = np.empty(0, dtype=int)
    overturned_in = np.empty(0, dtype=int)
    if settings.jurors:
        dissent = juror_votes != proposed
        overturn_share = dissent.mean(axis=0)
        overturned = triggered & (overturn_share >= settings.quorum)
        strength = np.sum(np.abs(juror_percentiles - cutoff) * dissent, axis=0) / np.maximum(dissent.sum(axis=0), 1)
        grants = np.flatnonzero(overturned & (proposed == 1))
        refusals = np.flatnonzero(overturned & (proposed == 0))
        count = min(len(grants), len(refusals))
        overturned_out = strongest(grants, strength, rank, count, descending_rank=True)
        overturned_in = strongest(refusals, strength, rank, count)
    decisions = proposed.copy()
    decisions[overturned_out] = 0
    decisions[overturned_in] = 1
    return JuryOutcome(decisions, proposed, triggered, reasons, votes, overturned, overturned_out, overturned_in)

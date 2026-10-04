from dataclasses import dataclass

import numpy as np

from .jury import percentile

SCOPES = ("reviewed", "all")
PAIRINGS = ("contradicted", "leaning")
MAX_ROUNDS = 3
MAX_APPLY_ROUNDS = 10
TRACE_SEPARATOR = " -> "


@dataclass(frozen=True)
class ReasoningSettings:
    variables: tuple[str, ...] = ("cote_r", "heures_travail", "consensus")
    evidence: float = 0.10
    margin: float = 0.05
    scope: str = "reviewed"
    pairing: str = "contradicted"
    apply_moves: bool = False


@dataclass(frozen=True)
class ReasoningOutcome:
    decisions: np.ndarray
    examined: np.ndarray
    leans: dict[str, np.ndarray]
    net: np.ndarray
    agreements: np.ndarray
    contradictions: np.ndarray
    moved_out: np.ndarray
    moved_in: np.ndarray
    unresolved: np.ndarray
    rounds: int
    traces: tuple[str, ...]


@dataclass(frozen=True)
class Deliberation:
    before: np.ndarray
    decisions: np.ndarray
    outcomes: tuple[str, ...]
    traces: tuple[str, ...]
    counts: dict


def strongest(indices, strength, count):
    return indices[np.lexsort((indices, -strength[indices]))[:count]]


def check_inputs(decisions, evidence, k, settings, examined):
    n = len(decisions)
    if not 0 <= k <= n:
        raise ValueError("k must be between zero and the applicant count")
    if not np.isin(decisions, (0, 1)).all() or int(decisions.sum()) != k:
        raise ValueError("decisions must be binary with exactly k grants")
    if settings.scope not in SCOPES:
        raise ValueError(f"Unknown reasoning scope: {settings.scope}")
    if settings.pairing not in PAIRINGS:
        raise ValueError(f"Unknown reasoning pairing: {settings.pairing}")
    if len(examined) != n:
        raise ValueError("Examined mask length must match the applicant count")
    for name in settings.variables:
        if name not in evidence:
            raise ValueError(f"Missing reasoning evidence: {name}")
        values = np.asarray(evidence[name], dtype=float)
        if len(values) != n:
            raise ValueError("Evidence lengths must match the applicant count")
        if not np.isfinite(values).all():
            raise ValueError(f"Evidence for {name} is not finite")


def verdict_word(verdict, belief):
    if verdict == 0:
        return "weak"
    return "agree" if verdict == belief else "contra"


def outcome_word(initial, final, unresolved, counterpart):
    if final != initial:
        side = "GRANT" if final == 1 else "REFUSE"
        return f"moved to {side} as counterpart" if counterpart else f"moved to {side}"
    if unresolved:
        return "contradicted, kept (no pair within budget)"
    return "kept"


def trace(index, variables, stack, verdicts, initial, final, agree, contra, net, unresolved, counterpart):
    steps = "; ".join(f"{name} {stack[v, index]:+.2f} {verdict_word(verdicts[v, index], initial)}"
                      for v, name in enumerate(variables))
    belief = "GRANT" if initial == 1 else "REFUSE"
    return (f"{steps} | {belief}: {agree} agree, {contra} contra, net {net:+.2f}"
            f"{TRACE_SEPARATOR}{outcome_word(initial, final, unresolved, counterpart)}")


def leaning(belief, net, examined, settings):
    return examined & (np.sign(net) == -belief) & (np.abs(net) >= settings.margin)


def wants_move(verdicts, belief, net, examined, settings):
    agree = (verdicts == belief).sum(axis=0)
    contra = (verdicts == -belief).sum(axis=0)
    return leaning(belief, net, examined, settings) & (contra > agree), agree, contra


def pair(outs, ins, belief, net, examined, settings, strength):
    count = min(len(outs), len(ins))
    if settings.pairing == "contradicted" or len(outs) == len(ins):
        return strongest(outs, strength, count), strongest(ins, strength, count), np.empty(0, dtype=int)
    short, side = (ins, -1) if len(ins) < len(outs) else (outs, 1)
    pool = np.flatnonzero(leaning(belief, net, examined, settings) & (belief == side))
    extra = strongest(np.setdiff1d(pool, short), strength, max(len(outs), len(ins)) - len(short))
    filled = np.concatenate([short, extra])
    count = min(len(outs), len(ins)) + len(extra)
    outs, ins = (outs, filled) if side == -1 else (filled, ins)
    return strongest(outs, strength, count), strongest(ins, strength, count), extra


def reason(decisions, evidence, k, settings, examined=None) -> ReasoningOutcome:
    decisions = np.asarray(decisions, dtype=int)
    n = len(decisions)
    examined = np.ones(n, dtype=bool) if examined is None else np.asarray(examined, dtype=bool)
    check_inputs(decisions, evidence, k, settings, examined)
    cutoff = 1 - k / n if n else 1.0
    variables = settings.variables
    stack = np.array([percentile(np.asarray(evidence[name], dtype=float)) - cutoff for name in variables]).reshape(len(variables), n)
    leans = {name: stack[v] for v, name in enumerate(variables)}
    net = stack.mean(axis=0) if variables else np.zeros(n)
    verdicts = np.sign(stack) * (np.abs(stack) >= settings.evidence)

    initial = np.where(decisions == 1, 1, -1)
    belief = initial.copy()
    moved_out = np.empty(0, dtype=int)
    moved_in = np.empty(0, dtype=int)
    counterpart = np.zeros(n, dtype=bool)
    rounds = 0
    strength = np.abs(net)
    while rounds < MAX_ROUNDS:
        candidates, _, _ = wants_move(verdicts, belief, net, examined, settings)
        outs = np.flatnonzero(candidates & (belief == 1))
        ins = np.flatnonzero(candidates & (belief == -1))
        outs, ins, extra = pair(outs, ins, belief, net, examined, settings, strength)
        if len(outs) == 0:
            break
        rounds += 1
        counterpart[extra] = True
        belief[outs], belief[ins] = -1, 1
        moved_out, moved_in = np.concatenate([moved_out, outs]), np.concatenate([moved_in, ins])

    unresolved, _, _ = wants_move(verdicts, belief, net, examined, settings)
    agree = (verdicts == initial).sum(axis=0)
    contra = (verdicts == -initial).sum(axis=0)
    traces = tuple(
        trace(i, variables, stack, verdicts, initial[i], belief[i], agree[i], contra[i], net[i], unresolved[i], counterpart[i])
        if examined[i] else ""
        for i in range(n)
    )
    final = (belief == 1).astype(int)
    return ReasoningOutcome(final, examined, leans, net, agree, contra, moved_out, moved_in, unresolved, rounds, traces)


def swapped(decisions, moved_out, moved_in):
    result = decisions.copy()
    result[moved_out], result[moved_in] = 0, 1
    return result


def apply_swaps(decisions, evidence, k, settings, examined, first, gate):
    current, applied = decisions.copy(), []
    outcome = first
    for round_number in range(1, MAX_APPLY_ROUNDS + 1):
        accepted = 0
        for out, into in zip(outcome.moved_out, outcome.moved_in):
            candidate = swapped(current, [out], [into])
            if gate is None or gate.admits(candidate, current):
                current = candidate
                applied.append((out, into, round_number))
                accepted += 1
        if gate is None or not accepted or gate.settled(current):
            break
        outcome = reason(current, evidence, k, settings, examined)
    return current, applied


def relabelled(text, outcome_text):
    return f"{text.rsplit(TRACE_SEPARATOR, 1)[0]}{TRACE_SEPARATOR}{outcome_text}"


def deliberate(decisions, evidence, k, settings: ReasoningSettings, examined=None, gate=None) -> Deliberation:
    decisions = np.asarray(decisions, dtype=int)
    first = reason(decisions, evidence, k, settings, examined)
    if settings.apply_moves:
        final, applied = apply_swaps(decisions, evidence, k, settings, examined, first, gate)
        withheld = "a guard would get worse"
    else:
        final, applied, withheld = decisions, [], "traces only"
    traces = list(first.traces)
    words = np.where(first.examined, "kept", "not_examined").astype(object)
    words[first.unresolved] = "contradicted_kept"
    for index in (*first.moved_out, *first.moved_in):
        traces[index] = relabelled(traces[index], f"contradicted, kept (move not applied: {withheld})")
        words[index] = "contradicted_kept"
    for out, into, round_number in applied:
        traces[out] = relabelled(traces[out], f"moved to REFUSE (round {round_number})")
        traces[into] = relabelled(traces[into], f"moved to GRANT (round {round_number})")
        words[out], words[into] = "moved_out", "moved_in"
    counts = {"examined": int(first.examined.sum()), "proposed_moves": len(first.moved_out),
              "moved_out": len(applied), "moved_in": len(applied),
              "contradicted_kept": int((words == "contradicted_kept").sum()),
              "rounds": max((round_number for *_, round_number in applied), default=0)}
    return Deliberation(decisions, final, tuple(words), tuple(traces), counts)

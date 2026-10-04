from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring import Check, Verdict, graded, verdict
from src.policy import (DROP_ALERT_RATIO, DROP_SLACK, GRANT_RATE_BANDS, GRANT_RATE_STRICT_BANDS, JUMP_ALERT_RATIO,
                        JUMP_WARN_RATIO, LIMITS, MIN_AGREEMENT_ALERT, MIN_AGREEMENT_WARN, REFERENCE_INCOME_WEIGHTS,
                        REVIEWER_NAMES, STRICT_LIMITS, allocate, is_remote, reference_decisions as rule_decisions,
                        signed_eo_gap, warn_level)

INTERSECTION_MIN_CELL = 100
HOURS_WINDOW = 2
HOURS_MIN_CELL = 30
R_STRATA = 10
PURE_R = "pure_r"


def reference_decisions(committee, batch, share):
    decisions = rule_decisions(committee, batch, share)
    decisions[PURE_R] = allocate(batch["cote_r_equivalent"].to_numpy(dtype=float), share)
    return decisions


def group_rate(values, mask):
    return values[mask].mean() if mask.any() else np.nan


def outside(value, band):
    return value < band[0] or value > band[1]


def strict_note(met, label):
    return f"{label} {'met' if met else 'missed'}"


def band_check(name, value, group, own, detail="", correctable=True):
    alert_band, strict = GRANT_RATE_BANDS[group], GRANT_RATE_STRICT_BANDS[group]
    warn_band = (warn_level(own, alert_band[0]), warn_level(own, alert_band[1]))
    status = ("ALERT" if np.isnan(value) or outside(value, alert_band)
              else "WARN" if outside(value, warn_band) else "OK")
    excess = float("inf") if np.isnan(value) else float(max(alert_band[0] - value, value - alert_band[1]))
    note = strict_note(not np.isnan(value) and not outside(value, strict), f"strict [{strict[0]}, {strict[1]}]")
    return Check(f"consensus: {name}", float(value),
                 f"warn outside [{warn_band[0]:.4f}, {warn_band[1]:.4f}], alert outside {alert_band}", status,
                 f"{detail}; {note}".lstrip("; "), correctable, excess)


def limit_check(name, value, key, own, detail="", higher_is_worse=True, correctable=True):
    strict = STRICT_LIMITS[key]
    met = not np.isnan(value) and (value <= strict if higher_is_worse else value >= strict)
    sign = ">" if higher_is_worse else "<"
    warn = warn_level(own, LIMITS[key])
    return graded(f"consensus: {name}", value, warn, LIMITS[key], higher_is_worse=higher_is_worse,
                  threshold=f"warn {sign} {warn:.4f}, alert {sign} {LIMITS[key]}",
                  detail=f"{detail}; {strict_note(met, f'strict {strict}')}".lstrip("; "), correctable=correctable)


def intersection(decisions, history, batch, remote):
    edges = history["revenu_familial_estime"].quantile([1 / 3, 2 / 3]).to_numpy()
    cells = pd.DataFrame({
        "decision": decisions,
        "remote": remote,
        "first_gen": batch["premiere_generation_universitaire"].to_numpy(),
        "income": np.digitize(batch["revenu_familial_estime"].to_numpy(), edges),
    }).groupby(["remote", "first_gen", "income"])["decision"].agg(["mean", "size"])
    cells = cells[cells["size"] >= INTERSECTION_MIN_CELL]
    gaps = (cells["mean"] - decisions.mean()).abs()
    if gaps.empty:
        return np.nan, "no cell reaches the minimum size"
    remote_flag, first_gen, income = (int(part) for part in gaps.idxmax())
    return float(gaps.max()), f"{len(gaps)} cells, worst remote={remote_flag} first_gen={first_gen} income_tercile={income + 1}"


def profile(decisions, references, history, batch, remote):
    centre, far = group_rate(decisions, remote == 0), group_rate(decisions, remote == 1)
    agreement = {name: float((decisions == references[name]).mean()) for name in REVIEWER_NAMES}
    gap, cell = intersection(decisions, history, batch, remote)
    return {"rates": {"overall": decisions.mean(), "centre": centre, "remote": far}, "dp": abs(centre - far),
            "ratio": far / centre, "merit_gap": signed_eo_gap(decisions, references[PURE_R], remote),
            "consensus_gap": abs(signed_eo_gap(decisions, references["consensus"], remote)),
            "intersection": gap, "cell": cell, "agreement": agreement}


def rate_checks(measured, own):
    return [band_check(f"grant rate {group}", value, group, own["rates"][group], correctable=group != "overall")
            for group, value in measured["rates"].items()]


def parity_checks(measured, own):
    rates = measured["rates"]
    detail = f"centre {rates['centre']:.1%}, remote {rates['remote']:.1%}"
    return [limit_check("demographic parity gap", measured["dp"], "max_dp_gap", own["dp"], detail),
            limit_check("impact ratio remote/centre", measured["ratio"], "min_impact_ratio", own["ratio"], detail,
                        higher_is_worse=False)]


def merit_checks(measured, own):
    gap = measured["merit_gap"]
    detail = f"TPR centre minus remote {gap:+.4f} among pure-R top-k"
    return [limit_check("opportunity gap vs merit (remote disadvantage)", gap, "max_eo_gap_vs_merit",
                        own["merit_gap"], detail),
            limit_check("opportunity overshoot vs merit (remote advantage)", -gap, "max_eo_overshoot_vs_merit",
                        -own["merit_gap"], detail)]


def consensus_gap_check(measured, own):
    return limit_check("opportunity gap vs consensus rule", measured["consensus_gap"], "max_eo_gap_vs_consensus",
                       own["consensus_gap"])


def intersection_check(measured, own):
    return limit_check("intersectional gap", measured["intersection"], "max_intersectional_gap", own["intersection"],
                       measured["cell"])


def agreement_checks(measured, own):
    agreement = measured["agreement"]
    worst = min(agreement, key=agreement.get)
    detail = " ".join(f"{name} {value:.3f}" for name, value in agreement.items())
    return [limit_check("mean agreement with reviewer rules", float(np.mean(list(agreement.values()))),
                        "min_mean_agreement_with_references", float(np.mean(list(own["agreement"].values()))), detail,
                        higher_is_worse=False, correctable=False),
            graded("consensus: min agreement with reviewer rules", agreement[worst], MIN_AGREEMENT_WARN,
                   MIN_AGREEMENT_ALERT, higher_is_worse=False,
                   threshold=f"warn < {MIN_AGREEMENT_WARN}, alert < {MIN_AGREEMENT_ALERT}", detail=f"lowest {worst}")]


def window_steps(decisions, batch):
    hours = batch["heures_travail_semaine"].to_numpy()
    strata = pd.qcut(batch["cote_r_equivalent"], R_STRATA, labels=False, duplicates="drop").to_numpy()
    steps = []
    for stratum in np.unique(strata):
        in_stratum = strata == stratum
        for start in range(int(hours.min()), int(hours.max()) + 1):
            before = in_stratum & (hours >= start) & (hours < start + HOURS_WINDOW)
            after = in_stratum & (hours >= start + HOURS_WINDOW) & (hours < start + 2 * HOURS_WINDOW)
            if before.sum() >= HOURS_MIN_CELL and after.sum() >= HOURS_MIN_CELL:
                steps.append(decisions[after].mean() - decisions[before].mean())
    return np.array(steps)


def largest(values):
    return float(values.max()) if len(values) else np.nan


def hours_shape_checks(decisions, batch, references):
    reference_steps = {name: window_steps(references[name], batch) for name in REFERENCE_INCOME_WEIGHTS}
    consensus_jump = largest(np.abs(reference_steps["consensus"]))
    reviewer_drop = max(largest(-reference_steps[name]) for name in REVIEWER_NAMES)
    steps = window_steps(decisions, batch)
    jump, drop = largest(np.abs(steps)), largest(-steps)
    drop_warn = reviewer_drop + DROP_SLACK
    return [graded("consensus: hours shape, largest jump between adjacent 2-hour windows", jump,
                   JUMP_WARN_RATIO * consensus_jump, JUMP_ALERT_RATIO * consensus_jump,
                   threshold=f"warn > {JUMP_WARN_RATIO} x consensus rule jump {consensus_jump:.3f}, "
                             f"alert > {JUMP_ALERT_RATIO} x",
                   detail=f"{len(steps)} window pairs within R deciles"),
            graded("consensus: hours shape, largest drop with more hours", drop, drop_warn,
                   DROP_ALERT_RATIO * reviewer_drop,
                   threshold=f"warn > {drop_warn:.3f} (reviewer rules' largest drop + {DROP_SLACK}), "
                             f"alert > {DROP_ALERT_RATIO} x {reviewer_drop:.3f}",
                   detail="grant rate must not fall as hours rise")]


def consensus_guard(committee, history, batch, share, decisions) -> Verdict:
    decisions = np.asarray(decisions)
    remote = is_remote(batch)
    references = reference_decisions(committee, batch, share)
    measured = profile(decisions, references, history, batch, remote)
    own = profile(references["consensus"], references, history, batch, remote)
    checks = [*rate_checks(measured, own), *merit_checks(measured, own), *parity_checks(measured, own),
              intersection_check(measured, own), *agreement_checks(measured, own),
              consensus_gap_check(measured, own), *hours_shape_checks(decisions, batch, references)]
    return verdict(checks)

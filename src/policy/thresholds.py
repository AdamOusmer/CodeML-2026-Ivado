# Declared policy thresholds from the five reviewer audits: docs/reviews/CONSENSUS.md
GRANT_RATE_BANDS = {"overall": (0.36, 0.44), "centre": (0.38, 0.43), "remote": (0.35, 0.42)}
GRANT_RATE_STRICT_BANDS = {"overall": (0.38, 0.42), "centre": (0.42, 0.43), "remote": (0.36, 0.38)}
LIMITS = {
    "max_eo_gap_vs_merit": 0.05,
    "max_eo_overshoot_vs_merit": 0.09,
    "max_eo_gap_vs_consensus": 0.06,
    "max_dp_gap": 0.08,
    "min_impact_ratio": 0.85,
    "max_intersectional_gap": 0.10,
    "min_mean_agreement_with_references": 0.93,
}
STRICT_LIMITS = {
    "max_eo_gap_vs_merit": 0.04,
    "max_eo_overshoot_vs_merit": 0.07,
    "max_eo_gap_vs_consensus": 0.03,
    "max_dp_gap": 0.06,
    "min_impact_ratio": 0.90,
    "max_intersectional_gap": 0.08,
    "min_mean_agreement_with_references": 0.96,
}
WARN_FRACTION = 0.9
MIN_AGREEMENT_WARN, MIN_AGREEMENT_ALERT = 0.92, 0.90
JUMP_WARN_RATIO, JUMP_ALERT_RATIO = 1.4, 1.7
DROP_SLACK, DROP_ALERT_RATIO = 0.02, 3.0


def warn_level(rule_value, limit):
    return rule_value + WARN_FRACTION * (limit - rule_value)

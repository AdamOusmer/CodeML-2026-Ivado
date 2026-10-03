# Verified findings (session of 2026-10-03)

Numbers computed on `data/` during development; revisions noted where they changed. Use these in the audit
notebook and pitch; re-run before quoting if code changed.

## 1. Bias diagnosis

- Groups: remote = Bas-Saint-Laurent, Cote-Nord, Gaspesie-Iles-de-la-Madeleine (40 % of history, 40.7 % of candidates).
- Historical grant rate 39.94 %; centre 48.4 % vs remote 27.3 % (gap 0.211). Per region: Montreal 48.3, Capitale-Nationale
  48.4, Bas-Saint-Laurent 26.9, Cote-Nord 26.3, Gaspesie 28.4.
- Profiles centre vs remote: R 27.99 vs 27.33; income 76k vs 56k; hours 9.0 vs 13.0; distance 20 vs 221 km; first-gen 26 % vs 44 %.
- Same R score, different outcome (grant rate centre / remote): R 26–28: 24.1 % / 6.7 %; R 28–30: 70.9 % / 40.8 %.
- Committee model (logistic, standardized): R +4.12, log income +0.80, hours +0.76, first-gen −0.03, programmes ≈ 0,
  remote −1.90 (odds × 0.15). Per-group fits give near-identical slopes (formal interaction test not yet run).
- Within each group, distance, sub-region and postal code add nothing (log-loss unchanged): the bias is one flat remote penalty.
- Natural threshold after full penalty removal grants 47.1 % (outside budget) → budget must be enforced by allocation.

## 2. Proxies

- Postal code → region purity 1.0 (each code in exactly one region).
- Single-feature AUC for remote vs centre: distance 0.997–0.998, hours 0.805, income 0.671, first-gen 0.59, R 0.53;
  hours + income 0.832; all legitimate features 0.845–0.854.
- Mutual information with group: distance 0.63, hours 0.155, income 0.062, first-gen 0.017, R 0.008.
- Organizer numbers reproduced: RF accuracy 88.1 %; parity gap 0.188 (all columns), 0.181 (no region), 0.173 (no region, no postal).

## 3. Model capacity (5-fold CV vs historical labels)

All within noise: logistic 88.74 %, tuned LightGBM 88.66 %, torch MLP on MPS 88.26–88.46 %, stacking 88.76 %;
R score alone 85.7 %. Bigger models do not help; labels are biased, so fidelity is not the target.

## 4. Hidden-reference hypotheses (red-team simulation)

Reference = top 40 % by the rule; closure / scaled utility for our pipeline (LR on corrected labels):
committee minus penalty 0.99 / 0.99; merit-only 0.88 / 0.86; merit within programme 0.93 / 0.86.
Need-based or distance-hardship references are implausible (baseline gap 0.46–0.51 vs brief's 0.270).
Rank blend 50/50 (jury) had the best worst case. Agreement peaks at exactly 40 % grant rate.

## 5. Mitigation results (10 splits, same budget, `resultats_pareto.csv` at 77ac0de)

| Method | centre / remote | EO gap vs corrected | EO gap vs merit | agree merit | agree committee |
|---|---|---|---|---|---|
| Production RF, natural threshold (anchor) | 45.8 / 26.5 | 0.253 | 0.247 | 0.944 | 0.878 |
| Production RF at budget | 47.4 / 28.7 | 0.211 | 0.188 | 0.946 | 0.876 |
| Drop proxies only | 44.5 / 33.0 | 0.100 | 0.070 | 0.925 | 0.875 |
| Best ExpGrad (eps 0.05) | 44.8 / 32.7 | 0.124 | 0.082 | 0.915 | 0.867 |
| ThresholdOptimizer (43.6 %, not budget-fixable) | 50.5 / 33.2 | 0.108 | 0.108 | 0.911 | 0.875 |
| Committee, penalty removal 1.0 | 42.2 / 36.5 | 0.000 | 0.009 | 0.927 | 0.868 |
| **Declared: jury 50/50** | 42.6 / 35.9 | 0.030 ± 0.011 | 0.006 ± 0.006 | 0.963 | 0.868 |

Key insight (keep numbers paired): fairlearn's equal-opportunity constraint conditions on the supplied labels; equalizing
TPR against biased historical labels does not equalize opportunity among deserving applicants (single split: ExpGrad
eps 0.3 historical gap 0.016 but merit gap 0.113). Fairness must be measured after allocation.

Tuner (worst case of hackathon-weighted score over corrected and merit references): merit weight 0.25 ≈ 0.928,
0.5 ≈ 0.922 at 3 splits — within noise; extremes (0 and 1) are partly scored against themselves.

## 6. Audit notebook outline (from diagnostic scout)

1. Data and groups, rates with Wilson CIs. 2. Baseline reproduction (88.1 %). 3. Why accuracy/TPR vs committee labels
mislead. 4. Raw disparity: bootstrap CI (parity gap 0.187, 95 % CI 0.154–0.220) and permutation test. 5. Explained vs
unexplained: grant rate by R bin, committee coefficients. 6. Counterfactual flip test, conditional parity (CMH).
7. Proxies: AUC bars, MI, correlation, postal crosstab, drop-column table. 8. Model drivers (permutation importance or SHAP).
9. Metric choice: equal opportunity vs parity, impossibility result. 10. Limitations.

## 7. Governance material (from governance scout; verify legal citations)

- Quebec Charter art. 10: region is not an enumerated ground; frame as indirect discrimination on "condition sociale".
  Art. 86: equal-access programmes would be the legal route for demographic parity (policy, not a model choice).
- Law 25 (private-sector act): automated-decision transparency, principal factors (article number to verify).
- Monitoring thresholds implemented: DP gap warn 0.10; impact ratio warn 0.80; EO gap warn 0.03 / alert 0.05;
  intersectional warn 0.15; proxy AUC drift alert +0.05; PSI warn 0.10 / alert 0.25.

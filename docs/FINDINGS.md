# FINDINGS (2026-10-03, numbers measured at 4d657c1)

Tags: [R] re-run this session (cmd given). [U] unverified, source: prior session (not reproducible without new code). Scratch outputs: /private/tmp/ff/.
Revision delta: decisions vs previous revision differ on 6 of 4,000 candidates; 1,598 grants unchanged.

## 1. Bias
| Item | Value | Tag |
|---|---|---|
| Baseline EO gap (scorer denominator) | 0.270 | [R] README.md:104, LISEZMOI.md:110 (brief constant; not recomputed) |
| Historical grant rate | 39.94 % | [R] pandas on data/donnees_demandes.csv |
| Remote share (BSL, Cote-Nord, Gaspesie) | 40.0 % hist; 40.7 % eval (1628/4000) | [R] check-data |
| Centre vs remote rate | 48.4 % vs 27.3 % (gap 0.211) | [R] |
| By region | MTL 48.3, CN 48.4, BSL 26.9, Cote-Nord 26.3, Gaspesie 28.4 | [R] |
| Mean R centre vs remote | 27.99 vs 27.33 | [R] |
| First-gen centre vs remote | 26 % vs 44 % | [R] |
| Income / hours / distance means | 76k/56k; 9.0/13.0 h; 20/221 km | [U] |
| Same R, different outcome (centre/remote) | R 26-28: 24.1 / 6.7 %; R 28-30: 70.4 / 40.7 % (old 70.9 / 40.8, bin edge diff) | [R] |
| Committee logit (standardized) | R +4.12, log income +0.80, hours +0.76, first-gen -0.03, programme ~0, remote -1.90 (odds x0.15) | [U] |
| Penalty is one flat remote term (distance, sub-region, postal add nothing in-group) | yes | [U] |
| Full penalty removal natural grant rate | 47.1 % (over budget; allocate to budget) | [U] |

Budget bound 36-44 % (`BUDGET_BOUNDS` in `src/policy/core.py`).

## 2. Proxies
| Item | Value | Tag |
|---|---|---|
| Postal code to region purity | 1.0 | [R] |
| Single-feature AUC remote vs centre | distance 0.998, hours 0.805, first-gen 0.59, income 0.307 (=0.693 flipped; old 0.671), R 0.439 (=0.561 flipped; old 0.53) | [R] raw AUC, orientation-dependent |
| hours+income 0.832; all legit features 0.845-0.854 | | [U] (monitor proxy_auc history on 3 scoring features = 0.845 [R]) |
| Mutual info: distance 0.63, hours 0.155, income 0.062, first-gen 0.017, R 0.008 | | [U] |
| Proxy drift (monitor, eval batch vs history) | AUC 0.845 -> 0.853, +0.008 (alert > +0.05) | [R] monitor |
| Feature drift max PSI (3 scoring features, histogram) | 0.012 (log_revenu, centre) | [R] monitor |
| Categorical drift max PSI | 0.038 (programme_etudes, Gaspesie, Genie +7.98 pts) OK | [R] monitor |
| Organizer RF acc 88.1 %; parity gap 0.188 / 0.181 (no region) / 0.173 (no region, no postal) | | [U] |

## 3. Model capacity
Claim: all models 87.5-88.8 % 5-fold CV, noise +-0.6. No script or doc in repo holds the source; only HANDOFF.md:57 and old FINDINGS.
| Model | Old | This session |
|---|---|---|
| Logistic | 88.74 | 87.96 [R] (my feature set, StratifiedKFold seed 0) |
| RF all columns | 88.1 (organizer) | 88.42 [R] (300 trees) |
| R score alone | 85.7 | 85.71 [R] |
| LightGBM tuned 88.66; torch MLP (MPS) 88.26-88.46; stacking 88.76 | | [U] |
Conclusion holds ([R] spread LR to RF 0.5 pt, inside noise): capacity not the lever; labels biased.

## 4. Reference hypotheses (hidden reference unknown)
- Merit reference: top R at same budget. Corrected reference: committee rule with regional penalty removed. Both built in src/policy/models.py `reference_labels`; monitor uses both [R].
- Red-team simulation, pipeline = LR on corrected labels, closure / scaled utility: committee minus penalty 0.99/0.99; merit-only 0.88/0.86; merit within programme 0.93/0.86 [U].
- Need-based / distance-hardship references implausible (baseline gap 0.46-0.51 vs brief 0.270) [U].
- Rank blend 50/50 best worst case; agreement peaks at 40 % grant rate [U; consistent with section 5 tuner].

## 5. 10-split results [R] `OMP_NUM_THREADS=2 uv run python -m src.main pareto --splits 10 --workers 2 --plain --no-log-file --out-dir /private/tmp/ff` (exit 0, 31 s). Regenerated after preprocessing (main model on 3 scoring features); values from `resultats_pareto.csv`.
| Method | centre / remote | EO vs corrected (std) | EO vs merit (std) | agree merit | acc hist |
|---|---|---|---|---|---|
| Production RF natural thr (anchor, rate 0.381) | 45.8 / 26.5 | 0.253 (.016) | 0.247 (.019) | 0.944 | 0.878 |
| Production RF at budget | 47.4 / 28.7 | 0.211 (.018) | 0.188 (.012) | 0.946 | 0.876 |
| Drop proxies only | 44.3 / 33.3 | 0.089 (.012) | 0.059 (.014) | 0.927 | 0.875 |
| ExpGrad eps 0.05 (eps 0.3/0.2/0.1 same ~0.110) | 44.5 / 33.1 | 0.110 (.022) | 0.078 (.019) | 0.917 | 0.867 |
| ThresholdOptimizer (rate 0.440, budget_ok False) | 51.2 / 33.4 | 0.104 (.016) | 0.104 (.012) | 0.911 | 0.876 |
| Committee, penalty removal 0.0 | 48.8 / 26.7 | 0.269 (.008) | 0.232 (.012) | 0.906 | 0.886 |
| Committee, penalty removal 0.6 (0.2/0.4/0.8 also run) | 45.0 / 32.4 | 0.113 (.012) | 0.087 (.013) | 0.925 | 0.880 |
| Committee, penalty removal 1.0 | 42.2 / 36.5 | 0.000 (0) | 0.009 (.007) | 0.927 | 0.868 |
| Jury merit 0 | 42.3 / 36.4 | 0.006 (.006) | 0.010 | 0.928 | 0.868 |
| Jury merit 0.25 | 42.5 / 36.0 | 0.021 (.008) | 0.007 | 0.945 | 0.870 |
| **Declared jury 50/50 (merit 0.5)** | 42.6 / 35.9 | **0.029 (.013)** | 0.007 (.005) | 0.963 | 0.867 |
| Jury merit 0.75 | 42.8 / 35.6 | 0.038 (.014) | 0.006 | 0.983 | 0.862 |
| Jury merit 1.0 | 42.9 / 35.4 | 0.042 (.019) | 0.000 | 1.000 | 0.856 |
All grant rate 0.399 (budget ok) except anchor 0.381, ThresholdOptimizer 0.440.

Tuner [R] `... src.main tune --splits 5 --workers 2 --plain --no-log-file --out-dir /private/tmp/ff` (regenerated, 5 splits, `resultats_tuner.csv`):
| merit w | eo corr | score corr | eo merit | score merit | worst case |
|---|---|---|---|---|---|
| 0 | 0.009 | 0.975 | 0.010 | 0.914 | 0.914 |
| 0.25 | 0.022 | 0.935 | 0.007 | 0.934 | **0.934** |
| 0.5 | 0.025 | 0.913 | 0.005 | 0.955 | 0.913 |
| 0.75 | 0.036 | 0.873 | 0.007 | 0.968 | 0.873 |
| 1.0 | 0.040 | 0.850 | 0.000 | 1.000 | 0.850 |
Old note: 0.25 ~0.928 vs 0.5 ~0.922 at 3 splits. At 5 splits 0.25 still best worst case (0.934); declared 0.5 (0.913) is 0.021 behind. Decision for orchestrator.
Insight: fairlearn EO conditions on supplied (biased) labels; equalizing TPR vs historical labels leaves merit gap (single split ExpGrad eps 0.3: hist gap 0.016, merit gap 0.113 [U]). Measure fairness after allocation.
Monitor on shipped predictions.csv [R]: grant 0.3995 (1,598 of 4,000); DP gap 0.059 (centre 42.4 % vs remote 36.4 %); EO vs merit 0.009; EO vs corrected 0.021; intersectional 0.065 (first-gen); impact ratio 0.791 (Cote-Nord 33.8 % vs Capitale-Nationale 42.8 %) = WARN (< 0.80); proxy drift AUC history 0.845 -> batch 0.853 (+0.008); feature drift max PSI 0.012 (log_revenu, centre); categorical drift max PSI 0.038 (programme_etudes, Gaspesie, Genie +7.98 pts) OK; overall WARN.

## 6. Audit outline (audit_rapport.ipynb) [U, plan]
1 Data, groups, Wilson CIs. 2 Baseline reproduction (88.1 %). 3 Why accuracy/TPR vs committee labels mislead. 4 Raw disparity: bootstrap CI (parity gap 0.187, CI 0.154-0.220) + permutation test. 5 Explained vs unexplained: rate by R bin, committee coefficients. 6 Counterfactual flip test, conditional parity (CMH). 7 Proxies: AUC bars, MI, correlation, postal crosstab, drop-column. 8 Model drivers (permutation importance or SHAP). 9 Metric choice: EO vs parity, impossibility. 10 Limitations (synthetic data, unknown reference, label bias, proxy residue).

## 7. Governance
Monitoring thresholds (threshold constants in `src/monitoring/checks.py`) [R]:
| Check | Warn | Alert |
|---|---|---|
| Grant rate (budget) | outside 36-44 % | same (ALERT) |
| DP gap centre vs remote | > 0.10 | - |
| Impact ratio low/high region | < 0.80 | - |
| EO gap vs merit / corrected / reviewed sample | > 0.03 | > 0.05 (correctable) |
| Largest intersectional gap | > 0.15 | - |
| Proxy AUC drift | > +0.05 | > +0.05 |
| Feature PSI (R, log income, hours per group (histogram)) | > 0.10 | > 0.25 |
| Categorical PSI (programme, first-gen per region) | > 0.10 | > 0.25 |
Note: declared config EO vs corrected 0.029 sits just under WARN 0.03 (strict `>`).
Legal [U, verify before citing]:
- Quebec Charter art. 10: region not enumerated; argue indirect discrimination on "condition sociale". Art. 86: equal-access programmes = legal route for demographic parity.
- Law 25: automated-decision transparency (inform person, principal factors, right to review). Article 12.1 UNVERIFIED: not checked against statute text; confirm before pitch.

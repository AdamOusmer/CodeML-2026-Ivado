# FINDINGS (2026-10-03, integration of PR #1, PR #2 and the validator jury)

Tags: [R] re-run this session (cmd given). [U] unverified, source: prior session (not reproducible without new code). Scratch outputs: /private/tmp/ff/.
Revision: validator jury replaces the rank-vote jury. Sections 1-4 unchanged (not re-run). Sections 5 and 7 re-run this session.
Acceptance: `OMP_NUM_THREADS=2 uv run python scripts/acceptance.py` -> 30/30 passed in 27 s (new check `jury_offset_monotone`).

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
- Rank blend 50/50 best worst case; agreement peaks at 40 % grant rate [U; prior rank-vote jury study, superseded by the validator jury, section 5].

## 5. 10-split results [R] `OMP_NUM_THREADS=2 uv run python model_corrige.py` (10 splits, exit 0, 37 s); values from `resultats_pareto.csv`.
| Method | centre / remote | EO vs corrected (std) | EO vs merit (std) | agree merit | acc hist |
|---|---|---|---|---|---|
| Production RF natural thr (anchor, rate 0.381) | 45.8 / 26.5 | 0.253 (0.016) | 0.247 (0.019) | 0.944 | 0.878 |
| Production RF at budget | 47.4 / 28.7 | 0.211 (0.018) | 0.188 (0.012) | 0.946 | 0.876 |
| Drop proxies only | 44.3 / 33.3 | 0.089 (0.012) | 0.059 (0.014) | 0.927 | 0.875 |
| ExpGrad eps 0.05 (eps 0.3/0.2/0.1 same ~0.110) | 44.5 / 33.1 | 0.110 (0.022) | 0.078 (0.019) | 0.917 | 0.867 |
| ThresholdOptimizer (rate 0.440, budget_ok False) | 51.2 / 33.4 | 0.104 (0.016) | 0.104 (0.012) | 0.911 | 0.876 |
| Committee, penalty removal 0.0 | 48.8 / 26.7 | 0.269 (0.008) | 0.232 (0.012) | 0.906 | 0.886 |
| Committee, penalty removal 0.6 (0.2/0.4/0.8 also run) | 45.0 / 32.4 | 0.113 (0.012) | 0.087 (0.013) | 0.925 | 0.880 |
| Committee, penalty removal 1.0 | 42.2 / 36.5 | 0.000 (0.000) | 0.009 (0.007) | 0.927 | 0.868 |
| Jury, band 0.0 | 42.4 / 36.2 | 0.013 (0.007) | 0.011 (0.007) | 0.937 | 0.870 |
| Jury, band 0.0125 | 42.4 / 36.2 | 0.014 (0.007) | 0.010 (0.008) | 0.939 | 0.870 |
| Jury, band 0.05 | 42.6 / 35.9 | 0.025 (0.014) | 0.009 (0.007) | 0.963 | 0.867 |
| Jury, band 0.1 | 42.8 / 35.6 | 0.034 (0.013) | 0.006 (0.005) | 0.986 | 0.861 |
| **Declared validator jury (band 0.025)** | 42.5 / 36.1 | 0.014 (0.007) | 0.010 (0.011) | 0.947 | 0.869 |
All grant rate 0.399 (budget ok) except anchor 0.381, ThresholdOptimizer 0.440. Jury bands 0.0, 0.025 and 0.05 are flagged `pareto False`; bands 0.0125 and 0.1 are `pareto True` (eo_gap vs acc_historical axes). Declared row = band 0.025.

Tuner [R] `OMP_NUM_THREADS=2 uv run python -m src.main tune --splits 10 --workers 2 --plain --no-log-file --out-dir .` (10 splits, 9 configs, `resultats_tuner.csv`, file order = worst_case descending):
| config | eo corr | score corr | eo merit | score merit | worst case |
|---|---|---|---|---|---|
| band 0.025, quorum 1.0, merit | 0.015 | 0.948 | 0.009 | 0.933 | 0.933 |
| band 0.025, quorum 0.5, merit+programme_merit | 0.014 | 0.950 | 0.009 | 0.932 | 0.932 |
| band 0.025, quorum 1.0, merit+programme_merit | 0.014 | 0.952 | 0.010 | 0.928 | 0.928 |
| band 0.0125, quorum 1.0, merit | 0.014 | 0.959 | 0.009 | 0.924 | 0.924 |
| band 0.0125, quorum 0.5, merit+programme_merit | 0.012 | 0.961 | 0.009 | 0.923 | 0.923 |
| band 0.0125, quorum 1.0, merit+programme_merit | 0.014 | 0.957 | 0.010 | 0.921 | 0.921 |
| band 0.05, quorum 1.0, merit | 0.023 | 0.916 | 0.008 | 0.949 | 0.916 |
| band 0.05, quorum 0.5, merit+programme_merit | 0.023 | 0.916 | 0.008 | 0.949 | 0.916 |
| band 0.05, quorum 1.0, merit+programme_merit | 0.025 | 0.914 | 0.009 | 0.945 | 0.914 |
Declared config (band 0.025, quorum 1.0, merit+programme_merit) is third by worst case (0.928) vs best `band 0.025, quorum 1.0, merit` (0.933); gap 0.005, inside split noise [no SE computed]. JURY_SPEC section 8 evidence (rank-vote vs validator, 10 splits) is from the design study, not re-run [U].
Insight: fairlearn EO conditions on supplied (biased) labels; equalizing TPR vs historical labels leaves merit gap (single split ExpGrad eps 0.3: hist gap 0.016, merit gap 0.113 [U]). Measure fairness after allocation.
Real-batch decide [R] `decision_record.json` from `model_corrige.py`: status published, actions [SELECT_CONFIG], offset 0.0, 1,598 grants; jury triggered 207 (near_cutoff 200, low_confidence 102, disagreement 2), overturned_out 36 = overturned_in 36 (36 swaps), `moved_ids` empty.
Monitor on shipped predictions.csv [R] `OMP_NUM_THREADS=2 uv run python -m src.main monitor --plain --no-log-file --history data/donnees_demandes.csv --batch data/candidats_evaluation.csv --decisions predictions.csv` (exit 0): grant 0.3995 (1,598 of 4,000); DP gap 0.057 (centre 42.3 % vs remote 36.5 %); EO vs merit 0.011; EO vs corrected 0.018; intersectional 0.064 (first-gen); impact ratio 0.819 (Cote-Nord 35.0 % vs Capitale-Nationale 42.8 %); proxy drift AUC 0.845 -> 0.853 (+0.008); feature drift max PSI 0.012 (log_revenu, centre); categorical drift max PSI 0.038 (programme_etudes, Gaspesie, Genie +7.98 pts); overall OK.
Offset reach [R] acceptance `jury_offset_monotone`: remote grants 587 / 595 / 606 at offset -0.10 / 0 / +0.10, grants = 1,598 at all 41 offsets.

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
Note: declared config EO vs corrected 0.014 (10-split Pareto) and 0.018 (real batch) sit below WARN 0.03.
Legal [U, verify before citing]:
- Quebec Charter art. 10: region not enumerated; argue indirect discrimination on "condition sociale". Art. 86: equal-access programmes = legal route for demographic parity.
- Law 25: automated-decision transparency (inform person, principal factors, right to review). Article 12.1 UNVERIFIED: not checked against statute text; confirm before pitch.

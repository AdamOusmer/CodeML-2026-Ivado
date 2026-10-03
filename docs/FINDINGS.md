# FINDINGS (2026-10-03, integration of PR #1, PR #2 and the validator jury, merit-only declared)

Tags: [R] re-run this session (cmd given). [U] unverified, source: prior session (not reproducible without new code). Scratch outputs: /private/tmp/ff/.
Revision: validator jury replaces the rank-vote jury; declared config now merit-only, low-confidence trigger removed, swap strength over dissenters only. Sections 1-4 re-verified at ac936d0 (not touched here). Sections 5 and 7 re-run this session.
Acceptance: `OMP_NUM_THREADS=2 uv run python scripts/acceptance.py` -> 30/30 passed in 26 s.

## 1. Bias
| Item | Value | Tag |
|---|---|---|
| Baseline EO gap (scorer denominator) | 0.270 | [R] README.md:104, LISEZMOI.md:110 (brief constant; not recomputed; historical labels vs corrected reference give 0.212, committee penalty-off 0.269 in section 5) |
| Historical grant rate | 39.94 % | [R] pandas on data/donnees_demandes.csv |
| Remote share (BSL, Cote-Nord, Gaspesie) | 40.0 % hist; 40.7 % eval (1628/4000) | [R] check-data |
| Centre vs remote rate | 48.4 % vs 27.3 % (gap 0.211) | [R] |
| By region | MTL 48.3, CN 48.4, BSL 26.9, Cote-Nord 26.3, Gaspesie 28.4 | [R] |
| Mean R centre vs remote | 27.99 vs 27.33 | [R] |
| First-gen centre vs remote | 26 % vs 44 % | [R] |
| Income / hours / distance means (centre/remote) | 76k/56k; 9.0/13.0 h; 20/221 km | [R] pandas groupby on is_remote |
| Same R, different outcome (centre/remote) | R 26-28: 24.1 / 6.7 %; R 28-30: 70.4 / 40.7 % (old 70.9 / 40.8, bin edge diff) | [R] |
| Committee logit (standardized) | R +4.12, log income +0.80, hours +0.76, first-gen -0.03, programme ~0 (+-0.02), remote -1.90 (odds x0.15) | [R] `CommitteeModel().fit(history).lr_.coef_` |
| Penalty is one flat remote term (distance, sub-region, postal add nothing in-group) | yes: 5-fold CV log-loss gain <= 0.0003 when adding distance, region or postal dummies (in-group and pooled) | [R] |
| Full penalty removal natural grant rate (corrected logit > 0) | 47.1 % on eval batch (47.075 %; 45.7 % on history in-sample); over budget, allocate to budget | [R] `corrected_logit(batch, 1.0) > 0` |

Budget bound 36-44 % (`BUDGET_BOUNDS` in `src/policy/core.py`).

## 2. Proxies
| Item | Value | Tag |
|---|---|---|
| Postal code to region purity | 1.0 | [R] |
| Single-feature AUC remote vs centre | distance 0.998, hours 0.805, first-gen 0.59, income 0.307 (=0.693 flipped; old 0.671), R 0.439 (=0.561 flipped; old 0.53) | [R] raw AUC, orientation-dependent |
| hours+income 0.842 (0.839 raw income; old 0.832); 3 scoring features 0.845; all committee features 0.854 | | [R] 3-fold CV logistic AUC on is_remote (monitor proxy_auc history = 0.845) |
| Mutual info (nats): distance 0.63, hours 0.156, income 0.064, first-gen 0.017, R 0.015 (old 0.008; kNN estimator noise 0.013-0.015, R is ~0 either way) | | [R] sklearn `mutual_info_classif` vs is_remote, random_state 0/1 |
| Proxy drift (monitor, eval batch vs history) | AUC 0.845 -> 0.853, +0.008 (alert > +0.05) | [R] monitor |
| Feature drift max PSI (3 scoring features, histogram) | 0.012 (log_revenu, centre) | [R] monitor |
| Categorical drift max PSI | 0.038 (programme_etudes, Gaspesie, Genie +7.98 pts) OK | [R] monitor |
| Organizer RF acc 88.1 %; parity gap 0.188 / 0.180 (no region) / 0.174 (no region, no postal) (old 0.181 / 0.173) | | [R] baseline_model.ipynb recipe: RF 300 trees leaf 20, split 0.3 seed 42, |centre - remote| selection rate |

## 3. Model capacity
Claim: all models 87.9-88.8 % 5-fold CV, noise +-0.6 (R alone 85.7 is the floor). No script or doc in repo holds the source; only HANDOFF.md:57 and old FINDINGS. Re-run: StratifiedKFold(5, shuffle, seed 0), accuracy.
| Model | Old | This session |
|---|---|---|
| Logistic | 88.74 | 88.76 [R] (committee features + remote, standardized; seeds 0-2: 88.67-88.76) |
| RF all columns | 88.1 (organizer holdout, [R] 88.13) | 88.42 [R] (300 trees, leaf 1); 87.88 with min_samples_leaf 20 |
| R score alone | 85.7 | 85.71 [R] |
| LightGBM tuned | 88.66 | 88.73 [R] (fold sd 0.49; nested: 12-combo grid num_leaves 4/8/16 x lr 0.03/0.1 x n_estimators 100/300, inner 3-fold accuracy per outer fold; picks leaves 4-8, lr 0.03 mostly, 300 trees; features = all columns one-hot as RF; subsample/colsample 0.8) |
| torch MLP (CPU) | 88.26-88.46 (MPS) | 88.51 [R] (fold sd 0.56; 2 hidden 64/32 ReLU, dropout 0.2, standardized all-column inputs, Adam, 15 % validation split, patience 15; seeds 0-2: 88.43 / 88.58 / 88.51) |
| Stacking LR+RF+LightGBM, LR meta (cv 5) | 88.76 | 88.81 [R] (fold sd 0.65; LR on all-column standardized, RF 300 trees, LightGBM at the fold's tuned params) |
| Logistic, all-column one-hot (same fold set) | | 88.69 [R] (fold sd 0.57) |
Scripts: session scratchpad (not in repo); sklearn stand-ins: HistGradientBoosting 88.16 [R]; LR+RF+HGB stacking 88.82 [R]; sklearn MLP (64,32) untuned 86.21 [R].
Conclusion holds ([R] LR, RF, LightGBM, MLP, stacking all 88.4-88.8, spread 0.4 pt vs fold sd 0.5-0.65; RF leaf 20 87.88): capacity not the lever; labels biased.

## 4. Reference hypotheses (hidden reference unknown)
- Merit reference: top R at same budget. Corrected reference: committee rule with regional penalty removed. Both built in src/policy/models.py `reference_labels`; monitor uses both [R].
- Red-team simulation, pipeline = LR on corrected labels (FairPipeline, no jury, allocate at budget), closure (1 - EO gap/0.270) / scaled utility, 10 splits: committee minus penalty 0.98/0.99 (old 0.99/0.99); merit-only 0.96/0.85 (old 0.88/0.86); merit within programme 0.97/0.85 (old 0.93/0.86) [R] `scaled_utility`, `eo_gap` per reference; closure definition assumed, old closure not reproduced for merit refs.
- Need-based / distance-hardship references implausible (baseline gap 0.46-0.51 vs brief 0.270) [U] not reproduced: definition unrecoverable (only docs hold it, first at 4d657c1; no code in any of 32 commits across all refs; searched git log -S hardship / need-based / jury_weights, git grep over every revision). Attempt, 5 splits, reference = top 40 % by lowest income / distance / income+distance / +hours / income+first-gen, baseline = historical labels or production RF at budget, EO gap by remote: 0.17-0.23, never 0.46-0.51; claim unsupported.
- Rank blend of main model and merit percentiles (old `Config.jury_weights`, code at 99d580e^): 50/50 NOT best worst case; merit weight 0.2 best (0.930), 0.25 0.927, 0.5 0.906, 0.0 0.915, 1.0 0.845 (worst of corrected / merit references, score = (20 closure + 15 scaled utility) / 35, 10 splits, seeds 0-9) [R] old code re-run, superseded by the validator jury section 5. Agreement peaks at 40 % grant rate [R] 50/50 jury at decision share 0.30-0.50 vs references fixed at 39.94 %: 0.963 corrected / 0.964 merit at 0.40 (0.960 at 0.42, 0.952 at 0.36); peak at reference share by construction.

## 5. 10-split results [R] `OMP_NUM_THREADS=2 uv run python model_corrige.py` (10 splits, exit 0); values from `resultats_pareto.csv`.
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
| Jury, band 0.0 | 42.3 / 36.4 | 0.006 (0.006) | 0.010 (0.008) | 0.928 | 0.868 |
| Jury, band 0.0125 | 42.4 / 36.2 | 0.010 (0.003) | 0.008 (0.008) | 0.939 | 0.870 |
| Jury, band 0.05 | 42.6 / 35.9 | 0.023 (0.013) | 0.008 (0.006) | 0.965 | 0.867 |
| Jury, band 0.1 | 42.9 / 35.5 | 0.034 (0.015) | 0.006 (0.003) | 0.988 | 0.861 |
| **Declared validator jury, merit (band 0.025)** | 42.5 / 36.1 | 0.015 (0.007) | 0.009 (0.010) | 0.948 | 0.869 |
All grant rate 0.399 (budget ok) except anchor 0.381, ThresholdOptimizer 0.440. Jury bands 0.0125 and 0.1 are `pareto True`; bands 0.0, 0.025 and 0.05 are `pareto False` (eo_gap vs acc_historical axes). Band sweep uses the declared juror set (merit), so its band 0.025 row equals the declared row.

Tuner [R] `OMP_NUM_THREADS=2 uv run python -m src.main tune --splits 10 --workers 2 --plain --no-log-file --out-dir .` (10 splits, 9 configs, `resultats_tuner.csv`, file order = worst_case descending):
| config | eo corr | score corr | eo merit | score merit | worst case |
|---|---|---|---|---|---|
| band 0.025, quorum 1.0, merit | 0.015 | 0.950 | 0.009 | 0.932 | 0.932 |
| band 0.025, quorum 0.5, merit+programme_merit | 0.014 | 0.951 | 0.009 | 0.931 | 0.931 |
| band 0.025, quorum 1.0, merit+programme_merit | 0.013 | 0.953 | 0.010 | 0.928 | 0.928 |
| band 0.0125, quorum 1.0, merit | 0.010 | 0.967 | 0.008 | 0.926 | 0.926 |
| band 0.0125, quorum 0.5, merit+programme_merit | 0.009 | 0.970 | 0.009 | 0.924 | 0.924 |
| band 0.0125, quorum 1.0, merit+programme_merit | 0.010 | 0.967 | 0.009 | 0.922 | 0.922 |
| band 0.05, quorum 1.0, merit | 0.023 | 0.916 | 0.008 | 0.949 | 0.916 |
| band 0.05, quorum 1.0, merit+programme_merit | 0.025 | 0.914 | 0.009 | 0.945 | 0.914 |
| band 0.05, quorum 0.5, merit+programme_merit | 0.024 | 0.913 | 0.008 | 0.948 | 0.913 |
Declared config (band 0.025, quorum 1.0, merit) is first by worst case (0.932). Merit-only vs the previous two-juror unanimous config: +0.004 on the table (0.932 vs 0.928); the paired evidence is in JURY_SPEC section 9 (stress reference +0.0057 ± 0.0037, 2-ref worst case +0.0034 ± 0.0039). Caveat: merit juror is also the merit reference. JURY_SPEC section 8 evidence (rank-vote vs validator, 10 splits) is from the design study, not re-run [U].
Insight: fairlearn EO conditions on supplied (biased) labels; equalizing TPR vs historical labels leaves merit gap (single split ExpGrad eps 0.3: hist gap 0.016, merit gap 0.113 [U]). Measure fairness after allocation.
Real-batch decide [R] `decision_record.json` from `model_corrige.py`: status published, actions [SELECT_CONFIG], offset 0.0, 1,598 grants (39.94 %); jury triggered 201 (near_cutoff 200, disagreement 2), overturned_out 37 = overturned_in 37 (37 swaps), `moved_ids` empty. `predictions.csv`: 4,000 rows, id order = candidate order, rate 0.3995.
Monitor on shipped predictions.csv [R] `OMP_NUM_THREADS=2 uv run python -m src.main monitor --plain --no-log-file` (defaults: supplied history, evaluation batch, `predictions.csv`; exit 0): grant 0.3995 (1,598 of 4,000); DP gap 0.058 (centre 42.3 % vs remote 36.5 %); EO vs merit 0.008; EO vs corrected 0.018; intersectional 0.064 (first-gen); impact ratio 0.819 (Cote-Nord 35.0 % vs Capitale-Nationale 42.8 %); proxy drift AUC 0.845 -> 0.853 (+0.008); feature drift max PSI 0.012 (log_revenu, centre); categorical drift max PSI 0.038 (programme_etudes, Gaspesie, Genie +7.98 pts); overall OK.
Offset reach [R] acceptance `jury_offset_monotone`: remote grants 586 / 594 / 606 at offset -0.10 / 0 / +0.10, grants = 1,598 at all 41 offsets. Forced drift study (remote R -1.5, session scratch) did not reach ALERT: corrector reach at 0.05 untested on a real drifted batch [U].

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
Note: declared config EO vs corrected 0.015 (10-split Pareto) and 0.018 (real batch) sit below WARN 0.03.
Legal [U, verify before citing]:
- Quebec Charter art. 10: region not enumerated; argue indirect discrimination on "condition sociale". Art. 86: equal-access programmes = legal route for demographic parity.
- Law 25: automated-decision transparency (inform person, principal factors, right to review). Article 12.1 UNVERIFIED: not checked against statute text; confirm before pitch.

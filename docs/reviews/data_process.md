# Reviewer: data_process (statistician / data-generating process lens)

## Committee fit (10,000 rows, 5-fold CV accuracy 0.887, matches the brief's "88%")
Logistic on raw units: R +1.38/pt, log income +1.78, hours +0.20/h, remote -1.94 (intercept -60.6). Per SD (R 2.99, logY 0.44, hours 3.9 h): R 4.2, logY 0.8, hours 0.8, remote -1.05 (binary SD). First-gen ~0 (-0.02 to -0.07), programme ~0 (all coefs <0.06; rates 0.37-0.43 are explained by R). Distance adds nothing once remote is in (it is a pure proxy, corr 0.82 with remote).

Shape tests: adding R^2, hours^2, log hours, hours>10, hours>15, income^2, remote x R, remote x hours, remote x income each moves CV accuracy by under 0.0005; GBM (0.886) does not beat the linear logit (0.887). Within-group fits give near-identical slopes (R 4.12 vs 4.18, hours 0.64 vs 0.69, income 0.73 vs 0.80). So committee = linear logit latent with a flat remote shift (about 1.4 R-points) plus Bernoulli noise (the 88.7% ceiling). No thresholds.

Candidates vs history: same distribution (R 27.80 vs 27.72, hours 10.61 vs 10.59, remote 41% vs 40%). Small covariate shift only; history-fitted models transfer.

## Likely organizer reference
Bias story: remote regions get lower R (-0.66), lower income (logY -0.31), more hours (+4.0 h), more first-gen, far more distance. Hours is positive in the committee in both groups, so it is a legitimate merit/need term (or at least not a penalty), and it favours remote applicants. The most probable unbiased reference is the committee latent without the remote shift: `1.38R + 1.78 logY + 0.20 h` (call A), thresholded near 40%. Alternatives: merit-only R (B), R plus income (C). Evidence for A: on candidates at 40%, "committee vs A" EO gap is 0.250 (A as truth), vs B 0.208, vs C 0.139. The brief's baseline gap is 0.270, closest to A. Agreement between rules at 40%: A-B 92.8%, A-C 94.6%, so utility risk from picking the wrong member is about 5-7 points of accuracy. Ranking by A also closes the gap against B (EO 0.035 vs B, -0.005 the other way).

Selection rates at 40% budget on candidates: A: remote 0.368 / centre 0.422; B: 0.359 / 0.430; C: 0.322 / 0.453. Parity cannot be reached without over-correcting: a remote group at 0.40 would deny centre candidates with higher merit. Counterfactual (committee without penalty) expected rate: remote 0.444 / centre 0.491 on all candidates, so reference rate for remote is below centre.

Uncertainty: shape and signs of the committee are solid; the reference is inferred. Hours sign in the reference (A vs C) is the main open question; first-gen or programme being a hidden positive/negative term cannot be excluded but has no support in the labels.

```json
{"reviewer":"data_process","criteria":{"cote_r":{"sign":"+","weight_vs_R":1.0},"log_revenu":{"sign":"+","weight_vs_R":0.19},"heures_travail":{"sign":"+","weight_vs_R":0.19,"shape":"linear"},"premiere_generation":{"sign":"0","weight_vs_R":0.0},"programme":{"adjust":true}},
"fair_reference_rule":"z(R) + 0.19*z(log income) + 0.19*z(hours)  (committee latent minus remote shift; fallback z(R) + 0.2*z(log income))",
"expected_grant_rate":{"overall":[0.36,0.44],"centre":[0.42,0.46],"remote":[0.32,0.38]},
"max_eo_gap_vs_merit":0.06,"max_dp_gap":0.14,"min_impact_ratio":0.70,"max_intersectional_gap":0.15,
"guard_rules":["overall grant rate in [0.38,0.42] and never outside [0.36,0.44]","grant rate remote/centre ratio >= 0.70 and DP gap <= 0.14 (reject DP gap < 0.02: over-correction)","EO gap vs z(R)+0.19*z(logY)+0.19*z(h) top-40% label <= 0.06, and vs R-only top-40% <= 0.08","decision agreement with merit-rule A >= 92%","no decision flips against the A ranking inside a score band unless same rule applied to both regions; remote flag must not be a feature","logistic residual check: coefficient of remote in decision ~ merit terms within [-0.3,+0.3] logit (committee was -1.94)","programme-level and first-gen x remote cell rates within 0.15 of overall"],
"red_flags":["any remote term or region-conditional threshold producing remote rate >= centre rate","remote rate near 0.27 (committee bias retained) or EO gap vs A near 0.25","grant rate at 36% or 44% edge by construction","distance or postal code used as a score term (proxy of remote)","over-correction: remote grants above merit-rule A rate by more than 3 points","first-gen or programme used to move decisions"],
"confidence":"medium"}
```

# Socio-economic need and access review (reviewer: need)

## Evidence (data/donnees_demandes.csv, n=10,000)
- Remote vs centre means: income 56.3k vs 76.1k, hours worked 13.0 vs 9.0 h/wk, first-generation 44% vs 26%, distance 221 vs 20 km, R 27.3 vs 28.0. Remote applicants are poorer and work more, so need is concentrated there (eval set shows the same pattern).
- Committee grant rate by income quintile (R flat at 27.6 to 27.8 across quintiles): 27%, 36%, 40%, 45%, 52%. The committee rewards wealth at equal merit. Standardized logistic coefficients (grant ~ R, log income, hours, first-gen): R 3.9, log income +0.96, hours +0.30 (+0.7 within each region group), first-gen -0.12 (about 0 within groups). Adding a remote dummy gives -0.95 SD, income stays +0.8, hours +0.77.
- Hours: raw grant rate falls slightly with hours (38%/41%/40%/38%/36%) only because hours proxy region; conditional on R and region the committee gives hours a positive weight (~0.18 R-SD per SD). That is need-aligned and acceptable. First-generation has no independent committee effect and little signal.
- Regional gap at equal R is large: R quintile 3 grants 38% centre vs 13% remote, quintile 4 80% vs 50%.

## Judgement
A bursary or loan exists to remove financial barriers, so a wealth premium is illegitimate: income must not raise the odds, and ideally carries a small need credit. Hours worked is a legitimate burden/need signal and a small positive weight is defensible (mostly region-neutral once R is held fixed). First-generation is a legitimate access criterion in principle but the data show no evidence for any weight; keep it at 0 rather than invent one. Region and distance get zero direct weight; the remote disadvantage is not need. Merit (R) remains dominant (R explains most of the outcome, corr 0.70), with programme used only to adjust R comparability.

Fair rule: grant the top ~40% on z(R) + 0.1*z(hours) - 0.05*z(log income), R adjusted for programme. Simulated on the 4,000 candidates: pure R gives centre 43.0% / remote 35.9%; adding hours 0.15 gives 40.4% / 39.5%. Pure R still leaves a rate gap because remote R is lower in the eval set; do not force parity beyond what R plus modest need produces. Anything above about 0.2 weight on hours or any positive income weight drifts from merit (agreement with pure R below 92%).

```json
{"reviewer":"need","criteria":{"cote_r":{"sign":"+","weight_vs_R":1.0},"log_revenu":{"sign":"-","weight_vs_R":-0.05},"heures_travail":{"sign":"+","weight_vs_R":0.1,"shape":"linear"},"premiere_generation":{"sign":"0","weight_vs_R":0.0},"programme":{"adjust":true}},
"fair_reference_rule":"z(R_programme_adjusted) + 0.1*z(hours) - 0.05*z(log income); grant top 40%",
"expected_grant_rate":{"overall":[0.38,0.42],"centre":[0.39,0.43],"remote":[0.35,0.41]},
"max_eo_gap_vs_merit":0.08,"max_dp_gap":0.08,"min_impact_ratio":0.85,"max_intersectional_gap":0.12,
"guard_rules":["Overall grant rate in 36-44%","Grant rate must be non-decreasing in R within each region group","At fixed R decile, grant rate must not rise with income quintile (spread <= 5 points); committee had 25-point spread","Remote-vs-centre grant-rate gap within each R decile <= 0.08","Applicants with higher hours at same R and region must not be granted less often than lower hours","No decision may use region or distance directly","Intersectional (remote x first-gen x low-income) grant rate >= 0.85 of overall within same R band"],
"red_flags":["Positive income weight (rewarding wealth)","Remote applicants at equal R granted far less than centre (historical 13% vs 38% at R quintile 3)","Penalising high hours worked or first-generation status","Region proxies (distance, postal code) used with net negative effect on remote","Overcorrection: remote granted above centre at equal R by large margin, or low-R applicants granted over high-R"],
"confidence":"medium"}
```

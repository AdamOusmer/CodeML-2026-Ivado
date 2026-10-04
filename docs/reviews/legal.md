# Legal and governance review (Quebec lens)

**Framing.** Region is not an enumerated ground in Charter of human rights and freedoms s.10, but « condition sociale » plausibly captures remoteness plus income and first-generation status (à vérifier: case law on territorial origin). Indirect discrimination (effet préjudiciable) is judged by effect, so proxies matter. s.86 (programmes d'accès à l'égalité) permits remedial measures but is not needed if the rule is merit-only. Loi 25 (s.12.1 of the Private-sector Act, à vérifier) requires notice that a decision is fully automated, the main factors used, and a right to human review. A public-style body must be able to explain each criterion.

**Evidence (10,000 historical rows).**
- Proxies of remote status: postal code AUC 1.00 (it encodes region), distance 0.998 (corr 0.82), hours 0.80 (corr 0.52), log income 0.69 (corr -0.33), first-gen 0.59. Dropping region cannot remove the signal.
- Within each region the committee's weights are the same: R 4.11 vs 4.17 logit per SD, log income 0.73 vs 0.80, hours 0.64 vs 0.68. Remote status adds a flat -2.16 logit penalty. First-gen has no effect (grant rate 0.27/0.27 remote, 0.48/0.49 centre).
- Same-R, different-outcome: R 29-31: centre 89.0% vs remote 65.4%; R 27-29: 45.6% vs 17.6%; R 31-33: 99.1% vs 93.8%. Top 40% by R: centre 89.2% vs remote 68.7%. The residual is region itself, not merit: not legally defensible.
- Evaluation set: 40.7% remote. R-only top 40%: centre 43.0%, remote 35.9%, impact ratio 0.83 (an R gap of 0.7 points alone justifies a gap of about 7 points, not 21).

**Criteria.** R is the only clearly job-related criterion, so it is the anchor. Hours worked is the one proxy with a plausible defence (work-study load as a hardship credit, positive and capped, not a penalty) and it closes the remaining 7-point R-driven gap (R+0.2h: centre 39.7%, remote 40.5%, IR 1.02). Income is region-correlated and gives a wealth advantage to the already advantaged (condition sociale): weight 0, or a hardship credit at most 0.1. Distance, postal code and region are excluded. First-gen has no historical effect: exclude it from the score and use it only as a monitored subgroup. Programme: adjust (programme mean R/grant rates differ by 1-6 points; compare within-programme). Hours must be a credit, never a penalty: a penalty would reproduce the historical pattern.

**Acceptable disparity.** Under indirect-discrimination doctrine the burden is on the institution to justify any effect. Use the 4/5 rule as heuristic: impact ratio at least 0.90 for remote vs centre in grants, with any residual gap explained by R. EO gap vs a merit reference at most 0.05. Strict demographic parity is not required (R means differ), but a gap beyond about 5 points must be explained.

**Governance.** Publish the criteria; notify applicants of automated decision and offer human review (Loi 25); log inputs and outputs; quarterly subgroup audit by region, income tercile and first-gen.

```json
{"reviewer":"legal","criteria":{"cote_r":{"sign":"+","weight_vs_R":1.0},"log_revenu":{"sign":"0","weight_vs_R":0.0},"heures_travail":{"sign":"+","weight_vs_R":0.2,"shape":"linear"},"premiere_generation":{"sign":"0","weight_vs_R":0.0},"programme":{"adjust":true}},
"fair_reference_rule":"z(R) + 0.2*z(hours), top 40%; no region, postal code, distance, income or first-gen terms",
"expected_grant_rate":{"overall":[0.38,0.42],"centre":[0.37,0.43],"remote":[0.36,0.42]},
"max_eo_gap_vs_merit":0.05,"max_dp_gap":0.07,"min_impact_ratio":0.9,"max_intersectional_gap":0.10,
"guard_rules":["no feature in the model with |corr| > 0.5 to region unless justified (distance, postal code excluded)","remote/centre grant-rate impact ratio >= 0.90 and absolute gap <= 7 points","within each R band (2-point bins) remote vs centre grant gap <= 5 points where n >= 30 per cell","EO gap vs merit reference (top 40% by R) <= 0.05","region x first-gen x income-tercile cells (n >= 100) within 10 points of overall rate","grant rate 36-44% overall, monotone in R: no candidate granted with lower R than a refused one in the same programme and region without logged reason","swapping region label alone changes no decision"],
"red_flags":["remote grant rate below 33% or ratio below 0.83 (copies historical penalty)","centre rate above 45% while remote below 38%","distance, postal code or region used directly or via lookalike encoding","hours or income used as a penalty","decisions where remote candidates with higher R are refused while centre candidates with lower R are granted","first-gen cell gap above 10 points","no explanation or human-review path for refused applicants"],
"confidence":"medium"}
```

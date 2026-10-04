# Merit and equal-opportunity review

**Lens:** a scholarship rewards academic merit; equally deserving applicants get equal chances whatever their region.

## Evidence (historical n=10,000; candidates n=4,000)
- R by group: remote 27.33 (sd 2.98) vs centre 27.99 (sd 2.99). The gap is 0.66 points, about 0.22 sd. R is not a region proxy in any serious way (corr with remote -0.11). Candidate set: 27.45 vs 28.04.
- Grant rate at equal R differs sharply. R in (28,30]: centre 70.9% vs remote 40.8%. R in (26,28]: 24.1% vs 6.7%. R in (30,32]: 95.8% vs 83.1%. Remote students are penalised at every R level.
- Logistic fit of the committee decision, standardized inputs: R +4.16 per sd, log income +0.80, hours +0.77, first-generation -0.05, distance +0.13, remote **-2.15**. The remote penalty is worth about 0.5 sd of R in logit terms (about 1.5 R points), on top of everything else.
- Region proxies: distance (corr 0.82; mean 221 km vs 20 km), hours worked (0.52; 13.0 vs 9.0 h), log income (-0.33; 56k vs 76k median-ish means), first-generation (0.19; 44% vs 26%). Postal code also leaks. Programme mix is similar by region.
- Programme: mean R is nearly flat across programmes (27.6 to 27.8) but committee grant rates vary from 36.5% (Arts) to 42.8% (Genie). That is a small programme effect not explained by R.
- Pure top-R at a 40% budget on candidates gives remote 35.9% vs centre 43.0%. This 7-point gap is legitimate: it reflects the R difference, not bias.

## Fair rule
Merit is R. Income, distance, first-generation and postal code get weight 0: they are not academic merit, and in the committee they act as hidden region proxies (income and hours carry positive weights although remote applicants have lower income, so those weights encode the penalty). Hours worked is a mild exception: reaching the same R while working about 4 extra hours per week shows more achievement per unit of study time. I allow a small linear credit of at most 0.15 sd; larger weights over-favour remote applicants (0.3 gives remote 43.6% vs centre 37.6%). Programme: rank R within programme, since nothing in R justifies a programme premium.

At weight 0.15 and a 40% budget, expected rates are remote 39.5% and centre 40.4%. Pure R gives remote 35.9% and centre 43.0%.

## Guards and flags
EO gap versus an R-merit reference (top 40% of R) should stay under 0.05. Rates by R band should match across regions within 5 points where each cell has at least 100 applicants.

Confidence is medium. No hidden labels exist. The hours credit and the programme handling are judgement calls.

```json
{"reviewer":"merit","criteria":{"cote_r":{"sign":"+","weight_vs_R":1.0},"log_revenu":{"sign":"0","weight_vs_R":0.0},"heures_travail":{"sign":"+","weight_vs_R":0.15,"shape":"linear"},"premiere_generation":{"sign":"0","weight_vs_R":0.0},"programme":{"adjust":true}},
"fair_reference_rule":"z(R) + 0.15*z(hours), R ranked within programme, top 40% granted",
"expected_grant_rate":{"overall":[0.36,0.44],"centre":[0.38,0.45],"remote":[0.33,0.42]},
"max_eo_gap_vs_merit":0.05,"max_dp_gap":0.08,"min_impact_ratio":0.85,"max_intersectional_gap":0.10,
"guard_rules":["Overall grant rate in [0.36,0.44]","Within each R band of width 2 with >=100 applicants per group, remote vs centre grant-rate difference <= 0.05","Monotone in R: no applicant is refused while a lower-R applicant of the same programme and hours band is granted","Remote vs centre grant rate difference <= 0.08","Region-blind: swapping the region label alone changes no decision","Mean R of granted remote applicants within 0.5 points of granted centre applicants","Per-programme grant rate within 5 points of the overall rate"],
"red_flags":["Remote grant rate below 0.33 at equal budget (committee penalty persists)","Remote grant rate above centre by more than 0.03 (reverse bias, over-correction beyond merit)","Positive weight on income or distance, which are proxies of region","Granting an applicant with lower R over a higher-R one in the same programme with no hours justification","Region-specific thresholds or quotas","Programme grant-rate spread over 8 points"],
"confidence":"medium"}
```

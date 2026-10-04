# Regional equity review (remote-student lens)

## Evidence (10,000 historical rows; 4,000 candidates)
- Grant rate and mean R by region: Montréal 4001 rows, R 27.99; Capitale-Nationale 1999, R 27.98; Bas-Saint-Laurent 1293, R 27.30; Côte-Nord 1178, R 27.33; Gaspésie 1529, R 27.35. R SD is about 3.0 everywhere. Remote R is about 0.65 points lower (0.22 SD).
- Grant rate at equal R, by R quintile Q1..Q5 (the middle three are the informative ones). Montréal 0.01/0.11/0.38/0.80/0.98, Capitale 0.01/0.10/0.37/0.79/0.98, BSL 0.00/0.02/0.10/0.53/0.90, Côte-Nord 0.00/0.03/0.13/0.47/0.91, Gaspésie 0.00/0.02/0.15/0.51/0.93. The penalty is uniform across the three remote regions and absent between the two centres. A 2-group "centre vs remote" split is therefore faithful.
- TPR among the top 39.9% by R: Montréal 0.894, Capitale 0.891, BSL 0.703, Gaspésie 0.700, Côte-Nord 0.657. This is the equal-opportunity gap, about 0.2, and it comes from the committee, not from merit.
- Committee logit on standardized features: R +4.15, log income +0.80, hours +0.77, first-gen -0.03, programme about 0, region dummies +/-0.35 to 0.45. Income and hours are the remote-correlated channels. Remote median distance is about 205 km against 17 km in the centres. Remote hours are 13.0 against 9.0 (the 20+ hour share is 2-3% remote and 0% in the centres). Remote log income is 10.84 against 11.15. Neither feature carries any R information (corr 0.03 and -0.05).
- Intersection (remote, first-gen, low income): grant rate 0.218. Centre, non-first-gen, low income: 0.397. Remote first-gen (income not low): 0.327. Remote status dominates and first-gen adds nothing (coefficient -0.03). Low income adds about -8 points within each region. That is the income channel, not a region effect.
- Programme: grant rates 0.365 to 0.428 with R means within 0.26. The programme effect is mostly an R-composition effect, so adjusting for it is optional and small.

## Fairness target
The right target is equal opportunity against merit, meaning equal grant probability at equal R, not outcome parity. R distributions differ slightly by region, so a pure R ranking already gives remote about 35.8% against centre about 42.8% (candidates, 39.9% budget). Forcing parity would deny merit-ranked centre applicants and cost utility. Closing the 0.2+ TPR gap does not require parity.

## Legitimacy of remote-correlated criteria
- Income: a higher-income-is-better weight rewards the existing advantage and penalizes the remote group (-0.3 in log income). It has no merit rationale. Weight 0, or slightly negative as a need factor, and never positive.
- Hours: the committee gives it a positive weight. Working 13 h/week against 9 while holding the same R is evidence of effort and need, and it is the only criterion that offsets distance and relocation cost. Allow a small positive weight, but not a threshold cliff (the 20+ band is about 100% granted in the centres, where almost nobody works that much).
- Distance and postal code are pure geography, so weight 0. Region is forbidden.

## Fair grant rates at 39.9% overall (4,000 candidates)
- Pure R rank: remote 35.8%, centre 42.8%.
- z(R)+0.2 z(hours): remote 40.5%, centre 39.5%, with per-region rates 38.7% to 41.7%. This is slightly pro-remote.
- Adding -0.2 z(log income) pushes remote to 43.6% and centre to 37.4%. This is too much compensation.
A defensible band is remote 0.35 to 0.42 and centre 0.38 to 0.43. A remote rate above 0.44 or a remote-minus-centre gap above +0.06 is over-correction. The historical 27.3% remote rate is unfair.

```json
{"reviewer":"regional","criteria":{"cote_r":{"sign":"+","weight_vs_R":1.0},"log_revenu":{"sign":"0","weight_vs_R":0.0},"heures_travail":{"sign":"+","weight_vs_R":0.15,"shape":"linear"},"premiere_generation":{"sign":"0","weight_vs_R":0.0},"programme":{"adjust":true}},
"fair_reference_rule":"z(R) + 0.15*z(hours); no region, income, distance or postal code; programme only as an R-composition adjustment",
"expected_grant_rate":{"overall":[0.36,0.44],"centre":[0.38,0.43],"remote":[0.35,0.42]},
"max_eo_gap_vs_merit":0.04,"max_dp_gap":0.06,"min_impact_ratio":0.88,"max_intersectional_gap":0.08,
"guard_rules":["TPR vs merit-top-39.9% (R-only rank): max difference across the 5 regions <= 0.05, and remote vs centre <= 0.04","Per-region grant rate within [0.33,0.45] for every one of the 5 regions; remote minus centre in [-0.06,+0.04]","Grant rate at equal R (R quintiles Q3-Q5, remote vs centre) within 0.08 in every bin","Intersectional cells (remote x first-gen x low income) grant rate >= 0.8 x the overall rate","Grant probability must be monotone non-decreasing in R within every region; no decision may be keyed on region, distance or postal code (shuffling these must change <1% of decisions)","Overall grant rate in [0.38,0.42]"],
"red_flags":["Remote grant rate below 0.34 (the committee penalty is still present), or above 0.44 (over-correction)","Positive weight on income, or any weight on distance or postal code, which keep the remote penalty","Hours used as a threshold or cliff rather than a small linear term","Equal opportunity closed via parity forcing, so that remote candidates below the merit cutoff are granted over higher-R centre candidates","Fairness measured only on a 2-group aggregate, hiding the gaps in Côte-Nord (0.66 TPR) or Gaspésie","Training on decision_octroi without removing the regional penalty (labels inherit it)"],
"confidence":"medium"}
```

# Consensus: what is considered fair

Five reviewers (merit, need, legal, data_process, regional). Machine-readable form: `consensus.json`. All numbers on the 4,000 candidates, k = 1598 (39.95%), features standardized with history mean/std (log income), top-k by score.

## Method
1. Criteria: sign by majority, weight = median of `weight_vs_R`, spread and dissent reported. Programme adjustment by majority (5/5 yes).
2. Consensus rule = median-weight formula; each reviewer's own rule kept as an ensemble of references.
3. Rate bands: [median of lows, median of highs] per group; intersection ("strict") also given.
4. Limits: median across reviewers; strictest value also given.
5. Guard rules and red flags deduplicated: raised by >= 2 reviewers = consensus, else minority (in JSON).
6. Decisions computed per rule and per existing file; "expected agreement with the hidden reference" = mean agreement with the 5 reviewer references (a proxy, not the true reference).

## Criteria
| Criterion | Consensus | Spread | Dissent |
|---|---|---|---|
| cote_r | +1.00 | 1.00 | none |
| log_revenu | 0.00 (4 of 5 sign 0 or negative) | -0.05 to +0.19 | need -0.05; data_process +0.19 |
| heures_travail | +0.15 (linear) | 0.10 to 0.20 | need 0.10, legal 0.20, data_process 0.19 |
| premiere_generation | 0.00 | 0 to 0 | none |
| programme | adjust R | 5/5 | regional: optional and small |

Consensus rule: `z(R) + 0.15*z(hours)`, top 1598. No region, distance, postal code, income or first-gen term. Programme adjustment (R centred by programme history mean) changes 0.85% of decisions, so it is not applied in the numbers below.

Dissents. data_process keeps income +0.19 (committee latent minus the remote shift); it is the outlier on every measure (agreement 94% with others, remote rate 0.370, impact ratio 0.88). need uses income -0.05 as a hardship credit. regional warns against threshold cliffs on hours: file W (hours above 10 h) is a cliff by design and is flagged (it passes every numeric limit, so the guard needs the shape check "hours linear, no cliff").

## Bands and limits
| Item | Consensus | Strict | Reviewer range |
|---|---|---|---|
| Overall rate | [0.36, 0.44] | [0.38, 0.42] | lows .36-.38, highs .42-.44 |
| Centre rate | [0.38, 0.43] | [0.42, 0.43] | .37-.42 / .43-.46 |
| Remote rate | [0.35, 0.42] | [0.36, 0.38] | .32-.36 / .38-.42 |
| max EO gap vs pure-R merit | 0.05 | 0.04 | .04-.08 |
| max DP gap | 0.08 | 0.06 | .06-.14 |
| min impact ratio (remote/centre) | 0.85 | 0.90 | .70-.90 |
| max intersectional gap | 0.10 | 0.08 | .08-.15 |
| max EO gap vs consensus | 0.06 | 0.03 | evidence below |
| min mean agreement with references | 0.93 | 0.96 | evidence below |
| max EO overshoot vs merit (added) | 0.09 | 0.07 | evidence below |

Added limit. EO vs pure R is signed (TPR centre minus TPR remote). The hours credit legitimately pushes remote above merit-only: the consensus rule itself scores -0.063 and legal's rule -0.087, so a two-sided 0.05 cap would fail the reviewers' own rules. The cap is therefore one-sided (0.05 disadvantage) plus an overshoot cap of 0.09 (strict 0.07).

Choice of the two evidence-based limits. EO vs consensus: the largest EO of any reviewer reference against the consensus is 0.051 (data_process), 0.025 (legal), 0.018 (need), 0 (merit, regional); limit 0.06 accepts all five references, strict 0.03 accepts all but data_process. Mean agreement: each reference's mean agreement with the other four is 0.976 (merit, regional), 0.969 (legal), 0.964 (need), 0.936 (data_process); limit 0.93 is the floor (lowest reference, rounded down), strict 0.96 is just under the median.

## Reference decisions on the candidates
| Rule | Overall | Centre | Remote | Impact ratio | EO vs pure R (signed) | Intersectional |
|---|---|---|---|---|---|---|
| pure R | .3995 | .428 | .358 | 0.837 | 0 | .084 |
| consensus (= merit = regional) | .400 | .404 | .393 | 0.971 | -0.063 | .057 |
| need | .400 | .410 | .385 | 0.938 | -0.052 | .078 |
| legal | .400 | .398 | .402 | 1.012 | -0.087 | .050 |
| data_process | .400 | .420 | .370 | 0.881 | -0.006 | .110 |

Pairwise agreement (decision identical):
| | merit | need | legal | data_process | regional | consensus |
|---|---|---|---|---|---|---|
| merit | 1 | .980 | .986 | .940 | 1.000 | 1.000 |
| need | .980 | 1 | .968 | .926 | .980 | .980 |
| legal | .986 | .968 | 1 | .938 | .986 | .986 |
| data_process | .940 | .926 | .938 | 1 | .940 | .940 |
Merit and regional rules are identical (same weights); consensus equals them. All references agree with pure R at 93-100% (consensus 95.6%).

## Existing decision files
Agreement with each reference (merit/need/legal/data_process/regional), rates, and limit checks (consensus limits; strict in last column).
| File | Rate c / r | IR | Agreement m/n/l/dp/r | Mean (min) | EO vs R (signed) | EO vs cons | Pass consensus | Strict fails |
|---|---|---|---|---|---|---|---|---|
| SUBMIT_integration_c595dd3 | .423 / .365 | 0.864 | .952/.941/.944/.978/.952 | .954 (.941) | -0.010 | 0.057 | all 10 pass (EO vs consensus 0.057 vs 0.06 and IR 0.864 vs 0.85 are marginal) | EO vs cons, IR, intersectional (.095), mean agreement |
| V_income_blind_pipeline_with_jury | .410 / .385 | 0.938 | .986/.983/.974/.940/.986 | .974 (.940) | -0.048 | 0.021 | all pass | none |
| W_hours_above_10h_best_fit | .399 / .400 | 1.002 | .966/.955/.968/.945/.966 | .960 (.945) | -0.051 | 0.003 | all pass | none; cliff flag (regional minority red flag) |
| J_merit_plus_hours_only | .400 / .399 | 0.999 | .990/.972/.996/.940/.990 | .977 (.940) | -0.079 | 0.017 | all pass | overshoot 0.079 > 0.07 |

All four hit 1598 grants (overall .3995). Mean agreement is a proxy for agreement with the hidden reference, not a measurement. SUBMIT (our declared pipeline) is the closest to data_process and the furthest from the others (remote 0.365, IR 0.864, remote disadvantage still visible in EO vs consensus 0.057); it passes only because the consensus limits are loose.

## Consensus guard rules and red flags (>= 2 reviewers)
- Overall rate in [0.36, 0.44]; region-blind (no region, distance, postal code); monotone in R; remote-vs-centre gap within R bands <= 0.08; |DP| <= 0.08 and IR >= 0.85; EO vs pure R <= 0.05 (disadvantage) with overshoot <= 0.09; intersectional cells within 0.10; agreement with pure R >= 92%; per-programme rate within 0.10 of overall.
- Income never positive; hours small linear credit (<= 0.2 sd), never a penalty or cliff.
- Red flags: remote rate < 0.33-0.34 (committee penalty kept); remote above centre by > 0.03-0.06 or > 0.44 (over-correction); positive income weight; distance/postal/region used or lookalike encoded; higher-R refused while lower-R granted; hours or income as a penalty; region thresholds, quotas, parity forcing.
- Minority items (single reviewer, in JSON): per-region bounds and cross-region TPR (regional), shuffle test and hours-cliff and label-training flags (regional), remote logit coefficient within +-0.3 (data_process), Loi 25 human-review and audit (legal), income-quintile spread at fixed R (need).

## Caveats
Reviewers had no hidden labels; the references are inferred. Median weights ignore that data_process (committee latent) may be the true reference: if it is, the consensus rule agrees with it at only 94%.

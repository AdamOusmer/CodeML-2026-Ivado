# Decision harness specification (v3, synced to HEAD 4d657c1)

Contract for the controller that wraps the fairness pipeline, decides a batch, audits it, and refuses to
publish unsafe decisions. "MUST" = contract. "DIVERGENCE:" = code differs from contract; code is the current truth.

## 1. Goals and non-goals
Goals
- Budget-valid decisions for a batch under one declared configuration.
- Publish only batches whose final audit is not ALERT.
- Replayable record of what was decided and why.
- `model_corrige.py` and `src/main.py` are thin composition roots over shared modules.

Non-goals
- No human gate, no review queue. Fully automated.
- No automatic policy change at decision time except the bounded `ADJUST_OFFSET`.
- No online learning (I7). No live tuning (I8).

## 2. Pipeline (order is normative)
1. Preprocess: `legitimate_features` only (cote_r, log revenue, work hours, first-generation, programme dummies).
   Region columns are kept for audit, never for default scoring.
2. Fit `CommitteeModel` on history (legit features + remote flag). Remove the committee's remote-region
   penalty: `corrected_logit(removal=1.0)`; allocate at budget -> corrected labels.
3. Main model: region-blind logistic regression (`StandardScaler` + `LogisticRegression`) on corrected labels.
4. Jury: weighted sum of percentile-rank votes `main_model` (model probability) and `merit` (`cote_r_equivalent`).
   Declared weights 0.5 / 0.5. Percentiles are computed within the scored batch.
5. Allocate: top `round(share * n)` scores, stable argsort.
6. Harness: audit -> (one bounded correction) -> publish or BLOCK.

Budget rule: `share = history["decision_octroi"].mean()` (39.94 % on supplied data, 10,000 rows;
1,598 grants on 4,000 candidates). MUST be read from data via `policy.core.budget_share`, never hardcoded.
MUST lie in `BUDGET_BOUNDS = (0.36, 0.44)`, else `ValueError`.
DIVERGENCE: `ValueError` is not in the handled list of `src/main.py:main`; it hits the generic handler, exit 1.

## 3. Module layout and ownership
| Path | Owns |
|---|---|
| `src/policy/regions.py` | `REGIONS`, `REMOTE_REGIONS`, `is_remote` (no sklearn) |
| `src/policy/core.py` | features, `BUDGET_BOUNDS`, `budget_share`, `allocate`, `percentile`, `logistic_regression`, `eo_gap` |
| `src/policy/models.py` | `Config`, `DECLARED_CONFIG`, `CommitteeModel`, `FairPipeline`, `reference_labels` |
| `src/evaluation/core.py` | `Candidate`, `evaluate`, `evaluate_split`, `run`, `summarize`, `scaled_utility` |
| `src/evaluation/candidates.py` | `default_candidates()`, candidate decide-fns, `pareto_report` (writes `resultats_pareto.csv`, `pareto_front.png`) |
| `src/evaluation/pareto.py` | `pareto_mask`, `plot` |
| `src/evaluation/tuner.py` | `SEARCH_SPACE`, `score_split`, `tune` (offline) |
| `src/monitoring/checks.py` | thresholds, `Check`, `run_checks`, `Verdict`, `verdict`, `overall_status` |
| `src/explain.py` | `explain` |
| `src/harness/record.py` | `ActionKind`, `Action`, `DecisionRecord` (single writer of outputs) |
| `src/harness/controller.py` | `decide`, `audit`, `fit_offset`, `OFFSET_GRID` |
| `src/preprocessing/validation.py` | `check-data` validation (own process pool) |
| `src/common/logging` | logging, `RunContext`, `get_logger` |
| `src/main.py`, `model_corrige.py` | composition roots / entry points |

`src/policy/__init__.py` MUST stay empty (validation subprocesses import `policy.regions` without sklearn).
DIVERGENCE: `reference_labels` lives in `policy/models.py:71`; earlier spec placed it in `core`.

## 4. Import graph
```
main.py, model_corrige.py -> any package (composition roots; lazy imports inside commands in main.py)
harness    -> monitoring, explain, policy, common
monitoring -> policy
explain    -> policy
evaluation -> policy            (evaluation.* may import each other)
preprocessing -> policy.regions, common
policy     -> policy only (+ numpy, pandas, scipy, scikit-learn)
common     -> stdlib, rich
```
Forbidden: any import of `main` or `model_corrige`; `policy` importing project modules outside `policy`;
`monitoring`, `evaluation`, `explain`, `preprocessing` importing each other; `harness` imported by anything but roots.
Verified with grep over `src/` and `model_corrige.py` at HEAD:
- No forbidden import found.
- DIVERGENCE (doc only): `policy/core.py:3` imports `scipy.stats.rankdata`; earlier spec said numpy/pandas/sklearn only.
- `model_corrige.py:18-19` imports `policy` directly beside `harness`/`evaluation`; allowed for a root.
- `harness` does not import `preprocessing` (earlier spec said it may).

## 5. Invariants
| Id | Invariant | Enforced by |
|---|---|---|
| I1 | Grants = round(share x n); 0.36 <= share <= 0.44 | `budget_share`, `allocate`, check "grant rate within budget" (ALERT, non-correctable) |
| I2 | Default scoring never reads `region_administrative`, `code_postal_3`, `distance_domicile_campus_km`; offset 0 => `score` = jury score | `legitimate_features`, `FairPipeline.score` |
| I3 | Region enters a decision only via `ADJUST_OFFSET`, abs(offset) <= 0.10, recorded; training labels use removal = 1.0 | `fit_offset` grid, `DecisionRecord` |
| I4 | Same history + batch => identical decisions, scores, record (record has no timestamps) | no RNG in decide path, stable sorts, deterministic LR |
| I5 | Blocked run never writes `predictions.csv` | `DecisionRecord.write` publishes only if `status == "published"` |
| I6 | Every harness action is one of `ActionKind` and appears in `record.actions` | `controller.decide` |
| I7 | Decisions are never fed back as training data; fit uses `history` only | `FairPipeline.fit(history)` |
| I8 | Tuner never changes live config; `DECLARED_CONFIG` is edited by the team only | `tuner.py` output is CSV only; nothing imports it into `decide` |

DIVERGENCE (I3): bound is enforced only by `OFFSET_GRID = linspace(-0.10, 0.10, 41)` (`controller.py:14`).
`FairPipeline.score/predict` accept any offset (`models.py:58`). Callers other than `fit_offset` can exceed it.

DIVERGENCE (I5): blocked run still writes `decision_record.json` and `explanations.csv` (`record.py:66-67`).
Only `predictions.csv` is withheld; a stale one from a prior run stays on disk by design.

## 6. Actions and flow
| Kind | Params (as coded) | When |
|---|---|---|
| `SELECT_CONFIG` | `{config: name}` | once, first (`controller.py:44`) |
| `ADJUST_OFFSET` | `{offset, moved: count, alerts: [check names]}` | first verdict ALERT and every ALERT check is `correctable` |
| `BLOCK` | `{checks, suggestion}` | final verdict ALERT |

```
FIT -> DECIDE(offset 0) -> AUDIT --OK/WARN--> PUBLISH
                             '--ALERT, all correctable--> ADJUST_OFFSET -> AUDIT --OK/WARN--> PUBLISH
                             '--ALERT otherwise-----------> BLOCK        '--ALERT--> BLOCK
```
- Exactly one correction at most. ALERT after the correction MUST end in BLOCK, exit 3, no fallback policy.
- `BLOCK.suggestion` = `MERIT_ONLY_SUGGESTION` text for humans; the harness MUST NOT apply it.
- Moved ids go in `record.moved_ids`; gaps before/after are readable from `verdicts[0]` vs `verdicts[1]`.
- DIVERGENCE: `SELECT_CONFIG` carries no Pareto evidence; `ADJUST_OFFSET.params` has `moved` count and `alerts`,
  not gaps or ids (`controller.py:44,55-56`). Earlier spec listed evidence, gaps, ids.
- If `fit_offset` returns 0, `ADJUST_OFFSET` is still recorded; second audit is identical and blocks.

Corrector (`fit_offset`): over `OFFSET_GRID` (41 points, [-0.10, 0.10]) pick the offset minimizing the mean
`eo_gap` vs `corrected` and `merit` references on the batch; ties on `round(gap, 3)` break to smaller abs(offset).
Offset is added to the jury score of remote-region applicants (`is_remote`), then `allocate` at the same budget.

## 7. Monitoring contract
`run_checks(history, batch, decisions, reviewed=None, eo_gap_alert=EO_GAP_ALERT) -> list[Check]`; `verdict(checks) -> Verdict`.
- Status = max over checks (OK < WARN < ALERT). `Verdict.correctable` = ALERT and all ALERT checks have `correctable`.
- ALERT-capable: budget (non-correctable); three opportunity-gap checks (merit, corrected, reviewed sample; `correctable`,
  warn 0.03, alert `EO_GAP_ALERT` 0.05); proxy AUC drift > +0.05 (non-correctable); feature PSI > 0.25 (non-correctable).
- WARN-only: demographic parity gap 0.10, impact ratio < 0.80, intersectional gap 0.15.
- `decide` does not pass `reviewed`; only `monitor --reviewed` does.
- References come from `policy.reference_labels(history, batch, share)`: `corrected`, `merit`, plus `historical` if batch has labels.
  Both proxy references are model-derived; tuner objective is partly circular (read every per-reference column).

## 8. Evaluation and tuner (offline)
- `pareto`: runs `default_candidates()` over K splits (CLI default 10, `model_corrige.py` `SPLITS = 10`), 70/30 stratified,
  seeds 0..K-1, ThreadPool `workers` (default 4). Candidates: production RF natural (anchor) and at budget,
  drop-proxies, ThresholdOptimizer, ExpGrad sweep, committee removal sweep, jury merit-weight sweep (`SEARCH_SPACE`),
  and `FairPipeline(DECLARED_CONFIG)`. Pareto axes: eo_gap vs acc_historical; only `budget_ok` rows eligible.
- `tune`: per `Config` in `SEARCH_SPACE` (merit weight 0, .25, .5, .75, 1; removal 1.0), per reference r in {corrected, merit}:
  `equity_r = 1 - gap_r / gap_r(committee, removal 0, at budget)`, `utility_r = scaled_utility`,
  `score_r = (20 equity_r + 15 utility_r) / 35`; `worst_case = min(score_corrected, score_merit)`. Output `resultats_tuner.csv`.
- Team reads table, edits `DECLARED_CONFIG` in `src/policy/models.py` by hand. Neither `tune` nor `pareto` feeds `decide`.
- Currently `DECLARED_CONFIG = Config("jury 50/50")` (removal 1.0, weights 0.5/0.5).

## 9. Explanations and record
`explain(pipeline, df, decisions, scores, offset, top=3)` -> columns `id_candidat, decision, score, merit_vote,
model_vote, offset` (offset x remote flag), `factor_1..3` (`"<feature> <+x.xx>"`, top by abs of coef x standardized value).

`DecisionRecord` fields: `config, share, status ("published"|"blocked"), ids, decisions, scores, offset, verdicts,
actions, moved_ids, explanations, region_rates, input_hashes` (sha256 per input file name).
`write(out_dir)` always writes `decision_record.json` (`summary()`: status, config, share, grants, applicants, offset,
actions, verdicts with checks, moved_ids, region_rates, input_hashes) and `explanations.csv`;
writes `predictions.csv` (`id_candidat, decision_octroi`) only when published (I5).
Per-applicant decisions and scores are in `explanations.csv`, not in the JSON.

## 10. Entry points and exit codes
- `check-data`, `monitor --history --batch --decisions [--reviewed] [--json]`, `decide --history --batch --out-dir [--json]`,
  `pareto --splits --workers --out-dir`, `tune --splits(5) --workers --out-dir`.
- Exit: 0 OK/published; 3 (`EXIT_ALERT`) monitor ALERT or decide blocked; 1 error; 2 bad arguments; 130 interrupted.
- `model_corrige.py`: `budget_share` -> `pareto_report` (10 splits, 4 workers) -> `decide` -> `record.write(ROOT)`;
  exit 3 if blocked. Same artifacts as `decide`.
- `decide(history, batch, input_hashes, config=DECLARED_CONFIG, eo_gap_alert=EO_GAP_ALERT) -> DecisionRecord` is pure:
  no file I/O, no printing; logs through `get_logger("harness")`.
  DIVERGENCE: earlier spec gave a `workers` parameter; code has none (`controller.py:41`).

## 11. Acceptance checks
- 4,000 rows, 1,598 grants (39.94 %), `offset == 0`, no `ADJUST_OFFSET`, status `published`; share read from data.
- Budget guard: history rate outside 36-44 % => `ValueError`, no output.
- Forced correctable ALERT (`eo_gap_alert` lowered, or remote jury scores shifted by -0.10) => `ADJUST_OFFSET` recorded,
  abs(offset) <= 0.10, `moved_ids` non-empty, grants still `round(share x n)`.
- Non-correctable ALERT (feature drift, e.g. a feature shifted so PSI > 0.25) => `BLOCK` without `ADJUST_OFFSET`, exit 3.
- ALERT after correction => `BLOCK`, exit 3; pre-existing `predictions.csv` byte-identical (I5).
- Replay twice => identical `predictions.csv`, `explanations.csv`, `decision_record.json` (I4).
- No proxy column read in default scoring: `score(df, 0)` identical when region/postal/distance columns are shuffled (I2).
- Offset-0 `explain` factors sum with intercept to `main_model_` `decision_function` within 1e-9.
- Import graph in section 4 holds (grep; no cycle; `policy/__init__.py` empty).
- `tune` and `pareto` write only their CSV/PNG; `decide` output unchanged by running them (I8).

## 12. Open risks
- Declared 50/50 jury rests on red-team simulation, not the hidden reference.
- Percentile votes depend on batch composition.
- Corrections inherit the proxy references' assumptions; both are derived from the audited committee.
- Offset bound is not enforced inside `FairPipeline` (I3 divergence).

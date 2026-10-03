# Decision harness specification (v4, synced to 576d28e + preprocessing nodes + validator jury)

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
1. Preprocess: `scoring_features` only (`cote_r, log_revenu, heures_travail`).
   Region columns are kept for audit, never for default scoring.
2. Fit `CommitteeModel` on history (the 8 `COMMITTEE_FEATURES` + remote flag). Remove the committee's remote-region
   penalty: `corrected_logit(removal=1.0)`; allocate at budget -> corrected labels.
3. Main model: region-blind logistic regression (`StandardScaler` + `LogisticRegression`) on corrected labels.
4. Validator jury (`policy.jury.validate`): the main model proposes the top `k = round(share * n)` (stable argsort).
   Jurors `merit` (`cote_r_equivalent`) and `programme_merit` (R z-scored within programme, history statistics) vote
   GRANT/REFUSE by batch percentile vs `1 - k/n`. Triggers (any fires): `near_cutoff` (band 0.025 x n around the cutoff rank),
   `low_confidence` (raw probability in 0.5 +- 0.15), `disagreement` (juror vs main percentile gap > 0.30).
   A triggered decision is overturned only if all jurors disagree (`quorum` 1.0, unanimous). Overturns are paired swaps
   (strongest first), so grants stay exactly `k`; unpaired overturns stand as proposed. Percentiles are computed within the scored batch.
5. Allocation is the jury's output; grants = `k` by construction.
6. Harness: audit -> (one bounded correction) -> publish or BLOCK.

Budget rule: `share = history["decision_octroi"].mean()` (39.94 % on supplied data, 10,000 rows;
1,598 grants on 4,000 candidates). MUST be read from data via `policy.core.budget_share`, never hardcoded.
MUST lie in `BUDGET_BOUNDS = (0.36, 0.44)`, else `ValueError`.
`decide` raises `InputError` (a `ValueError` subclass) on a budget outside the bounds or on missing/duplicate `id_candidat`.
CLI handles `InputError`, `DataValidationError`, `FileNotFoundError`, `KeyError` in one line, exit 1, no files; any other exception reaches the generic handler with traceback.
`KeyError` is still caught (missing CSV column); the required-column check is implemented by `src.preprocessing.frames.columns` (raises `DataValidationError`).

## 3. Module layout and ownership
| Path | Owns |
|---|---|
| `src/policy/regions.py` | `REGIONS`, `REMOTE_REGIONS`, `is_remote` (no sklearn) |
| `src/common/graph.py` | `Node`, `Graph`, `GraphError` (stdlib node runtime: dependency order, concurrency, add/replace/without) |
| `src/policy/schema.py` | `PROGRAMMES`, `TRANSFORMS`, `COMMITTEE_FEATURES`, `SCORING_FEATURES`, `feature_frame`, `committee_features`, `scoring_features` (no sklearn) |
| `src/policy/core.py` | `production_features`, `BUDGET_BOUNDS`, `budget_share`, `allocate`, `percentile`, `logistic_regression`, `eo_gap` (no longer owns features schema) |
| `src/policy/jury.py` | `JurySettings`, `JuryOutcome`, `validate` (numpy only; pure) |
| `src/policy/models.py` | `Config`, `DECLARED_CONFIG`, `CommitteeModel`, `FairPipeline`, `reference_labels` |
| `src/evaluation/core.py` | `Candidate`, `evaluate`, `evaluate_split`, `run`, `summarize`, `scaled_utility` (package-internal) |
| `src/evaluation/candidates.py` | `default_candidates()`, candidate decide-fns, `pareto_report` (returns table + figure; no file I/O) |
| `src/evaluation/pareto.py` | `pareto_mask`, `plot` (returns a matplotlib `Figure`; pyplot-free, thread-safe; no file I/O) |
| `src/evaluation/tuner.py` | `SEARCH_SPACE`, `score_split`, `tune` (offline) |
| `src/monitoring/checks.py` | thresholds, `Check`, checks as nodes, `CHECK_NODES`, `monitoring_graph`, `MONITORING`, `run_checks`, `Verdict`, `verdict`, `overall_status` |
| `src/explain.py` | `explain` |
| `src/harness/record.py` | `InputError`, `ActionKind`, `Action`, `DecisionRecord` (data + `summary()`; no file I/O) |
| `src/harness/controller.py` | `decide`, `audit`, `fit_offset`, `OFFSET_GRID` |
| `src/preprocessing/rules.py` | shared rules table + `DataValidationError` |
| `src/preprocessing/frames.py` | frame check nodes, `validate_frames`, `FrameReport` |
| `src/preprocessing/validation.py` | `check-data` validation (own process pool) |
| `src/pipelines/decision.py` | `DECISION_NODES`, `PARETO_NODES`, `DECISION`, `FULL`, `run_decision`, `run_full` (composition: read -> validate -> decide -> write; pareto branch concurrent) |
| `src/common/logging` | logging, `RunContext`, `get_logger` |
| `src/adapters/files.py` | all project file I/O: `Inputs`, `read_inputs`, `read_table`, `read_decisions`, `write_decision`, `write_table`, `save_figure`, `json_text` |
| `src/main.py`, `model_corrige.py` | composition roots / entry points |

`src/policy/__init__.py` MUST NOT import sklearn eagerly (validation subprocesses load `REGIONS` without it):
it exports through a PEP 562 `__getattr__` that imports only the submodule owning the requested name.

## 4. Boundaries, public API, import graph
Each package exposes its public API in `__init__.py` with `__all__`. Code outside a package imports only
`from src.<package> import name`; never `src.<package>.<module>`. Inside a package, relative imports.

| Package | Public API (`__all__`) |
|---|---|
| `src.policy` (lazy) | `REGIONS`, `REMOTE_REGIONS`, `is_remote`, `BUDGET_BOUNDS`, `budget_share`, `allocate`, `percentile`, `logistic_regression`, `eo_gap`, `PROGRAMMES`, `COMMITTEE_FEATURES`, `SCORING_FEATURES`, `feature_frame`, `committee_features`, `scoring_features`, `production_features`, `Config`, `DECLARED_CONFIG`, `CommitteeModel`, `FairPipeline`, `reference_labels`, `JurySettings`, `JuryOutcome`, `validate` |
| `src.monitoring` | `EO_GAP_ALERT`, `Check`, `Verdict`, `run_checks`, `verdict`, `overall_status`, `MONITORING`, `CHECK_NODES`, `monitoring_graph` |
| `src.explain` | `explain` |
| `src.evaluation` | `SEARCH_SPACE`, `tune`, `ParetoReport`, `pareto_report` |
| `src.harness` | `decide`, `InputError`, `DecisionRecord`, `Action`, `ActionKind` |
| `src.preprocessing` | `validate_datasets`, `DatasetReport`, `DataValidationError`, `validate_frames`, `FrameReport`, `Findings`, `FRAME_CHECKS`, `FRAME_CHECK_NODES`, `frame_graph` |
| `src.pipelines` | `DECISION_NODES`, `PARETO_NODES`, `DECISION`, `FULL`, `run_decision`, `run_full` |
| `src.adapters` | `Inputs`, `read_inputs`, `read_table`, `read_decisions`, `write_decision`, `write_table`, `save_figure`, `json_text` |
| `src.common.graph` | `Node`, `Graph`, `GraphError` |
| `src.common.logging` | unchanged |

Export rule: `__all__` lists names used outside the package, types its public functions return, and domain constants. Internals stay out.
Builders MAY add a name to `__all__` when a caller outside the package needs it; they MUST NOT import past it.

Dependency direction (arrow = may import):
```
main.py, model_corrige.py -> pipelines, adapters, harness, evaluation, monitoring, preprocessing, policy, common
pipelines     -> adapters, harness, evaluation, preprocessing, policy, common
adapters      -> harness (types only), common
harness       -> monitoring, explain, policy, common
monitoring    -> policy, common
explain       -> policy
evaluation    -> policy
preprocessing -> policy, common
policy        -> (numpy, pandas, scipy, scikit-learn only)
common        -> stdlib, rich
```
Forbidden: importing `main`, `model_corrige`, `adapters` or `pipelines` from any package; `monitoring`, `evaluation`, `explain`,
`preprocessing` importing each other; deep imports `src.<pkg>.<module>` across packages.
File I/O (`read_csv`, `to_csv`, `savefig`, `write_text`, `open`) only in `src/adapters/`, `src/main.py`
(argument parsing, stdout JSON), `src/common/logging` (log files) and `src/preprocessing/validation.py`
(streams the input CSVs it validates). Domain code returns data and figures.
The scanner also flags I/O calls `dump`, `to_json`, `to_parquet`, `to_pickle`, `save`, `savetxt`, `mkdir`, `write_bytes`, `unlink`, `rmtree` (plus the above and `read_text`, `read_bytes`) outside those files.
Import scan also catches submodule-name imports (`from src.policy import core`), `import_module` / `__import__` (except the `src/policy/__init__.py` lazy loader), and `src.<pkg>.<module>` attribute chains.
No abstract port classes: one filesystem adapter, plain functions. Add a Protocol only when a second backend exists.
Enforced by `scripts/acceptance.py` (static AST scan; proven by injecting a forbidden edge into a temp copy).

## 5. Invariants
| Id | Invariant | Enforced by |
|---|---|---|
| I1 | Grants = round(share x n); 0.36 <= share <= 0.44 | `budget_share`, `allocate`, check "grant rate within budget" (ALERT, non-correctable) |
| I2 | Default scoring never reads `region_administrative`, `code_postal_3`, `distance_domicile_campus_km`; offset shifts the main model probability for ranking only; jury triggers read the raw probability | `scoring_features`, `FairPipeline.decide`, `validate(..., ranking=)` |
| I3 | Region enters a decision only via `ADJUST_OFFSET` (shifts the ranking, not the triggers), abs(offset) <= 0.10, recorded; training labels use removal = 1.0 | `fit_offset` grid, `DecisionRecord` |
| I4 | Same history + batch => identical decisions, scores, record (record has no timestamps) | no RNG in decide path, stable sorts, deterministic LR |
| I5 | Blocked run never writes `predictions.csv` | `src.adapters.write_decision` writes `predictions.csv` only if `record.published` |
| I6 | Every harness action is one of `ActionKind` and appears in `record.actions` | `controller.decide` |
| I7 | Decisions are never fed back as training data; fit uses `history` only | `FairPipeline.fit(history)` |
| I8 | Tuner never changes live config; `DECLARED_CONFIG` is edited by the team only | `tuner.py` output is CSV only; nothing imports it into `decide` |

DIVERGENCE (I3): bound is enforced only by `controller.OFFSET_GRID = linspace(-0.10, 0.10, 41).round(3)`.
`FairPipeline.score` and `FairPipeline.predict` accept any offset. Callers other than `fit_offset` can exceed it.

I5 in practice: `src.adapters.write_decision` always writes `decision_record.json` and `explanations.csv` (the
audit trail of a blocked run) and writes `predictions.csv` only when published; an earlier file stays byte-identical.

## 6. Actions and flow
| Kind | Params (as coded) | When |
|---|---|---|
| `SELECT_CONFIG` | `{config: name}` | once, first (`controller.decide`) |
| `ADJUST_OFFSET` | `{offset, moved: count, alerts: [check names]}` | first verdict ALERT, every ALERT check `correctable`, and `fit_offset` != 0; re-runs the jury with the shifted ranking |
| `BLOCK` | `{checks, suggestion}` | final verdict ALERT |

```
FIT -> DECIDE(offset 0) -> AUDIT --OK/WARN--> PUBLISH
                             '--ALERT, all correctable--> ADJUST_OFFSET -> AUDIT --OK/WARN--> PUBLISH
                             '--ALERT otherwise-----------> BLOCK        '--ALERT--> BLOCK
```
- Exactly one correction at most. ALERT after the correction MUST end in BLOCK, exit 3, no fallback policy.
- `BLOCK.suggestion` = `MERIT_ONLY_SUGGESTION` text for humans; the harness MUST NOT apply it.
- DIVERGENCE: `moved` / `moved_ids` count post-jury decision differences (`controller.py`), so they include jury churn (swaps
  that change because the ranking shifted), not only offset-driven flips.
- Moved ids go in `record.moved_ids`; gaps before/after are readable from `verdicts[0]` vs `verdicts[1]`.
- DIVERGENCE: `SELECT_CONFIG` carries no Pareto evidence; `ADJUST_OFFSET.params` has `moved` count and `alerts`,
  not gaps or ids (`controller.decide`). Earlier spec listed evidence, gaps, ids.
- If `fit_offset` returns 0, no `ADJUST_OFFSET` is recorded; the run goes straight to BLOCK with reason
  "no offset in the allowed grid lowers the gap".
- A check whose metric cannot be computed (NaN, empty subgroup, single class) is a non-correctable ALERT, so it blocks.

Corrector (`fit_offset`): over `OFFSET_GRID` (41 points, [-0.10, 0.10], rounded to 3 decimals) pick the offset minimizing the larger
of the `eo_gap` vs `corrected` and vs `merit` (alerts fire on each gap); key `(round(max gap, 3), abs(offset), -offset)`,
so ties prefer the smaller and then the positive offset. A NaN gap scores infinity.
Offset is added to the main model probability of remote-region applicants (`is_remote`) as the ranking passed to `validate`; the jury then re-runs at the same budget.
Triggers and juror votes still read raw values (red-team: shifted triggers inverted the correction).

## 7. Monitoring contract
`run_checks(history, batch, decisions, reviewed=None, eo_gap_alert=EO_GAP_ALERT, *, graph=MONITORING, workers=4) -> list[Check]`; `verdict(checks) -> Verdict`.
- Status = max over checks (OK < WARN < ALERT). `Verdict.correctable` = ALERT and all ALERT checks have `correctable`.
- ALERT-capable: budget (non-correctable); three opportunity-gap checks (merit, corrected, reviewed sample; `correctable`,
  warn 0.03, alert `EO_GAP_ALERT` 0.05); proxy AUC drift > +0.05 (non-correctable); feature PSI > 0.25 (non-correctable); categorical drift, max PSI > 0.25 (non-correctable, blocks).
- WARN-only: demographic parity gap 0.10, impact ratio < 0.80, intersectional gap 0.15.
- `decide` does not pass `reviewed`; only `monitor --reviewed` does.
- References come from `policy.reference_labels(history, batch, share)`: `corrected`, `merit`, plus `historical` if batch has labels.
  Both proxy references are model-derived; tuner objective is partly circular (read every per-reference column).

## 8. Evaluation and tuner (offline)
- `pareto`: runs `default_candidates()` over K splits (CLI default 10, `model_corrige.py` `SPLITS = 10`), 70/30 stratified,
  seeds 0..K-1, ThreadPool `workers` (default 4). Candidates: production RF natural (anchor) and at budget,
  drop-proxies, ThresholdOptimizer, ExpGrad sweep, committee removal sweep, jury band sweep `BAND_SWEEP = (0, 0.0125, 0.025, 0.05, 0.10)`,
  and `FairPipeline(DECLARED_CONFIG)`. Pareto axes: eo_gap vs acc_historical; only `budget_ok` rows eligible.
- `tune`: per `Config` in `SEARCH_SPACE` (9 configs: band {0.0125, 0.025, 0.05} x quorum {0.5, 1.0} x jurors {merit; merit+programme_merit}, quorum 0.5 with one juror skipped as identical; removal 1.0), per reference r in {corrected, merit}:
  `equity_r = 1 - gap_r / gap_r(committee, removal 0, at budget)`, `utility_r = scaled_utility`,
  `score_r = (20 equity_r + 15 utility_r) / 35`; `worst_case = min(score_corrected, score_merit)`. Output `resultats_tuner.csv`.
- Team reads table, edits `DECLARED_CONFIG` in `src/policy/models.py` by hand. Neither `tune` nor `pareto` feeds `decide`.
- Currently `DECLARED_CONFIG = Config("validator jury")` (removal 1.0, `JurySettings()` defaults: band 0.025, conf 0.15, disagree 0.30, quorum 1.0, jurors merit + programme_merit).

## 9. Explanations and record
`explain(pipeline, df, decisions, scores, offset, top=3)` -> columns `id_candidat, decision, score, merit_vote,
model_vote, offset` (offset x remote flag), `validated` (bool), `trigger_reasons`, `juror_votes`, `jury_outcome`, `factor_1..3` (`"<feature> <+x.xx>"`, top by abs of coef x standardized value).
`jury_outcome` in `not_reviewed | confirmed | overturn_unpaired | overturned_out | overturned_in`.

`DecisionRecord` fields: `config, share, status ("published"|"blocked"), ids, decisions, scores, offset, verdicts,
actions, moved_ids, explanations, region_rates, input_hashes` (sha256 per input file name), `jury` (counts: `triggered`, `overturned_out`, `overturned_in`, `reasons` per trigger).
`src.adapters.write_decision(record, out_dir)` always writes `decision_record.json` (strict JSON via `src.adapters.json_text`: non-finite floats become null; `summary()`: status, config, share, grants, applicants, offset,
actions, verdicts with checks, moved_ids, region_rates, jury, input_hashes) and `explanations.csv`;
writes `predictions.csv` (`id_candidat, decision_octroi`) only when published (I5).
Per-applicant decisions and scores are in `explanations.csv`, not in the JSON.

## 10. Entry points and exit codes
- `check-data`, `monitor --history --batch --decisions [--reviewed] [--json]`, `decide --history --batch --out-dir [--json]`,
  `pareto --splits --workers --out-dir`, `tune --splits(5) --workers --out-dir`.
- Exit: 0 OK/published; 3 (`EXIT_ALERT`) monitor ALERT or decide blocked; 1 error; 2 bad arguments; 130 interrupted.
- `model_corrige.py`: `run_full` (`src.pipelines.FULL`): read -> `validate_frames` -> {`pareto_report` (10 splits,
  4 workers) ∥ `decide` -> `write_decision`}; in FULL `share` is taken from the decision record, so the Pareto branch
  starts after `decide`; exit 1 on `DataValidationError`/`InputError`/`FileNotFoundError`/`KeyError` (nothing written), 3 if blocked. `decide` CLI
  runs `src.pipelines.DECISION`. Same artifacts as `decide`.
- `decide(history, batch, input_hashes, config=DECLARED_CONFIG, eo_gap_alert=EO_GAP_ALERT) -> DecisionRecord` is pure:
  no file I/O, no printing; logs through `get_logger("harness")`.
  DIVERGENCE: earlier spec gave a `workers` parameter; code has none (`controller.decide`).

## 11. Acceptance checks
`OMP_NUM_THREADS=2 uv run python scripts/acceptance.py` runs 28 checks, all PASS:
1. `baseline`: 4,000 rows, 1,598 grants (39.94 %), `published`, no `ADJUST_OFFSET`, strict-JSON record, ids in batch order; share read from data.
2. `budget_guard`: history rate outside 36-44 % => exit 1, one ERROR line, no traceback, no files.
3. `duplicate_id`: duplicated `id_candidat` => exit 1, no traceback, no files.
4. `forced_correctable_alert`: `eo_gap_alert=0.015` => `ADJUST_OFFSET`, abs(offset) <= 0.10, `moved_ids` non-empty, grants = `round(share x n)`.
5. `zero_offset_block`: `fit_offset` patched to 0 => actions `[SELECT_CONFIG, BLOCK]`, reason `NO_OFFSET_REASON`, one verdict.
6. `drift_block`: `cote_r_equivalent` + 3 => exit 3, `blocked`, no `ADJUST_OFFSET`, BLOCK lists "feature drift, max PSI", pre-existing `predictions.csv` byte-identical (I5).
7. `categorical_drift`: `premiere_generation_universitaire` all 0 => exit 3, blocks via non-correctable "categorical drift, max PSI" ALERT (first-gen moved out of numeric PSI), no `ADJUST_OFFSET`.
8. `uncomputable_metric_block`: remote-only batch => exit 3, non-correctable ALERT with null value, no `ADJUST_OFFSET`.
9. `alert_after_correction`: `eo_gap_alert=-1.0` => `ADJUST_OFFSET` then `BLOCK`, `blocked`, sentinel `predictions.csv` intact (I5).
10. `replay`: two runs => identical `predictions.csv`, `explanations.csv`, `decision_record.json` (I4).
11. `region_blindness`: `score(df, 0)` identical when region/postal/distance columns are shuffled (I2).
12. `tuner_not_live`: `tune` writes only `resultats_tuner.csv`; `decide` output unchanged after it (I8).
13-16. `boundaries_deep_imports`, `boundaries_direction`, `boundaries_third_party`, `boundaries_file_io`: section 4 AST scan on the repo has no violations.
17. `boundaries_policy_lazy`: `from src.policy import REGIONS` loads no sklearn.
18. `boundaries_proof`: scanner catches injected violations in a temp copy and passes an allowed edge. Probes: deep import, wrong direction, `to_csv`, `json.dump`, `Path.mkdir`, `from src.policy import core`, `import_module`, `__import__`, `src.policy.core` attribute chain; allowed `from src.policy import eo_gap` stays clean.
19. `explain_identity`: offset-0 `explain` factors plus intercept equal `main_model_` `decision_function` within 1e-9; required columns present.
20. `frame_rejects`: `validate_frames` rejects invalid frames (`DataValidationError`).
21. `csv_income_zero`: CSV validator rejects income 0.
22. `region_invariance`: shuffled region/postal/distance columns leave `FairPipeline.score(batch, 0)` identical.
23. `schemas`: `committee_features` 8 columns, `scoring_features` 3, fixed order.
24. `categorical_drift_ok`: real batch gives one "categorical drift, max PSI" check, OK, value ~0.0383, Gaspesie Genie +7.98.
25. `monitoring_concurrency`: `run_checks` `workers=1` and `workers=4` give identical results.
26. `graph_runtime`: `src.common.graph` runtime contract (section 8.1 of PREPROCESSING_SPEC).
27. `cli_invalid_batch`: income 0 in batch => `decide` CLI exit 1, no traceback, nothing written, sentinel `predictions.csv` intact.
28. `extra_fields`: CLI rejects rows with more fields than header, exit 1, nothing written.

## 12. Open risks
- Declared jury rests on red-team simulation, not the hidden reference.
- Percentile votes depend on batch composition.
- Corrections inherit the proxy references' assumptions; both are derived from the audited committee.
- Offset bound is not enforced inside `FairPipeline` (I3 divergence).

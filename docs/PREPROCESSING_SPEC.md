# Preprocessing specification

Status: ready for implementation. Planner: main session (Opus). Independent review: Codex gpt-6.1-sol (read-only),
whose conclusions matched the planner's ablation. Conforms to `HARNESS_SPEC.md` §3–4 (public APIs, import direction,
file I/O only in adapters/entry points/validation). Evidence: §7 and `FINDINGS.md`.

## 1. Decisions

| Column | Committee reconstruction | Main-model scoring | Audit / monitoring | Why |
|---|---|---|---|---|
| `id_candidat` | — | — | alignment, output order | identifier |
| `cote_r_equivalent` | raw | raw | yes | academic merit, committee weight +4.12 |
| `revenu_familial_estime` | natural log | natural log | yes | committee +0.80; proxy AUC 0.67–0.69; must stay paired with hours |
| `heures_travail_semaine` | raw | raw | yes | committee +0.76; proxy AUC 0.81; must stay paired with income |
| `premiere_generation_universitaire` | binary | **dropped** | yes | weight −0.03; no measurable effect on any reference |
| `programme_etudes` | 4 dummies, base "Arts et lettres" | **dropped** | yes | weights ≤ 0.024; no measurable effect |
| `region_administrative` | remote indicator (to estimate and remove the penalty) | never | yes | sensitive attribute (I2) |
| `code_postal_3` | — | never | consistency check | 1:1 with region |
| `distance_domicile_campus_km` | — | never | yes | proxy AUC 0.998 |
| `decision_octroi` | target (history only) | never; corrected labels only | budget share | biased outcome |

Rules
- Income and hours are kept **together or not at all**. They are opposing proxies (remote: lower income, more hours);
  together the final score carries almost no region information beyond R (AUC 0.54). Dropping one alone sends it to
  0.67–0.70 or 0.36–0.40 and the worst-case score from 0.927 to 0.818 / 0.766.
- Committee reconstruction keeps the full 8-column schema so the regional penalty is estimated without omitted-variable
  bias; scoring uses the 3-column schema. Changes about 6/4,000 candidate decisions (measured).
- Transforms: natural log of income (skew 1.425 → −0.041); no winsorizing (1st/99th clipping gains 0.07 points);
  scalers fitted on the training partition only (already inside the policy pipelines).
- No proxy residualization: region-mean residualizing income/hours worsened EO gaps (0.042 / 0.017) and would read
  region at decision time (breaks I2). No reweighing: changed agreement by 0.03 points.

Known limitation (record in audit and governance, do not fix here): the committee rewards wealth and work hours.
Holding other inputs fixed, doubling income multiplies approval odds by 3.40; five more weekly hours by 2.70.
Removing the regional penalty preserves both. For a need-based award this is a second policy bias (construct validity).

## 2. Components

### 2.1 Feature schemas — `src/policy/core.py` (policy owns legitimacy)
- `COMMITTEE_FEATURES = ["cote_r", "log_revenu", "heures_travail", "premiere_generation", "prog_Genie",
  "prog_Sante", "prog_Sciences", "prog_Sciences sociales"]`
- `SCORING_FEATURES = ["cote_r", "log_revenu", "heures_travail"]`
- `committee_features(df) -> DataFrame` (8 columns, fixed order, fixed programme categories)
- `scoring_features(df) -> DataFrame` (3 columns)
- `legitimate_features` is replaced by these two; update every caller:
  `CommitteeModel` → `committee_features`; `FairPipeline` (fit, `model_probability`, `contributions`) → `scoring_features`;
  `monitoring` proxy AUC and PSI → `scoring_features`; evaluation candidates `drop_proxies`, ExpGrad,
  ThresholdOptimizer → `scoring_features`; `explain` unchanged (uses `pipeline.contributions`, now 3 factors).
- Export both functions and both constants from `src.policy.__all__` (lazy export rule unchanged).

### 2.2 Frame validation — `src/preprocessing/`
- `validate_frames(history, batch) -> FrameReport`, raising `DataValidationError` with the same message style as
  `validate_datasets`. Vectorized on DataFrames (the CSV streaming validator stays for `check-data`).
- Rules shared with the CSV validator from one table in `src/preprocessing/rules.py` (no duplicated constants):
  required columns (label only in history); unique IDs, disjoint history/batch IDs; `C[0-9]{6}` IDs; known
  programmes and regions (reject unknown, never default to the baseline category); `A1A` postal format
  `[A-Z][0-9][A-Z]` (ASCII digits only);
  binary fields in {0, 1}; R in [15, 40]; hours in [0, 168]; distance ≥ 0; **income strictly > 0**
  (the current validator accepts 0 and `log(0) = -inf`); every value finite after transforms.
- Warnings, not errors, in `FrameReport.warnings`: postal prefix whose region differs from history's mapping;
  numeric values outside the history range (candidates exceed it: R 40.00 > 38.41, income 347,143 > 293,879).
- `FrameReport(history_rows, batch_rows, warnings)`; add `validate_frames`, `FrameReport` to `src.preprocessing.__all__`.
- Imports: `src.policy` (regions, programme list), `src.common`. No file I/O.

### 2.3 Entry points — gatekeeper
- `src/main.py` `decide` and `model_corrige.py` call `validate_frames(history, batch)` after reading and before
  `decide(...)`; on `DataValidationError` exit 1 with the message; log every warning.
- The harness stays free of preprocessing imports (HARNESS_SPEC §4).

### 2.4 Categorical drift — `src/monitoring/checks.py`
- Add a check for categorical shift per group (programme shares and first-generation rate, history vs batch):
  categorical PSI over category frequencies, warn > 0.10, alert > 0.25, not correctable. Numeric histogram PSI must
  no longer be applied to binary columns (it can merge 0/1 into one bin and report zero drift).
- Monitored numeric features become `SCORING_FEATURES`.

## 3. Work packages

| WP | Files owned | Model | Depends | Done when |
|---|---|---|---|---|
| P1 Feature schemas + callers | `src/policy/core.py`, `src/policy/models.py`, `src/policy/__init__.py`, `src/evaluation/candidates.py`, `src/evaluation/tuner.py` if touched | Opus | — | all callers on the new functions; `model_corrige.py` grants 1,598; decisions differ from the previous revision on ≤ 10 candidates |
| P2 Rules table + frame validator | `src/preprocessing/rules.py`, `src/preprocessing/frames.py`, `src/preprocessing/validation.py`, `src/preprocessing/__init__.py` | Codex | — | CSV and frame validators share `rules.py`; income 0 rejected by both |
| P3 Gatekeeper wiring | `src/main.py`, `model_corrige.py` | Opus | P1, P2 | invalid batch → exit 1, no files written |
| P4 Categorical drift + scoring features in monitoring | `src/monitoring/checks.py`, `src/monitoring/__init__.py` | Sonnet | P1 | categorical check present; real batch still OK/WARN as today |
| P5 Acceptance additions | `scripts/acceptance.py` | Sonnet | P3, P4 | checks in §4 pass |
| P6 Refresh numbers | `docs/FINDINGS.md`, regenerate `resultats_pareto.csv`, `pareto_front.png`, `resultats_tuner.csv` | Sonnet | P5 | tables match the new revision |

P1 ∥ P2, then P3 ∥ P4, then P5, then P6. One writer per file; commit per WP; one Python process per agent,
`OMP_NUM_THREADS=2`. The orchestrator had uncommitted changes in `src/` when this spec was written: rebase the
WPs on its latest commit before starting.

## 4. Acceptance checks
- Reject: income 0, unknown programme, unknown region, NaN or inf in any feature, duplicate or overlapping IDs.
- Invariance (I2): shuffling `region_administrative`, `code_postal_3` and `distance_domicile_campus_km` across batch
  rows leaves `FairPipeline.score(batch, offset=0)` identical.
- Schemas: `committee_features` 8 columns, `scoring_features` 3 columns, fixed order, history and batch aligned.
- Budget: 1,598 grants for 4,000 candidates (historical share 39.94 %); IDs in candidate order; replay identical.
- Categorical drift check reports per-group programme shift (Gaspésie Génie share +7.98 points, PSI ≈ 0.038 → OK).

## 5. Out of scope
Need-based reweighting of income/hours (policy change, not preprocessing); outlier clipping; imputation (no missing
values exist; any missing value is a validation error).

## 6. Data facts (both files, verified)
No missing values, duplicate feature rows, overlapping IDs, zero income/hours/distance. Ranges history → candidates:
R 16.78–38.41 → 17.04–40.00; income 13,364–293,879 → 13,236–347,143; hours 1–29 → 1–26. Remote share 40.0 % → 40.7 %.
Within-region max continuous PSI: Montreal 0.013, Capitale-Nationale 0.029, Bas-Saint-Laurent 0.040, Côte-Nord 0.054,
Gaspésie 0.040.

## 7. Evidence (10 splits, references fixed to the full committee; jury merit weight 0.25)

| Scoring features | Worst case (2 refs) | Worst case (3 refs) | Region AUC beyond R |
|---|---|---|---|
| Full (R, income, hours, first-gen, programme) | 0.927 | 0.922 | 0.540 |
| R, income, hours | 0.926 | 0.922 | 0.543 |
| Drop income | 0.818 | 0.813 | 0.672 |
| Drop hours | 0.766 | 0.766 | 0.373 |
| Drop income and hours | 0.843 | 0.843 | 0.484 |
| R only | 0.845 | 0.845 | 0.489 |

## 8. Node architecture (supersedes the file split in §2–3 where they differ)

Every multi-step flow is a declared graph of nodes. Order comes from data dependencies, not list position, so steps
can be added, replaced, removed or reordered by editing one node tuple. Independent nodes run concurrently.

### 8.1 Runtime — `src/common/graph.py` (stdlib only)
- `Node(name, fn, inputs=(), after=())`, frozen dataclass. The node's artifact is named `name`. `fn` receives the
  artifacts named in `inputs`, positionally. `after` lists node names that must finish first without passing data.
- `GraphError(ValueError)`.
- `Graph(nodes)`: rejects duplicate names, `after` naming no node, and cycles. An input that names no node is a seed.
  `nodes`, `seeds` (frozenset), `order()` (topological, ties broken by declaration order),
  `add(*nodes)`, `replace(node)` (same name, same position), `without(*names)` all return a new `Graph`.
- `run(seeds, *, targets=None, workers=1) -> dict`: runs the targets and their transitive dependencies (all nodes when
  `targets` is None). Before running anything it raises `GraphError` for an unknown target, a missing seed, a seed
  that shares a node's name, or `workers < 1`. `workers == 1` runs inline in `order()`; otherwise a
  `ThreadPoolExecutor` runs every ready node. The first node exception stops scheduling and is re-raised unchanged
  after running nodes finish: queued nodes are skipped once a failure or interrupt is seen, and active nodes are drained before `run` raises. Returns the seeds plus every produced artifact.

### 8.2 Policy schema — `src/policy/schema.py` (numpy, pandas only; keeps `REGIONS`-style lazy import sklearn-free)
- `PROGRAMMES` (moved from `core.py`, base category first), `TRANSFORMS: dict[str, Callable[[DataFrame], Series]]`
  (one entry per derived column, committee order), `COMMITTEE_FEATURES = list(TRANSFORMS)`, `SCORING_FEATURES`.
- `feature_frame(df, schema)`, `committee_features(df)`, `scoring_features(df)`; columns in schema order, index = `df.index`.
- `legitimate_features` is deleted. Callers per §2.1.

### 8.3 Frame validation — `src/preprocessing/`
- `rules.py`: the single rules table shared by the CSV and frame validators: `DataValidationError` (moved here),
  `FEATURE_COLUMNS`, `LABEL_COLUMN`, `ID_PATTERN`, `POSTAL_PATTERN`, `CATEGORIES`, `BINARY_COLUMNS`,
  `Bound(low, high=None, strict_low=False)` with `accepts(values)` (scalar or array; finite and within bounds) and
  `describe()`, `NUMERIC_BOUNDS` (income `Bound(0, strict_low=True)`), `MAX_REPORTED_ISSUES`, `required_columns(labeled)`,
  `binary_columns(labeled)`.
- `frames.py`: `Findings(issues=(), warnings=())`; check nodes over seeds `history`, `batch`, each returning `Findings`
  (`columns` raises instead, structure first); `FRAME_CHECK_NODES`; `frame_graph(nodes=FRAME_CHECK_NODES)` appends the
  consolidation node `report`; `FRAME_CHECKS = frame_graph()`; `FrameReport(history_rows, batch_rows, warnings)`;
  `validate_frames(history, batch, *, graph=FRAME_CHECKS, workers=4) -> FrameReport`.

### 8.4 Monitoring — `src/monitoring/checks.py`
- Each check is a node over seeds `history`, `batch`, `decisions`, `reviewed`, `eo_gap_alert`; shared intermediates
  (`remote`, `references`, proxy AUCs, feature frames) are nodes, so the two proxy AUC fits run concurrently.
  `CHECK_NODES`, `monitoring_graph(nodes)` appends the consolidation node `checks` (declared order, `None` skipped),
  `MONITORING`; `run_checks(..., *, graph=MONITORING, workers=4)` keeps its signature and output order.
- New check `categorical drift, max PSI`: per region, `programme_etudes` and `premiere_generation_universitaire`,
  categorical PSI, warn > 0.10, alert > 0.25, not correctable. Numeric drift: histogram PSI on `SCORING_FEATURES` only.

### 8.5 Composition — `src/pipelines/` (new package, the gatekeeper)
- `DECISION_NODES`: `inputs` (adapter read) → `history`, `batch` → `frame_report` (`validate_frames`) →
  `frame_warnings` (logs each warning) → `record` (`decide`, after `frame_report`) → `written` (adapter write).
  `PARETO_NODES`: `share` (taken from `record`, so the Pareto branch starts after `decide`), `pareto`, `pareto_written`, all after `frame_report`. `DECISION`, `FULL` graphs;
  `run_decision(...)`, `run_full(...)`. In `FULL` the Pareto branch and the decision branch run concurrently.
- Entry points call these; nothing is written when validation fails. `src.evaluation.plot` builds a pyplot-free
  `Figure` so it is safe off the main thread.

### 8.6 Import direction changes
`monitoring -> policy, common`; `pipelines -> adapters, harness, evaluation, preprocessing, policy, common`;
`main.py`, `model_corrige.py` additionally `-> pipelines`. No package imports `pipelines`.

### 8.7 Work packages (Sonnet builders, one writer per file; Opus orchestrates and verifies)
| WP | Files |
|---|---|
| N1 | `src/common/graph.py` |
| N2 | `src/policy/schema.py`, `src/policy/__init__.py` |
| N3 | `src/policy/core.py`, `src/policy/models.py` |
| N4 | `src/evaluation/candidates.py`, `src/evaluation/pareto.py` |
| N5 | `src/preprocessing/rules.py`, `src/preprocessing/validation.py` |
| N6 | `src/preprocessing/frames.py`, `src/preprocessing/__init__.py` |
| N7 | `src/monitoring/checks.py`, `src/monitoring/__init__.py` |
| N8 | `src/pipelines/decision.py`, `src/pipelines/__init__.py` |
| N9 | `src/main.py`, `model_corrige.py` |
| N10 | `scripts/acceptance.py` |
| N11 | `docs/HARNESS_SPEC.md`, `docs/FINDINGS.md` |

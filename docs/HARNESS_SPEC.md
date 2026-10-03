# Decision harness specification (v2, after review)

Scope: the controller that wraps the fairness pipeline, takes allocation decisions for a batch,
and refuses to publish unsafe ones.

## 1. Goals and non-goals

Goals
- Budget-valid decisions for a batch under a declared, justified policy configuration.
- Every published batch satisfies section 2, otherwise publication is blocked.
- A complete, replayable record of what was decided and why.
- `model_corrige.py` (deliverable) and the CLI stay thin entry points over shared modules.

Non-goals
- The decision path is fully automated: no human gate. The configuration is declared (50/50 jury)
  and justified by evaluation data; the review queue is an informational output.
- The tuner is an offline experiment tool (6.2). It never sets the live configuration: every proxy
  reference is itself a candidate, so its objective is partly circular (review finding: optimizing it
  moved 64 decisions toward the audited committee). Its table informs the declared choice.
- No automatic policy change at decision time beyond the bounded `ADJUST_OFFSET`.
- No online learning, no training on the harness's own decisions.

## 2. Invariants

| Id | Invariant | Enforced by |
|---|---|---|
| I1 | Grants = round(share x n), share = historical grant rate, 0.36 <= share <= 0.44 | `policy.budget_share`, `policy.allocate`, Auditor |
| I2 | Default scoring never reads `region_administrative`, `code_postal_3` or `distance_domicile_campus_km` | `policy.legitimate_features`, `FairPipeline.score` returns the jury score when offset = 0 |
| I3 | Region enters a decision only through `ADJUST_OFFSET`, abs(offset) <= 0.10, recorded and disclosed; training labels have the full penalty removed (removal = 1.0) | Corrector, `Config`, DecisionRecord |
| I4 | Same inputs => identical decisions and record (excluding timestamps) | fixed seeds, stable sorts |
| I5 | A blocked run never overwrites a published `predictions.csv` | `DecisionRecord.write` (single writer) |
| I6 | Every harness action belongs to section 4 and appears in the record | controller loop |

## 3. Roles

| Role | Input | Output | May change |
|---|---|---|---|
| Gatekeeper | history CSV, batch CSV | validated frames or error | nothing |
| Decider | history, batch, declared `Config` | scores, decisions | nothing |
| Auditor | history, batch, decisions | `Verdict` | nothing |
| Corrector | `Verdict`, batch, jury scores | new decisions + `Action` | bounded offset, or block |
| Escalator | ranks, decisions, moved ids | review queue | nothing; flags only |
| Recorder | everything above | `DecisionRecord` | nothing |

## 4. Actions (closed set)

| Kind | Params | When |
|---|---|---|
| `SELECT_CONFIG` | config, evidence (Pareto summary row) | once, at start |
| `ADJUST_OFFSET` | offset, gaps before/after, moved ids | ALERT where every alerting check is correctable |
| `BLOCK` | failing checks, suggested human action | non-correctable ALERT, or ALERT after one correction |
| `ESCALATE` | count by reason | always |

## 5. Flow

```
VALIDATE -> SELECT_CONFIG -> DECIDE -> AUDIT --OK/WARN--> ESCALATE -> PUBLISH
                                         '--ALERT (correctable)--> ADJUST_OFFSET -> AUDIT --OK/WARN--> ESCALATE -> PUBLISH
                                         '--ALERT (otherwise)----> BLOCK         '--ALERT--> BLOCK
```
Correctable = opportunity-gap checks only (`Check.correctable = True`). Budget, proxy drift and
feature drift ALERTs block: data shift or a broken budget needs a human. A block reason may suggest
the merit-only policy for human sign-off; the harness never applies it.

## 6. Components

### 6.1 Policy — domain core, imports only numpy, pandas, scikit-learn
`src/policy/__init__.py` stays empty (validation subprocesses import `policy.regions` without sklearn).
- `regions.py`: `REGIONS`, `REMOTE_REGIONS`, `is_remote(df)`; `preprocessing.validation` imports it.
- `core.py`: `legitimate_features`, `production_features`, `BUDGET_BOUNDS`, `budget_share`, `allocate`,
  `percentile`, `eo_gap`, `reference_labels(history, target, share) -> {historical?, corrected, merit}`
  (`historical` only when the target has `decision_octroi`).
- `models.py`:
  - `Config(frozen)`: `removal: float = 1.0`, `jury_weights = {"main_model": 0.5, "merit": 0.5}`, `name`.
  - `CommitteeModel`: unchanged.
  - `FairPipeline(config, share)`: `fit(history)`, `jury_score(df)`, `score(df, offset=0.0)`,
    `predict(df, offset=0.0)`, `contributions(df) -> DataFrame` (coef x standardized value per feature).
  - `DECLARED_CONFIG = Config(name="jury 50/50")`.

### 6.2 Evaluation — imports policy
- `core.py`: `Candidate(name, family, decide, setting)`, `evaluate(decisions, references, remote) -> dict`,
  `scaled_utility`, `run(history, candidates, share, splits, workers)`, `summarize(results)`.
- `pareto.py`: `pareto_mask`, `plot`.
- Candidates: baseline anchor (natural threshold), baseline at budget, drop-proxies, ExpGrad sweep,
  ThresholdOptimizer, removal sweep, merit-weight sweep, and `FairPipeline(DECLARED_CONFIG)` with offset 0.
- `tuner.py` (offline): `tune(history, share, configs, splits=5, workers=4) -> DataFrame`. Per config, per
  reference r in {corrected, merit}: equity_r = 1 - gap_r / gap_r(committee at budget), utility_r = scaled
  agreement, score_r = (20 equity_r + 15 utility_r) / 35; reports every score_r, the worst case, and mean gaps.
  Search space: merit weight in {0, 0.25, 0.5, 0.75, 1}, removal = 1.0. Output only; never feeds `decide`.

### 6.3 Monitoring — imports policy
- `checks.py`: existing `run_checks(history, batch, decisions, reviewed=None)`, references from
  `policy.reference_labels`; `Check.correctable` set for the opportunity-gap checks; `Verdict(status, checks)`.
- Thresholds unchanged; `EO_GAP_ALERT` injectable for acceptance tests.

### 6.4 Decision outputs — imports policy
- `decision.py`:
  - `review_queue(ids, scores, decisions, band_share=0.05, moved_ids=())`: k = granted count;
    near cutoff = rank distance abs(rank - k) <= round(band_share x n); plus every moved id.
    Columns: id_candidat, rank, score, decision, cutoff_distance, reason (`near_cutoff` | `moved_by_correction`).
  - `explain(pipeline, df, decisions, scores, offset, top=3)`: id, decision, score, merit_vote, model_vote,
    applied offset, factor_1..3 from `pipeline.contributions`. Identity with decision_function at 1e-9.

### 6.5 Harness — imports policy, monitoring, decision, preprocessing
- `record.py`: `ActionKind`, `Action`, `DecisionRecord(config, share, status, decisions, scores, offset,
  verdicts, actions, review, explanations, region_rates, input_hashes)`;
  `write(out_dir)` writes JSON + CSVs and writes `predictions.csv` only when published (I5).
- `controller.py`: `decide(history, batch, input_hashes, config=DECLARED_CONFIG, workers=4) -> DecisionRecord`.
  Pure: no file I/O except through `record.write`, no printing, logs via `get_logger("harness")`.

### 6.6 Entry points (composition roots)
- `src/main.py`: `check-data`, `monitor`, `decide --history --batch --out-dir`, `pareto --splits --workers`,
  `tune --splits --workers` (writes `resultats_tuner.csv`).
  Exit codes: 0 published / OK, 3 blocked or ALERT (existing `EXIT_ALERT`), 1 error, 2 bad arguments, 130 interrupted.
- `model_corrige.py`: declares the Pareto candidates, runs evaluation (10 splits), then `harness.decide` and
  `record.write`. Same artifacts as the CLI.

## 7. Dependency rules
```
main.py, model_corrige.py -> harness -> {monitoring, decision, preprocessing} -> policy
main.py, model_corrige.py -> evaluation -> policy
common.logging: importable anywhere
```
Forbidden: importing `main` or `model_corrige`; `policy` importing project modules; `monitoring`,
`evaluation`, `decision` importing each other.

## 8. Resources and determinism
At most 4 threads; validation keeps its 2 processes. Splits seeded 0..K-1, RF `random_state=42`,
stable argsort everywhere.

## 9. Governance (hackathon scope)
Artifacts only, no reviewer workflow: `explanations.csv` gives principal factors per applicant
(Law 25 art. 12.1, verify numbering before citing); `review_queue.csv` lists the applicants a human
should look at. Human review itself is described in the pitch, not built.

## 10. Migration (each step keeps predictions.csv valid: 4,000 rows, 36-44 %, candidate order)
1. `src/policy` from `model_corrige.py`; offset leaves `FairPipeline` (default 0).
2. `src/evaluation` absorbs `src/experiments/harness.py` and `compare_methods`/`stability`/`plot_pareto`.
3. Monitoring on `policy.reference_labels`, `Check.correctable`, `Verdict`.
4. `src/decision.py` from the bake-off (Codex cutoff logic, Opus structure), adapted to 6.4.
5. `src/harness`, CLI `decide`/`pareto`, thin `model_corrige.py`.

## 11. Acceptance checks
- 1,598 grants for 4,000 candidates (share 39.94 %); default offset 0; no proxy column read (I2).
- Forced correctable ALERT (remote jury scores shifted by -0.10) -> `ADJUST_OFFSET` recorded, moved ids escalated.
- Drift ALERT (R scores +3) -> `BLOCK`, exit 3, existing `predictions.csv` unchanged (I5).
- Replay gives identical `predictions.csv` and record minus timestamps (I4).
- Import graph obeys section 7.

## 12. Open risks
- The declared 50/50 jury rests on the red-team simulation, not on the hidden reference.
- Percentile votes depend on the batch cohort.
- Batch-level corrections inherit the proxy references' assumptions.

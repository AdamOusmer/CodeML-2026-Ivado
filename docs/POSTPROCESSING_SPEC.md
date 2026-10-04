# Post-processing specification

Status: ready for implementation. Scope: everything after the main model scores a batch, up to publication:
proposal at budget, jury validation, audit, bounded correction, output guard, publication.
Conforms to `HARNESS_SPEC.md` (invariants I1–I8, actions, exit codes), `JURY_SPEC.md` (validator jury, approved:
merit + programme merit, both must agree) and the node architecture of `PREPROCESSING_SPEC.md` §8.

## 1. Stages (normative order)

| # | Stage | Input → output | Rule |
|---|---|---|---|
| 1 | Proposal | main score, k → proposed decisions | `allocate(score, share)`, k = round(share x n); ties by main score then batch order (stable) |
| 2 | Jury | proposal, main probability, juror scores → `JuryOutcome` | `JURY_SPEC.md` §3–4; swaps keep k |
| 3 | Audit | history, batch, decisions → `Verdict` | `src.monitoring.run_checks` |
| 4 | Correction | correctable ALERT → offset, re-run 1–3 | at most once; offset from `fit_offset` (max-gap objective, grid [−0.10, 0.10]); offset shifts the main score of remote applicants, then the jury validates the new proposal |
| 5 | Output guard | ids, decisions, batch → issues | §3; any issue → `BLOCK` (non-correctable) |
| 6 | Publication | record → files | `predictions.csv` only when published (I5) |

Stage order is fixed: the jury always validates the final proposal (after any offset), and the output guard
always runs last, after correction.

## 2. Bound enforcement (closes HARNESS_SPEC I3 divergence)

- `OFFSET_BOUND = 0.10` in `src/policy/models.py`; `FairPipeline.score`, `predict` and `decide` raise `ValueError`
  when the offset is not finite or `abs(offset) > OFFSET_BOUND`. `OFFSET_GRID` in the harness derives from it.

## 3. Output guard — `src/harness/output_guard.py`

`output_issues(batch, ids, decisions, k) -> list[str]`, empty when valid:
- `ids` equal `batch["id_candidat"]` in order; unique; length n.
- decisions are 1-D integer (not bool) in {0, 1}; sum == k; rate within `BUDGET_BOUNDS`.
- missing values (ids or decisions) are reported as issues, never raised.
Non-empty → `BLOCK` with params `{"checks": ["output"], "issues": [...]}`, exit 3, nothing published.

## 4. Node graph — `src/harness/postprocessing.py`

Declared with `src.common.graph` (harness may import common). Seeds: `pipeline`, `history`, `batch`, `share`,
`eo_gap_alert`.

| Node | Inputs | Output |
|---|---|---|
| `outcome` | pipeline, batch | `pipeline.decide(batch)`: `JuryOutcome` at offset 0 |
| `verdict` | history, batch, outcome, eo_gap_alert | first `Verdict` |
| `offset` | pipeline, history, batch, share, verdict | 0.0 unless verdict is correctable ALERT, then `fit_offset` |
| `final_outcome` | pipeline, batch, offset, outcome | `outcome` when offset is 0, else `pipeline.decide(batch, offset)` |
| `final_verdict` | history, batch, final_outcome, offset, verdict, eo_gap_alert | `verdict` when offset is 0, else re-audit |
| `output` | batch, final_outcome, share; after `final_verdict` | issues list |
| `status` | final_verdict, output | `published` or `blocked` |

Nodes return their input unchanged when their condition is false (pass-through), so the graph is static.
One decision function (`pipeline.decide`) serves both paths, so `fit_offset` optimises the published path.
`POSTPROCESSING_NODES`, `postprocessing_graph(nodes)`; `controller.decide` runs it and builds the record.

## 5. Record and explanations

- `DecisionRecord` adds: `proposed` (array), `jury` (triggered, per-trigger counts, overturned in/out),
  `output_issues`, `offset_moved_ids` (moves caused by the offset) and `jury_moved_ids` (swaps), kept separate.
- `explanations.csv` adds `proposed_decision`, `final_rank`, and the jury columns of `JURY_SPEC.md` §5.4.
- Actions unchanged (`SELECT_CONFIG`, `ADJUST_OFFSET`, `BLOCK`); jury swaps are part of the decision, not an action.

## 6. Work packages

| WP | Files owned | Model | Depends | Done when |
|---|---|---|---|---|
| Q1 Offset bound | `src/policy/models.py` (bound only), `src/harness/controller.py` (grid from bound) | Codex | J2 | offset 0.11 raises; grid unchanged |
| Q2 Output guard | `src/harness/output_guard.py` | Codex | — | each §3 rule rejects a crafted bad output |
| Q3 Post-processing graph | `src/harness/postprocessing.py`, `src/harness/controller.py` | Opus | J2, Q1, Q2 | default run published, 1,598 grants, jury counts recorded |
| Q4 Record + explanations fields | `src/harness/record.py`, `src/explain.py`, `src/adapters/files.py` if the writer needs it | Sonnet | Q3, J4 | §5 fields present |
| Q5 Acceptance | `scripts/acceptance.py` | Sonnet | Q4 | §7 passes |

## 7. Acceptance checks
- Default batch: published, 1,598 grants, offset 0, jury swaps recorded, output issues empty.
- Forced correctable ALERT: offset applied, jury re-run on the shifted proposal, `offset_moved_ids` and
  `jury_moved_ids` both recorded, grant count still 1,598.
- Value 2 and k + 1 grants injected end-to-end (replaced `final_outcome` node): each → BLOCK, one BLOCK action, no
  `predictions.csv`. Duplicate id and wrong order are proven on `output_issues` directly (in the harness ids come
  from the batch; duplicate batch ids are rejected earlier by `check_ids` / frame validation, exit 1). Exit 3 for a
  output BLOCK through the CLI is not exercised (CLI cannot inject).
- Offset beyond the bound raises; the harness never requests one.
- Replay identical (I4); shuffled region/postal/distance with offset 0 gives identical decisions (I2).

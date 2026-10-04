# Handoff for the implementing orchestrator

Revision: `77ac0de` on `main` (local). GitHub remote `AdamOusmer/CodeML-2026-Ivado` (private) holds only `5a126b2`;
pushing requires the user's explicit approval. Working tree clean at handoff, except untracked `data/` (git-ignored, required).

Read first: [`HARNESS_SPEC.md`](HARNESS_SPEC.md) (design contract) and [`FINDINGS.md`](FINDINGS.md) (verified numbers).
The original brief is `README.md` (top part) and `consignes-fr.pdf`.

## 1. Objective

24-hour hackathon (CodeML 2026 / IVADO, challenge "ÉquiAlgo"). Grant decisions for 4,000 applicants that close the
regional equal-opportunity gap at a fixed budget, plus audit and governance deliverables.

Scoring: 35 automated points vs a hidden reference (equity 20 = share of baseline EO gap 0.270 closed; utility 15),
zero if grant rate leaves 36–44 %. Jury: diagnostics 25, governance/ethics 25, pitch + code quality 15.

Required deliverables (repo root): `predictions.csv` ✅, `model_corrige.py` with Pareto plot ✅,
`audit_rapport.ipynb` ❌, `presentation.pdf` ❌. A production monitoring plan is also expected (jury).

## 2. Current state (verified at 77ac0de)

| Area | Path | State | Evidence |
|---|---|---|---|
| Domain policy | `src/policy/{regions,core,models}.py` | done | pipeline output identical before/after extraction except 6 decisions (offset removed by design) |
| Evaluation + tuner + candidates | `src/evaluation/{core,pareto,tuner,candidates}.py` | done | `model_corrige.py` 10-split Pareto run, 35 s |
| Monitoring checks | `src/monitoring/checks.py` | done | `monitor` output unchanged; forced ALERT correctable=True; drift ALERT correctable=False |
| Explanations | `src/explain.py` | done (Codex) | logit identity max error 1.4e-14 |
| Harness | `src/harness/{record,controller}.py` | done, partially verified | default path publishes 1,598/4,000; correction and block paths NOT yet exercised |
| CLI | `src/main.py`: `check-data`, `monitor`, `decide`, `pareto`, `tune` | done | `decide` exit 0 + 3 files; `tune --splits 2` exit 0; `pareto` runs via `model_corrige.py` path |
| Deliverable | `model_corrige.py` (thin) | done | 1,598 grants, 39.95 %, ids in candidate order, exit 0 |
| Data validation | `src/preprocessing/validation.py` | done | parallel processes, 0.16 s |
| Logging | `src/common/logging/runtime.py` | done | |

Run (Python 3.12 via uv; `requirements.txt` kept for judges using pip):
```bash
uv sync
uv run python -m src.main check-data
OMP_NUM_THREADS=2 uv run python model_corrige.py
uv run python -m src.main decide --out-dir <dir>
uv run python -m src.main tune --splits 10
uv run python -m src.main monitor
```

## 3. Decisions already made by the user (do not relitigate)

1. Pipeline: preprocess → remove committee's regional penalty from labels → region-blind logistic regression →
   jury (rank vote: main model + merit) → allocate at budget → harness audit/correct/block. Jury comes last.
2. Budget = historical grant rate (39.94 %), read from data, bounded 36–44 %; never hardcoded, never "natural" threshold
   (natural threshold after full correction gives 47.1 % → zero score).
3. Fully automated system: no human gate, no review queue. Knobs (removal, jury weights) are for the team via `tune`.
4. Tuner is an offline team tool; it never sets the live config. `DECLARED_CONFIG` is set by the team in `src/policy/models.py`.
5. Region at decision time only through the Corrector's bounded `ADJUST_OFFSET` (|offset| ≤ 0.10), recorded.
6. No automatic fallback policy; ALERT after one correction → BLOCK (exit 3, `predictions.csv` untouched).
7. Spec layout (`src/policy`, `src/evaluation`, `src/monitoring`, `src/explain.py`, `src/harness`); user's earlier
   `src/pipeline/*` sketch was removed with approval.
8. No automated tests (user choice); verification by scripts/acceptance runs. Hackathon scope: minimal human-validation features.
9. No bigger models / GPU / Neural Engine: capacity test showed all models within 87.5–88.8 % (noise ±0.6).
10. `data/` is never committed.

## 4. Remaining work packages

| WP | Task | Suggested model | Depends | Done when |
|---|---|---|---|---|
| 7 | Cross-model review of `5a126b2..HEAD`: `codex review`; Fable red-team on harness invariants | Codex + Fable | — | findings triaged, P0/P1 fixed |
| 8 | Acceptance checks, spec §11: forced correctable ALERT (`decide(..., eo_gap_alert=0.015)` or remote score shift) → `ADJUST_OFFSET` recorded; drift (batch R +3) → BLOCK, exit 3, existing `predictions.csv` unchanged (I5); replay determinism (I4); import-graph rule (§7) | Opus | 7 | all pass, evidence in PR/commit message |
| 9 | Choose `DECLARED_CONFIG`: run `tune --splits 10`, present table, user picks (0.25 vs 0.5 merit weight within noise so far) | Opus (present), user decides | — | user choice recorded in spec |
| 10 | `audit_rapport.ipynb` (10-section outline in FINDINGS §6), sklearn only, no statsmodels | Sonnet | — | runs top to bottom on `data/` |
| 11 | README: replace top with our solution, commands, artifacts; keep organizer brief below | Sonnet | 8 | commands verified |
| 12 | Monitoring plan section (governance) in README or `docs/` — thresholds already in `src/monitoring/checks.py` | Fable draft, Sonnet edit | 8 | |
| 13 | `presentation.pdf`, 5-minute pitch | Fable narrative | 10 | |
| 14 | Cleanup: delete stale `resultats_stabilite.csv` (no longer produced); decide whether to drop `monitor --reviewed` (human-review feature, out of hackathon scope); update spec for `src/evaluation/candidates.py` | Sonnet | — | |
| 15 | Push to GitHub | user approval required | all | |

## 5. Working rules (user preferences, enforced so far)

- Code style: no comments or docstrings (module docstring of `model_corrige.py` kept for the jury), clear names,
  small direct functions, no unused code, no speculative abstractions.
- One writer per file; each delegate gets explicit file ownership; commit after each WP.
- Machine: M1 Pro, 16 GB RAM near full. One Python process per agent, `OMP_NUM_THREADS=2`, ≤ 4 threads.
- `codex exec` in a background shell needs `< /dev/null` or it hangs on stdin.
- Ask before adding dependencies, pushing, creating PRs. `requirements.txt` and `pyproject.toml` must stay in sync.
- After any change touching decisions: `predictions.csv` 4,000 rows, 36–44 %, ids in candidate order.

## 6. Known gaps and risks

- Harness correction/block paths unexercised (WP8).
- Declared config: EO gap vs corrected reference 0.030 ± 0.011 (10 splits) sits at the WARN threshold 0.03.
- All fairness references are proxies; the hidden reference is unknown (see FINDINGS §4).
- Monitor WARN on real batch: impact ratio 0.795 (Cote-Nord 33.8 % vs Capitale-Nationale 42.5 %).
- Percentile votes depend on the batch cohort.
- Law 25 article number (12.1) not verified; verify before citing.

## 7. Next step

Run WP7 (`codex review` on `5a126b2..HEAD`, read-only) and WP9 (`tune --splits 10`) in parallel, then WP8.

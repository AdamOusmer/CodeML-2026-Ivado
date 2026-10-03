# Handoff for the implementing orchestrator

Revision verified: `4d657c1` on `main`. Working tree clean (`git status --short` empty); `data/` and `logs/` git-ignored.
Remote `origin/main` = `5a126b2` only; local is ahead 7 (`git branch -vv`, no fetch done). Push needs user approval.
Remote repo `AdamOusmer/CodeML-2026-Ivado` (private) — from prior session, unverified here.

Read first: [`HARNESS_SPEC.md`](HARNESS_SPEC.md) (design contract), [`FINDINGS.md`](FINDINGS.md) (verified numbers; not repeated here).
Original brief: `README.md` (top), `consignes-fr.pdf`.

## 0. Objective

24h hackathon (CodeML 2026 / IVADO, "ÉquiAlgo"). Grant decisions for 4,000 applicants, close regional equal-opportunity gap at fixed budget, plus audit + governance.
Scoring: 35 auto points vs hidden reference (equity 20 = share of baseline EO gap 0.270 closed; utility 15); zero if grant rate outside 36–44 %.
Jury: diagnostics 25, governance/ethics 25, pitch + code quality 15.
Deliverables (repo root): `predictions.csv` done, `model_corrige.py` + Pareto plot done, `audit_rapport.ipynb` missing, `presentation.pdf` missing. Monitoring plan expected by jury.

## 1. Current state

Tags: [R] = reproduced this session at `4d657c1` with command shown. [U] = unverified (prior session).

| Area | Path | State | Evidence |
|---|---|---|---|
| Domain policy | `src/policy/{regions,core,models}.py` | done | `DECLARED_CONFIG = Config("jury 50/50")` (`models.py:19`) [R, grep]. Output identical pre/post extraction except 6 decisions (offset removed by design) [U] |
| Evaluation core/pareto/tuner/candidates | `src/evaluation/{core,pareto,tuner,candidates}.py` | done | `candidates.py:89` registers declared pipeline [R, grep]. `--help` lists `pareto`, `tune` [R]. `pareto --splits 10` 31 s, `tune --splits 10` exit 0 [R, FINDINGS §5] |
| Monitoring | `src/monitoring/checks.py` | done | `monitor --plain --no-log-file` exit 0, overall WARN: only impact ratio 0.795 (<0.8) warns; budget 0.400 (1,598/4,000), EO vs corrected 0.022, drift OK [R]. Forced ALERT correctable=True / drift ALERT correctable=False [U] |
| Explanations | `src/explain.py` | done (Codex) | `decide` wrote `explanations.csv` [R]. Logit identity max err 1.4e-14 [U] |
| Harness | `src/harness/{record,controller}.py` | done, partial | `decide` exit 0: "published; 1598 of 4000 granted; wrote decision_record.json, explanations.csv, predictions.csv"; log "SELECT_CONFIG: declared configuration jury 50/50" [R]. Correction and block paths NOT exercised (WP8) |
| CLI | `src/main.py` | done | `--help` exit 0, commands `check-data monitor decide pareto tune`; flags `-v -q --plain --no-progress --log-dir --no-log-file` [R]. `check-data` exit 0 [R]. `monitor --reviewed CSV` still present (`main.py:72`) |
| Deliverable | `model_corrige.py` (thin; imports `pareto_report`, `decide`) | done | Not re-run this session (writes repo-root files). Prior: 1,598 grants, 39.95 %, exit 0 [U]. Equivalent `decide` output checked below [R] |
| Data validation | `src/preprocessing/validation.py` | done | `check-data` exit 0, regional counts table printed [R]. Parallel, 0.16 s [U] |
| Logging | `src/common/logging/runtime.py` | done | each CLI run writes timestamped file in `logs/` unless `--no-log-file` [R] |

`decide` output check [R] (`decide --out-dir /private/tmp/equialgo_handoff_check`, exit 0): `predictions.csv` columns `id_candidat,decision_octroi`; 4,000 rows; grant rate 0.3995 (in 36–44 %); id order identical to `data/candidats_evaluation.csv`.

## 2. How to run

Python 3.12 via uv. `requirements.txt` kept for judges using pip.
```bash
uv sync
uv run python -m src.main --help
uv run python -m src.main check-data
OMP_NUM_THREADS=2 uv run python -m src.main monitor --plain --no-log-file
OMP_NUM_THREADS=2 uv run python -m src.main decide --out-dir <dir>      # safe: not repo root
OMP_NUM_THREADS=2 uv run python -m src.main tune --splits 10            # slow, offline
OMP_NUM_THREADS=2 uv run python model_corrige.py                        # overwrites root outputs
```
`decide`/`model_corrige.py` write `predictions.csv`, `decision_record.json`, `explanations.csv`; `pareto`/`model_corrige.py` also `resultats_*.csv`, `pareto_front.png`. For checks always use `--out-dir` outside repo.
Exit codes: 3 = BLOCK (spec). `decide` publish = 0.

## 3. Decisions by the user (do not relitigate)

1. Pipeline: preprocess → remove committee's regional penalty from labels → region-blind logistic regression → jury (rank vote: main model + merit) → allocate at budget → harness audit/correct/block. Jury last.
2. Budget = historical grant rate (39.94 %), read from data, bounded 36–44 %; never hardcoded, never "natural" threshold (natural threshold after full correction gives 47.1 % → zero score).
3. Fully automated: no human gate, no review queue. Knobs (removal, jury weights) are for the team via `tune`.
4. Tuner = offline team tool; never sets live config. `DECLARED_CONFIG` set by team in `src/policy/models.py`.
5. Region at decision time only via Corrector's bounded `ADJUST_OFFSET` (|offset| ≤ 0.10), recorded.
6. No automatic fallback policy; ALERT after one correction → BLOCK (exit 3, `predictions.csv` untouched).
7. Spec layout (`src/policy`, `src/evaluation`, `src/monitoring`, `src/explain.py`, `src/harness`); earlier `src/pipeline/*` sketch removed with approval.
8. No automated tests (user choice); verify by scripts/acceptance runs. Minimal human-validation features (hackathon scope).
9. No bigger models / GPU / Neural Engine: capacity test, all models 87.5–88.8 % (noise ±0.6). Re-run: LR 87.96, RF 88.42, R-only 85.71; conclusion holds, no repo script reproduces the original figures (FINDINGS §3).
10. `data/` never committed.

## 4. Remaining work packages

| WP | Task | Model | Depends | Done when |
|---|---|---|---|---|
| 7 | Cross-model review of `5a126b2..HEAD`: `codex review`; Fable red-team on harness invariants | Codex + Fable | — | findings triaged; P0/P1 fixed; list of triage decisions in commit message |
| 8 | Acceptance, spec §11: forced correctable ALERT (`decide(..., eo_gap_alert=0.015)` or remote score shift) → `ADJUST_OFFSET` recorded; drift (batch R +3) → BLOCK, exit 3, existing `predictions.csv` byte-identical (I5); replay determinism (I4); import-graph rule (§7) | Opus | 7 | all four pass; evidence (commands + exit codes + hashes) in commit message |
| 9 | Choose `DECLARED_CONFIG`: `tune --splits 10`, present table, user picks (10 splits: worst case merit 0.25 = 0.929 vs current 50/50 = 0.904, FINDINGS §5) | Opus presents, user decides | — | user choice written in `HARNESS_SPEC.md` and `models.py` matches |
| 10 | `audit_rapport.ipynb` (10-section outline in FINDINGS §6), sklearn only, no statsmodels | Sonnet | — | `jupyter nbconvert --execute` runs top to bottom on `data/`, exit 0, no errors in cells |
| 11 | README: replace top with our solution, commands, artifacts; keep organizer brief below | Sonnet | 8 | every command in README run once, exit 0 |
| 12 | Monitoring plan (governance) in README or `docs/MONITORING_PLAN.md`: per check in `checks.py` — metric, threshold, owner, action on WARN/ALERT, cadence, how BLOCK is lifted, Law 25 reference only if verified | Fable draft, Sonnet edit | 8 | every threshold matches `checks.py` (grep-checked); each check has owner + action; one page max; user reads it |
| 13 | `presentation.pdf`, 5-minute pitch | Fable narrative | 10 | PDF renders; ≤ 5 min script; numbers match FINDINGS |
| 14 | Cleanup: delete stale `resultats_stabilite.csv` (still in root; no longer produced — [R] `ls`); decide whether to drop `monitor --reviewed`; update spec for `src/evaluation/candidates.py` | Sonnet | — | stale file gone; decision on `--reviewed` recorded; spec lists candidates module |
| 15 | Push to GitHub | user approval required | all | user approves target; remote head = local head |

## 5. Working rules

- Code style: no comments/docstrings (module docstring of `model_corrige.py` kept for jury), clear names, small direct functions, no unused code, no speculative abstractions.
- One writer per file; each delegate gets explicit file ownership; commit after each WP (commit only when user allows).
- Machine: M1 Pro, 16 GB RAM near full. One Python process per agent, `OMP_NUM_THREADS=2`, ≤ 4 threads.
- `codex exec` in background shell needs `< /dev/null` or it hangs on stdin.
- Ask before adding dependencies, pushing, creating PRs. `requirements.txt` and `pyproject.toml` stay in sync.
- After any decision-touching change: `predictions.csv` 4,000 rows, 36–44 %, ids in candidate order.
- Verification runs write outside repo root (`--out-dir /private/tmp/...`).

## 6. Known gaps and risks

- Harness correction/block paths unexercised (WP8).
- Declared config EO gap vs corrected reference 0.030 ± 0.011 (10 splits) at WARN threshold 0.03 [U]; this session single-batch value 0.022 [R].
- All fairness references are proxies; hidden reference unknown (FINDINGS §4).
- Monitor WARN on real batch: impact ratio 0.795, Cote-Nord 33.8 % vs Capitale-Nationale 42.5 % [R].
- Percentile votes depend on batch cohort.
- Law 25 article number (12.1) unverified; verify before citing.
- `model_corrige.py` not re-run at `4d657c1` (writes repo-root files).
- Spec divergences (HARNESS_SPEC `DIVERGENCE:` lines): `fit_offset` can return 0 → identical re-audit → BLOCK; blocked run leaves an earlier `predictions.csv` on disk; offset bound enforced only by `OFFSET_GRID`, not by `FairPipeline`.
- Missing deliverables: `audit_rapport.ipynb`, `presentation.pdf`.
- `logs/` accumulates one file per CLI run (ignored by git).

## 7. Next step

Run WP7 (`codex review 5a126b2..HEAD`, read-only) and WP9 (present FINDINGS §5 table, user picks merit weight) in parallel, then WP8.

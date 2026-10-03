# Handoff for the implementing orchestrator

Branch state: PR AdamOusmer/CodeML-2026-Ivado#1 (`feat/decision-harness`) + PR #2 (`feat/preprocessing-nodes`) + validator jury (`feat/validator-jury`, rebased on the integration replay of the jury commits); `origin/main` = `4d657c1`. `data/` and `logs/` git-ignored.

Read first: [`HARNESS_SPEC.md`](HARNESS_SPEC.md) (design contract), [`FINDINGS.md`](FINDINGS.md) (verified numbers; not repeated here).
Original brief: `README.md` (top), `consignes-fr.pdf`.

## 0. Objective

24h hackathon (CodeML 2026 / IVADO, "ÉquiAlgo"). Grant decisions for 4,000 applicants, close regional equal-opportunity gap at fixed budget, plus audit + governance.
Scoring: 35 auto points vs hidden reference (equity 20 = share of baseline EO gap 0.270 closed; utility 15); zero if grant rate outside 36–44 %.
Jury: diagnostics 25, governance/ethics 25, pitch + code quality 15.
Deliverables (repo root): `predictions.csv` done, `model_corrige.py` + Pareto plot done, `audit_rapport.ipynb` missing, `presentation.pdf` missing. Monitoring plan expected by jury.

## 1. Current state

Tags: [R] = reproduced on `feat/validator-jury` after the rebase with command shown. [U] = unverified (prior session).

| Area | Path | State | Evidence |
|---|---|---|---|
| Domain policy | `src/policy/{regions,schema,core,jury,models}.py` | done | `DECLARED_CONFIG = Config("validator jury, merit", jury=JurySettings(jurors=("merit",)))` (`models.py:21`) [R, grep]. Validator jury: triggers near_cutoff + disagreement; strength over dissenting jurors (JURY_SPEC §9) |
| Evaluation core/pareto/tuner/candidates | `src/evaluation/{core,pareto,tuner,candidates}.py` | done | declared pipeline registered in `candidates.py`. `model_corrige.py` (10 splits) and `tune --splits 10 --workers 2` exit 0 [R, FINDINGS §5] |
| Monitoring | `src/monitoring/checks.py` | done | `monitor --plain --no-log-file` exit 0, overall OK: budget 0.400 (1,598/4,000), impact ratio 0.819, EO vs corrected 0.018, EO vs merit 0.008, drift OK [R]. Forced ALERT correctable / drift ALERT non-correctable: acceptance [R] |
| Explanations | `src/explain.py` | done (Codex) | `decide` wrote `explanations.csv` [R]. Logit identity max err 1.4e-14 [U] |
| Harness | `src/harness/{record,controller}.py` | done, partial | `decide` exit 0, published, 1,598 of 4,000 granted, jury triggered 201, 37 swaps [R]. Correction and block paths exercised by acceptance (`forced_correctable_alert`, `drift_block`, `alert_after_correction`) [R]. Corrector reach on a real drifted batch untested |
| CLI | `src/main.py` | done | `--help` exit 0, commands `check-data monitor decide pareto tune`; flags `-v -q --plain --no-progress --log-dir --no-log-file` [R]. `check-data` exit 0 [R]. `monitor --reviewed CSV` still present (`main.py:72`) |
| Deliverable | `model_corrige.py` (thin; `run_full`) | done | Re-run at this revision, exit 0: 1,598 grants, 39.95 % [R]; root outputs regenerated |
| Data validation | `src/preprocessing/validation.py` | done | `check-data` exit 0, regional counts table printed [R]. Parallel, 0.16 s [U] |
| Logging | `src/common/logging/runtime.py` | done | each CLI run writes timestamped file in `logs/` unless `--no-log-file` [R] |

`predictions.csv` check [R] (`OMP_NUM_THREADS=2 uv run python model_corrige.py`): columns `id_candidat,decision_octroi`; 4,000 rows; grant rate 0.3995 (in 36–44 %); id order identical to `data/candidats_evaluation.csv`.
Acceptance [R] `OMP_NUM_THREADS=2 uv run python scripts/acceptance.py`: 30/30 passed in 26 s.

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

1. Pipeline: preprocess → remove committee's regional penalty from labels → region-blind logistic regression → validator jury (main model proposes, merit juror validates near the cutoff or on strong disagreement; paired swaps) → allocate at budget → harness audit/correct/block. Jury last.
2. Budget = historical grant rate (39.94 %), read from data, bounded 36–44 %; never hardcoded, never "natural" threshold (natural threshold after full correction gives 47.1 % → zero score).
3. Fully automated: no human gate, no review queue. Knobs (removal, jury band/quorum/jurors) are for the team via `tune`.
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
| 7 | ✅ Done: Codex gpt-6.1-sol + Fable review of `5a126b2..HEAD`; all P1/P2 fixed (fit_offset max-gap objective, NaN metrics → blocking ALERT, binary PSI, qcut edges, id validation, clean ValueError exit) | Codex + Fable | — | done |
| 8 | ✅ Done: `OMP_NUM_THREADS=2 uv run python scripts/acceptance.py` → ACCEPTANCE_COUNT PASS (baseline, budget guard, duplicate id, correction, drift BLOCK + I5 sentinel, I4 replay, I2, I8, explain identity, boundaries with injected-edge proof, jury checks) | Sonnet | 7 | done |
| 9 | ✅ Done: user picked merit-only jury; `DECLARED_CONFIG` in `models.py`, HARNESS_SPEC §8, JURY_SPEC §9 (evidence), acceptance rerun | Opus presents, user decides | — | done |
| 10 | `audit_rapport.ipynb` (10-section outline in FINDINGS §6), sklearn only, no statsmodels | Sonnet | — | `jupyter nbconvert --execute` runs top to bottom on `data/`, exit 0, no errors in cells |
| 11 | README: replace top with our solution, commands, artifacts; keep organizer brief below | Sonnet | 8 | every command in README run once, exit 0 |
| 12 | Monitoring plan (governance) in README or `docs/MONITORING_PLAN.md`: per check in `checks.py` — metric, threshold, owner, action on WARN/ALERT, cadence, how BLOCK is lifted, Law 25 reference only if verified | Fable draft, Sonnet edit | 8 | every threshold matches `checks.py` (grep-checked); each check has owner + action; one page max; user reads it |
| 13 | `presentation.pdf`, 5-minute pitch | Fable narrative | 10 | PDF renders; ≤ 5 min script; numbers match FINDINGS |
| 14 | Cleanup: stale `resultats_stabilite.csv` gone ([R] `ls`), spec lists candidates module; still open: decide whether to drop `monitor --reviewed` (`main.py:72`) | Sonnet | — | decision on `--reviewed` recorded |
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

- Corrector reach at the 0.05 production threshold is untested on a real drifted batch (forced drift study, remote R -1.5, did not reach ALERT; `eo_gap_alert=0.015` forced test blocked earlier).
- Merit juror is also the merit reference: merit-reference scores partly self-agreement (JURY_SPEC §8–10). Declared merit-only chosen by user on stress-reference evidence.
- All fairness references are proxies; hidden reference unknown (FINDINGS §4).
- Percentile votes depend on batch cohort.
- Law 25 article number (12.1) unverified; verify before citing.
- Spec divergences (HARNESS_SPEC `DIVERGENCE:` lines): `fit_offset` can return 0 → direct BLOCK, no `ADJUST_OFFSET`; blocked run leaves an earlier `predictions.csv` on disk; offset bound enforced only by `OFFSET_GRID`, not by `FairPipeline`.
- Missing deliverables: `audit_rapport.ipynb`, `presentation.pdf`.
- `logs/` accumulates one file per CLI run (ignored by git).

## 7. Next step

Commit the WP9 change and regenerated outputs when the user allows. Then WPs 10–13 (audit notebook, README, monitoring plan, presentation), decide `monitor --reviewed` (WP14), push only with approval (WP15).

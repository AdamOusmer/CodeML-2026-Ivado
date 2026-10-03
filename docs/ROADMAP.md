# Roadmap (execution order, approved by the user)

Specs: `HARNESS_SPEC.md`, `PREPROCESSING_SPEC.md` (data, P1–P6), `LABEL_CORRECTION_SPEC.md` (fairness pre, L1–L3),
`JURY_SPEC.md` (validator jury, J1–J6), `POSTPROCESSING_SPEC.md` (Q1–Q5). Status of WP7–WP15: `HANDOFF.md`.

Approved decisions: jury = merit + programme merit, both must agree; main model = logistic regression (stacking,
boosting, RF tested and rejected, `FINDINGS.md`); fully automated decision path; knobs set by the team via `tune`.

## Waves (one writer per file; commit per WP; `OMP_NUM_THREADS=2`, one Python process per agent)

| Wave | Parallel work packages | Notes |
|---|---|---|
| 1 | P1 (Opus) · P2 (Codex) · J1 (Codex) · Q2 (Codex) · WP10 audit notebook (Sonnet) | P1 owns `src/policy/models.py` |
| 2 | L1 (Opus) · P3 (Opus, after P1+P2) · P4 (Sonnet) | L1 takes `models.py` after P1 |
| 3 | J2 (Opus) | takes `models.py` after L1 |
| 4 | J4 (Sonnet) · J5 (Sonnet) · Q1 (Codex) | |
| 5 | Q3 (Opus) · J3 (Opus, same files as Q3: one WP) | harness integration |
| 6 | L2 · Q4 (Sonnet) | record and explanations fields |
| 7 | WP9: run `tune --splits 10` on the jury knobs, user sets `DECLARED_CONFIG` | user decision |
| 8 | P5 · L3 · J6 · Q5 acceptance (Sonnet) → P6 refresh numbers and result files | full acceptance green |
| 9 | WP12 monitoring plan (Fable draft) · WP11 README (Sonnet) · WP13 presentation (Fable) | quote final numbers |
| 10 | WP14 cleanup · WP15 push (user approval) | |

`src/policy/models.py` order: P1 → L1 → J2 → Q1 (bound only). `src/harness/controller.py` order: Q1 (grid from bound) → J3/Q3.

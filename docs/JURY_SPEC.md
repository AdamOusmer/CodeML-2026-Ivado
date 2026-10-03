# Jury specification (validator jury)

Status: implemented (HEAD ac936d0 + uncommitted declared-config change in `src/policy/models.py`). Replaces the rank-vote jury (`Config.jury_weights`) in `FairPipeline`.
Conforms to `HARNESS_SPEC.md` (§3–5 boundaries, invariants) and `PREPROCESSING_SPEC.md` (scoring features).
Evidence: §8. Study scripts: session scratchpad `validator_jury_study.py`, `jury_study.py` (not in the repo).

## 1. Concept

The main model decides; jurors validate. Jurors are independent models with a different viewpoint. They are consulted
only for decisions that need validation, and they can overturn the main model only by a heavy majority. The budget is
fixed, so every overturn is a swap that keeps the grant count exact.

```
main model scores everyone -> proposes top-k (k = round(share x n))
  -> trigger: does this decision need validation?
       no  -> stands
       yes -> each juror votes CONFIRM / OVERTURN
  -> verdict: heavy majority overturns
  -> swaps: overturned grants out, overturned refusals in, paired by juror strength, count stays k
```

## 2. Jurors

| Juror | Score | Fitted on | Why |
|---|---|---|---|
| `merit` | `cote_r_equivalent` | nothing | academic merit viewpoint; the main model weighs income and hours, merit does not |
| `programme_merit` | (R − mean R of programme) / std R of programme | history programme statistics, fixed at `fit` | protects strong students in harder-graded programmes |

Juror vote: `GRANT` if the juror's percentile within the batch is above `1 - k/n`, else `REFUSE`.
A juror votes `OVERTURN` when its vote differs from the main model's proposal.

Rejected after testing (do not add): gradient boosting on corrected labels (agrees with the main model, vetoes useful
overturns: swaps 30 → 8, worst case −0.009), random forest (ranks 96 % identical to the main LR).

No juror reads `region_administrative`, `code_postal_3` or `distance_domicile_campus_km` (I2 holds).

## 3. Triggers (a decision is validated if any trigger fires)

| Trigger | Rule | Default |
|---|---|---|
| Near cutoff | abs(decision rank − (k − 0.5)) ≤ `band` × n (decision rank = rank of main probability + offset) | `band = 0.025` (≈ 100 of 4,000 each side) |
| ~~Low confidence~~ | removed (see below) | - |
| Strong disagreement | max over jurors of abs(juror percentile − main percentile) > `disagree` | `disagree = 0.30` |

Low-confidence trigger (probability in 0.5 ± 0.15) removed with `JurySettings.conf`. Evidence (10 splits): disabling it
changes the worst-case score by +0.0005 ± 0.0010. The window 0.35–0.65 lies below the batch cutoff
probability (about 0.68), so it flagged only refusals; cutoff-centered windows were identical to disabled because `near_cutoff` already covers them.

`band` dominates: 0.025 → worst case 0.929, 0.05 → 0.915, 0.10 → 0.872. Do not widen without re-running the study.

## 4. Verdict and swaps

- Overturn when the share of jurors voting `OVERTURN` ≥ `quorum`; `quorum = 1.0` (unanimous; with two jurors, both must agree).
- Strength of an overturn = mean over jurors voting `OVERTURN` of abs(juror percentile − (1 − k/n)); confirming jurors do not count.
- Swaps: `s = min(#overturned grants, #overturned refusals)`; remove the `s` strongest overturned grants, add the
  `s` strongest overturned refusals; unpaired overturns stand as proposed. Grant count stays exactly k (I1).
- Deterministic: stable sorts; ties in strength broken by main rank.

## 5. Components

### 5.1 Policy — `src/policy/jury.py` (pure, numpy only)
- `JurySettings(frozen)`: `band=0.025`, `disagree=0.30`, `quorum=1.0`, `jurors=("merit", "programme_merit")` (class default; declared config overrides to `("merit",)`).
- `JuryOutcome`: `decisions`, `proposed`, `triggered` (bool array), `reasons` (per applicant: subset of
  `near_cutoff`, `disagreement`), `votes` (juror → array), `overturned_out`, `overturned_in` (index arrays).
- `validate(main_probability, juror_scores: dict[str, array], k, settings) -> JuryOutcome`.

### 5.2 Policy — `src/policy/models.py`
- `Config`: replace `jury_weights` with `jury: JurySettings`; `DECLARED_CONFIG = Config("validator jury, merit", jury=JurySettings(jurors=("merit",)))`.
- `FairPipeline.fit` also stores programme statistics from history (`programme_stats_`).
- `FairPipeline.juror_scores(df) -> dict`; `FairPipeline.decide(df, offset=0.0) -> JuryOutcome`;
  `predict(df, offset)` returns `decide(...).decisions`; `score(df, offset)` returns the main model's score
  (plus offset when non-zero), used for ranking and explanations.
- Export `JurySettings`, `JuryOutcome`, `validate` through `src.policy.__all__` (lazy export rule).

### 5.3 Harness
- Decider = `pipeline.decide(batch)`; Corrector `ADJUST_OFFSET` shifts the main score, then re-runs the jury.
- `DecisionRecord` gains `jury`: counts of triggered, overturned out, overturned in, and per-trigger counts.

### 5.4 Explanations — `src/explain.py`
- Add columns `validated` (bool), `trigger_reasons`, `juror_votes` (e.g. `merit:GRANT;programme_merit:GRANT`),
  `jury_outcome` (`confirmed` | `overturned_in` | `overturned_out` | `not_reviewed`).

### 5.5 Evaluation and tuner
- `SEARCH_SPACE` becomes a grid over `band ∈ {0.0125, 0.025, 0.05}`, `quorum ∈ {0.5, 1.0}`,
  `jurors ∈ {(merit,), (merit, programme_merit)}`; tuner stays offline, the team sets `DECLARED_CONFIG`.
- Candidates: replace the merit-weight sweep with a band sweep `{0, 0.0125, 0.025, 0.05, 0.10}` for the Pareto plot.

## 6. Work packages

| WP | Files owned | Model | Depends | Done when |
|---|---|---|---|---|
| J1 `jury.py` | `src/policy/jury.py` | Codex | — | unit-level script: grant count always k; zero triggers → decisions == proposal |
| J2 Pipeline integration | `src/policy/models.py`, `src/policy/__init__.py` | Opus | J1, PREPROCESSING P1 | `model_corrige.py` 1,598 grants |
| J3 Harness + record | `src/harness/*` | Opus | J2 | record shows jury counts; correction re-runs the jury |
| J4 Explanations | `src/explain.py` | Sonnet | J2 | new columns; identity check still 1e-9 |
| J5 Tuner + candidates | `src/evaluation/tuner.py`, `src/evaluation/candidates.py` | Sonnet | J2 | `tune` and Pareto run on the new knobs |
| J6 Acceptance + findings | `scripts/acceptance.py`, `docs/FINDINGS.md`, regenerated result files | Sonnet | J3–J5 | §7 passes; tables refreshed |

J1 can start now. J2 waits for PREPROCESSING P1 (both edit `src/policy/models.py`). One writer per file.

## 7. Acceptance checks
- Grant count exactly k for every setting, including `band = 0` with other triggers on, and zero triggers.
- Invariance: shuffling region, postal code and distance leaves `decide(batch).decisions` unchanged (I2).
- Swap symmetry: `len(overturned_out) == len(overturned_in)`.
- Determinism: replay identical decisions and outcome arrays.
- On the real batch: triggered ≈ 150–250, swaps ≈ 20–40; report the exact numbers in the record (current: triggered 201, 37 swaps).

## 8. Evidence (10 splits; paired differences ± 2 SE)

| Design | Worst case 2 refs | Worst case 3 refs | Stress ref | Reviewed | Swaps |
|---|---|---|---|---|---|
| Main model only | 0.912 | 0.905 | 0.813 | 0 | 0 |
| Rank vote 0.75 / 0.25 (previous) | 0.926 | 0.922 | 0.833 | all | — |
| Validator, merit only, band 2.5 % | 0.929 | 0.925 | 0.837 | 150 | 29.6 |
| Validator, merit + programme, unanimous | 0.925 | 0.921 | 0.833 | 150 | 27.9 |
| Validator, + gradient boosting, heavy majority | 0.920 | 0.917 | 0.827 | 150 | 8.2 |

Validator (merit) vs main only: +0.017 ± 0.011 (worst case 2 refs), +0.024 ± 0.011 (stress). Vs previous rank vote:
+0.003 ± 0.005 (tie). Merit + programme vs merit only: −0.004, within noise; chosen to make the heavy majority
meaningful (two independent merit viewpoints must agree). Stress reference = committee rule with income and hours
weights reversed, used by no juror. Caveat: the merit juror is also a reference, so merit-reference scores are partly
self-agreement.

## 9. Implementation notes (divergences from the design above)
- `validate(main_probability, juror_scores, k, settings, ranking=None)`: `ranking` (default the probability) orders the proposal;
  `near_cutoff` uses the rank of `ranking` (main probability + offset on remote applicants), so review sits at the boundary of the decision actually being made;
  `disagreement` uses the raw main percentile; juror votes read juror scores only. Orchestrator choice: at offset +0.03, raw-rank `near_cutoff` would review 14 different
  applicants and change 8 final decisions (Codex flagged the ambiguity). Earlier red-team: shifting all triggers inverted the correction.
  `FairPipeline.decide(df, offset)` passes `probability + offset * is_remote` as `ranking`.
- `JuryOutcome` has an extra `overturned` mask (all overturns, paired or not). Unpaired overturns keep the proposal and are labelled
  `overturn_unpaired` in `explanations.csv`.
- `overturned_out` ties in strength are broken by worse main rank first; `overturned_in` by better main rank first.
- Non-finite probabilities, ranking or juror scores raise `ValueError`.
- Tuner skips quorum 0.5 with a single juror (identical to quorum 1.0): 9 configs, not 12.
- Acceptance adds `jury_offset_monotone`: grants == k at all 41 offsets; remote grants non-decreasing within 2 applicants, and reach (+0.10 minus −0.10) >= 10; forced correctable alert must record `ADJUST_OFFSET` with 0 < |offset| <= 0.10.
- `near_cutoff` deliberately uses the decision rank (offset included). Tried the raw model rank per Codex review: the merit juror then cancels the offset entirely
  (remote grants 594 at every offset, EO gap constant, forced alert → BLOCK), so the corrector would be dead. Reverted; acceptance now asserts `ADJUST_OFFSET` and offset reach >= 10.

- Swap strength averages |juror percentile − cutoff| over dissenting jurors only (OVERTURN voters), so a confirming juror no longer dilutes it. Quorum stays 1.0.
- Low-confidence trigger removed (`JurySettings.conf` gone, `JURY_REASONS = (near_cutoff, disagreement)`); evidence in §3.
- `ADJUST_OFFSET.params` gains `offset_moved` (proposal changes from the offset alone, before the jury); `moved` stays the final diff.
- Declared config is merit-only (user decision, WP9): `DECLARED_CONFIG = Config("validator jury, merit", jury=JurySettings(jurors=("merit",)))`.
  Evidence, 10 splits, paired vs the previous two-juror unanimous config, ±2 SE: stress reference +0.0057 ± 0.0037; worst case over 2 refs +0.0034 ± 0.0039;
  quorum 0.5 two-juror on stress +0.0047 ± 0.0057; main model only on stress −0.016 ± 0.013. Caveat: the merit juror is also the merit reference.
  Real batch: triggered 201 (near_cutoff 200, disagreement 2), 37 swaps.

## 10. Open questions
Resolved:
- Low-confidence window: trigger removed (§3).
- Strength with `quorum` < 1: fixed, averages over dissenting jurors only (§9).
- Offset scale: kept on probability. Percentile scale reaches −58..+56 remote grants at ±0.10, too coarse for the 0.005 grid; probability scale −8..+11 remote grants,
  monotone, 23 distinct decision sets over 41 points (acceptance `jury_offset_monotone`: 586 / 594 / 606 at −0.10 / 0 / +0.10).

Open:
- Corrector reach at the 0.05 alert threshold is untested on a real drifted batch (forced drift study, remote R −1.5, did not reach ALERT).
- Merit self-agreement: the merit juror is also the merit reference, so merit-reference scores are partly self-agreement.

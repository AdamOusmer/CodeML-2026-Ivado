# Fairness pre-processing specification (label correction)

Status: ready for implementation. Scope: the stage between data preprocessing (`PREPROCESSING_SPEC.md`) and the main
model: estimate the committee's regional penalty and produce corrected training labels.
Conforms to `HARNESS_SPEC.md` (I1, I3, I7; policy imports numpy, pandas, scipy, scikit-learn only) and the node
architecture of `PREPROCESSING_SPEC.md` §8.

## 1. Method

1. Fit the committee model on history: logistic regression on standardized `committee_features` (8 columns) plus a
   remote indicator, target `decision_octroi`.
2. Corrected logit = committee logit with the remote term removed (`removal = 1.0`).
3. Corrected labels = top k by corrected logit, k = round(share x n_history) (3,994 of 10,000).
4. The main model trains on corrected labels only (I7: never on decisions it produced).

Assumptions, each tested and recorded (§3): the committee applies the same rule to both groups apart from one additive
penalty; the penalty is non-zero; the remaining features are the committee's criteria (their legitimacy is a policy
question, recorded as a limitation in `PREPROCESSING_SPEC.md` §1).

## 2. Components — `src/policy/label_correction.py` (pure)

- `LabelCorrection(frozen)`: `labels` (int array), `penalty` (float), `k` (int), `flipped_in` (index array: refused
  historically, positive after correction), `flipped_out` (granted historically, negative after correction).
- `correct_labels(history, share, removal=1.0) -> LabelCorrection`. Uses `CommitteeModel`; deterministic.
- `CorrectionReport(frozen)`: `penalty`, `penalty_ci` (2.5 %, 97.5 %), `slope_test_statistic`, `slope_test_df`,
  `slope_test_p`, `flipped_in`, `flipped_out`, `flipped_in_remote`, `flipped_out_centre`.
- `correction_report(history, share, n_boot=200, seed=0, workers=4) -> CorrectionReport`:
  - Penalty CI: nonparametric bootstrap of the committee fit (resample rows with `numpy.random.default_rng(seed)`),
    fits in a `ThreadPoolExecutor(max_workers=workers)`.
  - Slope-equality test: likelihood-ratio test between the committee model and the same model plus
    remote x feature interactions, both unpenalized (`LogisticRegression(C=np.inf)`; `penalty=None` is deprecated in
    scikit-learn 1.8), statistic = 2 x (log-loss sum difference), df = 8, p from `scipy.stats.chi2.sf`.
- `FairPipeline.fit` calls `correct_labels` and stores `label_correction_`. The report is not on the decision path.
- Export `LabelCorrection`, `correct_labels`, `CorrectionReport`, `correction_report` via `src.policy.__all__` (lazy).

## 3. Guards (recorded, never silently change the policy)

| Condition | Effect |
|---|---|
| `penalty >= 0` (no penalty against remote) | corrected labels equal the committee's own ranking; record warning `no regional penalty found` |
| penalty CI includes 0 | record warning `penalty not distinguishable from zero` |
| `slope_test_p < 0.05` | record warning `group-specific committee rules; single-penalty correction may be incomplete` |
| `len(flipped_in) != len(flipped_out)` | impossible by construction (fixed k); raise `ValueError` |

Warnings go to `DecisionRecord.warnings`; none blocks a decision.

## 4. Composition (node graph)

In `src/pipelines/` (`PREPROCESSING_SPEC.md` §8.5) add node `correction_report` with input `history` and `share`,
after `frame_report`, running concurrently with the decision branch; `record` includes its summary.

## 5. Work packages

| WP | Files owned | Model | Depends | Done when |
|---|---|---|---|---|
| L1 `label_correction.py` + `FairPipeline.fit` wiring | `src/policy/label_correction.py`, `src/policy/models.py`, `src/policy/__init__.py` | Opus | PREPROCESSING P1 | corrected labels identical to the current pipeline's; k = 3,994 |
| L2 Report node + record field | `src/pipelines/*`, `src/harness/record.py` | Sonnet | L1, J3 | record shows penalty, CI, slope test, flips |
| L3 Acceptance | `scripts/acceptance.py` | Sonnet | L2 | §6 passes |

`src/policy/models.py` is also edited by PREPROCESSING P1 and JURY J2: order P1 → L1 → J2, one writer at a time.

## 6. Acceptance checks (current data)

- Penalty −1.93 (unpenalized) / −1.90 (default L2), bootstrap 95 % CI ≈ [−2.07, −1.73]; CI excludes 0.
- Slope-equality test: statistic 4.38, df 8, p ≈ 0.82 (assumption holds).
- k = 3,994; flips 651 in, 651 out; remote flipped in 432; centre flipped out 577.
- Corrected label rates: centre 42.4 %, remote 36.2 % (historical: 48.4 % / 27.3 %).
- Report replay with the same seed is identical; the decision path does not run the bootstrap.

## 7. Out of scope

Massaging, reweighing (tested: no gain), group-specific correction (slope test does not support it), changing which
committee criteria count (policy decision, not preprocessing).

# Stage 6 — freeze and external-evaluation package

Generated: `2026-11-15T20:43:24+00:00`. No fit, no tuning, no main result overwrite.

## 1. Frozen candidates

| Candidate | Case | Roles | Rows | Prediction SHA | Evidence |
|---|---|---|---:|---|---|
| s1a | S1 | level_reference, submission_candidate | 525 | `84f994045a10e480…` | S1 reference; block-CV baseline; lower-tail MAE best among block candidates |
| global_ridge_geometry_correction | S1 | spatial_challenger, submission_candidate | 525 | `150ca4cbd78bd6cf…` | promising on 100/200/500 block schemes; temporal law not identified |
| global_ridge_geometry_time_scaled | S1 | spatial_challenger | 525 | `f959edddafebf552…` | historical correction/time-scaled identical because dt_train is common; not a winner |
| physics_priors | S2 | level_reference, submission_candidate | 1475 | `c2aa262998449d1e…` | strict train 2018 only; target level 2.809539; target accuracy unavailable |
| spatial_no_temperature_centered | S2 | spatial_challenger | 1475 | `5c933e6c6c2d229b…` | centering fixes level shift; strict/reference metrics do not establish target gain |
| spatial_centered_corrected_temperature | S2 | spatial_challenger | 1475 | `abc72732c04d9545…` | strict/post-cutoff metrics weaker than physics; retained as challenger only |

Level reference, spatial challenger and submission candidate are explicit roles, not winner claims.

## 2. Explicit S1/S2 combinations

| Pair | S1 | S2 | Rows | result SHA |
|---|---|---|---:|---|
| s1a__physics_priors | s1a | physics_priors | 2000 | `4fd624c7b605c7bf…` |
| global_ridge_geometry_correction__physics_priors | global_ridge_geometry_correction | physics_priors | 2000 | `66e3909ae8e11729…` |
| global_ridge_geometry_time_scaled__physics_priors | global_ridge_geometry_time_scaled | physics_priors | 2000 | `e6a5d757329ee8ca…` |
| s1a__spatial_no_temperature_centered | s1a | spatial_no_temperature_centered | 2000 | `59b22ddcef954344…` |
| s1a__spatial_centered_corrected_temperature | s1a | spatial_centered_corrected_temperature | 2000 | `2327b9fe37bce72b…` |

Only these predeclared pairs were assembled; no automatic Cartesian product was run.

## 3. Contract checks

- every result: 2000 unique keys, 525 S1 + 1475 S2, finite, within bounds;
- selected S1 and S2 halves preserve source predictions exactly (`max_abs=0`);
- constant/near-constant diagnostics are saved per pair;
- no main `result.csv` was overwritten.

## 4. Artifact-only recheck

- new-process max abs differences: `{'global_ridge_geometry_correction': 0.0, 'global_ridge_geometry_time_scaled': 0.0, 'physics_priors': 0.0, 'spatial_no_temperature_centered': 0.0, 'spatial_centered_corrected_temperature': 0.0}`; no fit, no labels.
- Full/partial inference is covered by Stage 2/3 tests.

## 5. External evaluator

```bash
C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.evaluate_external \
  --predictions experiments/targeted_cycle/frozen_candidates/combinations/<pair>/result.csv \
  --truth <separate_truth.csv> --output-root experiments/targeted_cycle/external_evaluation/<run>
```
Truth columns: `id,segment_id,date,thickness_measured` (mm). Truth is never merged into inference features.
No external labels were available in Stage 6, so no external accuracy is claimed.

## 6. What was fixed / confirmed / rejected

- Fixed: full-grid thermal scenario before label joins; full-grid centered S2 spatial; physical block protocol.
- Confirmed: S1 Ridge shows positive spatial/level signal on 100/200/500 block schemes; S2 physics remains level reference.
- Rejected as improvements: S2 spatial/temperature on strict/retrospective metrics; affine as spatial candidate; S1 time law as identifiable from OOF.

## 7. What cannot be established without new data

- Hidden target accuracy and the official score formula;
- whether S1 time scaling is correct;
- cross-environment S2 improvement on the target;
- campaign comparability, repairs, instrument/threshold metadata;
- two-campaign S1B advantage (single prior campaign in training).

## 8. Tests

- `PIPETWIN_NEXT_DATA_ROOT=... C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m unittest tests.test_stage_6_freeze -v`: 4 tests, OK.
- Full Stage 2–6 run: 30 tests, OK.
- Проверены candidate roles/hashes, explicit pairs, 2000-key contracts, exact halves, artifact-only recheck и external evaluator on synthetic truth.

## 9. Commands and paths

```bash
C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_freeze \
  --root . --data-root C:\Users\Администратор\Documents\PipeTwin_next\experiments\next_cycle\data_generation --output-root experiments/targeted_cycle/frozen_candidates
```

- `experiments/targeted_cycle/frozen_candidates/candidate_freeze.json`
- `experiments/targeted_cycle/frozen_candidates/predictions/`
- `experiments/targeted_cycle/frozen_candidates/combinations/`
- `experiments/targeted_cycle/frozen_candidates/recheck/`
- `experiments/targeted_cycle/stage_6_report_ru.md`
- `experiments/targeted_cycle/report_ru.md`
- `experiments/targeted_cycle/mentor_questions_ru.md`

Stage 6 stops here; no messages were sent.

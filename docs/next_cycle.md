# Next cycle: S1 diagnostics and strict S2 candidates

## Data generation

The old `processed/target_segments_grid.csv` had an extra `delta_t_c` column and
did not match its manifest. It was not used and its checksum was not rewritten.
A fresh consistent generation was produced with the standard pipeline:

```bash
python -m src.data_prep --output-root experiments/next_cycle/data_generation
```

All 25 CSV/Parquet pairs pass the manifest checks, the target grid has 7
columns, and key counts are 525 (S1) / 1475 (S2). S1A/S1B/S1C were reproduced on
this generation with zero numerical difference in `result.csv`; the old S1C
artifact hash is unchanged
(`e998606bc75b97ff9b88360b0d6a11a10dfefa466090abfa7ca3afbe61ae83cf`).

## S1 experiments

Fit data: 357 rows, three simple sections (4 / 32 / 321), one campaign
`2022-07-06`, one parent `MAIN-2`. All preprocessing and models are fit inside
leave-one-simple-section-out folds. The core learner uses **one** last
measurement baseline; it does not receive both `s1b_prediction` and
`last_vtd_thickness_mm`.

Candidates:

- controls: S1A, S1B, old S1C, train median, nonnegative affine calibration;
- `ridge_history`: last + history/neighbour features;
- `ridge_geometry`: history + signed/absolute slope, local low, relative position;
- `rate_signed` / `rate_nonnegative`: target `(last - next) / dt`, prediction
  `last - rate_hat * actual_dt`, with actual production dt;
- `catboost`: fixed depth=2, iterations=200, lr=0.03, l2=10, seed=17, row-uniform;
- mixtures `last + lambda * ridge_geometry_delta`, lambda in {0, 0.25, 0.5, 1};
- weight ablation: row-uniform, inverse-sqrt(section size), inverse(section size);
- block CV: 100 m blocks with 20 m buffer on the target pipe (same-pipe spatial
  interpolation), not a future-campaign validation.

Result: no learned candidate improved both within-section Pearson² and MAE over
S1A. S1A remains the S1 default. The report is
`experiments/next_cycle/s1/report_ru.md`; tables are
`metrics_pooled.json`, `metrics_per_fold.csv`, `weight_ablation.csv`,
`block_cv/metrics.json`, and `selected_candidates.json`.

The old S1C signal loss is explained and reproduced: duplicate baselines,
constant horizon, constant `last-minus-min`/campaign count, and section weights
with Kish ESS ≈31.65. `rate_signed` OOF is numerically identical to the delta
model because training dt is constant; production differs only because actual
dt differs. Block CV shows a positive same-pipe interpolation signal that LOPO
does not confirm across sections.

## S2 experiments

Strict train: cold-start labels strictly before `2022-07-01`, i.e. only
`2018-09-23` (372 rows, 3 simple sections, one parent `MAIN-2`). The 2023-08-12
campaigns are used only in a frozen post-cutoff retrospective; they do not
enter fit, preprocessing, reference conditions, spatial gate, feature selection
or model choice.

Candidates:

- `constant_effective_rate`: nonnegative constant thinning from nominal, fit on train;
- `physics_priors`: existing environment-aware physics with declared priors and
  fitted `r_ref`, `own_mean` gap policy;
- `physics_velocity_consistent`: `v_liquid_true` in both reference/factors and
  exposure intervals (sensitivity; it collapses to the same `r_ref` after fit);
- `physics_spatial`: train-only within-section residual Ridge on profile features;
- `physics_spatial_temperature`: same plus `-ΔT` from the existing steady-state
  `heat_transfer.py` scenario (a proxy, not a measured 29-year profile);
- `catboost_residual`: fixed CatBoost residual challenger;
- `v2_reference`: unchanged old S2 result, production reference only.

Strict OOF results: constant MAE 0.4398, physics 0.4544, spatial 0.4482,
temperature 0.4521, CatBoost 0.4427; within-section signal is tiny (0.004–0.012
r²) and the 2023 retrospective gets worse when spatial/temperature corrections
are added (physics MAE 0.2905 vs spatial 0.3188 / temperature 0.3178). Physics
level is retained as the S2 default; no target labels exist, so target accuracy
is not claimed.

Domain shift is severe: strict nominal 8 mm / age 18.42 / warm-wet-fast;
target 4.5 mm / age 29.03 / cold-dry-slow; strict hydraulic coverage 17.5%,
target 13.3%. See `experiments/next_cycle/s2/domain_shift.json`.

## Artifacts

```text
experiments/next_cycle/
  experiment_registry.json
  audit/data_audit.json, audit/audit_recheck.json
  data_generation/
  repro_s1abc/
  s1/  (training table, folds, OOF, metrics, weight ablation, block CV, candidates, figures, report)
  s2/  (strict train, eligibility, folds, OOF, metrics, domain shift, retrospective, candidates, figures, report)
  assembled/s1a__physics_priors/result.csv
  report_ru.md
```

Each trained candidate has `model_artifact.joblib`, `model_metadata.json`,
`predictions_s1.csv` or `predictions_s2.csv`, diagnostics, and an explicit
`result.csv` with the other half fixed. The assembly is an explicit candidate
pair, not a promotion of a best production result.

## Commands

```bash
python -m src.data_prep --output-root experiments/next_cycle/data_generation

python -m src.ml.next_audit --root . \
  --data-root experiments/next_cycle/data_generation \
  --output-root experiments/next_cycle/audit

python -m src.ml.next_s1 train --root . \
  --data-root experiments/next_cycle/data_generation \
  --output-root experiments/next_cycle/s1
python -m src.ml.next_s1 predict --root . \
  --data-root experiments/next_cycle/data_generation \
  --output-root experiments/next_cycle/s1_predict \
  --candidate ridge_geometry

python -m src.ml.next_s2 train --root . \
  --data-root experiments/next_cycle/data_generation \
  --output-root experiments/next_cycle/s2
python -m src.ml.next_s2 predict --root . \
  --data-root experiments/next_cycle/data_generation \
  --output-root experiments/next_cycle/s2_predict \
  --candidate physics_priors

python -m src.ml.next_compare --root . --output-root experiments/next_cycle \
  --s1-candidate s1a --s2-candidate physics_priors

# sequential
python -m src.ml.next_all --root . \
  --data-root experiments/next_cycle/data_generation \
  --output-root experiments/next_cycle
```

## Honest limitations

- No strict forward-time S1 folds; LOPO and block CV are spatial diagnostics.
- S1B production uses two historical campaigns while S1C training has one.
- S2 strict train has one parent and one campaign; leave-one-parent-out is
  unavailable. Target labels are absent.
- The temperature profile is a declared steady-state scenario, not a measured
  historical temperature field; coefficients/boundary parameters are priors.
- The 2023 retrospective is frozen and excluded from selection.
- No push/deploy and no replacement of `experiments/ml_v2/result.csv`.

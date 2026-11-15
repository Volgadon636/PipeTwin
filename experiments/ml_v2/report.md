# PipeTwin ML v2: environment-aware physics and gated spatial term

Checked branch `model_v2`, commit `0c7f143536a00591ff315b32d1b73312a92d23b5`.

## What changed against v1

1. **Cold start is environment-aware.** v1 fitted one constant rate on warm, wet MAIN-2 pipes and extrapolated it to the cold, dry, slow 1250009583. v2 keeps the multi-parameter rate structure of the technical specification (temperature, watercut, velocity), fits only the reference rate and declares the remaining exponents as explicit priors with a sensitivity grid.
2. **A within-pipe profile deviation is allowed only with out-of-sample evidence.** The profile ridge model is gated by leave-one-campaign-out correlation and shrunk by 0.50.
3. **Between-VTD rates separate comparable and non-comparable inspections.** Pairs whose campaign density differs by more than the declared ratio are not silently pooled with dense campaigns. Post-cutoff comparable pairs are reported as a diagnostic only.
4. **Validation is leave-one-pipe-out / leave-one-campaign-out.** The repository still has no forward-time held-parent folded evidence before either production cutoff; that limitation remains and is not replaced by random segment splits.

## Cold-start leave-one-pipe-out evidence

Each first-inspection campaign is predicted by a model fitted on the other pipes. The metrics below are therefore out-of-pipe, not in-sample. They measure model behaviour on ten campaigns, not accuracy on the hidden 1250009583 inspection.

| Model | Held-out campaigns | MAE (mm) | RMSE (mm) | Bias (mm) | Mean Pearson² |
|---|---:|---:|---:|---:|---:|
| C0 | 10 | 1.251 | 1.299 | +1.251 | undefined (constant) |
| M0 | 10 | 0.328 | 0.390 | -0.047 | undefined (constant) |
| M1 | 10 | 0.349 | 0.460 | +0.243 | undefined (constant) |
| M1S | 10 | 0.356 | 0.461 | +0.247 | 0.154 |

A per-campaign mean is dominated by the three tiny campaigns (n = 1-4). The pooled out-of-pipe view below removes each campaign mean first and is the number to quote:

| Model | Rows | MAE (mm) | RMSE (mm) | Bias (mm) | Pooled within-pipe Pearson² |
|---|---:|---:|---:|---:|---:|
| C0 | 466 | 1.676 | 1.799 | +1.676 | undefined |
| M0 | 466 | 0.433 | 0.538 | +0.047 | undefined |
| M1 | 466 | 0.429 | 0.561 | -0.098 | undefined |
| M1S | 466 | 0.429 | 0.558 | -0.094 | 0.0166 |

Constant predictions (C0/M0/M1) have zero within-pipe variance, so their Pearson correlation is undefined by definition. Only the profile term (M1S) can produce a within-pipe ranking at all. Its pooled value is 0.0166, i.e. small and not hidden below. The cold-start *level* is unchanged by the profile term.

### Cross-environment transfer (the actual production direction)

Leave-one-pipe-out mixes warm/wet and cold/dry pipes in every training fold. Production does not: 1250009583 is cold and dry, while its only pre-cutoff first-inspection labels are warm and wet. The table below trains on one parent and predicts the other, which is the closest available test of that transfer. It is a two-direction check, not a hidden-test estimate.

| Held-out parent | Pipe | Campaign | n | MAE (mm) | Bias (mm) | Pearson² |
|---|---|---:|---:|---:|---:|---:|
| MAIN-2 | 1250009490 | 2018-09-23 | 4 | 0.547 | -0.547 | 0.433 |
| MAIN-2 | 1250011604 | 2018-09-23 | 34 | 0.832 | -0.803 | 0.099 |
| MAIN-2 | 1250022871 | 2018-09-23 | 334 | 0.974 | -0.948 | 0.013 |
| MAIN-3 | 1250009584 | 2023-08-12 | 19 | 0.281 | +0.281 | 0.082 |
| MAIN-3 | 1250009587 | 2023-08-12 | 3 | 0.289 | +0.289 | 0.285 |
| MAIN-3 | 1250028931 | 2023-08-12 | 4 | 0.660 | +0.660 | 0.113 |
| MAIN-3 | 1250028932 | 2023-08-12 | 9 | 0.351 | +0.351 | 0.125 |
| MAIN-3 | 1250028933 | 2023-08-12 | 1 | 0.447 | +0.447 | undefined |
| MAIN-3 | 1250028934 | 2023-08-12 | 31 | 0.282 | +0.281 | 0.051 |
| MAIN-3 | 1250028935 | 2023-08-12 | 27 | 0.263 | +0.263 | 0.001 |

Spatial gate: usable = **True**; pooled leave-one-campaign-out Pearson r = 0.12630352253952526. Reason: positive out-of-sample correlation.

## Production forecasts

- `1250022871` (between VTD, 2023-12-01): rate basis **all_pre_cutoff_pairs_no_comparable_basis**, rate = 0 mm/year, 357 pairs, 4 segments with a borrowed prior baseline.
  Post-cutoff comparable diagnostic: {'pairs': 47, 'pipes': [1250011604], 'rate_mm_year': 0.2278362573099415, 'mean_change_mm': -0.32000000000000006, 'prior_persistence_pearson_squared': 0.006984869858758343, 'dt_years': [1.405]}. 
  Leave-one-pipe-out pair check (each pipe predicted from the others, cutoff-respecting):

| Model | Rows | MAE (mm) | RMSE (mm) | Bias (mm) | Pooled within-pipe Pearson² |
|---|---:|---:|---:|---:|---:|
| C0 | 357 | 0.498 | 0.652 | +0.049 | 0.1133 |
| M0 | 357 | 0.498 | 0.652 | +0.049 | 0.1133 |
| M0dense | 357 | 0.498 | 0.652 | +0.049 | 0.1133 |

- `1250009583` (cold start, 2022-07-01): training rows 372, pipes [1250009490, 1250011604, 1250022871], campaigns ['2018-09-23']; reference rate 0.105552 mm/year, mean exposure 16.015 reference-years, mean environment factor 0.549. Mean predicted thickness 2.810 mm.
  Archive-training sensitivity (adds the later MAIN-3 campaign): reference rate 0.110185 mm/year, mean predicted thickness 2.735 mm.
  Declared scenario band (model uncertainty, not calibrated): {'scenarios': 243, 'thickness_p10_mm': 1.4369980740221282, 'thickness_median_mm': 2.6607198661683147, 'thickness_p90_mm': 3.468742407830294, 'r_ref_p10_mm_year': 0.049510450476126275, 'r_ref_p90_mm_year': 1.1506393447720418, 'note': 'Model-scenario range from declared priors; not a calibrated confidence interval.'}

## Honest limitations

- The hidden inspections of the two target pipes are not in the repository. No hidden-test accuracy is claimed.
- The environment exponents are priors. With two confounded first-inspection environments they cannot be identified from data; the grid shows how much the level moves.
- Segment-level minima persist poorly between inspections. Leave-one-campaign-out spatial correlation is small; the profile term changes ranking, not necessarily absolute accuracy.
- Missing early operating history (29-year cold-start pipe, ~49 % hydraulic coverage) is represented by an explicit `own_mean` gap assumption and a lower/upper scenario, never by invented measurements.
- Repairs, pigging, inhibition, chemistry, cathodic protection and external corrosion remain unavailable or undated; the fitted rate absorbs them as an effective segment minima rate.
- The between-VTD production rate is zero because every pre-cutoff pair is non-comparable (357 pairs, 0 comparable). The one dense cross-pipe pair (2022-07-06 -> 2023-12-01) shows a real drop but lies on the production cutoff: rate 0.2278362573099415 mm/year, prior-thickness persistence Pearson² 0.006984869858758343. It is disclosed, not used. This is the single most important unresolved S1 question for the mentors.

## Artifacts

- `result.csv`: exactly `id,segment_id,thickness_pred` (2,000 rows).
- `prediction_diagnostics.csv`: bounds, intervals, model metadata per segment.
- `candidate_predictions.csv`: C0, M0, M1/M1S and sensitivity variants.
- `folds.csv`: cold-start leave-one-pipe-out metrics per campaign.
- `folds_cross_environment.csv`: two-fold warm/cold parent transfer metrics.
- `scenario_sensitivity_cold_start.csv`: the declared prior grid values.
- `spatial_cv.json`: spatial cross-validation and gate decision.
- `inspection_pairs.csv`: every consecutive VTD pair with density ratio and comparability.
- `pairs_leave_one_pipe_metrics.csv`, `metrics.json`, `models.json`, `run_manifest.json`.

```bash
python -m src.ml.experiment_v2 --output-root experiments/ml_v2
```

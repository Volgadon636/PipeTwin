# Stage 5 — temporal interpretations of the fixed S1 Ridge

Generated: `2026-11-15T20:36:29+00:00`. No model fit, no parameter tuning. Stage 3/4 artifacts are loaded read-only.

## 1. Actual horizons

- training rows: **357**, unique train dt: **3.783709787817 years**.
- production own-prior segments: **521**. dt counts: `{'1.404517453799': 508, '5.188227241615': 13}`.
- production dt is computed from each segment own last VTD date; 4 fallback rows have no own dt.
- common train dt is true, so `dt_target/dt_train` is defined; the OOF cannot identify a time law.

## 2. Historical OOF temporal variants

Stage 3 OOF all have the same dt. Therefore correction and time-scaled are numerically identical; conservative is a fixed 0.25 shrinkage. The OOF table cannot choose between 1 and 2.

| Base | Variant | n | MAE | bias | Pearson r | Pearson r² | R² | top10 global | lower-tail global MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global_ridge_geometry | correction | 321 | 0.418703 | 0.008153 | 0.403056 | 0.162454 | 0.159937 | 0.454545 | 1.008966 |
| global_ridge_geometry | time_scaled | 321 | 0.418703 | 0.008153 | 0.403056 | 0.162454 | 0.159937 | 0.454545 | 1.008966 |
| global_ridge_geometry | conservative_0_25 | 321 | 0.466157 | 0.049434 | 0.367327 | 0.134929 | -0.054835 | 0.393939 | 0.915436 |
| target_ridge_geometry | correction | 321 | 0.421395 | 0.002710 | 0.401766 | 0.161416 | 0.154457 | 0.454545 | 0.985686 |
| target_ridge_geometry | time_scaled | 321 | 0.421395 | 0.002710 | 0.401766 | 0.161416 | 0.154457 | 0.454545 | 0.985686 |
| target_ridge_geometry | conservative_0_25 | 321 | 0.467201 | 0.048074 | 0.368590 | 0.135859 | -0.057671 | 0.393939 | 0.910427 |

The correction and time-scaled OOF predictions are identical by construction; this is not evidence that either time law is correct.

## 3. Production variants (global and target Ridge)

| Base | Variant | mean | std pred | std |pred-last| | min | max | above last raw | above nominal raw | clipping | fallback | mean shift after clip |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global_ridge_geometry | correction | 6.060359 | 0.296444 | 0.217856 | 4.798129 | 7.049606 | 147 | 0 | 0 | 4 | 0.000000 |
| global_ridge_geometry | time_scaled | 6.161803 | 0.443370 | 0.105031 | 4.065792 | 6.909500 | 147 | 0 | 0 | 4 | 0.000000 |
| global_ridge_geometry | conservative_0_25 | 6.183868 | 0.480652 | 0.054464 | 3.887585 | 7.004351 | 147 | 0 | 0 | 4 | 0.000000 |
| target_ridge_geometry | correction | 6.063419 | 0.320796 | 0.213561 | 4.707544 | 7.145787 | 145 | 0 | 0 | 4 | 0.000000 |
| target_ridge_geometry | time_scaled | 6.162530 | 0.451335 | 0.103738 | 4.036481 | 6.921842 | 145 | 0 | 0 | 4 | 0.000000 |
| target_ridge_geometry | conservative_0_25 | 6.184633 | 0.485517 | 0.053390 | 3.867844 | 7.012663 | 145 | 0 | 0 | 4 | 0.000000 |

No unconditional ban on prediction > last was applied. Raw predictions and clipped predictions are both saved; only the established `[0, nominal]` bounds are enforced.

### |prediction − last| distribution

| Base | Variant | mean | median | q05 | q95 | max |
|---|---|---:|---:|---:|---:|---:|
| global_ridge_geometry | correction | 0.309406 | 0.278649 | 0.025004 | 0.662361 | 1.470341 |
| global_ridge_geometry | time_scaled | 0.122820 | 0.110134 | 0.009282 | 0.258382 | 1.012139 |
| global_ridge_geometry | conservative_0_25 | 0.077352 | 0.069662 | 0.006251 | 0.165590 | 0.367585 |
| target_ridge_geometry | correction | 0.302589 | 0.274070 | 0.024948 | 0.645717 | 1.391376 |
| target_ridge_geometry | time_scaled | 0.120322 | 0.107462 | 0.009349 | 0.257391 | 0.961221 |
| target_ridge_geometry | conservative_0_25 | 0.075647 | 0.068517 | 0.006237 | 0.161429 | 0.347844 |

## 4. Differences between temporal variants

| Base | Comparison | mean | median | q05 | q95 | max | nonzero rows |
|---|---|---:|---:|---:|---:|---:|---:|
| global_ridge_geometry | time_scaled_minus_correction | 0.101443 | 0.123593 | -0.264533 | 0.390816 | 0.694034 | 521 |
| global_ridge_geometry | conservative_minus_correction | 0.123508 | 0.151031 | -0.318219 | 0.468617 | 0.827809 | 521 |
| global_ridge_geometry | time_scaled_minus_conservative | -0.022065 | -0.025043 | -0.080279 | 0.054387 | 0.827604 | 521 |
| target_ridge_geometry | time_scaled_minus_correction | 0.099110 | 0.117451 | -0.272975 | 0.389944 | 0.700135 | 521 |
| target_ridge_geometry | conservative_minus_correction | 0.121213 | 0.146581 | -0.333217 | 0.465437 | 0.835086 | 521 |
| target_ridge_geometry | time_scaled_minus_conservative | -0.022103 | -0.024119 | -0.078643 | 0.054260 | 0.785969 | 521 |

### Maximum absolute difference vs correction

| Base | Variant | max abs difference, mm |
|---|---|---:|
| global_ridge_geometry | correction | 0.000000 |
| global_ridge_geometry | time_scaled | 0.924549 |
| global_ridge_geometry | conservative_0_25 | 1.102756 |
| target_ridge_geometry | correction | 0.000000 |
| target_ridge_geometry | time_scaled | 0.874896 |
| target_ridge_geometry | conservative_0_25 | 1.043532 |

## 5. Thin-tail behavior

Thin set: **53** lowest `last_mm` segments; tie order by segment_id.

| Base variant | Thin mean pred | Thin mean Δ vs last | Thin above last |
|---|---:|---:|---:|
| global_ridge_geometry_correction | 5.483756 | 0.469144 | 52 |
| global_ridge_geometry_time_scaled | 5.216880 | 0.202267 | 52 |
| global_ridge_geometry_conservative_0_25 | 5.131899 | 0.117286 | 52 |

Thin-tail file: `thin_segments_global_ridge.csv`. Fallback rows are explicitly marked; Ridge is not applied to them.

## 6. Proposal for freeze choice

- Reference: **s1a**.
- Challengers (max 2):
  - `global_ridge_geometry_correction`: `h_last + g(X)` on `global_ridge_geometry`
  - `global_ridge_geometry_time_scaled`: `h_last + (dt_target/dt_train)*g(X)` on `global_ridge_geometry`
- Sensitivity, not frozen: ['conservative_0_25 (fixed shrinkage; not selected by OOF)', 'target_ridge_geometry variants (local fit; no clear advantage in Stage 4)']
- Basis: Common train dt permits the time-scaled expression; OOF cannot distinguish correction from time-scaled because all historical dt equal. No temporal winner is declared.
- OOF does not select a time law; the proposal is for our explicit choice, not a winner claim.

## 7. Artifacts and commands

```text
experiments/targeted_cycle/stage_5_temporal/
  dt_audit.json, oof_temporal_metrics.csv, oof_temporal_details.json
  production/<base>_diagnostics.csv, production/<base>_<variant>_predictions.csv
  production_summary.json, differences.json, thin_segments_global_ridge.csv
  proposal.json, run_manifest.json
```

```bash
C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_s1_temporal --root . --data-root C:/Users/Администратор/Documents/PipeTwin_next/experiments/next_cycle/data_generation --output-root experiments/targeted_cycle/stage_5_temporal
```

## 8. Tests

- `PIPETWIN_NEXT_DATA_ROOT=... C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m unittest tests.test_stage_5_temporal -v`: 6 tests, OK.
- Full Stage 2–5 test run: 26 tests, OK.
- Проверены actual dt, fallback без Ridge, bounds, raw/clipped columns, выше-last без запрета, OOF identity correction/time-scaled и proposal <= 2.

## 9. Limitations

- Only one historical dt; OOF is identical for correction and time-scaled.
- Historical campaign is one campaign; target horizon differs.
- Production graph is not validation; target labels are unavailable.
- Stage 5 does not choose a winner and does not run Stage 6.

# PipeTwin ML baseline run

Checked branch `main`, commit `5f06c32e682fe43433213d6a51087733ccc27570`.

## Result status

A reproducible provisional M0/W1 forecast is exported. This is not a validated final ML solution. There are no eligible forward-time held-parent-pipeline folds for a fitted model before either production cutoff. C0 scores measure no-thinning baseline error only. No hidden-test metric is known.

W1 gives equal total weight to each parent pipeline, then each campaign within that parent. M0 fits a nonnegative constant effective rate by weighted least squares in millimeters. W0 and W2 are sensitivity experiments, not selected winners. W2 doubles weights below the training 10th-percentile thickness. Negative observed losses remain in fitting and are counted in models.json.

## Independent evidence

| Case | Target rows | Training rows | Parent pipelines | Fitted rate (mm/year) | Forward fitted-model folds |
|---|---:|---:|---:|---:|---:|
| between_vtd | 525 | 357 | 1 | 0.0130336 | 0 |
| cold_start | 1475 | 372 | 1 | 0.10198 | 0 |

## Measured campaign inventory

| Case | Parent | Date | Rows | Simple sections |
|---|---|---|---:|---:|
| between_vtd | MAIN-2 | 2022-07-06 | 561 | 3 |
| between_vtd | MAIN-2 | 2023-12-01 | 53 | 2 |
| cold_start | MAIN-2 | 2018-09-23 | 372 | 3 |
| cold_start | MAIN-3 | 2023-08-12 | 94 | 7 |

Rows at or after each production cutoff are excluded from that target fit and evaluation. MAIN-2 simple sections are not three independent pipes; MAIN-3 supplies seven simple sections. The 2022-07-06 survey is unavailable for the 2022-07-01 cold-start forecast.

## Baseline and exposure limitations

- S1 uses the last measured minimum separately per segment. Missing baselines borrow the nearest prior measured segment and its date, with lower segment_id breaking ties. The source, distance, timestamp and missing flag are exported. A 100 m block-masking diagnostic tests reconstruction of prior thickness; it does not independently validate the four genuinely unmeasured segments.
- S2 assumes nominal thickness at commissioning. The prediction is constant along this pipe: M0 does not establish a spatial risk ranking. All tied ranks are disclosed; the top-10% export uses segment_id solely as a deterministic tie-break, not as evidence of higher risk.
- Exposure uses a maximum 24-hour forward hold from each valid calculation. The next calculation, even an invalid one, terminates the preceding interval. No backward fill or unrestricted carry across gaps occurs. observed_years means supported under this assumption, not continuously measured exposure. Unknown years and longest gaps are reported on the entire forecast horizon.
- M0 extrapolates a learned constant rate across the full horizon, including unknown regime history. It does not use exposure integrals. M1 code supports observed-only and train-reference gap assumptions but is gated off for this dataset. M2/M3/M4, angle, coatings and chemistry are deferred.
- No confidence intervals are exported: independent residual calibration is unavailable.
- Repair/replacement histories and within-inspection spatial alignment are unverified. Clipping to the measured/assumed baseline can hide apparent thickness increases; unclipped predictions are retained.

## Provenance and environment

All read CSVs match preprocessing checksums. Stale preprocessing code hashes: ['config.py'].
The published report states 270 preprocessing checks. That is a prior report, not evidence of a rerun in this experiment. See docs/ml_model_card.md for tests actually run and environment limitations.
Executed versions: {'python': '3.12.14', 'pandas': '2.2.3', 'numpy': '2.3.5'}. This is not the environment pinned in requirements.txt.

## Artifacts and next command

- `source_audit.json`: exact schemas, missing counts, dates, join keys, examples and source mappings.
- `training_windows.csv`: reconstructed historical features, label dates, starts and exposure coverage.
- `metrics.json`, `validation_predictions.csv`: strict forward held-parent evaluations, including blocked fits.
- `models.json`, `run_manifest.json`: coefficients, weights, limitations, versions and hashes.
- `candidate_predictions.csv`: C0 and M0/W0/W1/W2 sensitivity predictions.
- `result.csv`: exactly the three official fields. Diagnostics and risk ranks are separate.

```bash
python -m src.ml.experiment --output-root experiments/ml_baseline
```

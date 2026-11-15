# Ingestion v2 verification record

This record documents actual commands and observed outputs.  It is not a claim of
hidden-test accuracy, calibrated intervals or independent forward-fold success.

## Environment and scope

- Python 3.13.11; pandas 3.0.6, NumPy 2.5.4, PyArrow 26.0.1 (`requirements.txt`).
- Current archive: 16 passport rows, 20,218 supplied profile records (38 exact
  duplicates removed), 48,291 supplied VTD records (251 out-of-pipe positions
  quarantined), 67 UZT records, 495 monitoring records and 39,052 hydraulic rows.
- Verified full grid: 4,668 segments.  Target grid recomputed from passport
  lengths: 525 + 1,475 = **2,000** segments.
- The root repository's dirty `processed/target_segments_grid.csv` was **not**
  overwritten.  Verification outputs were written to `scratch/pipeline_test` and
  the renamed-layout run to `scratch/layout_old_out`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Result: **91 tests passed** (62 existing/reused + 29 new ingestion tests).
The tests cover encoding/delimiter/header loss, identifiers, dates, source
classification including both meanings of `regime.csv`, duplicate and conflicting
copies, optional-source runs, units, grid endpoints, VTD eligibility, persistence,
future invariance, cold-start isolation, Reynolds/thermal assumptions, metric
failure cases, mentor revision/idempotency, template detection, priority export,
cache keys and failure-safe publication.

## Full archive run

```bash
python -m src.data_prep --output-root scratch/pipeline_test
```

Observed: **306 checks passed**, **25 CSV/Parquet pairs verified**, 2,000 target
segments.  The leakage audit passed.  Source inventory:

| Role | File | Rows |
|---|---|---:|
| passport | passport.csv | 16 |
| profile | profile.csv | 20,218 supplied / 20,180 retained |
| VTD | vtd.csv | 48,291 supplied / 48,040 retained |
| UZT | uzt.csv | 67 |
| hydraulic_regime | tech_mode_...ids16.csv | 39,052 |
| corrosion_monitoring | regime.csv | 495 |

The `sample_result.csv` candidate in the supplied organizer folder is a Techmod
table, not a three-column template.  No authoritative template was found, so
`reports/template_validation.json` explicitly validates against the derived grid.

## Renamed organizer layout

The same semantic inputs were run with organizer names
(`pot_south.csv`, `altitudes_south.csv`, `vtd_south_train.csv`, `uzt_south.csv`,
`repcorr_south.csv`, and legacy `sample_result.csv` as hydraulic input):

```bash
python -m src.data_prep --data-dir scratch/layout_old --output-root scratch/layout_old_out
```

Result: the same **306 checks** and byte-semantic equality of the target features,
target grid, VTD/UZT histories, profile features, training targets, full grid and
monitoring table against the canonical layout.  Source classification correctly
selected `hydraulic_regime` for `sample_result.csv` and
`corrosion_monitoring` for `repcorr_south.csv`.

## Model compatibility/evaluation

The v1 and v2 experiments were rerun against the verified output root:

```bash
python -m src.ml.experiment --root scratch/pipeline_test --output-root scratch/exp_v1_test
python -m src.ml.experiment_v2 --root scratch/pipeline_test --output-root scratch/exp_v2_test
```

Both produced 2,000 validated predictions with keys exactly matching the derived
grid.  Their prediction values were **identical** to the previously published v1
and v2 results on unchanged accepted inputs (maximum absolute difference 0.0).
The additive metric fields are the only semantic change in the metric JSON.  The
separate UZT priority export is `scratch/exp_v2_test/uzt_priority.csv`; it contains
the required ranking fields and is not part of `result.csv`.

Observed v2 historical/cross-environment diagnostics (not hidden-test scores):

| Evaluation block | Model | n | MAE mm | RMSE mm | Bias mm | Pooled within-pipe Pearson² |
|---|---|---:|---:|---:|---:|---:|
| Between-VTD prior pairs | C0/M0 | 357 | 0.4982 | 0.6523 | 0.0493 | 0.1133 |
| Cold-start cross-environment | M1S | 466 | 0.8239 | 0.9875 | -0.6823 | 0.0149 |

The reports still show no eligible independent forward fitted-model fold before
the production cutoffs.  These numbers are sensitivity/process diagnostics, not
validated hidden-test accuracy.  Cold-start predictions remain spatially weak and
the UZT priority table marks constant/tied predictions as unsupported for spatial
ranking.

## Temperature scenario

```bash
python scripts/calculate_segment_temperature.py \
  --features scratch/pipeline_test/processed/target_features.csv \
  --output-root scratch/temperature_test --sensitivity
```

Observed: 56,000 scenario rows (base + 27 declared sensitivity points), all
scenario-valid on this archive.  Base scenario cooling ranged from about
-0.018 to -0.0017 °C per segment on the cold-start pipe and about
-0.0032 to -0.0024 °C on the between-VTD pipe.  These are explicitly assumed
parameters (ground 5 °C, U=1 W/m²K, cp=3,500 J/kgK, roughness 5e-5 m), not
measured temperatures.  The verified target grid checksum was unchanged.

## Mentor evidence

```bash
python -m src.mentor_ingest --output docs/mentor
```

Observed: 10 sources, 7 with text, 713 automatic candidates, 15 curated decisions,
8 requirements, 13 open questions, 8 actions, zero excerpts not found, zero
messages sent.  All source dates/participants remain null because the supplied
notes do not state them.  Requirements that have observed passing tests are marked
`verified`; blocked/unresolved science remains visible in
`docs/mentor/open_questions.csv`.

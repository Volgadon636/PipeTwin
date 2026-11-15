# ML compatibility and impact

M0 was run with the original commit's source and isolated baseline data, then corrected source and isolated corrected data, in separate Python processes with identical W1 settings. parser_ml_impact.py extracts baseline src from Git without switching the checkout. This tests code-and-data compatibility, not just two data copies through the same code.

| Scenario | Eligible rate-fit rows | Baseline and corrected fitted rate (mm/year) | Prediction change |
|---|---:|---:|---:|
| between_vtd, 1250022871, 2023-12-01 | 357 | 0.013033550759625047 | 0 |
| cold_start, 1250009583, 2022-07-01 | 372 | 0.10197980994208126 | 0 |

All 1,080 reconstructed training-window rows compare equal, including initial thickness/horizons/labels, available spatial, temperature and hydraulic features, masks and missingness. All 2,000 target predictions compare equal; maximum absolute difference is 0 mm. See ML_IMPACT.json and baseline/corrected training-window/prediction CSVs.

The 38 corrected monitoring flags affect rows already excluded for other reasons. Target chemical features remain missing. Training targets, target features and mapped hydraulic/profile tables are semantically unchanged. Existing cold ML make_snapshot already clears the archive flag; fixing the lower-level API prevents leakage for direct and future callers rather than changing current M0 output.

No model selection, Ea selection, hyperparameter optimization or accuracy claim was made. LightGBM native runtime is unavailable in this local environment (libomp dependency); M0 does not need it and completed. No boosted-model retraining was needed for the unchanged training matrices. Independent checks cover 68 target/historical preprocessing requests, including inspection-day cutoffs, and the suite covers future-data mutation. These do not cure unverified static-input availability or limited independent longitudinal cohorts.

Use the corrected preprocessing code and an explicitly selected corrected artifact root for further work. Existing published artifacts remain preserved for review; this audit did not replace them or publish competition predictions.

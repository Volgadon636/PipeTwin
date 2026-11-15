# Stage 04 QA

Branch `model_v02`, HEAD `d0ee12c25cc5a1d27b00f77b2cb7ee4fbc4b85b2`.

Fixed Ea40 development provenance: Explicit user instruction: Proceed on to stage 4 retaining the value of EA as 40.

Protected files checked: 245; changed: 0. Published input mismatches: 0. Protocol unchanged: True.

The initial `.gitignore`, `.mcp.json` and uncommitted Stage 03.1 work were included in the protected inventory. No existing source, test, data, configuration, prior experiment or official result was edited.

## Actual test runs

- `tests_stage04.txt`: exit 0; Ran 17 tests in 1.225s; OK
- `tests_ml.txt`: exit 0; Ran 17 tests in 10.261s; OK
- `tests_temperature_regression.txt`: exit 1; FAIL: test_legacy_factor_and_baseline_files_unchanged (experiments.temperature_v02_02.test_temperature.TemperatureTests.test_legacy_factor_and_baseline_files_unchanged); Ran 43 tests in 16.507s; FAILED (failures=1)

Stage 04 checks passed: True. All requested regressions passed: False.

Known repository limitation: the unchanged Stage 02 preservation test hashes raw CRLF baseline bytes against a historical LF hash. Stage 04 copies B0 byte-for-byte and separately verifies exact reconstructed M0/EA00 identity at the original exporter's ten decimal places. The saved Stage 03.1 full regression log also records a preprocessing nullable-boolean setup error; preprocessing was not rerun for this isolated model change and that historical failure is not represented as a current pass.

The first Stage 04 run stopped at an overstrict 1e-12 comparison against B0 before candidate fitting. Source inspection confirmed src/ml/experiment.py exports ten decimal places; the corrected check verifies that exact original serialization. The failed log is retained as run_attempt01_B0_precision.txt. No B0 values or training rules changed.

EA40 remains a user-chosen development hypothesis. No independent folds, no target-label evaluation and no production promotion. Export schema and physical bounds are tested for all 30 candidate files. EA40 result.csv is the reference-gap assumption; observed-only is explicitly marked as an incomplete descriptor.

GO for exploratory Stage 05 ablations; NO-GO for independently validated-performance claims or production promotion.

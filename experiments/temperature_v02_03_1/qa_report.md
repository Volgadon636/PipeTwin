# Stage 03.1 QA and decision

Branch `model_v02`, HEAD `d0ee12c25cc5a1d27b00f77b2cb7ee4fbc4b85b2`. Initial user changes: `.mcp.json`, untracked `.github/`; preserved.

Protected files checked: 201; changed: 1; scientific/source/data/baseline changes: 0. Inputs checked: 8; mismatches: 0. Frozen preregistration unchanged: True.

A `.gitignore` addition (`.github`) appeared during execution after the initial clean inventory. No Stage 03.1 command edits that file. It is preserved and disclosed as a concurrent ancillary change; the original before-hash inventory is not rewritten to hide it. Consequently the blanket all-protected-files-unchanged gate remains false.

The first execution failed before data loading because Windows default cp1252 could not read the existing UTF-8 manifest. Reproduction uses `-X utf8`; original source and loader were not changed. See run_attempt_01_encoding_failure.txt.

The initial verifier incorrectly compared raw-byte input hashes against the loader's CRLF-normalized hashes. The corrected verifier calls src.storage.sha256 for that contract and separately checks exact bytes for all protected files. No input checksum was bypassed.

## between_vtd: candidate_set_retained

357 eligible rows; 1 parent; 1 campaign; 3 sections; 3 section/windows. Latest label: 2022-07-06 00:00:00. Negative measured thinning: 177 rows.

Admissible independent folds: 0; usable section holdouts: 3. One-day support: 80.136394%; unknown share: 19.863606%. Missing-history rows: 0.

Failed selection gates: unique_primary_best_both_policies.

Retained: EA00_null, EA25_low, EA40_mid, EA55_elevated, EA70_high.

## cold_start: candidate_set_retained

372 eligible rows; 1 parent; 1 campaign; 3 sections; 3 section/windows. Latest label: 2018-09-23 00:00:00. Negative measured thinning: 0 rows.

Admissible independent folds: 0; usable section holdouts: 3. One-day support: 17.526759%; unknown share: 82.473241%. Missing-history rows: 0.

Failed selection gates: unique_primary_best_both_policies.

Retained: EA00_null, EA25_low, EA40_mid, EA55_elevated, EA70_high.

## Interpretation and Stage 04

The experiment does not justify a unique selection under the frozen engineering tolerances; it cannot rule out a physical candidate merely from its numerical rank. All retained hypotheses remain compatible with this limited evidence. Cold-start support is sparse; between-VTD exposure shape differences are small. Practical held-out ties and shared parent/campaign histories prevent identification. All 60 sensitivity matrices have effective rank 2; rank deficiency is not the reason selection was withheld. See findings.md for the numerical interpretation.

GO for isolated Stage 04 development with the retained candidate grid and explicit gap assumptions. NO-GO for claiming a scientifically selected Ea or promoting production forecasts. EA40 can be software continuity only. Obtain independent campaigns/parents spanning varied, densely measured temperatures, paired W/v, reliable baseline and repeated thickness labels before making a transfer claim.

## Actual tests

- `tests_stage03_1.txt`: exit 0; Ran 18 tests in 3.023s; OK
- `tests_regression.txt`: exit 1; ERROR: setUpClass (test_preprocessing.DatasetTests); FAIL: test_legacy_factor_and_baseline_files_unchanged (experiments.temperature_v02_02.test_temperature.TemperatureTests.test_legacy_factor_and_baseline_files_unchanged); Ran 58 tests in 37.973s; FAILED (failures=1, errors=1)

Stage 03.1 tests/input/preservation checks passed: True. Full regression suite passed: False.

Existing regression limitations: the Stage 02 baseline test hashes raw CRLF bytes against an LF hash; both baseline files retain their exact pre-experiment bytes and their normalized hashes match the expected value. Preprocessing DatasetTests setup raises `TypeError: boolean value of NA is ambiguous` at src/data_cleaner.py:132 with this installed pandas/Arrow environment. Those unchanged source paths do not import Stage 03.1. No prior tests, parser, data or baselines were edited to make them pass. Full failure traces are retained in tests_regression.txt.

The 13.3086% prior cold-start production support remains cited handoff context; no production reconstruction was required for historical selection. Production segment counts are regression-tested, not counted as independent experiments.

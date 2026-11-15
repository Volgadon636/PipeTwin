# Implementation impact report

This report is generated from the requirement matrix; it does not claim that blocked science is resolved.

## Implemented / verified components

- **REQ-0001** [verified] Write exactly id,segment_id,thickness_pred; one unique finite numeric mm prediction per requested segment; no extra index or diagnostic columns. → `src/ml/submission.py::validate_submission; src/ml/submission.py::export_result`; tests: `tests/test_ml.py::test_submission_rejects_duplicate_extra_missing_and_nonfinite; tests/test_ingestion.py::ExportTests`
- **REQ-0002** [verified] The forecast target is minimum residual wall thickness in millimetres, not defect depth, nominal wall, corrosion rate or a unitless score. → `src/segment_mapper.py::map_vtd_to_segments`; tests: `tests/test_preprocessing.py::test_original_measured_minimum`
- **REQ-0003** [verified] Keep the UZT inspection-priority table separate from result.csv. → `src/ml/submission.py::uzt_priority_table`; tests: `tests/test_ingestion.py::ExportTests`
- **REQ-0004** [verified] Report Pearson correlation squared, signed Pearson r, separate R2, MAE, RMSE, bias and MAPE with explicit undefined/near-zero policy. → `src/ml/evaluation.py::metrics`; tests: `tests/test_ingestion.py::MetricTests`
- **REQ-0005** [verified] Apply time eligibility before persistence, aggregation, imputation or feature selection; timestamp < prediction date at 00:00. → `src/temporal.py::hydraulic_asof; src/segment_mapper.py::build_features_asof`; tests: `tests/test_preprocessing.py::test_future_injection_cannot_change_features; tests/test_preprocessing.py::test_third_future_day_does_not_validate_two_day_anomaly`
- **REQ-0006** [verified] Cold start excludes target-pipe VTD/UZT-derived inputs even if present in the archive. → `src/segment_mapper.py::build_features_asof`; tests: `tests/test_preprocessing.py::test_cold_start_excludes_diagnostic_history`
- **REQ-0007** [verified] Preserve source bytes and publish only after checks; source changes during a run fail the run. → `src/data_prep.py; src/storage.py::publish_tree`; tests: `tests/test_ingestion.py::PublicationTests`
- **REQ-0008** [verified] Account explicitly for every supplied field and every rejected/masked value; preserve unknown archival columns. → `src/schema_registry.py::build_feature_inventory; src/quality.py`; tests: `tests/test_ingestion.py::FeatureInventoryTests`

## Blocked or unresolved

- **Q-0001**: Is an authoritative submission grid/template available, and what is the endpoint rule for the final incomplete segment? Evidence needed: Organizer-provided sample_result/template or written endpoint rule
- **Q-0002**: What are the exact units and physical basis of pressure, viscosity, q/q_oil/q_gas, d_mm, t_mix, CO2 and inhibitor measurements? Evidence needed: Field dictionary or instrument/export documentation with units and reference conditions
- **Q-0003**: Are CO2, КВЧ/mechanical impurities and inhibitor fields concentrations, fractions, doses or scores, and at what timestamp/reference condition? Evidence needed: Measurement method, units, sampling point and availability time
- **Q-0004**: Can you provide the historical RepCorr ID/direction crosswalk with effective dates? The current join rejects 412 unknown IDs and 83 direction conflicts and has zero usable target records. Evidence needed: Authoritative ID/direction mapping with validity intervals; approved spatial/temporal transfer policy
- **Q-0005**: Why are UZT thicknesses above passport nominal, and are they measured on a different wall reference or method? Evidence needed: UZT method, calibration block, measurement location and wall reference
- **Q-0006**: What critical/rejection thickness and ranking policy applies at the forecast date? Evidence needed: Approved critical thickness formula or explicit statement that none is provided
- **Q-0007**: Is there repair/replacement history, and how does it reset segment history? Evidence needed: Dated repair/replacement records with intervals
- **Q-0008**: Are same-day 2023-12-01 diagnostics allowed? The current conservative cutoff excludes the entire day. Evidence needed: Observation time and organizer rule for same-day measurements
- **Q-0009**: What is the train/test role of 1250009583, and which cold-start calibration evidence is permitted? Evidence needed: Organizer split file or explicit statement
- **Q-0010**: What measured ground temperature/thermal properties and flow basis should be used, or is a temperature scenario desired at all? Evidence needed: Measured/declared parameters and applicability range
- **Q-0011**: Are campaign detection and apparent thickening complete, or are there missing/extra observations and repairs? Evidence needed: Campaign coverage, tool calibration and repair records
- **Q-0012**: How should defect coordinates/identities be matched across VTD campaigns, if at all? Evidence needed: Defect start/centre convention, tool orientation and matching tolerance
- **Q-0013**: Should the starting state use the minimum of the 2018 and 2022 inspections or the latest available inspection? The team selected minimum; prove it on identical splits. Evidence needed: Paired forward-time comparison with pre-cutoff features

## Rerun requirements when inputs change

- Re-ingest mentor notes after a new note/revision: `python -m src.mentor_ingest`.
- Rebuild preprocessing after any source/config/schema/rule change: `python -m src.data_prep --output-root <new>`. 
- Re-run model/evaluation after feature or label changes; do not reuse an earlier cutoff's features.
- Temperature scenario is explicit and isolated: `python scripts/calculate_segment_temperature.py --sensitivity`.
- Submission validation must rerun against the derived grid or the authoritative template if one appears.

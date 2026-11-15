# PipeTwin parser and preprocessing audit

**Decision: CONDITIONAL PASS for the corrected parser and the restricted uses below.** Nine confirmed implementation defects were fixed. The supplied records reconcile, independent computational checks pass, and baseline predictions are unchanged. This is not certification of the measurements, physical corrosion model, or future unseen imports.

## Scope and authoritative baseline

Audited branch: `model_v02`; baseline commit: `2522d5fc2409ea54c5c6962f187a32c01dd0e7e3`. Fetch succeeded and `origin/model_v02` matched this commit. Remediation branch: `codex/parser-audit-2522d5f`, based directly on that commit. Existing untracked `docs/stage_prompts/Analysis.md` was preserved. No remediation commit was created: permission to stage the completed audit changes was declined. The audit files and fixes remain reviewable in the working tree on the dedicated branch; HEAD remains the baseline commit.

The attached audit prompt, README, cleaning rules, configuration, all ten preprocessing modules, preprocessing tests and relevant ML consumer/window code were inspected. The repository graph was used first for scope and later for impact/test relationships. DEPENDENCY_MAP.md documents producers, schemas, keys and loss at each transition. FILE_INVENTORY.csv records actual discovered files and Git-tracked status; no filenames were inferred from an older package.

Before edits, the documented `python -m src.data_prep` entry point was run with a separate output root. It passed 270 checks, wrote 22 verified CSV/Parquet pairs, generated 4,668 grid segments, 1,080 training targets and 2,000 target rows. The original 72 tests passed. Baseline source/config/artifact hashes are in before_manifest.json and baseline_run_manifest.json; exact schemas and versions are included. Environment: Python 3.14.2, pandas 3.0.6, NumPy 2.5.4, PyArrow 26.0.1.

Published artifacts were not accepted as ground truth. Their manifest does not exactly match the audited source: config.py and storage.py hashes differ. Twenty-one of 22 published tables reproduce semantically under explicit tolerances. quality_issues has the same keys/count but 22 different floating-point diagnostic string renderings, at approximately 1e-13 m, listed in PUBLISHED_BASELINE_DIFFERENCES.csv. The published run used Python 3.13.11; this audit used 3.14.2. Those strings do not change quarantine decisions or measurements. This explains the observed discrepancy without claiming a proven historical cause or assigning an unsupported published commit.

## Verification results

The final suite passes 88 tests (72 existing plus 16 new). The independent scripts record 18,743 source/aggregate/profile checks and 108,165 extended checks, including exact source IDs and 68 historical/current requests; these are computational invariants, not independent statistical samples. Both failure lists are empty. Repeated builds match all 44 CSV/Parquet byte streams. final_reproduction.txt captures the end-to-end reproduction run.

## Source reconciliation

| Source | Raw logical records | Retained | Excluded duplicates | Quarantined |
|---|---:|---:|---:|---:|
| passport.csv | 16 | 16 | 0 | 0 |
| profile.csv | 20,218 | 20,180 | 38 | 0 |
| vtd.csv | 48,291 | 48,040 | 0 | 251 |
| uzt.csv | 67 | 67 | 0 | 0 |
| regime.csv | 495 | 495 | 0 | 0 |
| tech_mode_for_embedings_with_length_with_temperatures_ids16.csv | 39,052 | 39,052 | 0 | 0 |
| Total | 108,139 | 107,850 | 38 | 251 |

Every record has a disjoint accounting state and source ordinal in SOURCE_TO_OUTPUT_RECONCILIATION.csv. Source-row identity, not only total counts, is checked. `source_row` means logical CSV record number including the header; physical line numbers can differ for quoted multiline records. Aggregation lineage for each retained VTD/UZT observation is in MEASUREMENT_LINEAGE.csv.

COLUMN_TRANSFORMATION_AUDIT.csv accounts for all 426 source columns. Independent csv.reader checks match source widths/counts; Decimal numeric conversion checks match every retained or quarantined typed numeric field, with exact integer-ID checks separately. Dates were independently parsed with Python datetime, and all archival strings were compared except documented missing tokens. There are zero unexpected conversion losses and zero invalid nonmissing values needing a parse-failure record in these supplied inputs. FAILED_CONVERSIONS.csv and NUMERIC_CONVERSION_DISAGREEMENTS.csv capture those categories, including empty results.

Encoding/CSV tests cover UTF-8, BOM, CP1251, Cyrillic, decimal commas, quotes, embedded delimiters/newlines, empty fields, trailing empty headers, malformed widths, original aliases/conflicts, and column-name collisions. Ambiguous slash dates are not guessed. Invalid supported-format dates become traceable missing values; mixed ISO/Russian date formats are tested.

## Confirmed defects and remediation

The nine confirmed implementation findings comprise **1 BLOCKER, 4 CRITICAL and 4 MAJOR**; no MINOR or DOCUMENTATION-only implementation findings were registered. Severity describes the potential failure, not a claim that all nine affected existing competition outputs.

| ID | Severity | Defect and correction | Existing artifact impact |
|---|---|---|---|
| PA-01 | CRITICAL | Large IDs rounded through float; parse exact Int64 with Decimal | Synthetic reproduction; no supplied IDs affected |
| PA-02 | MAJOR | Duplicate headers silently renamed by pandas; reject ambiguous headers | Synthetic reproduction |
| PA-03 | CRITICAL | Extra fields allowed implicit index/column shifts; validate logical widths | Synthetic reproduction |
| PA-04 | MAJOR | Missing exposure dates did not flag invalid duration; explicitly mark missing duration | 38 monitoring flags and 38 added issue records |
| PA-05 | CRITICAL | Persistent temperatures below absolute zero remained usable; hard Kelvin bound | Synthetic reproduction; no supplied values affected |
| PA-06 | MAJOR | Whole-second CSV formatting broke fractional timestamp export; retain nanoseconds | Synthetic reproduction; date presentation changes only |
| PA-07 | MAJOR | Float-tolerant validator missed integer corruption; exact integer comparison | Synthetic reproduction |
| PA-08 | BLOCKER | Public as-of API merged archive-wide diagnostic conflict into cold/earlier age features; restrict conflict to allowed history | Synthetic leakage reproduction; current targets/windows unchanged |
| PA-09 | CRITICAL | Mid-publication failure left mixed generations; batch rollback and reader-visible incomplete marker | Failure injection reproduction |

DEFECT_REGISTER.csv contains exact reproduction procedures, source examples, original/actual/expected values, counts, IDs, root causes and regression names. regressions_before.txt demonstrates PA-01..08 failing before correction; publication_before.txt separately demonstrates PA-09. malformed_width_before.txt additionally shows the original parser silently changing source ID 1 to parsed ID 2 on an over-wide record. All 88 current tests pass (tests_after.txt). No scientific ambiguity was fixed by altering a measurement or guessing a date/unit.

## Independent transformation verification

The first reference run reported two aggregate failures caused by pandas nullable-versus-ordinary integer index dtype equality, not differing keys or values. The reference now checks exact key sets and aligned numeric values; both reconstructions pass. The original log is retained in reconciliation_run.txt. No production assertion was weakened.

**VTD and UZT.** Independently reconstructed every point's segment using Decimal floor and passport length, and all aggregate minima/counts/maximum depths/flags/eligibility. The eligible VTD subset exactly equals training_targets. Measured residual thickness is never replaced by nominal thickness or a depth formula. The special 2018-09-23 fraction cohort has 2,473 retained normalized rows across three IDs; per-ID median fraction residual error is 0.002 mm, versus approximately 1.38–1.81 mm under the percent interpretation. Other cohorts support percent interpretation, not a blanket x100 rule. Raw depths and measured residuals remain unchanged; only normalized depth changes. VTD_COHORT_AUDIT.csv and VTD_FEATURE_COHORTS.csv separate campaigns and feature types; VTD_DEPTH_DISAGREEMENTS.csv lists the six retained residual/depth discrepancies.

There are 13,485 retained observations before commissioning, 14,239 local-versus-nominal wall mismatches, and 344 missing residuals. These are source issues, not inferred parser bugs. Pre-commissioning measurements remain archival and are excluded from labels; missing thickness is never filled with nominal/zero. Source categories include internal/external corrosion, clusters, manufacturing defects and service features. Counts are observations rather than independent physical defects. Long defects retain the documented reference-point assignment; extent-based mapping requires an organizer rule.

All 67 UZT measurements exceed the passport nominal tolerance and remain stored but ineligible. Their raw minima/means and eligible minima/counts reconcile. Angle and other unmodeled columns remain unchanged; their engineering meaning is not invented.

**Grid/profile.** Reconstructed all 4,668 grid rows independently. Exact 10 m boundaries, immediately adjacent floating-point values, short final segments, zero and final endpoint, and rejected negative/outside values are tested. No source IDs or segments changed. Piecewise-linear heights were independently implemented with bisect; trapezoidal integral means, slopes, endpoint/center/extreme heights and extrapolation flags match all segments. Exact duplicate profile exclusions match source identities; conflicting same-coordinate elevations fail rather than average incompatible data. Twelve segments are extrapolated and 61 raw steps are steep; flags are preserved. Survey accuracy and accumulation interpretations remain conditional.

**Monitoring/hydraulics.** Independently checked every monitoring ID/direction/exposure eligibility and every hydraulic ID/date eligibility/diameter conversion against passport. All hydraulic diameter scales agree with passport after conditional x1000; nominal/outside/inside subtype remains unresolved. Monitoring yields no eligible chemical observations, so no cross-pipe medians or corrosion rates are invented. Gravimetric and nonselected chemical columns remain archival.

Hydraulic reference code uses explicit Python calendar-day sets/runs rather than the production persistence helper. It verifies every usable value for 68 target/historical requests, including hard invalid quantities, paired high-flow masking, missing calendar days and cutoff boundaries. Independently reconstructed daily medians, means, sample standard deviations, valid-day counts, source dates, coverage and maximum gaps match final features. Related modeled fields are masked for the documented high-flow dependency; the provenance of other calculated archival fields cannot be certified without the source model. Current target coverage is approximately 69.17% / 51.49% of each observed span, with maximum gaps of 195 / 398 days. No full-operation-history mean or imputation is claimed.

**Temporal isolation and joins.** Checked 68 current and historical requests, covering all pipes with VTD history, multiple inspection cutoffs and both scenarios. Independent latest-eligible-history selections and static/profile/hydraulic/monitoring joins agree. Strict `source_date < cutoff` excludes the entire prediction day when cutoff is midnight. Existing synthetic tests check future injections, a third future persistence day, missing past history and cross-pipe independence. Cold excludes direct diagnostic thickness and indirect archive-wide flags. Derived pit_corrosion_intensity and passport diagnostic summaries are excluded. Static passport/profile availability remains an explicit data limitation.

**Serialization/publication.** All 22 pairs pass semantic CSV/Parquet validation and exact Parquet dtype checks. Independent raw-value/aggregate references complement rather than repeat that production validator. Integer IDs and nanosecond dates are exact. Continuous floating comparisons use atol=1e-10 and rtol=1e-12 to allow 15-significant-digit CSV formatting and harmless arithmetic order differences; profile integration uses absolute 1e-9 m. These are numerical tolerances, not measurement uncertainty. The second corrected run is byte-identical for all 44 CSV/Parquet files. Failure injection verifies rollback; marker-aware readers reject interrupted roots. Simultaneous arbitrary readers and power-loss atomicity are not guaranteed.

## Before/after and ML impact

TABLE_COMPARISON.csv covers all 22 generated tables, with exact added/removed keys, schema/type/missingness changes and published-baseline comparison. CELL_DIFFERENCES.csv contains 38 changed exposure flags; ADDED_REMOVED_ROWS.csv contains the 38 resulting issue rows. Detailed per-table files include these additions. No rows or columns were removed; no schema/type changes remain; no other data values, assignments, eligibility, labels, feature values or missingness changed. CSV date presentation differences are benign ISO-compatible serialization differences.

Baseline and corrected M0 were run with their respective source code in separate processes under identical settings. All 1,080 historical training windows and 2,000 target predictions match. Between-inspection fit uses 357 rows at 0.013033550759625047 mm/year; cold-start uses 372 rows at 0.10197980994208126 mm/year. Maximum prediction difference: 0 mm. ML_IMPACT_REPORT.md distinguishes parser correctness from model accuracy and records the unavailable native LightGBM runtime; no model redesign/retraining was required.

Original data, configuration and published artifacts retain their pre-audit hashes (IMMUTABILITY.json). Corrected bulk artifacts remain locally under corrected/, with baseline/ and repeated/ preserved. These redundant builds are excluded from Git; their manifests, checks, evidence, scripts and reports are prepared with the audit changes. The corrected datasets were not substituted into the main published directories.

## Release gate by downstream use

| Use | Decision | Restriction |
|---|---|---|
| VTD remaining-thickness targets | CONDITIONAL PASS | Trust measured minima under documented eligibility and point-assignment semantics; resolve source inconsistencies before wider physical interpretation. |
| Historical corrosion-rate estimation | CONDITIONAL PASS | Only permitted chronology and explicit baseline assumptions; not verified physical rates for conflicting/replaced pipes or full lifetimes. |
| Spatial segmentation | PASS for implemented convention | Complete exact 10 m grid and point assignments reconcile; organizer extent semantics remain a separate question. |
| Profile-derived features | CONDITIONAL PASS | Numerical interpolation/integration verified; extrapolated/steep survey regions need engineering review. |
| Hydraulic and temperature features | CONDITIONAL PASS | Strict-cutoff descriptive observed-day features only; no guessed units, unobserved-history dose, or certified Ea/thermal physics. |
| Corrosion-monitoring measurements | CONDITIONAL PASS for preservation/exclusion | No current eligible monitoring feature data; do not use unmatched observations as training covariates or rate truth. |
| between_vtd predictions | CONDITIONAL PASS | Corrected pipeline compatible, M0 unchanged; source and validation limitations remain. |
| cold_start predictions | CONDITIONAL PASS | Direct and indirect diagnostics excluded; nominal baseline and static availability assumptions remain. |

No known confirmed implementation blocker remains in the scoped fixes/checks. The conditional decision reflects unresolved source/scientific assumptions detailed in UNRESOLVED_AMBIGUITIES.md, not weakened failing checks. A strange measurement is not proof of a parser error, and passing these tests is not proof of universal correctness.

## Reproduction and next actions

Run `.venv/bin/python scripts/reproduce_parser_audit.py` from the repository to extract the exact baseline source without switching branches, reproduce baseline/corrected/repeat builds in isolated audit roots, run independent checks, compare M0 and run the full tests. Original before_manifest.json is the immutable reference; do not replace it with post-change hashes. Individual scripts and captured command logs are also retained. Corrected source hashes and complete output schemas are in corrected_run_manifest.json.

Use corrected code/artifact roots for further ML work. Before publishing a replacement root, review the documented delta and select the corrected generation; this audit leaves original published outputs intact. Obtain organizer confirmation of chronology/replacements, UZT reference measurements, monitoring IDs/exposure, engineering units/temperature meaning and defect extent rules. Keep ambiguous fields excluded or explicitly flagged until resolved. Do not treat this audit as evidence to select Ea or as validation of corrosion prediction accuracy.

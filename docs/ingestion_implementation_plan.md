# Ingestion and preprocessing implementation plan

Status: implementation plan, written before code changes on the current working tree.

## 1. Baseline and guardrails

- Repository: `C:\Users\Honor\Desktop\PipeTwin` (git). The separate
  `C:\Users\Honor\Documents\PipeTwin` folder is an evidence/analysis workspace, not
  a second application repository.
- Baseline observed before edits: 62 tests pass; `python -m src.data_prep` reports
  270 checks, 22 CSV/Parquet pairs and 2,000 target rows.
- Dirty state preserved: `processed/target_segments_grid.csv` (contains an
  illustrative `delta_t_c` column from the old temperature script) and untracked
  `scripts/`. No reset, checkout or clean will be run.
- We will publish verification runs to `scratch/` or a temporary directory, never
  over the user's dirty published files.
- The scientific target remains measured **minimum residual wall thickness in mm**
  per 10 m segment on the forecast date. No parser change may turn it into nominal
  thickness, defect depth, a rate, or an unlabelled score.

## 2. Existing components to extend (not replace)

| Existing component | Extension |
|---|---|
| `src/config.py` | Keep legacy keys/public constants. Add semantic source IDs and mark `regime.csv` as the legacy key for **corrosion monitoring**; Techmod remains `TECH_FILE`. |
| `src/format_detector.py` | Keep `parse_numeric`, `robust_parse_dates`, `resolve_sources` behaviour for callers/tests. Add loss-aware table reading, source classification and audit metadata. |
| `src/data_cleaner.py` | Keep rule names and flags. Add unit decisions, row dispositions, monitoring/join audits and field-level status. |
| `src/grid_builder.py` | Keep the half-open 10 m grid and endpoint convention. Add explicit tolerance and length reconciliation diagnostics. |
| `src/segment_mapper.py` | Keep `build_features_asof` and existing output columns. Add derived coverage/eligibility fields, no target leakage. |
| `src/temporal.py` | Keep cutoff-first filtering and three-calendar-day rule. Add window/coverage/age diagnostics. |
| `src/storage.py` | Keep CSV/Parquet round-trip validation. Add failure-safe tree publication and spreadsheet-safe text export. |
| `src/data_prep.py` | Keep CLI and output names. Add inventories, lineage, manifest and template validation. |
| `src/ml/*` | Keep model interfaces and current baselines. Extend metrics/submission helpers without changing model fitting. |
| `scripts/calculate_segment_temperature.py` | Audit and replace the grid mutation with a separate, explicit scenario artifact. |

## 3. New components

1. `src/source_registry.py`
   - Semantic source types: `passport`, `profile`, `vtd`, `uzt`,
     `hydraulic_regime`, `corrosion_monitoring`, `submission_template`.
   - Schema fingerprints independent of filename; configured aliases only as hints.
   - Identical-copy deduplication; conflicting copies fail unless explicitly
     selected. Filename/schema conflicts are reported, not silently resolved.
   - Required/optional source policy per command. Cold start does not require
     target VTD/UZT; missing monitoring is usable with missingness flags.

2. `src/schema_registry.py`
   - Versioned contracts with required/optional fields, aliases, dtypes, expected
     units, nullability, checks and feature eligibility.
   - Full field inventory for every raw column, including archival/unmapped fields.

3. `src/parsing.py`
   - Encoding/delimiter detection with scored candidates.
   - Duplicate raw-header and canonical-collision detection before pandas.
   - Exact identifier parsing (`1250013653.0` accepted only as an exact integer),
     date parsing per value, ambiguous slash-date quarantine, decimal comma /
     thousands / NBSP handling, censored values (`<0.1`) never coerced to zero.
   - Logical CSV record numbers and malformed-record quarantine with reasons.

4. `src/units.py`
   - Explicit conversion policies. Only passport-checked `d_mm` m→mm and the
     documented 2018 defect-depth fraction→percent rule convert. Unverified
     pressure/viscosity/CO2/КВЧ units stay unresolved.

5. `src/quality.py`
   - Disjoint row disposition counts, schema drift, unit report, monitoring/join
     cardinality, coverage and field-level feature inventory.
   - One record may carry several flags but is counted in exactly one disposition.

6. `src/lineage.py`
   - Feature/column → source hashes → rule/config version → record references.
   - No giant per-cell copies; aggregate lineage IDs.

7. `src/leakage.py`
   - Automated cutoff, label-derived, static-passport, cold-start,
     imputation/scaling and parent-group audits.

8. `src/heat_transfer.py` and `src/fluid.py`
   - Strict Reynolds helper (dynamic vs kinematic viscosity explicit).
   - Steady-state heat-loss scenario with continuity, actual segment length,
     positive mass flow and explicit assumption flags. Never auto-merged into the
     verified grid or target features.

9. `src/mentor_ingest.py`
   - Deterministic ingestion of supplied notes/transcripts; source hashes,
     line locators and verbatim excerpts.
   - Stable `DEC/Q/ACT/REQ` IDs, revisions, idempotency, requirement matrix,
     change report, follow-up agenda/questions and implementation impact.
   - Curated accepted records are separate from machine-generated candidates.
     Unknown dates/participants remain unknown; drafts are never sent.

10. `src/cache.py`
    - Cache key from complete dependency hashes: source bytes, config/schema/rules,
      requests/cutoffs and code revision. Optional, off by default.

## 4. Data contracts

- Every source record gets `source_row` (logical record ordinal, 2-based for
  compatibility), `source_record_number` (1-based data record) and a safe physical
  line reference where available.
- Raw values remain immutable. Normalized values, usable masks, flags and issue
  references are separate. CSV exports remain analytical; spreadsheet-safe text is
  a separate report and escapes formula-leading cells without changing analytics.
- Feature state vocabulary: `measured`, `source_calculated`, `derived_verified`,
  `assumed_scenario`, `available_but_excluded`, `unresolved_semantics`,
  `not_provided`.
- A record-level validity flag is never collapsed into a field-level status.
- `regime.csv` remains the legacy filename key for **corrosion monitoring**;
  Techmod is a separate `hydraulic_regime` source. Schema fingerprints decide what
  a file means. A Techmod file named `regime.csv` is reported as a filename/schema
  conflict and is not loaded into the monitoring adapter.

## 5. Pipeline output additions

- `reports/source_inventory.json`
- `reports/schema_mapping.json`
- `reports/feature_inventory.csv`
- `reports/unit_resolution.json`
- `reports/row_reconciliation.json`
- `reports/join_audit.json`
- `reports/leakage_audit.json`
- `reports/template_validation.json`
- `reports/lineage.json`
- `docs/mentor/{source_inventory.json,requirements.csv,decisions.csv,open_questions.csv,follow_up.md,change_report.md,implementation_impact.md}`
- `processed/rejected_records.*`
- `processed/temperature_scenarios.*` only when the scenario script is explicitly run.
- Existing artifact names stay stable. `target_segments_grid.*` remains the verified
  grid without scenario columns; no `delta_t_c` is written there by new runs.

Publication is failure-safe: all artifacts are serialized and checked in a staging
tree, existing files are backed up, files are published, the manifest is written
last, and a failure rolls back the previous published set.

## 6. Phase checkpoints

- **A evidence/contract** — plan, source inventory, field inventory, baseline hashes.
- **B adapters/compatibility** — classification tests, `regime.csv` ambiguity tests,
  old/new layout equivalence, deterministic parsing tests.
- **C quality/alignment** — unit report, grid/measurement tests, row reconciliation,
  monitoring join audit.
- **D as-of features** — cutoff features, coverage indicators, leakage tests,
  isolated thermal scenario.
- **E mentor workflow** — idempotent ingestion, revisions/conflicts, synthetic
  fixture, follow-up drafts.
- **F evaluation/exports** — complete metric definitions, template/key validation,
  UZT priority table, baseline comparisons.
- **G verification/handoff** — full tests, isolated full-data run, publication-failure
  test, README/migration notes, final report.

## 7. Acceptance gates

- Existing 62 tests remain green.
- New required behaviours in prompt §14 have explicit tests.
- Full run on the current archive produces 2,000 target rows only after verified
  lengths/segments; counts are written as observed values, not forced constants.
- Current regression checkpoints (16/20,180/48,040/67/495/39,052/4,668/2,000/251)
  are investigated and reported; changes are explained, not hidden.
- No future row, target-pipe diagnostic, hidden label, global imputation, or
  cross-pipe chemistry transfer can change an earlier-cutoff feature.
- No missing scientific input is replaced by an invented value.

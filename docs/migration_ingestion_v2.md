# Ingestion v2 compatibility and migration

## What stayed stable

- `python -m src.data_prep` still builds the verified grid, target grid, VTD/UZT
  histories, profile features, target features, training targets, quality issues,
  pipes summary and run manifest.
- `build_features_asof(cleaned, grid, profile, vtd_history, uzt_history, requests, config)`
  keeps its signature and returns the same artifact keys plus new diagnostic
  columns.
- `result.csv` remains exactly `id,segment_id,thickness_pred`, with no index or
  diagnostic columns.
- The public VTD label remains the minimum **eligible measured residual wall
  thickness**, not nominal wall, defect depth or corrosion rate.
- The endpoint convention is unchanged: a point exactly at pipe length belongs
  to the final 10 m segment; no extra endpoint segment is created.

## `regime.csv` compatibility map

| Legacy filename | Semantic source ID | Adapter | Meaning |
|---|---|---|---|
| `regime.csv` | `corrosion_monitoring` | monitoring adapter | RepCorr/corrosion/chemistry monitoring |
| `repcorr_south.csv` | `corrosion_monitoring` | monitoring adapter | Same source under organizer name |
| `tech_mode_for_embedings_with_length_with_temperatures_ids16.csv` | `hydraulic_regime` | hydraulic adapter | Techmod daily regime |
| `sample_result.csv` | depends on contents | hydraulic adapter **or** template adapter | Old archive used this name for Techmod; only exact `id,segment_id,thickness_pred` is a submission template |

The old loader contract is intentionally preserved: **`regime.csv` remains an
alias for monitoring, not Techmod**.  Renaming Techmod to `data/regime.csv`
without changing the source means the file is classified as `hydraulic_regime`
by its columns and a filename/schema conflict is reported.  It is not silently
loaded as monitoring.

If two semantically different files claim one role (for example Techmod under
both `regime.csv` and `sample_result.csv`), discovery fails.  Select one with an
explicit mapping in the `source_overrides` argument to `run_pipeline`; there is
no preference for a filename, modification time or numbered download suffix.
Identical copies may be deduplicated with provenance.

## Output additions

New reports are additive:

```text
reports/source_inventory.json
reports/schema_mapping.json
reports/feature_inventory.csv / .parquet
reports/unit_resolution.json
reports/row_reconciliation.json
reports/join_audit.json
reports/date_coverage.json
reports/spatial_coverage.json
reports/vtd_campaigns.json
reports/feature_coverage.json
reports/persistence_filter.json
reports/leakage_audit.json
reports/template_validation.json
reports/lineage.json
reports/data_quality.json
processed/rejected_records.*
processed/lineage_keys.*
```

`processed/target_features.*` gains explicit derived/diagnostic columns:
`age_days`, `d_inner_assumed_mm`, `hydraulic_observed_days`,
`hydraulic_latest_age_days`, `hydraulic_window_start`, `hydraulic_expected_days`,
`hydraulic_coverage_since_start`, `hydraulic_unobserved_days_since_start`,
`water_fraction_mean`, `water_fraction_valid_days`, `prior_vtd_campaign_count`,
`prior_vtd_min_thickness_mm`, `prior_vtd_min_date`.

`history_date_conflict_flag` is **removed from model features**.  It is an
archive-wide diagnostic that can summarize observations outside the cutoff; it is
still retained in the cleaned passport for audit and used internally to mask age.

`target_segments_grid.*` is the verified geometry only.  Temperature output is
written by `scripts/calculate_segment_temperature.py` to
`processed/temperature_scenarios.*`; the old behavior of writing `delta_t_c` into
the target grid is deprecated because it mixed a scenario into the verified grid.

## Intentional behaviour changes

- Dates with an unresolved slash convention are quarantined unless a locale is
  declared.  Dot dates and ISO dates remain accepted per value.
- `1,234` is treated as ambiguous unless a decimal/thousands convention is
  declared; it is not silently converted.  Decimal comma values such as `1,5`
  and Russian day-first dates remain supported.
- Censored values such as `<0.1` remain missing with a `censored` reason, never
  numeric zero.
- Schema drift columns are preserved in the archival layer and reported; they are
  not silently dropped by pandas.
- A failed publication rolls back the previous complete published tree.  The
  manifest is written last.
- `RepositoryData` can audit a separate verified output root, but the root
  repository's dirty `processed/target_segments_grid.csv` is deliberately not
  overwritten by verification runs.

## Verification performed

- Existing and new tests: 91 passed.
- Full archive canonical run: 306 checks, 25 CSV/Parquet pairs, 2,000 target
  segments, leakage audit passed.
- Organizer-name layout (`pot_south.csv`, `altitudes_south.csv`,
  `vtd_south_train.csv`, `uzt_south.csv`, `repcorr_south.csv`, legacy
  `sample_result.csv` as Techmod): same 306 checks and byte-semantic equality of
  `target_features`, grids, histories, profile, training targets and monitoring
  tables against the canonical layout.
- v1 and v2 model predictions on unchanged accepted inputs are identical to the
  previously published results; only additive metric/diagnostic fields changed.

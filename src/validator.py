"""Content checks, not just dimensions. These execute even with python -O."""
import numpy as np
import pandas as pd
from .grid_builder import assign_segments


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def validate_and_summarize(grid, tables, cleaned, requests):
    checks = []
    def check(condition, message):
        require(condition, message)
        checks.append(message)

    passport = cleaned['passport.csv'].set_index('id')
    check(not grid.duplicated(['id','segment_id']).any(), 'grid_unique_keys')
    for pid, part in grid.groupby('id'):
        length = round(float(passport.loc[pid, 'length_km'])*1000, 6)
        check(np.array_equal(part.segment_id, np.arange(len(part))), f'grid_contiguous_ids_{pid}')
        check(np.allclose(part.start_m.iloc[1:],part.end_m.iloc[:-1],rtol=0,atol=1e-8), f'grid_no_gaps_{pid}')
        check(part.start_m.iloc[0]==0 and abs(part.end_m.iloc[-1]-length)<1e-8
              and abs(part.length_m.sum()-length)<1e-7
              and part.length_m.gt(0).all() and part.length_m.le(10).all(), f'grid_lengths_{pid}')

    for name in ('vtd_by_segment','uzt_by_segment'):
        df = tables[name]
        check(not df.duplicated(['id','date','segment_id']).any(), name+'_unique_keys')
        check(df.merge(grid[['id','segment_id']], on=['id','segment_id'], how='left', indicator=True)._merge.eq('both').all(), name+'_valid_keys')
    # Independent target reconciliation against individual measurements in cleaned VTD.
    mapped = assign_segments(cleaned['vtd.csv'], 'distance', grid)
    expected = mapped.groupby(['id','date','segment_id']).residual_for_target_mm.min()
    actual = tables['vtd_by_segment'].set_index(['id','date','segment_id']).min_residual_thickness_mm
    check(expected.index.equals(actual.index) and np.allclose(expected.to_numpy(float),actual.to_numpy(float),equal_nan=True), 'vtd_matches_measured_minimum')

    profile = tables['profile_by_segment']
    check(not profile.duplicated(['id','segment_id']).any() and len(profile)==len(grid), 'profile_complete_unique')
    check(profile[['z_start','z_end','z_mean','slope_m_per_m']].notna().all().all(), 'profile_finite_coverage')
    check(np.allclose(profile.slope_m_per_m,(profile.z_end-profile.z_start)/profile.length_m), 'profile_slope_uses_actual_distance')

    features = tables['target_features']
    check(not features.duplicated(['id','prediction_date','segment_id']).any(), 'features_unique_keys')
    expected_n = sum(int(grid.id.eq(r['id']).sum()) for r in requests)
    check(len(features)==expected_n, 'features_complete_target_grid')
    check(set(features.id)=={r['id'] for r in requests}, 'features_only_requested_ids')
    for field in ('hydraulic_source_date_max','monitoring_source_date_max','last_vtd_date','last_uzt_date'):
        check((features[field].isna() | features[field].lt(features.prediction_date)).all(), field+'_before_prediction')
    cold = features['case'].eq('cold_start')
    check(features.loc[cold,['last_vtd_thickness_mm','last_uzt_thickness_mm']].isna().all().all(), 'cold_start_has_no_diagnostic_history')
    check('pit_corrosion_mean' not in features and 'pit_corrosion_intensity' not in features, 'unverified_corrosion_model_excluded')
    check(tables['hydraulic_asof'].date.lt(tables['hydraulic_asof'].prediction_date).all(), 'all_hydraulic_inputs_before_prediction')
    for name, df in tables.items():
        for col in df.select_dtypes(include='number'):
            check(not np.isinf(df[col].to_numpy(dtype=float,na_value=np.nan)).any(), f'{name}.{col}_no_infinity')

    summary = grid.groupby('id', as_index=False).agg(length_m=('length_m','sum'),total_segments=('segment_id','size'))
    vtd = tables['vtd_by_segment']
    uzt = tables['uzt_by_segment']
    summary['segments_with_measured_vtd'] = summary.id.map(vtd[vtd.thickness_measurement_count.gt(0)].groupby('id').segment_id.nunique()).fillna(0).astype(int)
    summary['segments_with_eligible_vtd'] = summary.id.map(vtd[vtd.target_eligible].groupby('id').segment_id.nunique()).fillna(0).astype(int)
    summary['segments_with_eligible_uzt'] = summary.id.map(uzt[uzt.eligible_measurement_count.gt(0)].groupby('id').segment_id.nunique()).fillna(0).astype(int)
    summary['is_prediction_target'] = summary.id.isin(features.id)
    # Dates matter: operational coverage is reported only for actual target feature snapshots.
    counts = features[features.hydraulic_days_count.gt(0)].groupby('id').segment_id.nunique()
    summary['target_segments_with_regime'] = summary.id.map(counts).astype('Int64')
    return summary, {'passed':True, 'checks_passed':len(checks), 'checks':checks,
                     'target_rows':len(features), 'all_grid_rows':len(grid)}

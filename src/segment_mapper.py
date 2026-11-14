"""Observed thickness histories and date-specific features for complete segment grids."""
import numpy as np
import pandas as pd

from .config import TECH_FILE, STATIC_FEATURES
from .grid_builder import assign_segments
from .temporal import hydraulic_asof, hydraulic_summary


def map_vtd_to_segments(vtd, grid):
    df = assign_segments(vtd, 'distance', grid)
    result = df.groupby(['id', 'date', 'segment_id'], as_index=False).agg(
        observation_count=('source_row','size'),
        thickness_measurement_count=('residual_for_target_mm','count'),
        min_residual_thickness_mm=('residual_for_target_mm','min'),
        max_defect_depth_pct=('defect_depth_pct','max'),
        before_commissioning_flag=('before_commissioning_flag','any'),
        commissioning_unknown_flag=('commissioning_unknown_flag','any'),
        wall_nominal_mismatch_flag=('wall_nominal_mismatch_flag','any'),
        heldout_or_future_flag=('heldout_or_future_flag','any'),
    )
    result['target_eligible'] = (result.thickness_measurement_count.gt(0)
        & ~result.before_commissioning_flag & ~result.commissioning_unknown_flag & ~result.heldout_or_future_flag)
    return result


def map_uzt_to_segments(uzt, grid):
    df = assign_segments(uzt, 'distance', grid)
    df['eligible_thickness_mm'] = df.residual_wall_thickness.where(df.measurement_eligible)
    return df.groupby(['id', 'date', 'segment_id'], as_index=False).agg(
        measurement_count=('residual_wall_thickness','count'),
        min_residual_thickness_mm=('residual_wall_thickness','min'),
        mean_residual_thickness_mm=('residual_wall_thickness','mean'),
        eligible_measurement_count=('eligible_thickness_mm','count'),
        eligible_min_thickness_mm=('eligible_thickness_mm','min'),
        above_nominal_flag=('above_nominal_flag','any'),
    )


def map_profile_to_segments(profile, grid):
    rows = []
    for pid, pipe_grid in grid.groupby('id', sort=True):
        observations = profile[profile.id.eq(pid)].groupby('total_distance', as_index=False).agg(z=('z','mean'))
        observations = observations.sort_values('total_distance')
        if observations.empty:
            raise ValueError(f'No valid elevation profile for {pid}')
        x = observations.total_distance.to_numpy(float)
        z = observations.z.to_numpy(float)
        for segment in pipe_grid.itertuples():
            start, end = segment.start_m, segment.end_m
            inside = x[(x>start)&(x<end)]
            nodes = np.r_[start, inside, end]
            heights = np.interp(nodes, x, z)
            area = np.sum((heights[:-1]+heights[1:])*0.5*np.diff(nodes))
            count = int(((x>=start)&((x<end) | ((end==pipe_grid.end_m.max()) & (x<=end+0.05)))).sum())
            rows.append({
                'id': int(pid), 'segment_id': int(segment.segment_id), 'points_count': count,
                'z_start': float(heights[0]), 'z_end': float(heights[-1]),
                'z_center': float(np.interp((start+end)/2, x, z)),
                'z_mean': float(area/(end-start)), 'z_min': float(heights.min()), 'z_max': float(heights.max()),
                'slope_m_per_m': float((heights[-1]-heights[0])/(end-start)),
                'profile_extrapolated_flag': bool(start<x[0] or end>x[-1]),
                'max_profile_gap_m': float(np.diff(x[max(0,np.searchsorted(x,start)-1):min(len(x),np.searchsorted(x,end)+1)]).max())
                    if len(x[max(0,np.searchsorted(x,start)-1):min(len(x),np.searchsorted(x,end)+1)])>1 else np.nan,
            })
    return grid.merge(pd.DataFrame(rows), on=['id','segment_id'], validate='one_to_one')


def map_monitoring_asof(monitoring, pid, cutoff):
    candidates = monitoring[monitoring.id.eq(pid) & monitoring.sample_removal_date.lt(cutoff)]
    usable = candidates[candidates.observation_eligible]
    row = {'id': pid, 'monitoring_candidate_records': len(candidates),
           'monitoring_records_count': len(usable), 'monitoring_source_date_max': usable.sample_removal_date.max()}
    for col in ('ph','co2_water','total_mineralization'):
        row[col+'_median'] = usable[col].median()
    return pd.DataFrame([row])


def build_features_asof(cleaned, grid, profile, vtd_history, uzt_history, requests, config):
    features, regimes, monitoring_tables, quality, hydraulic_rows = [], [], [], [], []
    for request in requests:
        pid, cutoff = int(request['id']), pd.Timestamp(request['prediction_date'])
        case = request['case']
        if case not in ('between_vtd','cold_start'):
            raise ValueError(f'Unknown scenario: {case}')
        if not grid.id.eq(pid).any():
            raise ValueError(f'Target ID {pid} is not in passport')
        target = profile[profile.id.eq(pid)].copy()
        target['prediction_date'] = cutoff
        target['case'] = case
        # Only explicit static fields. Current diagnostic summaries are never model features.
        static = cleaned['passport.csv'][STATIC_FEATURES+['history_date_conflict_flag']]
        target = target.merge(static, on='id', validate='many_to_one')
        target['age_years'] = (cutoff-target.commissioning_date).dt.total_seconds()/(86400*365.25)
        target.loc[target.age_years.lt(0) | target.history_date_conflict_flag, 'age_years'] = np.nan

        tech, tech_log = hydraulic_asof(cleaned[TECH_FILE], [pid], cutoff, config)
        stats = hydraulic_summary(tech, [pid])
        reg = grid[grid.id.eq(pid)].merge(stats, on='id', validate='many_to_one')
        reg['prediction_date'] = cutoff
        regimes.append(reg)
        quality.append(tech_log)
        hydraulic_rows.append(tech)
        target = target.merge(stats, on='id', validate='many_to_one')

        monitor = map_monitoring_asof(cleaned['regime.csv'], pid, cutoff)
        monitor['prediction_date'] = cutoff
        monitoring_tables.append(grid[grid.id.eq(pid)].merge(monitor, on='id', validate='many_to_one'))
        target = target.merge(monitor.drop(columns='prediction_date'), on='id', validate='many_to_one')

        # Cold start excludes every VTD/UZT observation, even if present in input files.
        vh = vtd_history[vtd_history.id.eq(pid) & vtd_history.date.lt(cutoff) & vtd_history.target_eligible]
        uh = uzt_history[uzt_history.id.eq(pid) & uzt_history.date.lt(cutoff) & uzt_history.eligible_measurement_count.gt(0)]
        if case == 'cold_start':
            vh, uh = vh.iloc[:0], uh.iloc[:0]
        vh = vh.sort_values('date').drop_duplicates(['id','segment_id'], keep='last')
        vh = vh[['id','segment_id','date','min_residual_thickness_mm']].rename(columns={
            'date':'last_vtd_date', 'min_residual_thickness_mm':'last_vtd_thickness_mm'})
        uh = uh.sort_values('date').drop_duplicates(['id','segment_id'], keep='last')
        uh = uh[['id','segment_id','date','eligible_min_thickness_mm']].rename(columns={
            'date':'last_uzt_date', 'eligible_min_thickness_mm':'last_uzt_thickness_mm'})
        target = target.merge(vh, on=['id','segment_id'], how='left', validate='one_to_one')
        target = target.merge(uh, on=['id','segment_id'], how='left', validate='one_to_one')
        target['has_previous_vtd'] = target.last_vtd_thickness_mm.notna()
        target['years_since_vtd'] = (cutoff-target.last_vtd_date).dt.total_seconds()/(86400*365.25)
        # Missing past measurements stay missing; nominal thickness is already a separate feature.
        features.append(target)
    if not features:
        raise ValueError('At least one prediction request is required')
    return {
        'target_features': pd.concat(features, ignore_index=True),
        'regime_mapped': pd.concat(regimes, ignore_index=True),
        'monitoring_mapped': pd.concat(monitoring_tables, ignore_index=True),
        'hydraulic_asof': pd.concat(hydraulic_rows, ignore_index=True),
        'hydraulic_quality': pd.concat(quality, ignore_index=True),
    }

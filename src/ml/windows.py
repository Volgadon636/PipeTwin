"""Historical X/y reconstruction through the existing as-of API."""
import numpy as np
import pandas as pd

from src.segment_mapper import build_features_asof

YEAR_SECONDS = 86400 * 365.25
KEYS = ['id', 'segment_id', 'prediction_date', 'case']


def add_baselines(features):
    """Use measured S1 starts, explicit nearest-prior fallback, nominal S2 starts.

    A fallback copies a nearby *prior* measurement and its timestamp, never a
    target label. It is an assumption about an unmeasured segment, not a reading.
    """
    frame = features.copy()
    frame['h0_mm'] = frame.nominal_thickness_mm
    frame['start_date'] = frame.commissioning_date
    frame['baseline_source'] = 'nominal_at_commissioning'
    frame['baseline_missing'] = False
    frame['baseline_neighbor_segment_id'] = np.nan
    frame['baseline_neighbor_distance_m'] = np.nan
    for _, part in frame.groupby(['id', 'prediction_date', 'case'], sort=True):
        if part['case'].iloc[0] == 'cold_start':
            continue
        known = part[part.last_vtd_thickness_mm.notna()].sort_values('segment_id')
        frame.loc[known.index, 'h0_mm'] = known.last_vtd_thickness_mm
        frame.loc[known.index, 'start_date'] = known.last_vtd_date
        frame.loc[known.index, 'baseline_source'] = 'measured_prior_vtd'
        missing = part.index.difference(known.index)
        frame.loc[missing, 'baseline_missing'] = True
        for index in missing:
            if known.empty:
                frame.loc[index, 'baseline_source'] = 'nominal_no_prior_vtd'
                continue
            distances = (known.start_m - frame.at[index, 'start_m']).abs()
            neighbor = known.loc[distances.idxmin()]  # ties: lower segment_id
            frame.loc[index, ['h0_mm', 'start_date', 'baseline_source',
                              'baseline_neighbor_segment_id', 'baseline_neighbor_distance_m']] = [
                neighbor.last_vtd_thickness_mm, neighbor.last_vtd_date,
                'nearest_prior_vtd_assumption', neighbor.segment_id, distances.min()]
    frame['horizon_years'] = (frame.prediction_date - frame.start_date).dt.total_seconds() / YEAR_SECONDS
    if not np.isfinite(frame[['h0_mm', 'horizon_years']].to_numpy(float)).all():
        raise ValueError('Missing baseline or commissioning date')
    if frame.h0_mm.le(0).any() or frame.horizon_years.le(0).any():
        raise ValueError('Nonpositive initial thickness or forecast horizon')
    return frame


def make_snapshot(data, request):
    cleaned = data.cleaned
    vtd = data.read('processed/vtd_by_segment')
    uzt = data.read('processed/uzt_by_segment')
    if request['case'] == 'cold_start':
        vtd, uzt = vtd.iloc[:0], uzt.iloc[:0]
        # The archive-wide conflict flag is diagnostic-derived. It cannot
        # influence cold-start features, even indirectly through age_years.
        cleaned['passport.csv']['history_date_conflict_flag'] = False
    snapshot = build_features_asof(
        cleaned, data.read('processed/segments_grid'),
        data.read('processed/profile_by_segment'), vtd, uzt, [request], data.config)
    features = snapshot['target_features']
    for column in ('last_vtd_date', 'last_uzt_date', 'hydraulic_source_date_max', 'monitoring_source_date_max'):
        if not (features[column].isna() | features[column].lt(features.prediction_date)).all():
            raise ValueError(f'Future feature: {column}')
    if request['case'] == 'cold_start':
        if features[['last_vtd_thickness_mm', 'last_uzt_thickness_mm']].notna().any().any():
            raise ValueError('Cold-start diagnostic leakage')
    features = add_baselines(features)
    features['main_pipeline_id'] = features.id.map(data.parent_groups())
    return features, snapshot['hydraulic_asof']


def build_training_windows(data):
    """Attach measured labels only after rebuilding each historical snapshot.

    S2 uses each permitted simple section's first eligible inspection. Parent
    pipelines, not the simple-section IDs, define independent validation groups.
    S1 includes campaigns with at least one genuine repeat observation; missing
    segment baselines remain visible but are excluded from rate fitting.
    """
    labels = data.read('processed/training_targets')
    forbidden = {r['id'] for r in data.config['targets'] if r['case'] == 'cold_start'}
    labels = labels[~labels.id.isin(forbidden) & labels.target_eligible].copy()
    first_dates = labels.groupby('id').date.min()
    frames, hydraulic = [], {}
    for (pid, date), truth in labels.groupby(['id', 'date'], sort=True):
        cases = ['cold_start'] if date == first_dates.loc[pid] else ['between_vtd']
        for case in cases:
            request = dict(id=int(pid), prediction_date=str(date.date()), case=case)
            features, rows = make_snapshot(data, request)
            if case == 'between_vtd' and not features.has_previous_vtd.any():
                continue
            measured = truth[['id', 'segment_id', 'min_residual_thickness_mm']]
            window = features.merge(measured, on=['id', 'segment_id'], validate='one_to_one')
            window['campaign_id'] = window.main_pipeline_id.astype(str) + ':' + date.strftime('%Y-%m-%d')
            frames.append(window)
            hydraulic[(int(pid), date)] = rows
    if not frames:
        raise ValueError('No eligible historical windows')
    windows = pd.concat(frames, ignore_index=True)
    if windows.duplicated(KEYS).any():
        raise ValueError('Duplicate historical training keys')
    return windows, hydraulic


def eligible_training(windows, cutoff, case, excluded_parent=None):
    result = windows[windows['case'].eq(case) & windows.prediction_date.lt(pd.Timestamp(cutoff))]
    if excluded_parent is not None:
        result = result[~result.main_pipeline_id.eq(excluded_parent)]
    if case == 'between_vtd':
        result = result[~result.baseline_missing]
    return result.copy()

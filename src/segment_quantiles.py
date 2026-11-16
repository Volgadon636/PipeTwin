"""VTD segment state and retrospective quantile comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .grid_builder import assign_segments

KEYS = ['id', 'segment_id']
QUANTILES = {'rw_min': 0., 'rw_q01': .01, 'rw_q05': .05, 'rw_q10': .10, 'rw_q50': .50}
SIGNALS = [*QUANTILES, 'rw_mean']


def _number(frame: pd.DataFrame, name: str) -> pd.Series:
    values = frame.get(name, pd.Series(np.nan, index=frame.index))
    return pd.to_numeric(values, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)


def extract_segment_state(defects: pd.DataFrame, grid: pd.DataFrame, nominal_mm: float,
                          coordinate: str = 'distance',
                          covered_m: tuple[float, float] | None = None) -> pd.DataFrame:
    """Aggregate one inspected section in its supplied coordinate frame.

    A zero-defect segment inside the inspected footprint receives a nominal
    feature with a censoring flag. Explicit defects with unknown thickness
    remain missing unless their depth supports a separate derived feature.
    """
    if grid.empty or grid.id.nunique() != 1 or grid.segment_id.duplicated().any():
        raise ValueError('Expected a complete grid of one pipe section')
    if not np.isfinite(nominal_mm) or nominal_mm <= 0:
        raise ValueError('Nominal thickness must be finite and positive')
    frame = defects.copy()
    if frame.empty:
        frame = frame.reindex(columns=frame.columns.union(['id', coordinate]))
    x = _number(frame, coordinate)
    if not frame.id.eq(grid.id.iloc[0]).all():
        raise ValueError('VTD rows must belong to the section being aggregated')
    in_bounds = x.between(0, grid.end_m.max())
    frame = assign_segments(frame.loc[in_bounds].assign(_position=x.loc[in_bounds]), '_position', grid)
    observed = _number(frame, 'residual_wall_thickness')
    measured = (_number(frame, 'residual_for_target_mm') if 'residual_for_target_mm' in frame
                else observed.where(observed.gt(0) & observed.le(nominal_mm + .02)))
    depth_mm = _number(frame, 'defect_depth')
    depth_pct = _number(frame, 'defect_depth_pct')
    d = depth_mm.where(depth_mm.between(0, nominal_mm, inclusive='left')).fillna(
        (nominal_mm * depth_pct / 100).where(depth_pct.between(0, 100, inclusive='left')))
    explicit = frame.get('defect_type', pd.Series('', index=frame.index, dtype='string'))
    descriptor = frame.get('trait', pd.Series('', index=frame.index, dtype='string'))
    description = frame.get('feature_description', pd.Series('', index=frame.index, dtype='string'))
    metal_loss = (descriptor.fillna('').str.lower().str.contains('потер')
                  | description.fillna('').str.lower().str.contains('потер'))
    detected = (observed.notna() | measured.notna() | depth_mm.notna() | depth_pct.notna()
                | explicit.fillna('').str.strip().ne('') | metal_loss)
    derived = (nominal_mm - d).where(observed.isna() & measured.isna())
    state = frame.loc[detected, ['segment_id']].assign(
        rw=measured.fillna(derived), measured=measured,
        derived=derived, depth_mm=d.fillna((nominal_mm - measured).clip(lower=0)))
    groups = state.groupby('segment_id', sort=True)
    summary = groups.agg(defect_count=('rw', 'size'),
                         valid_thickness_count=('rw', 'count'),
                         measured_thickness_count=('measured', 'count'),
                         derived_thickness_count=('derived', 'count'),
                         rw_mean=('rw', 'mean'), max_depth_mm=('depth_mm', 'max'))
    quantiles = groups.rw.quantile(list(QUANTILES.values()), interpolation='linear').unstack()
    quantiles = quantiles.reindex(columns=list(QUANTILES.values()))
    quantiles.columns = list(QUANTILES)
    out = grid.set_index('segment_id').join(summary).join(quantiles).sort_index()
    counts = ['defect_count', 'valid_thickness_count', 'measured_thickness_count',
              'derived_thickness_count']
    out[counts] = out[counts].fillna(0).astype('int64')
    lo, hi = covered_m if covered_m is not None else (0., float(grid.end_m.max()))
    out['full_segment_coverage'] = out.start_m.ge(lo - 1e-9) & out.end_m.le(hi + 1e-9)
    out['nominal_fallback'] = out.defect_count.eq(0) & out.full_segment_coverage
    out.loc[out.nominal_fallback, SIGNALS] = nominal_mm
    out.loc[out.nominal_fallback, 'max_depth_mm'] = 0.
    return out.reset_index()


def run_quantile_ablation(truth: pd.DataFrame, candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Score identical measured labels and keys for every signal in each cohort."""
    actual = truth.set_index(KEYS).min_residual_thickness_mm.rename('actual')
    parts = [actual]
    for name, frame in candidates.items():
        columns = list(QUANTILES) + ['nominal_fallback', 'measured_thickness_count',
                                    'derived_thickness_count']
        parts.append(frame.set_index(KEYS)[columns].add_prefix(f'{name}:'))
    same = pd.concat(parts, axis=1, join='inner')
    same = same.dropna(subset=['actual', *[f'{name}:{signal}' for name in candidates
                                         for signal in QUANTILES]])
    measured = pd.Series(True, index=same.index)
    for name in candidates:
        measured &= (same[f'{name}:measured_thickness_count'].gt(0)
                     & same[f'{name}:derived_thickness_count'].eq(0))
    results = []
    for cohort, subset in [('all_available', same), ('measured_only', same.loc[measured])]:
        y = subset.actual
        for name in candidates:
            for signal, q in QUANTILES.items():
                x = subset[f'{name}:{signal}']
                r = x.corr(y) if len(y) >= 3 and x.nunique() > 1 and y.nunique() > 1 else None
                results.append({
                    'cohort': cohort, 'coordinate_frame': name, 'signal': signal,
                    'quantile': q, 'n': len(y), 'reference_measured_n': len(actual),
                    'nominal_segments': int(subset[f'{name}:nominal_fallback'].sum()),
                    'derived_segments': int(subset[f'{name}:derived_thickness_count'].gt(0).sum()),
                    'pearson_r': r, 'pearson_r2': None if r is None else r * r,
                    'mae_mm': float((y - x).abs().mean()) if len(y) else None,
                })
    return pd.DataFrame(results)


def prepare_quantile_outputs(cleaned: dict, grid: pd.DataFrame, measured: pd.DataFrame,
                             requests: list, aligned_tables: dict,
                             alignment_report: dict | None) -> dict[str, pd.DataFrame]:
    """Publish diagnostic states and strictly earlier VTD features for S1."""
    vtd = cleaned['vtd.csv']
    valid = vtd.loc[~(vtd.before_commissioning_flag | vtd.commissioning_unknown_flag
                      | vtd.heldout_or_future_flag)]
    nominal = cleaned['passport.csv'].set_index('id').nominal_thickness_mm
    grids = {pid: rows for pid, rows in grid.groupby('id', sort=False)}
    states = {(pid, day): extract_segment_state(run, grids[pid], nominal.loc[pid]).assign(date=day)
              for (pid, day), run in valid.groupby(['id', 'date'], sort=True)}
    empty_state = extract_segment_state(valid.iloc[:0], next(iter(grids.values())),
                                        nominal.iloc[0]).iloc[:0].assign(date=pd.NaT)
    result = {'vtd_segment_state': pd.concat(states.values(), ignore_index=True) if states else empty_state}

    aligned = {}
    if alignment_report is not None:
        points = aligned_tables['vtd_odometer_coordinates']
        cleaned_rows = valid.drop(columns=['distance']).set_index(['id', 'date', 'source_row'])
        for pair in alignment_report['pairs']:
            pid = pair['id']
            source = pd.Timestamp(pair['alignment_source_date'])
            reference = pd.Timestamp(pair['alignment_reference_date'])
            if pair['method'] == 'unresolved' or (pid, source) not in states:
                continue
            subset = points[points.id.eq(pid) & points.alignment_source_date.eq(source)
                            & points.alignment_reference_date.eq(reference)]
            rows = subset.join(cleaned_rows, on=['id', 'date', 'source_row'], how='inner',
                               validate='one_to_one')
            aligned[pid, source, reference] = extract_segment_state(
                rows, grids[pid], nominal.loc[pid], coordinate='distance_aligned',
                covered_m=tuple(pair['aligned_source_bounds_m'])).assign(
                    date=source, alignment_reference_date=reference)
    if aligned:
        result['vtd_aligned_segment_state'] = pd.concat(aligned.values(), ignore_index=True)

    ablations = []
    observed = measured[measured.target_eligible].groupby(['id', 'date'], sort=False)
    for pid, group in result['vtd_segment_state'].groupby('id', sort=True):
        dates = sorted(group.date.unique())
        for source, reference in zip(dates, dates[1:]):
            key = (pid, reference)
            if key not in observed.groups:
                continue
            candidates = {'distance': states[pid, source]}
            if (pid, source, reference) in aligned:
                candidates['distance_aligned'] = aligned[pid, source, reference]
            ablations.append(run_quantile_ablation(observed.get_group(key), candidates).assign(
                id=pid, source_date=source, reference_date=reference))
    if ablations:
        result['vtd_quantile_ablation'] = pd.concat(ablations, ignore_index=True)

    features = []
    state_columns = [*SIGNALS, 'defect_count', 'valid_thickness_count', 'measured_thickness_count',
                     'derived_thickness_count', 'max_depth_mm', 'nominal_fallback']
    for request in requests:
        if request['case'] != 'between_vtd':
            continue
        pid, cutoff = int(request['id']), pd.Timestamp(request['prediction_date'])
        dates = [day for section, day in states if section == pid and day < cutoff]
        source = max(dates) if dates else pd.NaT
        history = states[pid, source] if dates else empty_state
        frame = grids[pid][KEYS].merge(history[KEYS + state_columns], on=KEYS,
                                      how='left', validate='one_to_one')
        features.append(frame.assign(prediction_date=cutoff, source_vtd_date=source))
    if features:
        result['target_features_vtd_quantiles'] = pd.concat(features, ignore_index=True)
    return result

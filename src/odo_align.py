"""Align ILI odometers before binning on the physical segment grid."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .grid_builder import assign_segments


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column].astype('string').str.replace(',', '.', regex=False),
                         errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)


class OdometerAligner:
    """Fit a monotone, piecewise-linear map for two runs of the same section.

    By default uses the local Distance field. Callers must select the correct
    reference frame and check its availability before using the map in a forecast.
    Beyond the anchors the nearest estimated shift is held constant.
    """

    def __init__(self, coordinate: str = 'distance', length_tolerance_m: float = 0.5,
                 max_shift_m: float = 20.0, step_m: float = 0.2,
                 window_m: float = 100.0, method: str = 'auto') -> None:
        if not (0.05 <= step_m <= 0.25 and 0 < max_shift_m < window_m / 2
                and length_tolerance_m > 0):
            raise ValueError('Invalid odometer alignment parameters')
        if method not in ('auto', 'density'):
            raise ValueError('Odometer method must be auto or density')
        self.coordinate = coordinate
        self.length_tolerance_m = length_tolerance_m
        self.max_shift_m = max_shift_m
        self.step_m = step_m
        self.window_m = window_m
        self.method = method

    def _joints(self, frame: pd.DataFrame) -> pd.DataFrame:
        joint = frame.get('section_number', pd.Series(pd.NA, index=frame.index, dtype='string'))
        joint = joint.astype('string').str.strip().str.replace(r'(?<=\d)\.0+$', '', regex=True)
        relative = _numeric(frame, 'relative_distance')
        start = _numeric(frame, self.coordinate) - relative.where(relative.ge(0))
        length = _numeric(frame, 'pipe_length')
        rows = pd.DataFrame({'joint_id': joint, 'start': start, 'length': length.where(length.gt(0))})
        rows = rows.loc[rows.joint_id.ne('').fillna(False)].dropna(subset=['start'])
        return rows.groupby('joint_id').agg(position=('start', 'median'),
                                              count=('start', 'size'),
                                              scatter=('start', lambda x: (x - x.median()).abs().median()),
                                              joint_length=('length', 'median'))

    def _filter(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if pairs.empty:
            return pairs
        pairs = pairs.sort_values('source_m').copy()
        shift = pairs.shift_m
        deviation = (shift - shift.rolling(5, center=True, min_periods=1).median()).abs()
        pairs = pairs[deviation.le(max(1., 4 * deviation.median()))
                      & shift.abs().le(self.max_shift_m)]
        pairs = pairs.drop_duplicates('source_m', keep=False).drop_duplicates('reference_m', keep=False)
        if (pairs.reference_m.diff().dropna() <= 0).any():
            return pairs.iloc[:0]
        return pairs

    def _density_anchors(self, ref: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
        def defect_positions(frame: pd.DataFrame) -> np.ndarray:
            field = ('residual_for_target_mm' if 'residual_for_target_mm' in frame
                     else 'residual_wall_thickness')
            return _numeric(frame.loc[_numeric(frame, field).notna()]
                            if field in frame else frame, self.coordinate).dropna().to_numpy()

        r, s = defect_positions(ref), defect_positions(src)
        columns = ['source_m', 'reference_m', 'shift_m', 'density_score', 'density_peak_margin']
        if min(len(r), len(s)) < 8:
            return pd.DataFrame(columns=columns)
        start, end = max(0., min(r.min(), s.min())), max(r.max(), s.max())
        points = []
        lag_limit = round(self.max_shift_m / self.step_m)
        smoothing = np.ones(max(3, round(1 / self.step_m)))
        for left in np.arange(start, end, self.window_m / 2):
            right = min(left + self.window_m, end)
            if right - left < self.window_m / 2:
                break
            edges = np.arange(left, right + self.step_m, self.step_m)
            rh, sh = np.histogram(r, edges)[0], np.histogram(s, edges)[0]
            if min(rh.sum(), sh.sum()) < 8:
                continue
            a, b = np.convolve(rh, smoothing, mode='same'), np.convolve(sh, smoothing, mode='same')
            a, b = a - a.mean(), b - b.mean()
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if not norm:
                continue
            lags = np.arange(-len(a) + 1, len(a))
            permitted = np.abs(lags) <= lag_limit
            scores = np.correlate(a, b, mode='full')[permitted] / norm
            lags = lags[permitted]
            best = int(np.argmax(scores))
            alternatives = np.abs(lags - lags[best]) * self.step_m > 1.0
            second = scores[alternatives].max(initial=-1.)
            margin = scores[best] - second
            shift = lags[best] * self.step_m
            if scores[best] >= 0.30 and margin >= 0.04 and abs(shift) < self.max_shift_m:
                center = (left + right) / 2
                points.append((center, center + shift, shift, scores[best], margin))
        return self._filter(pd.DataFrame(points, columns=columns))

    def fit(self, df_ref: pd.DataFrame, df_src: pd.DataFrame) -> OdometerAligner:
        """Fit with welds, with local density correlation as a fallback."""
        if self.coordinate not in df_ref or self.coordinate not in df_src:
            raise ValueError(f'Missing odometer column: {self.coordinate}')
        for field in ('id', 'date'):
            if any(field in frame and frame[field].nunique(dropna=False) > 1
                   for frame in (df_ref, df_src)):
                raise ValueError('Fit expects one section and one campaign per frame')
        if ('id' in df_ref and 'id' in df_src and len(df_ref) and len(df_src)
                and df_ref.id.iloc[0] != df_src.id.iloc[0]):
            raise ValueError('Source and reference must be the same section')
        src, ref = self._joints(df_src), self._joints(df_ref)
        pairs = src.join(ref, how='inner', lsuffix='_source', rsuffix='_reference')
        self.weld_counts_ = {'source': len(src), 'reference': len(ref), 'common_ids': len(pairs)}
        length_delta = (pairs.joint_length_source - pairs.joint_length_reference).abs()
        usable = (pairs.count_source.ge(2) & pairs.count_reference.ge(2)
                  & pairs.scatter_source.le(0.5) & pairs.scatter_reference.le(0.5)
                  & (length_delta.isna() | length_delta.le(self.length_tolerance_m)))
        joints = pairs.loc[usable].rename(columns={'position_source': 'source_m',
                                                  'position_reference': 'reference_m'}).reset_index()
        joints['shift_m'] = joints.reference_m - joints.source_m
        joints = self._filter(joints)
        self.weld_counts_['accepted'] = len(joints)
        use_joints = self.method == 'auto' and len(joints) >= 2
        anchors = joints if use_joints else self._density_anchors(df_ref, df_src)
        self.method_ = 'joints' if use_joints else ('density' if len(anchors) >= 2 else 'unresolved')
        self.anchors_ = anchors if self.method_ != 'unresolved' else anchors.iloc[:0]
        return self

    def transform(self, distances: np.ndarray) -> np.ndarray:
        """Apply the fitted map. Preserve NaN and leave unresolved coordinates in place."""
        if not hasattr(self, 'anchors_'):
            raise ValueError('Fit odometer alignment before transforming distances')
        x = np.asarray(distances, dtype=float)
        if self.anchors_.empty:
            return x.copy()
        return x + np.interp(x, self.anchors_.source_m, self.anchors_.shift_m)

    def fit_transform(self, df_ref: pd.DataFrame, df_src: pd.DataFrame) -> pd.DataFrame:
        """Return source rows with separate aligned coordinates and shifts."""
        self.fit(df_ref, df_src)
        x = _numeric(df_src, self.coordinate)
        aligned = self.transform(x.to_numpy())
        return df_src.assign(distance_aligned=aligned, odometer_shift_m=aligned - x)


def evaluate_alignment_quality(aligner: OdometerAligner, source: pd.DataFrame,
                               reference: pd.DataFrame, grid: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compare defect counts and measured minima on matched physical segment keys."""
    def aggregate(rows: pd.DataFrame, coordinate: str, suffix: str) -> pd.DataFrame:
        x = _numeric(rows, coordinate)
        valid = x.between(0, grid.end_m.max()) & rows.target_eligible
        binned = assign_segments(rows.loc[valid].assign(_position=x.loc[valid]), '_position', grid)
        grouped = binned.groupby('segment_id').residual_for_target_mm
        return grouped.agg(**{f'count_{suffix}': 'size', f'min_mm_{suffix}': 'min'})

    profile = grid[['id', 'segment_id', 'start_m', 'end_m']].set_index('segment_id')
    profile = profile.join(aggregate(source, aligner.coordinate, 'before'))
    profile = profile.join(aggregate(source, 'distance_aligned', 'after'))
    profile = profile.join(aggregate(reference, aligner.coordinate, 'ref'))
    counts = ['count_before', 'count_after', 'count_ref']
    profile[counts] = profile[counts].fillna(0).astype(int)
    metrics = {}
    for field in ('count', 'min_mm'):
        same = profile[[f'{field}_{key}' for key in ('before', 'after', 'ref')]].dropna()
        result = {'matched_segments': len(same)}
        for name in ('before', 'after'):
            x, y = same[f'{field}_{name}'], same[f'{field}_ref']
            r = x.corr(y) if len(same) >= 3 and x.nunique() > 1 and y.nunique() > 1 else None
            result[name] = {'pearson_r': r, 'pearson_r2': None if r is None else r * r}
        metrics[field] = result
    shifts = source.odometer_shift_m.dropna()
    drift = (None if shifts.empty or aligner.method_ == 'unresolved' else
             {'min': float(shifts.min()), 'max': float(shifts.max()),
              'mean': float(shifts.mean()), 'std': float(shifts.std(ddof=0))})
    return {'method': aligner.method_, 'welds': aligner.weld_counts_,
            'anchors_used': len(aligner.anchors_), 'drift_m': drift, 'comparison': metrics}, profile.reset_index()


def prepare_odometer_outputs(cleaned: dict, grid: pd.DataFrame, profile: pd.DataFrame,
                             uzt: pd.DataFrame, requests: list, config: dict) -> tuple[dict, dict]:
    """Publish explicit coordinate-frame artifacts and optional causal snapshots."""
    from .segment_mapper import build_features_asof, map_vtd_to_segments

    vtd = cleaned['vtd.csv']
    segments, profiles, observations, reports = [], [], [], []
    feature_groups = {}
    seen = set()
    request_keys = {(int(r['id']), pd.Timestamp(r['prediction_date'])): r for r in requests}
    plan = config['vtd_odometer_alignment']
    if not isinstance(plan, list):
        raise ValueError('Odometer alignment plan must be a list of run pairs')
    for pair in plan:
        pid = int(pair['id'])
        source_date = pd.Timestamp(pair['source_date'])
        reference_date = pd.Timestamp(pair['reference_date'])
        key = (pid, source_date, reference_date)
        if key in seen or not source_date < reference_date:
            raise ValueError('Alignment pairs must be unique, with source before reference')
        seen.add(key)
        cutoff = pd.Timestamp(pair['feature_cutoff']) if 'feature_cutoff' in pair else None
        if cutoff is not None:
            request = request_keys.get((pid, cutoff))
            if not reference_date < cutoff or request is None or request['case'] != 'between_vtd':
                raise ValueError('Aligned features require a matching S1 request and reference strictly before cutoff')
        source = vtd[vtd.id.eq(pid) & vtd.date.eq(source_date)]
        reference = vtd[vtd.id.eq(pid) & vtd.date.eq(reference_date)]
        section_grid = grid[grid.id.eq(pid)]
        if source.empty or reference.empty or section_grid.empty:
            raise ValueError(f'Missing cleaned VTD run or grid for alignment pair {key}')
        if cutoff is not None and (source.heldout_or_future_flag.any()
                                   or reference.heldout_or_future_flag.any()):
            raise ValueError('Aligned forecast features cannot use a held-out VTD run')
        aligner = OdometerAligner(method=pair.get('method', 'auto'))
        moved = aligner.fit_transform(reference, source)
        in_bounds = moved.distance_aligned.between(0, section_grid.end_m.max())
        diagnostics, rows = evaluate_alignment_quality(aligner, moved, reference, section_grid)
        tags = {'alignment_source_date': source_date, 'alignment_reference_date': reference_date}
        reports.append({'id': pid, **{name: str(day.date()) for name, day in tags.items()},
                        'feature_cutoff': None if cutoff is None else str(cutoff.date()),
                        'out_of_bounds': int((~in_bounds).sum()),
                        'aligned_source_bounds_m': aligner.transform(
                            [0., float(section_grid.end_m.max())]).tolist(),
                        **diagnostics, 'anchor_table': aligner.anchors_.astype(object).where(
                            aligner.anchors_.notna(), None).to_dict('records')})
        profiles.append(rows.assign(**tags))
        observations.append(moved[['id', 'date', 'source_row', 'distance', 'distance_aligned',
                                   'odometer_shift_m']].assign(
                                       inside_grid=in_bounds, method=aligner.method_, **tags))
        if aligner.method_ == 'unresolved':
            if cutoff is not None:
                raise ValueError(f'Could not establish a verified coordinate map for {key}')
            continue
        aligned = map_vtd_to_segments(moved.loc[in_bounds], section_grid, coordinate='distance_aligned')
        segments.append(aligned.assign(**tags, alignment_method=aligner.method_))
        if cutoff is not None:
            group = feature_groups.setdefault((pid, cutoff), (reference_date, []))
            if group[0] != reference_date:
                raise ValueError('One coordinate frame is required per aligned snapshot')
            group[1].append(aligned)
    tables = {'vtd_alignment_profiles': pd.concat(profiles, ignore_index=True),
              'vtd_odometer_coordinates': pd.concat(observations, ignore_index=True)}
    if segments:
        tables['vtd_aligned_by_segment'] = pd.concat(segments, ignore_index=True)
    features = []
    for (pid, cutoff), (reference_date, aligned) in feature_groups.items():
        section_grid = grid[grid.id.eq(pid)]
        reference = vtd[vtd.id.eq(pid) & vtd.date.eq(reference_date)]
        history = pd.concat([*aligned, map_vtd_to_segments(reference, section_grid)])
        if history.duplicated(['id', 'segment_id', 'date']).any() or not history.date.lt(cutoff).all():
            raise ValueError('Aligned snapshot history violates unique keys or cutoff')
        built = build_features_asof(cleaned, grid, profile, history, uzt,
                                    [request_keys[pid, cutoff]], config)['target_features']
        features.append(built.assign(alignment_reference_date=reference_date))
    if features:
        tables['target_features_odo_aligned'] = pd.concat(features, ignore_index=True)
    return tables, {'coordinate': 'Distance', 'pairs': reports,
                    'policy': 'Archive pairs are diagnostic. Only explicit pre-cutoff frames '
                              'can create separate aligned feature snapshots.'}

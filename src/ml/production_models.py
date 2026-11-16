"""Production implementations of H18 (between VTD) and H31 (cold start)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

from src.grid_builder import assign_segments
from .experiment_v2 import merge_spatial, profile_spatial
from .exposure import attach_exposure
from .next_common import fit_ridge, make_weights, predict_ridge, s1_fit_rows, s1_training_table
from .s1_alignment import KEYS, predict_section
from .s1_contract import RIDGE_FEATURES, S1_CASE, S1_CUTOFF, S2_ID
from .s1_joint_alignment import map_coordinates
from .s1_model import fit_s1_ridge, predict_s1_ridge
from .s2_next import fit_spatial_residual
from .targeted_s1 import CORE_FEATURES
from .targeted_s2 import SPATIAL_FEATURES, S2_CUTOFF, _apply_candidate, _fit_physics, _predict_physics
from .windows import make_snapshot

FORECAST_KEYS = [*KEYS, 'prediction_date']
H18_FEATURES = [
    'prior_log_median_length', 'prior_log_median_width',
    'prior_median_minus_min_mm', 'prior_p90_minus_p10_mm',
    'prior_reported_metal_loss_fraction', 'prior_external_fraction',
]
H31_PARAMETERS = {
    'length_fraction': 0.25,
    'feature_length_per_sqrt_dimension': 1.,
    'spatial_variance': 4.,
    'noise_to_linear_variance': 10.,
}


def point_summaries(raw, grid, cutoff, source_date, anchors=None, *, anchor_extrapolation='reject'):
    """Use eligible measurements from one earlier inspection of one section."""
    (pid,) = grid.id.unique()
    if pid == S2_ID or pd.isna(source_date) or not source_date < cutoff:
        raise ValueError('H18 requires an earlier own inspection outside the cold-start pipe')
    selected = raw.loc[raw.id.eq(pid) & raw.date.eq(source_date) & raw.target_eligible
                       & raw.residual_for_target_mm.notna()].copy()
    if anchors is not None:
        # Require coverage of the full grid and of every mapped input point.
        if anchor_extrapolation == 'reject':
            map_coordinates([float(grid.start_m.min()), float(grid.end_m.max())], anchors, outside='reject')
        selected['point_m'] = map_coordinates(selected.distance, anchors, outside=anchor_extrapolation)
    else:
        selected['point_m'] = selected.distance
    inside = selected.point_m.between(0., float(grid.end_m.max()))
    points = assign_segments(selected.loc[inside], 'point_m', grid)
    description = points.feature_description.fillna('').str.strip().str.casefold()
    side = points.defect_type.fillna('').str.strip().str.casefold()
    points = points.assign(
        length_log=np.log1p(points.defect_length.where(points.defect_length.gt(0))),
        width_log=np.log1p(points.defect_width.where(points.defect_width.gt(0))),
        metal_loss=description.str.startswith(('потеря металла', 'корроз')),
        external=side.str.startswith('внеш').where(side.ne('')),
    )
    groups = points.groupby(KEYS)
    thickness = groups.residual_for_target_mm
    stats = groups.agg(
        prior_direct_mm=('residual_for_target_mm', 'min'),
        prior_log_median_length=('length_log', 'median'),
        prior_log_median_width=('width_log', 'median'),
        prior_reported_metal_loss_fraction=('metal_loss', 'mean'),
        prior_external_fraction=('external', 'mean'),
    ).assign(
        prior_p05_mm=thickness.quantile(.05),
        prior_median_minus_min_mm=thickness.median() - thickness.min(),
        prior_p90_minus_p10_mm=thickness.quantile(.9) - thickness.quantile(.1),
    )
    return stats.reset_index(), int((~inside).sum())


def s1_training(data, raw, grid):
    fit = s1_fit_rows(s1_training_table(data))
    if not fit.prediction_date.lt(S1_CUTOFF).all() or fit.id.eq(S2_ID).any():
        raise ValueError('Invalid S1 training history')
    parts = []
    for (pid, cutoff, source), labels in fit.groupby(['id', 'prediction_date', 'last_vtd_date']):
        section = grid.loc[grid.id.eq(pid)]
        points, _ = point_summaries(raw, section, cutoff, source)
        part = labels.merge(points, on=KEYS, how='left', validate='one_to_one')
        np.testing.assert_allclose(part.prior_direct_mm, part.last_mm, atol=1e-9, rtol=0.)
        parts.append(part)
    history = pd.concat(parts, ignore_index=True)
    if len(history) != len(fit) or history.duplicated(FORECAST_KEYS).any():
        raise ValueError('H18 changed the eligible historical key set')
    return history


def fit_s1_models(train):
    if (train.empty or not train.case.eq(S1_CASE).all() or train.id.eq(S2_ID).any()
            or not train.prediction_date.lt(S1_CUTOFF).all()
            or not train.last_vtd_date.lt(train.prediction_date).all()):
        raise ValueError('H18 requires S1 training labels before cutoff and strictly earlier own inspections')
    return {
        'baseline': fit_ridge(train, CORE_FEATURES, 'target_delta_mm', make_weights(train, 'row_uniform'),
                              alpha=10., baseline='last_mm', model_name='global_ridge_geometry'),
        's1c': fit_s1_ridge(train),
        'H18': fit_s1_ridge(train, feature_order=RIDGE_FEATURES + H18_FEATURES,
                            model_name='H18_geometry_type_distribution'),
    }


def s1_prediction_frame(frame, raw, grid, anchors=None, *, anchor_extrapolation='reject'):
    expected = pd.MultiIndex.from_frame(grid[KEYS])
    actual = pd.MultiIndex.from_frame(frame[KEYS])
    if (frame.empty or grid.empty or grid.id.nunique() != 1
            or frame.prediction_date.nunique() != 1 or actual.has_duplicates
            or expected.has_duplicates or len(actual) != len(expected)
            or len(expected.difference(actual)) or len(actual.difference(expected))):
        raise ValueError('H18 requires the complete section grid for profile normalization')
    (cutoff,) = frame.prediction_date.unique()
    (pid,) = frame.id.unique()
    if pid == S2_ID or not frame.case.eq(S1_CASE).all():
        raise ValueError('H18 is only available for S1 between inspections')
    past = raw.loc[raw.id.eq(pid) & raw.date.lt(cutoff) & raw.target_eligible
                   & raw.residual_for_target_mm.notna()]
    source = past.date.max()
    points, dropped = point_summaries(past, grid, cutoff, source, anchors,
                                      anchor_extrapolation=anchor_extrapolation)
    fallback, _ = predict_section(frame, past, grid, cutoff, 0.)
    enriched = (frame.merge(points, on=KEYS, how='left', validate='one_to_one')
                .merge(fallback[KEYS + ['thickness_pred']].rename(columns={'thickness_pred': 'prior_fallback_mm'}),
                       on=KEYS, validate='one_to_one'))
    direct = enriched.prior_direct_mm.notna()
    minimum = enriched.prior_direct_mm.fillna(enriched.prior_fallback_mm)
    quantile = enriched.prior_p05_mm.fillna(enriched.prior_fallback_mm)
    width = quantile.std(ddof=0)
    adjusted = (minimum.mean() + (quantile - quantile.mean()) * minimum.std(ddof=0) / width
                if width > 0 else minimum)
    base = ((minimum + adjusted) / 2.).clip(lower=0., upper=enriched.nominal_thickness_mm)
    return enriched.assign(joint_blend=base, direct_h18_measurement=direct, prior_source_date=source), {
        'prior_source_date': str(source.date()),
        'coordinate_source': 'verified_anchors' if anchors is not None else 'published_distance',
        'direct_rows': int(direct.sum()), 'fallback_rows': int((~direct).sum()),
        'observations_outside_grid': dropped,
    }


def predict_s1_models(models, frame):
    if frame.empty or frame.case.ne(S1_CASE).any() or frame.id.eq(S2_ID).any():
        raise ValueError('H18 is only available for S1 between inspections')
    own = frame.has_own_prior_vtd.to_numpy(bool)
    baseline = frame.s1b_prediction.to_numpy(float).copy()
    if own.any():
        baseline[own] = predict_ridge(models['baseline'], frame.loc[own])
    correction = (predict_s1_ridge(models['H18'], frame).ridge_delta_mm
                  - predict_s1_ridge(models['s1c'], frame).ridge_delta_mm)
    correction = np.where(frame.direct_h18_measurement, correction, 0.)
    candidate = np.clip(frame.joint_blend.to_numpy(float) + correction, 0., frame.nominal_thickness_mm)
    return frame[FORECAST_KEYS].assign(
        baseline=baseline, joint_blend=frame.joint_blend, H18=candidate,
        h18_correction_mm=correction, h18_direct=frame.direct_h18_measurement,
    )


def cold_grid(data, pid, date, spatial):
    snapshot, hydraulic = make_snapshot(
        data, {'id': int(pid), 'prediction_date': str(date.date()), 'case': 'cold_start'})
    full, exposure = attach_exposure(snapshot, {(int(pid), date): hydraulic})
    return merge_spatial(full, spatial).assign(
        forecast_grid_rows=len(full), forecast_grid_extent_m=float(full.end_m.max())), exposure[0]


def cold_training(data, assessment=False):
    """Only first eligible labels before S2 cutoff enter production training."""
    measured = data.read('processed/training_targets')
    labels = measured.loc[measured.target_eligible & measured.id.ne(S2_ID)].copy()
    labels = labels.loc[labels.date.eq(labels.groupby('id').date.transform('min'))]
    if not assessment:
        labels = labels.loc[labels.date.lt(S2_CUTOFF)]
    labels = labels.rename(columns={'min_residual_thickness_mm': 'next_mm'})
    spatial = profile_spatial(data.read('processed/profile_by_segment'))
    grids, intervals, parts = {}, {}, []
    for (pid, date), part in labels.groupby(['id', 'date'], sort=True):
        grids[pid, date], intervals[pid, date] = cold_grid(data, pid, date, spatial)
        parts.append(grids[pid, date].merge(part[KEYS + ['next_mm']], on=KEYS, validate='one_to_one'))
    fit = pd.concat(parts, ignore_index=True)
    fit['campaign_id'] = fit.main_pipeline_id.astype(str) + ':' + fit.prediction_date.dt.strftime('%Y-%m-%d')
    if len(fit) != len(labels) or fit.duplicated(FORECAST_KEYS).any():
        raise ValueError('Cold-start training label join changed eligible keys')
    return fit, grids, intervals


def gp_covariance(x, coordinate, train_x, train_coordinate, config):
    distance = np.sqrt(3.) * np.abs(coordinate[:, None] - train_coordinate) / config['length_fraction']
    scale = x.shape[1] * config['feature_length_per_sqrt_dimension'] ** 2
    feature_distance = cdist(x, train_x, metric='sqeuclidean') / scale
    spatial = (1. + distance) * np.exp(-distance - .5 * feature_distance)
    return x @ train_x.T + config['spatial_variance'] * spatial


def fit_s2_models(train, intervals, config, features=SPATIAL_FEATURES):
    if not train.prediction_date.lt(S2_CUTOFF).all() or train.id.eq(S2_ID).any():
        raise ValueError('H31 can train only on other sections with labels before the S2 cutoff')
    if not train.case.eq('cold_start').all():
        raise ValueError('H31 requires cold-start training rows')
    physics = _fit_physics(train, intervals)
    level = _predict_physics(physics, train, intervals)
    baseline = fit_spatial_residual(train, level, features)
    design = baseline['preprocessor'].transform(train)
    coordinate = train.position.to_numpy(float)
    residual = train.next_mm.to_numpy(float) - level
    groups = [train.id.to_numpy(), train.prediction_date.to_numpy()]
    centered = residual - pd.Series(residual).groupby(groups).transform('mean').to_numpy(float)
    mean = float(centered.mean())
    covariance = gp_covariance(design, coordinate, design, coordinate, config)
    covariance.flat[::len(design) + 1] += config['noise_to_linear_variance']
    dual = cho_solve(cho_factor(covariance, lower=True), centered - mean)
    return {'physics': physics, 'baseline': baseline,
            'H31': {'design': design, 'coordinate': coordinate, 'dual': dual, 'mean': mean,
                    'parameters': dict(config)}}


def predict_s2_models(models, frame, intervals):
    if (frame.id.nunique() != 1 or frame.prediction_date.nunique() != 1
            or not frame.case.eq('cold_start').all()):
        raise ValueError('H31 requires a single cold-start section and forecast date')
    pid, date = frame.id.iloc[0], frame.prediction_date.iloc[0]
    if frame[['last_vtd_thickness_mm', 'last_uzt_thickness_mm']].notna().any().any():
        raise ValueError('H31 does not accept own inspection features')
    ordered = frame.sort_values('start_m')
    starts, ends = ordered.start_m.to_numpy(float), ordered.end_m.to_numpy(float)
    if (not frame.forecast_grid_rows.eq(len(frame)).all()
            or not np.allclose(frame.forecast_grid_extent_m, ends[-1], atol=1e-8, rtol=0.)
            or frame.duplicated(KEYS).any() or frame.length_m.le(0.).any()
            or not np.allclose(ends - starts, ordered.length_m, atol=1e-8, rtol=0.)
            or not np.allclose(ends[:-1], starts[1:], atol=1e-8, rtol=0.)
            or not np.isclose(starts[0], 0.)):
        raise ValueError('H31 requires the complete section grid for spatial centering')
    if pid == S2_ID and date != S2_CUTOFF:
        raise ValueError('Unexpected date for the target S2 cold-start forecast')
    physics = _predict_physics(models['physics'], frame, intervals)
    base = _apply_candidate(physics, predict_ridge(models['baseline'], frame), frame, True)
    design = models['baseline']['preprocessor'].transform(frame)
    gp = models['H31']
    shape = gp['mean'] + gp_covariance(design, frame.position.to_numpy(float), gp['design'],
                                      gp['coordinate'], gp['parameters']) @ gp['dual']
    candidate = _apply_candidate(physics, shape, frame, True)
    return frame[FORECAST_KEYS].assign(
        baseline=base['h_clipped'], H31=candidate['h_clipped'], h31_centered_mm=candidate['g_centered'],
        h31_clipped=candidate['clipping'],
    )

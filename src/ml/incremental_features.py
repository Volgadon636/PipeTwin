"""Optional historical inspection and static geometry features."""
import numpy as np
import pandas as pd

from src.grid_builder import assign_segments

SUPPORT = ['prior_defect_count_log', 'prior_measured_count_log',
           'prior_low_count', 'prior_min_boundary_m']
DISTRIBUTION = ['prior_min_mm', 'prior_p05_mm', 'prior_p10_mm', 'prior_p05_minus_min_mm',
                'prior_neighbor_mean_mm', 'prior_neighbor_min_mm', 'prior_neighbor_count']
DEFECT_STATS = {'support': SUPPORT, 'full': SUPPORT + DISTRIBUTION}
SCALES_M = (100, 300, 1000)


def prior_inspection_stats(vtd, grid, forecasts):
    """Aggregate only inspections referenced as strictly prior by a forecast."""
    prior = forecasts.loc[forecasts.has_own_prior_vtd,
                          ['id', 'last_vtd_date', 'prediction_date']]
    if not prior.last_vtd_date.lt(prior.prediction_date).all():
        raise ValueError('Expected an own inspection strictly before the forecast date')
    eligible_dates = prior[['id', 'last_vtd_date']].drop_duplicates()
    observations = vtd.merge(eligible_dates, left_on=['id', 'date'],
                             right_on=['id', 'last_vtd_date'], validate='many_to_one')
    lengths = observations.id.map(grid.groupby('id').end_m.max())
    observations = observations[observations.distance.between(0, lengths + 1e-9)]
    observations = assign_segments(observations, 'distance', grid)
    observations = observations.merge(grid[['id', 'segment_id', 'start_m', 'end_m']],
                                      on=['id', 'segment_id'], validate='many_to_one')
    observations['defect'] = observations.defect_type.fillna('').str.strip().ne('')
    observations['thickness'] = observations.residual_for_target_mm.where(
        observations.target_eligible)
    observations['boundary_m'] = np.minimum(
        observations.distance - observations.start_m,
        observations.end_m - observations.distance).clip(lower=0)
    keys = ['id', 'last_vtd_date', 'segment_id']
    groups = observations.groupby(keys, sort=True)
    stats = groups.agg(defect_count=('defect', 'sum'),
                       measured_count=('thickness', 'count'),
                       prior_min_mm=('thickness', 'min')).reset_index()
    values = observations.dropna(subset=['thickness'])
    minima = (values.sort_values(['thickness', 'distance'], kind='stable')
              .drop_duplicates(keys))
    stats = stats.merge(minima[keys + ['boundary_m']], on=keys, how='left',
                        validate='one_to_one').rename(columns={'boundary_m': 'prior_min_boundary_m'})
    for percentile in (5, 10):
        column = f'prior_p{percentile:02d}_mm'
        quantile = groups.thickness.quantile(percentile / 100.0).rename(column)
        stats = stats.merge(quantile.reset_index(), on=keys, how='left', validate='one_to_one')
        stats[column] = stats[column].where(stats.measured_count.ge(2))
    stats['prior_p05_minus_min_mm'] = stats.prior_p05_mm - stats.prior_min_mm
    stats['prior_defect_count_log'] = np.log1p(stats.defect_count)
    stats['prior_measured_count_log'] = np.log1p(stats.measured_count)
    stats['prior_low_count'] = stats.measured_count.lt(2).astype(float)
    for offset in (-1, 1):
        neighbor = stats[keys + ['prior_min_mm']].copy()
        neighbor['segment_id'] -= offset
        stats = stats.merge(neighbor.rename(columns={'prior_min_mm': f'neighbor_{offset}'}),
                            on=keys, how='left', validate='one_to_one')
    neighbors = stats[['neighbor_-1', 'neighbor_1']]
    stats['prior_neighbor_mean_mm'] = neighbors.mean(axis=1)
    stats['prior_neighbor_min_mm'] = neighbors.min(axis=1)
    stats['prior_neighbor_count'] = neighbors.count(axis=1)
    return stats[keys + SUPPORT + DISTRIBUTION]


def attach_defect_stats(frame, stats):
    keys = ['id', 'last_vtd_date', 'segment_id']
    result = frame.join(stats.set_index(keys), on=keys, validate='many_to_one')
    own = result.has_own_prior_vtd
    result.loc[~own, SUPPORT + DISTRIBUTION] = np.nan
    if result.loc[own, 'prior_min_mm'].isna().any() or not np.allclose(
            result.loc[own, 'prior_min_mm'], result.loc[own, 'last_mm'], rtol=0, atol=1e-9):
        raise ValueError('Historical stats disagree with the original prior measured minimum')
    return result


def elevation_columns(scales):
    return [f'{name}_{width}m' for width in scales
            for name in ('elevation_smooth', 'elevation_relative', 'elevation_slope')]


def multiscale_elevation(profile, scales):
    """Use physical windows and segment overlap as the elevation weight."""
    parts = []
    for _, section in profile.groupby('id', sort=True):
        section = section.sort_values('start_m')
        start = section.start_m.to_numpy(float)
        end = section.end_m.to_numpy(float)
        centers = (start + end) / 2
        elevation = section.z_center.to_numpy(float)
        output = section[['id', 'segment_id']].copy()
        for width in scales:
            smooth = np.full(len(section), np.nan)
            slope = np.full(len(section), np.nan)
            left = centers - width / 2
            right = centers + width / 2
            first = np.searchsorted(end, left, side='right')
            last = np.searchsorted(start, right, side='left')
            for index, (lo, hi) in enumerate(zip(first, last)):
                window = slice(lo, hi)
                weights = (np.minimum(end[window], right[index])
                           - np.maximum(start[window], left[index]))
                valid = np.isfinite(elevation[window]) & (weights > 0)
                if valid.any():
                    weight = weights[valid]
                    z = elevation[window][valid]
                    x = centers[window][valid]
                    smooth[index] = np.average(z, weights=weight)
                    dx = x - np.average(x, weights=weight)
                    denominator = np.dot(weight, dx ** 2)
                    if denominator > 0:
                        slope[index] = np.dot(weight * dx, z - smooth[index]) / denominator
            output[f'elevation_smooth_{width}m'] = smooth
            output[f'elevation_relative_{width}m'] = elevation - smooth
            output[f'elevation_slope_{width}m'] = slope
        parts.append(output)
    return pd.concat(parts, ignore_index=True)

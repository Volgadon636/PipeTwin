"""Shared as-of contract for S1A, S1B and S1C.

The contract is deliberately built on the existing ``build_features_asof`` /
``make_snapshot`` pathway.  For every historical answer date ``T`` the features
are reconstructed from eligible VTD rows with ``date < T`` only.  The production
S1 request uses the same code with ``T = 2023-12-01``.

Baseline definitions
--------------------
S1A
    Last eligible own prior VTD thickness.  If the segment has no own prior
    measurement, the existing repository fallback is used: the last eligible
    prior VTD of the nearest measured segment (same pipe only, lower
    ``segment_id`` on an exact distance tie).  If the whole pipe has no prior
    VTD, the existing nominal-at-commissioning fallback is retained.
S1B
    Minimum over all eligible own prior VTD campaigns.  On an exact minimum tie
    the earliest source date wins.  If there is no own prior measurement the
    S1A fallback value is reused unchanged, so the S1B-vs-S1A comparison is
    isolated to the new minimum rule.  No extra velocity term is applied.

S1C uses ``s1b_prediction`` as its baseline ``b`` and learns
``measured_thickness - b``.  Rows without an own prior measurement are excluded
from the Ridge fit and keep the S1B fallback explicitly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .windows import YEAR_SECONDS, build_training_windows, make_snapshot
from .spatial import spatial_features

S1_ID = 1250022871
S2_ID = 1250009583
S1_CUTOFF = pd.Timestamp('2023-12-01')
S1_CASE = 'between_vtd'
TARGET_COLUMN = 'min_residual_thickness_mm'
NEIGHBOR_RADIUS_M = 20.0
CONTRACT_VERSION = 's1abc-v1'

PROFILE_SOURCE_COLUMNS = ['id', 'segment_id', 'start_m', 'end_m', 'z_center', 'slope_m_per_m']
PROFILE_FEATURES = ['slope_m_per_m', 'abs_slope_m_per_m', 'relative_position', 'local_low_m']

# Declared order.  Constants and all-empty columns are handled at fit time by
# the preprocessor; this order is stored in the artifact and checked at predict.
RIDGE_FEATURES = [
    's1b_prediction',
    'last_vtd_thickness_mm',
    'age_last_vtd_years',
    'age_min_vtd_years',
    'prior_vtd_campaign_count',
    'last_minus_min_mm',
    'neighbor_prior_median_mm',
    'neighbor_prior_count',
    'neighbor_missing',
    'slope_m_per_m',
    'abs_slope_m_per_m',
    'relative_position',
    'local_low_m',
]

EXPECTED_S1_SEGMENTS = 525
EXPECTED_S2_SEGMENTS = 1475
EXPECTED_TOTAL_SEGMENTS = 2000


def add_profile_features(profile: pd.DataFrame) -> pd.DataFrame:
    """Compute the small, audited profile block used by S1C.

    Only geometry and elevation are used; no defect angles or individual
    defect matching.  The 21-segment rolling median is the same local relief
    definition already used by ``src.ml.spatial``.
    """
    missing = set(PROFILE_SOURCE_COLUMNS) - set(profile.columns)
    if missing:
        raise ValueError(f'Profile is missing columns: {sorted(missing)}')
    base = profile[PROFILE_SOURCE_COLUMNS].copy()
    features = spatial_features(base)
    # ``spatial_features`` keeps the input ``slope_m_per_m`` column and adds a
    # computed ``slope``.  Drop the input copy before renaming to avoid duplicate
    # column labels.
    features = features.drop(columns=['slope_m_per_m'])
    features = features.rename(columns={
        'slope': 'slope_m_per_m',
        'abs_slope': 'abs_slope_m_per_m',
        'position': 'relative_position',
        'local_low': 'local_low_m',
        'pool_depth': 'pool_depth_m',
    })
    for column in PROFILE_FEATURES:
        if not np.isfinite(features[column].to_numpy(float)).all():
            raise ValueError(f'Nonfinite profile feature: {column}')
    return features[['id', 'segment_id'] + PROFILE_FEATURES + ['pool_depth_m']]


def _latest_prior_history(vtd: pd.DataFrame, pid: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    history = vtd[
        vtd.id.eq(pid)
        & vtd.date.lt(cutoff)
        & vtd.target_eligible.fillna(False)
        & vtd.min_residual_thickness_mm.notna()
    ].copy()
    if history.empty:
        return history
    history['date'] = pd.to_datetime(history.date)
    return (history.sort_values(['segment_id', 'date'])
            .drop_duplicates(['segment_id'], keep='last')
            .reset_index(drop=True))


def neighbor_prior_features(vtd: pd.DataFrame, profile: pd.DataFrame, pid: int,
                            cutoff: pd.Timestamp,
                            radius_m: float = NEIGHBOR_RADIUS_M) -> pd.DataFrame:
    """Median of *latest eligible prior* thickness of nearby same-pipe segments.

    The neighbour value is the latest measurement available strictly before
    ``cutoff``.  Own segment is excluded.  No other pipe and no future row can
    enter.  Radius is measured between segment centres, inclusive.
    """
    target = profile[profile.id.eq(pid)][['id', 'segment_id', 'start_m', 'end_m']].copy()
    if target.empty:
        raise ValueError(f'No profile rows for pipe {pid}')
    target['center_m'] = (target.start_m.astype(float) + target.end_m.astype(float)) / 2.0
    latest = _latest_prior_history(vtd, pid, cutoff)
    if latest.empty:
        target['neighbor_prior_median_mm'] = np.nan
        target['neighbor_prior_count'] = 0
        target['neighbor_missing'] = True
        return target[['id', 'segment_id', 'neighbor_prior_median_mm',
                       'neighbor_prior_count', 'neighbor_missing']]
    positions = (profile[profile.id.eq(pid)][['segment_id', 'start_m', 'end_m']]
                 .assign(center_m=lambda x: (x.start_m.astype(float) + x.end_m.astype(float)) / 2.0)
                 [['segment_id', 'center_m']])
    latest = latest.merge(positions, on='segment_id', how='left', validate='one_to_one')
    if latest.center_m.isna().any():
        raise ValueError('Prior VTD segment is missing profile geometry')
    rows = []
    for row in target.itertuples():
        candidate = latest[
            latest.segment_id.ne(row.segment_id)
            & (latest.center_m - row.center_m).abs().le(radius_m + 1e-9)
        ]
        values = candidate.min_residual_thickness_mm.to_numpy(float)
        rows.append({
            'id': int(pid), 'segment_id': int(row.segment_id),
            'neighbor_prior_median_mm': float(np.median(values)) if len(values) else np.nan,
            'neighbor_prior_count': int(len(values)),
            'neighbor_missing': bool(len(values) == 0),
        })
    return pd.DataFrame(rows)


def enrich_s1(frame: pd.DataFrame, vtd: pd.DataFrame, profile: pd.DataFrame,
              cutoff: pd.Timestamp) -> pd.DataFrame:
    """Attach S1A/S1B baselines, local history and profile features to one snapshot.

    ``frame`` must be one pipe's as-of snapshot built by ``make_snapshot`` at
    ``cutoff``.  The function never reads labels; it only uses the snapshot's
    past-history columns and eligible VTD rows strictly before ``cutoff``.
    """
    cutoff = pd.Timestamp(cutoff)
    required = {
        'id', 'segment_id', 'start_m', 'end_m', 'prediction_date', 'h0_mm',
        'baseline_source', 'baseline_missing', 'last_vtd_thickness_mm',
        'last_vtd_date', 'prior_vtd_min_thickness_mm', 'prior_vtd_min_date',
        'prior_vtd_campaign_count', 'baseline_neighbor_segment_id',
        'baseline_neighbor_distance_m', 'start_date', 'nominal_thickness_mm',
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Snapshot is missing contract columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('Cannot enrich an empty S1 snapshot')
    if frame.id.nunique() != 1:
        raise ValueError('The S1 contract accepts exactly one pipe at a time')
    pid = int(frame.id.iloc[0])
    frame = frame.copy()
    if not np.isfinite(frame[['h0_mm', 'nominal_thickness_mm']].to_numpy(float)).all():
        raise ValueError('Nonfinite baseline or nominal thickness')
    if (frame.h0_mm <= 0).any() or (frame.nominal_thickness_mm <= 0).any():
        raise ValueError('Nonpositive baseline or nominal thickness')
    for column, label in [('last_vtd_date', 'last VTD date'),
                          ('prior_vtd_min_date', 'minimum VTD date')]:
        dates = frame[column]
        if (dates.notna() & dates.ge(cutoff)).any():
            raise ValueError(f'{label} is not strictly before the cutoff')

    own = frame.prior_vtd_min_thickness_mm.notna()
    if not frame.loc[own, 'last_vtd_thickness_mm'].notna().all():
        raise ValueError('Own prior minimum without an own last measurement')
    frame['has_own_prior_vtd'] = own
    frame['s1a_prediction'] = frame.h0_mm.astype(float)
    frame['s1a_source'] = frame.baseline_source.astype('string')
    frame['s1a_missing'] = frame.baseline_missing.fillna(True).astype(bool)
    frame['s1b_prediction'] = np.where(
        own, frame.prior_vtd_min_thickness_mm.astype(float),
        frame.s1a_prediction.to_numpy(float))
    frame['s1b_source'] = np.where(
        own, 'own_prior_vtd_min', frame.s1a_source.to_numpy(object))
    frame['s1b_missing'] = ~own
    frame['baseline_min_mm'] = frame.s1b_prediction
    frame['age_last_vtd_years'] = (
        (cutoff - frame.last_vtd_date).dt.total_seconds() / YEAR_SECONDS)
    frame['age_min_vtd_years'] = (
        (cutoff - frame.prior_vtd_min_date).dt.total_seconds() / YEAR_SECONDS)
    frame['prior_vtd_campaign_count'] = pd.to_numeric(
        frame.prior_vtd_campaign_count, errors='coerce').astype(float)
    frame['last_minus_min_mm'] = (
        frame.last_vtd_thickness_mm.astype(float)
        - frame.prior_vtd_min_thickness_mm.astype(float))

    # Explicit fallback provenance.  Own rows keep the own dates; fallback rows
    # copy the neighbour's last measurement timestamp, never a target label.
    frame['fallback_segment_id'] = np.where(
        ~own, frame.baseline_neighbor_segment_id, np.nan)
    frame['fallback_distance_m'] = np.where(
        ~own, frame.baseline_neighbor_distance_m, np.nan)
    frame['fallback_measurement_date'] = frame.start_date.where(~own)
    frame['fallback_thickness_mm'] = frame.s1a_prediction.where(~own)
    frame['own_last_thickness_mm'] = frame.last_vtd_thickness_mm.where(own)
    frame['own_last_date'] = frame.last_vtd_date.where(own)
    frame['own_min_thickness_mm'] = frame.prior_vtd_min_thickness_mm.where(own)
    frame['own_min_date'] = frame.prior_vtd_min_date.where(own)
    frame['own_min_tie_policy'] = 'earliest_source_date_on_exact_minimum_tie'

    if (frame.loc[own, 's1b_prediction'] > frame.loc[own, 's1a_prediction'] + 1e-9).any():
        raise ValueError('S1B exceeds S1A on a segment with its own prior measurement')
    if not np.isfinite(frame[['s1a_prediction', 's1b_prediction']].to_numpy(float)).all():
        raise ValueError('Nonfinite S1A/S1B baseline')
    if (frame[['s1a_prediction', 's1b_prediction']] <= 0).any().any():
        raise ValueError('Nonpositive S1A/S1B baseline')

    profile_features = add_profile_features(profile[profile.id.eq(pid)])
    neighbor_features = neighbor_prior_features(vtd, profile, pid, cutoff)
    before = len(frame)
    # The as-of snapshot already contains a raw ``slope_m_per_m`` column.  Drop
    # the overlapping profile block before merging the audited computed block so
    # merge suffixes cannot silently rename a model feature.
    overlap = [c for c in PROFILE_FEATURES + ['pool_depth_m'] if c in frame.columns]
    frame = frame.drop(columns=overlap)
    frame = frame.merge(profile_features, on=['id', 'segment_id'],
                        how='left', validate='one_to_one')
    frame = frame.merge(neighbor_features, on=['id', 'segment_id'],
                        how='left', validate='one_to_one')
    if len(frame) != before:
        raise ValueError('Profile neighbourhood merge changed the S1 row count')
    if frame[PROFILE_FEATURES + ['neighbor_prior_median_mm',
                                 'neighbor_prior_count']].isna().all().any():
        # All-NaN is valid only for the neighbour median when no history exists;
        # profile features and counts must be present.
        if frame[PROFILE_FEATURES + ['neighbor_prior_count']].isna().any().any():
            raise ValueError('Missing profile/neighbour features after merge')
    frame['neighbor_prior_count'] = frame.neighbor_prior_count.astype(int)
    frame['neighbor_missing'] = frame.neighbor_missing.astype(bool)
    return frame.reset_index(drop=True)


def target_key_grid(data) -> pd.DataFrame:
    """Published target keys derived from manifest-verified geometry.

    ``processed/target_segments_grid.csv`` in this checkout fails its manifest
    checksum and is deliberately not used.  The canonical ``segments_grid`` and
    ``target_features`` artifacts both pass; the key set is therefore taken from
    geometry and cross-checked against the published feature rows.
    """
    geometry = data.read('processed/segments_grid')
    keys = geometry[geometry.id.isin([S1_ID, S2_ID])][['id', 'segment_id']].copy()
    keys = keys.sort_values(['id', 'segment_id']).reset_index(drop=True)
    counts = keys.groupby('id').size().to_dict()
    if counts.get(S1_ID) != EXPECTED_S1_SEGMENTS:
        raise ValueError(f'Expected {EXPECTED_S1_SEGMENTS} S1 keys from geometry, got {counts.get(S1_ID)}')
    if counts.get(S2_ID) != EXPECTED_S2_SEGMENTS:
        raise ValueError(f'Expected {EXPECTED_S2_SEGMENTS} S2 keys from geometry, got {counts.get(S2_ID)}')
    published = data.read('processed/target_features')[['id', 'segment_id']].copy()
    published = published.sort_values(['id', 'segment_id']).reset_index(drop=True)
    if not keys.equals(published):
        raise ValueError('Geometry keys do not match published target_features keys')
    return keys


def production_frame(data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the production S1 snapshot and its expected key grid."""
    request = {'id': S1_ID, 'prediction_date': str(S1_CUTOFF.date()),
               'case': S1_CASE}
    features, _ = make_snapshot(data, request)
    expected = target_key_grid(data)
    expected = expected[expected.id.eq(S1_ID)][['id', 'segment_id']].copy()
    expected = expected.sort_values(['id', 'segment_id']).reset_index(drop=True)
    if len(expected) != EXPECTED_S1_SEGMENTS:
        raise ValueError(f'Expected {EXPECTED_S1_SEGMENTS} S1 keys, got {len(expected)}')
    actual = features[['id', 'segment_id']].sort_values(
        ['id', 'segment_id']).reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError('Production S1 key set does not match the target grid')
    enriched = enrich_s1(features, data.read('processed/vtd_by_segment'),
                         data.read('processed/profile_by_segment'), S1_CUTOFF)
    enriched = enriched.sort_values(['id', 'segment_id']).reset_index(drop=True)
    if not enriched[['id', 'segment_id']].equals(expected):
        raise ValueError('Enriched S1 key set does not match the target grid')
    return enriched, expected


def build_s1_training_frame(data, cutoff: pd.Timestamp = S1_CUTOFF) -> pd.DataFrame:
    """Rebuild every permissible historical S1 snapshot and attach its label.

    Only labels strictly before ``cutoff`` are kept.  Features for a label at
    ``T`` are rebuilt from VTD rows with ``date < T`` by ``build_training_windows``.
    """
    cutoff = pd.Timestamp(cutoff)
    windows, _ = build_training_windows(data)
    eligible = windows[
        windows['case'].eq(S1_CASE)
        & windows.prediction_date.lt(cutoff)
    ].copy()
    if eligible.empty:
        raise ValueError('No eligible historical S1 windows before the cutoff')
    vtd = data.read('processed/vtd_by_segment')
    profile = data.read('processed/profile_by_segment')
    parts = []
    for (pid, date), part in eligible.groupby(['id', 'prediction_date'], sort=True):
        if pd.Timestamp(date) >= cutoff:
            raise ValueError('Training snapshot is not strictly before the cutoff')
        enriched = enrich_s1(part, vtd, profile, pd.Timestamp(date))
        enriched['training_cutoff'] = cutoff
        parts.append(enriched)
    training = pd.concat(parts, ignore_index=True)
    training['fit_eligible'] = training.has_own_prior_vtd.astype(bool)
    training['target_delta_mm'] = (
        training[TARGET_COLUMN].astype(float) - training.s1b_prediction.astype(float))
    training['target_delta_from_s1a_mm'] = (
        training[TARGET_COLUMN].astype(float) - training.s1a_prediction.astype(float))
    # Explicit leakage guards, not just documentation.
    if not training.prediction_date.lt(cutoff).all():
        raise ValueError('Historical label on/after production cutoff')
    for column in ('last_vtd_date', 'prior_vtd_min_date'):
        if (training[column].notna()
                & training[column].ge(training.prediction_date)).any():
            raise ValueError(f'Feature {column} is not strictly before its label date')
    if training.duplicated(['id', 'segment_id', 'prediction_date']).any():
        raise ValueError('Duplicate historical S1 keys')
    if not np.isfinite(training[['target_delta_mm',
                                 'target_delta_from_s1a_mm']].to_numpy(float)).all():
        raise ValueError('Nonfinite historical S1 target')
    return training


def load_result_source(path: Path, target_grid: pd.DataFrame, pid: int) -> pd.DataFrame:
    """Load and validate one pipe's predictions from an explicit result.csv."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Result source not found: {path}')
    frame = pd.read_csv(path)
    expected_columns = ['id', 'segment_id', 'thickness_pred']
    if list(frame.columns) != expected_columns:
        raise ValueError(f'{path}: expected columns {expected_columns}')
    expected = target_grid[target_grid.id.eq(pid)][['id', 'segment_id']].copy()
    expected = expected.sort_values(['id', 'segment_id']).reset_index(drop=True)
    actual = frame[frame.id.eq(pid)][['id', 'segment_id']].sort_values(
        ['id', 'segment_id']).reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError(f'{path}: key set for {pid} does not match the target grid')
    part = frame[frame.id.eq(pid)].copy()
    if part.thickness_pred.isna().any() or not np.isfinite(
            part.thickness_pred.to_numpy(float)).all():
        raise ValueError(f'{path}: nonfinite prediction for {pid}')
    return part.sort_values(['id', 'segment_id']).reset_index(drop=True)


def load_s2_source(path: Path, target_grid: pd.DataFrame) -> pd.DataFrame:
    """Load the unchanged, verified S2 block; never synthesize missing rows."""
    part = load_result_source(path, target_grid, S2_ID)
    if len(part) != EXPECTED_S2_SEGMENTS:
        raise ValueError(f'Expected {EXPECTED_S2_SEGMENTS} S2 rows, got {len(part)}')
    return part


def load_v2_s1_source(path: Path, target_grid: pd.DataFrame) -> pd.DataFrame:
    part = load_result_source(path, target_grid, S1_ID)
    if len(part) != EXPECTED_S1_SEGMENTS:
        raise ValueError(f'Expected {EXPECTED_S1_SEGMENTS} v2 S1 rows, got {len(part)}')
    return part


def nominal_frame(data) -> pd.DataFrame:
    """Nominal thickness for every published target key, used only for bounds."""
    features = data.read('processed/target_features')
    frame = features[['id', 'segment_id', 'nominal_thickness_mm']].copy()
    if frame.duplicated(['id', 'segment_id']).any():
        raise ValueError('Duplicate target key in nominal thickness table')
    if not np.isfinite(frame.nominal_thickness_mm.to_numpy(float)).all():
        raise ValueError('Nonfinite nominal thickness')
    if (frame.nominal_thickness_mm <= 0).any():
        raise ValueError('Nonpositive nominal thickness')
    return frame


def combine_with_s2(s1: pd.DataFrame, s2: pd.DataFrame,
                    target_grid: pd.DataFrame, bounds: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach the unchanged S2 block and validate all 2,000 submission keys.

    ``bounds`` must contain ``h0_mm`` (positive physical/assumed baseline) and
    an explicit ``upper_mm`` policy.  S1C passes nominal thickness for S1 and
    the original cold-start h0/nominal for S2; S1A/S1B pass h0 for both.
    """
    if list(s1.columns) != ['id', 'segment_id', 'thickness_pred']:
        raise ValueError('S1 frame must have exactly id,segment_id,thickness_pred')
    if list(s2.columns) != ['id', 'segment_id', 'thickness_pred']:
        raise ValueError('S2 frame must have exactly id,segment_id,thickness_pred')
    combined = pd.concat([
        s1[['id', 'segment_id', 'thickness_pred']],
        s2[['id', 'segment_id', 'thickness_pred']],
    ], ignore_index=True)
    if len(combined) != EXPECTED_TOTAL_SEGMENTS:
        raise ValueError(f'Expected {EXPECTED_TOTAL_SEGMENTS} rows, got {len(combined)}')
    combined = combined.sort_values(['id', 'segment_id']).reset_index(drop=True)
    for column in ('h0_mm', 'upper_mm'):
        if column not in bounds:
            raise ValueError(f'Bounds table is missing {column}')
    if (bounds.h0_mm <= 0).any() or (bounds.upper_mm <= 0).any():
        raise ValueError('Nonpositive bound')
    if (bounds.upper_mm < bounds.h0_mm - 1e-9).any():
        raise ValueError('Upper prediction bound is below h0')
    s2_bounds = bounds[bounds.id.eq(S2_ID)]
    if (s2_bounds.upper_mm - s2_bounds.h0_mm).abs().max() > 1e-9:
        raise ValueError('S2 must keep its original cold-start h0 bound')
    from .submission import validate_submission
    validation = validate_submission(combined, target_grid, bounds,
                                     upper_column='upper_mm')
    return combined, validation


def verify_s2_unchanged(path: Path, source: pd.DataFrame) -> dict:
    """Compare the written S2 block with its explicit source; report max diff."""
    written = pd.read_csv(path)
    part = written[written.id.eq(S2_ID)][['id', 'segment_id', 'thickness_pred']]
    part = part.sort_values(['id', 'segment_id']).reset_index(drop=True)
    expected = source.sort_values(['id', 'segment_id']).reset_index(drop=True)
    if not part[['id', 'segment_id']].equals(expected[['id', 'segment_id']]):
        raise ValueError('Written S2 keys differ from the source')
    difference = np.abs(part.thickness_pred.to_numpy(float)
                        - expected.thickness_pred.to_numpy(float))
    return {'rows': int(len(part)), 'max_abs_difference': float(difference.max()),
            'exact_numeric_match': bool(np.all(difference == 0.0)),
            'source_rows': int(len(expected))}

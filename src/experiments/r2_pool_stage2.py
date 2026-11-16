"""Run H06–H10 against the frozen stage 1 benchmarks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.experiments.r2_pool_stage1 import KEY
from src.experiments.r2_pool_stage1_photo import score_comparison
from src.experiments.r2_pool_stage1_strong import freeze_checkout
from src.ml import s1_joint_alignment as joint
from src.ml.data import RepositoryData
from src.ml.next_common import (
    fit_ridge, json_safe, predict_ridge, s1_fit_rows, s1_training_table, versions, write_csv,
)
from src.ml.s1_alignment import predict_section, section_snapshot
from src.ml.s1_contract import S1_CUTOFF, S1_ID, S2_ID, YEAR_SECONDS, add_profile_features
from src.ml.targeted_s2 import S2_CUTOFF
from src.storage import sha256, write_json

PROTOCOL = Path('experiments/r2_pool/protocol_batch_02.json')
PASSPORT = Path('data/passport.csv')
SOURCE = Path('docs/mentor/follow_up.md')
PEER = 'S1_peer_2023'


def inventory(data, raw, mapped):
    requests = {S1_ID: S1_CUTOFF, S2_ID: S2_CUTOFF, **dict.fromkeys(joint.PEERS, S1_CUTOFF)}
    repair_fields = ['repair_method', 'repair_time_years', 'recommended_repair_method',
                     'prerepair_classification']
    repair_counts = {str(pid): raw.loc[raw.id.eq(pid) & raw.date.lt(date), repair_fields]
                     .notna().sum().to_dict() for pid, date in requests.items()}
    passport = pd.read_csv(PASSPORT, low_memory=False)
    repair_columns = [str(name) for name in passport if 'ремонт' in name.casefold()
                      and passport[name].notna().any()]
    calibration_columns = [str(name) for name in passport
                           if any(term in name.casefold() for term in
                                  ('дефектоскоп', 'калибровк', 'точность втк', 'снаряд'))
                           and passport[name].notna().any()]
    by_campaign = mapped.groupby('date', sort=True).agg(
        rows=('id', 'size'), sections=('id', 'nunique'), eligible=('target_eligible', 'sum'),
        before_commissioning=('before_commissioning_flag', 'sum')).reset_index()
    by_campaign['date'] = by_campaign.date.dt.date.astype(str)
    strict = mapped[mapped.date.lt(S2_CUTOFF) & mapped.target_eligible & ~mapped.id.eq(S2_ID)]
    s2_parents = strict.id.map(data.parent_groups())
    numbers = raw.groupby('date', sort=True).inspection_number.count()
    return {
        'H06': {'vtd_nonempty_before_cutoff': repair_counts,
                'passport_populated_repair_fields': repair_columns,
                'verified_dated_spatial_repair_records': 0, 'open_question': 'Q-0007'},
        'H07': {'campaigns': by_campaign.to_dict('records'),
                's2_strict_rows': len(strict), 's2_strict_parent_count': int(s2_parents.nunique()),
                'verified_new_campaigns': 0, 'open_question': 'Q-0011'},
        'H08': {'inspection_number_nonempty_by_date': {str(d.date()): int(n) for d, n in numbers.items()},
                'passport_populated_quality_fields': calibration_columns,
                'verified_quality_weights': 0, 'open_question': 'Q-0011'},
    }


def peer_inputs(data, raw, archive):
    """Rebuild the original folds and replay the joint forecast on their full grids."""
    grid = data.read('processed/segments_grid')
    profile = data.read('processed/profile_by_segment')
    anchors = joint.joint_anchors(raw, joint.PRIOR_DATE, S1_CUTOFF, joint.PEERS)
    folds = pd.read_csv(archive / 'split_keys.csv', parse_dates=['prediction_date'])
    if set(folds.role) != {'test'} or not folds.prediction_date.eq(S1_CUTOFF).all():
        raise ValueError('Wrong frozen S1 split')
    fold_specs = json.loads((archive / 'anchor_folds.json').read_bytes())
    prepared = {}
    for pid in joint.PEERS:
        section = grid[grid.id.eq(pid)].sort_values('segment_id')
        fallback, _ = predict_section(section_snapshot(data, pid), raw, section, S1_CUTOFF, 0)
        terrain = add_profile_features(profile[profile.id.eq(pid)])
        prepared[pid] = (section, fallback, terrain)
    parts = []
    for fold in fold_specs:
        pid = fold['id']
        fold_name = f"{pid}:{fold['fold']}"
        held = folds.loc[folds.fold.eq(fold_name), KEY + ['fold', 'supported']]
        subset = joint.heldout_anchors(anchors[anchors.id.eq(pid)],
                                       fold['lo_with_buffer_m'], fold['hi_with_buffer_m'])
        if len(subset) != fold['anchors_used'] or not held.supported.eq(fold['supported']).all():
            raise ValueError(f'Anchor support changed in {fold_name}')
        if not fold['supported']:
            continue
        section, fallback, terrain = prepared[pid]
        forecast, _ = joint.predict_aligned(raw, section, fallback, subset)
        columns = joint.KEYS + ['joint_blend', 'joint_min', 'aligned_measurement', 'nominal_thickness_mm']
        part = held.merge(forecast[columns], on=joint.KEYS, how='left', validate='one_to_one')
        parts.append(part.merge(terrain, on=joint.KEYS, how='left', validate='one_to_one'))
    features = pd.concat(parts, ignore_index=True)
    features['prediction_date'] = features.prediction_date.astype('datetime64[ns]')
    return features


def history_mix(features, mapped, cfg):
    older, latest = (pd.Timestamp(cfg[key]) for key in ('older_campaign', 'latest_campaign'))
    if not (older < latest == joint.PRIOR_DATE and features.prediction_date.gt(latest).all()):
        raise ValueError('The historical campaigns must both precede evaluation')
    historical = mapped.loc[mapped.date.eq(older) & mapped.target_eligible,
                            joint.KEYS + ['min_residual_thickness_mm']].rename(
                                columns={'min_residual_thickness_mm': 'older_mm'})
    pair = features.merge(historical, on=joint.KEYS, how='left', validate='one_to_one')
    both = pair.aligned_measurement & pair.older_mm.notna()
    minimum = np.minimum(pair.older_mm, pair.joint_min)
    adjustment = ((1 - cfg['last_weight']) * (minimum - pair.joint_min)).where(both, 0.)
    return pair[KEY + ['fold', 'joint_min', 'older_mm']].assign(
        two_direct_campaigns=both, delta_mm=adjustment,
        prediction=(pair.joint_blend + adjustment).clip(0., upper=pair.nominal_thickness_mm))


def train_rates(training, cutoff, held_id, cfg):
    train = training.loc[training.prediction_date.lt(cutoff) & training.id.ne(held_id)].copy()
    if (train.empty or not train.last_vtd_date.lt(train.prediction_date).all()
            or not np.isfinite(train.rate_mm_year).all()):
        raise ValueError('Expected eligible historical rate pairs from other sections')
    median = float(train.rate_mm_year.median())
    low, high = train.rate_mm_year.quantile(cfg['clip_quantiles']).tolist()
    train['robust_rate'] = train.rate_mm_year.clip(low, high)
    model = fit_ridge(train, cfg['features'], 'robust_rate', np.ones(len(train)),
                      cfg['ridge_alpha'], model_name='H10')
    return model, median, (low, high), train


def shrink_rates(features, training, cfg):
    parts, fit_keys, model_metadata = [], [], []
    for (pid, cutoff), frame in features.groupby(['id', 'prediction_date'], sort=True):
        model, median, (low, high), train = train_rates(training, cutoff, pid, cfg)
        inputs = frame.assign(last_mm=frame.joint_min)
        local = np.clip(predict_ridge(model, inputs), low, high)
        rate = np.where(frame.aligned_measurement, median + cfg['local_weight'] * (local - median), median)
        horizon = (cutoff - joint.PRIOR_DATE).total_seconds() / YEAR_SECONDS
        upper = frame.nominal_thickness_mm
        parts.append(frame[KEY + ['fold', 'aligned_measurement']].assign(
            local_rate_mm_year=local, median_rate_mm_year=median, shrunk_rate_mm_year=rate,
            prediction=(frame.joint_blend - horizon * rate).clip(0., upper=upper),
            median_control=(frame.joint_blend - horizon * median).clip(0., upper=upper)))
        model_metadata.append({'held_id': int(pid), 'date': str(cutoff.date()),
                               'median_rate_mm_year': median, 'rate_limits_mm_year': [low, high],
                               'horizon_years': horizon, **model['metadata']})
        fit_keys.append(train[KEY + ['last_vtd_date', 'rate_mm_year', 'robust_rate']].assign(
            held_id=pid, evaluation_date=cutoff))
    return pd.concat(parts, ignore_index=True), pd.concat(fit_keys, ignore_index=True), model_metadata


def read_baseline(folder, spec):
    frame = pd.read_csv(folder / 'validation_predictions.csv', parse_dates=['prediction_date'])
    frame['prediction_date'] = frame.prediction_date.astype('datetime64[ns]')
    if len(frame) != spec['n'] or frame.duplicated(KEY).any() or not np.isfinite(
            frame[['next_mm', 'prediction']].to_numpy()).all():
        raise ValueError('Invalid frozen baseline')
    return frame


def compare(base, candidate, spec, folder):
    proposed = base[KEY + ['next_mm']].merge(candidate[KEY + ['prediction']], on=KEY,
                                            how='outer', validate='one_to_one')
    return score_comparison(base, proposed, spec, 1, folder)


def run(protocol_path, output):
    tick = perf_counter()
    protocol = json.loads(protocol_path.read_bytes())
    baseline_protocol = Path(protocol['baseline_protocol'])
    baseline = json.loads(baseline_protocol.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'source_snapshot.json', freeze_checkout(output))
    data = RepositoryData(Path(baseline['build']))
    raw = data.read('data/cleaned/vtd')
    mapped = data.read('processed/vtd_by_segment')
    parent = data.parent_groups()
    if parent.loc[list(joint.PEERS)].nunique() != 1:
        raise ValueError('Unexpected parent groups in the frozen S1 benchmark')
    audits = inventory(data, raw, mapped)
    archive = Path(protocol['baseline_root'])
    scheme = baseline['schemes'][PEER]
    base = read_baseline(archive / PEER, scheme)
    features = peer_inputs(data, raw, archive / PEER)
    paired = base.merge(features, on=KEY, how='outer', validate='one_to_one')
    np.testing.assert_allclose(paired.prediction, paired.joint_blend, atol=baseline['archive_tolerance_mm'],
                               rtol=0., equal_nan=False)
    features = paired.drop(columns=['next_mm', 'prediction'])
    # Use the same unrounded replay for baseline and variants to avoid rounding artefacts.
    base = paired[KEY + ['next_mm', 'joint_blend']].rename(columns={'joint_blend': 'prediction'})
    setup = protocol['hypotheses']
    h09 = history_mix(features, mapped, setup['H09'])
    training = s1_fit_rows(s1_training_table(data))
    h10, fit_keys, metadata = shrink_rates(features, training, setup['H10'])
    evaluations = {}
    for name, predictions in [('H09', h09), ('H10', h10)]:
        path = output / 'hypotheses' / name
        write_csv(predictions, path / 'prediction_inputs.csv', float_format='%.15g')
        evaluations[name] = compare(base, predictions, scheme, path / PEER)
        print(f"{name}: S1 n={len(base)}, delta={evaluations[name]['delta_pearson_squared']}", flush=True)
    cmp = h10.drop(columns='prediction').rename(columns={'median_control': 'prediction'})
    control = compare(base, cmp, scheme, output / 'hypotheses/H10/median_control')
    local = compare(base[KEY + ['next_mm']].merge(cmp[KEY + ['prediction']], on=KEY),
                    h10, {**scheme, 'r2': control['candidate']['pearson_squared']},
                    output / 'hypotheses/H10/vs_median_control')
    write_csv(fit_keys, output / 'hypotheses/H10/training_keys.csv')
    write_json(output / 'hypotheses/H10/fits.json', metadata)
    peer_history = mapped.loc[mapped.id.isin(joint.PEERS) & mapped.target_eligible
                              & mapped.date.lt(S1_CUTOFF)]
    counts = peer_history.groupby(['id', 'date']).size().rename('eligible_rows').reset_index()
    audits['H09'] = {'eligible_previous_segments': counts.astype({'date': str}).to_dict('records'),
                     'rows_with_two_direct_inputs': int(h09.two_direct_campaigns.sum())}
    audits['H10'] = {'historical_rate_pairs': len(training),
                     'folds': metadata, 'median_control': control,
                     'vs_median_control': local}
    benchmarks = {}
    for name, details in baseline['schemes'].items():
        frozen = read_baseline(archive / name, details)
        r = float(frozen.next_mm.corr(frozen.prediction))
        np.testing.assert_allclose(r ** 2, details['r2'], atol=1e-8, rtol=0.)
        benchmarks[name] = {'pearson_r': r, 'pearson_squared': r ** 2, 'n': len(frozen)}
    states = {key: 'blocked_data' if key in ('H06', 'H07', 'H08') else 'implemented'
              for key in setup}
    for key, config in setup.items():
        path = output / 'hypotheses' / key
        write_json(path / 'config.json', config)
        write_json(path / 'summary.json', json_safe({
            'state': states[key], 'audit': audits[key], 'evaluation': evaluations.get(key),
            'evidence': 'exploratory' if key in evaluations else 'not_evaluated'}))
    write_json(output / 'stage_results.json', json_safe({
        'benchmarks': benchmarks, 'audits': audits, 'evaluation': evaluations, 'states': states}))
    inputs = [baseline_protocol, PASSPORT, SOURCE, *[p for name in baseline['schemes']
              for p in (archive / name).glob('*.csv')], archive / PEER / 'anchor_folds.json']
    write_json(output / 'run_manifest.json', {
        'protocol_sha256': sha256(protocol_path), 'code_sha256': sha256(Path(__file__)),
        'data_manifest_sha256': sha256(data.root / 'reports/run_manifest.json'),
        'input_sha256': {**data.hashes, **{path.as_posix(): sha256(path) for path in inputs}},
        'environment': versions(), 'seed': protocol['seed'], 'seconds': perf_counter() - tick,
        'command': f'"{sys.executable}" -B -m src.experiments.r2_pool_stage2 '
                   f'--protocol "{protocol_path}" --output "{output}"',
        'status': 'completed',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)

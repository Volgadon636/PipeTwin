"""Replay photo benchmarks and audit H01–H05 without changing published inputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.experiments.r2_pool_stage1 import IDS, KEY, checked_comparison, scores, units
from src.experiments.r2_pool_stage1_strong import (
    filter_audit, fit_s1_scheme, fit_s2_scheme, freeze_checkout, mapped_labels,
    prediction, split_frames,
)
from src.ml import s1_joint_alignment as joint
from src.ml.data import RepositoryData
from src.ml.evaluation import metrics
from src.ml.incremental import _s1_stats
from src.ml.incremental_features import attach_defect_stats
from src.ml.next_common import json_safe, s1_fit_rows, s1_training_table, versions, write_csv
from src.ml.s1_contract import S1_CUTOFF, S1_ID, S2_ID, target_key_grid
from src.ml.s2_terrain import VARIANTS, terrain_features
from src.ml.targeted_s2 import (
    S2_CUTOFF, _predict_candidate, fit_candidate_artifacts, prepare_full_grid,
)
from src.storage import sha256, write_json

PROTOCOL = Path('experiments/r2_pool/protocol_photo_v3.json')
MENTOR = Path('docs/mentor/sources/SRC-M2-STRICT-REVIEW.md')


def parse_audit(data, raw):
    audit = units(data, raw)
    fields = {'Wall Thickness': 'wall_thickness', 'Relative Distance': 'relative_distance',
              'Section Number': 'section_number'}
    with Path('data/vtd.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    reference = pd.DataFrame([rows[i - 1] for i in raw.source_record_number], index=raw.index)
    for source, cleaned in fields.items():
        value = pd.to_numeric(reference[source].str.strip().str.replace(',', '.', regex=False),
                              errors='coerce').astype(float)
        equal = np.isclose(value, raw[cleaned].astype(float), atol=1e-9, rtol=1e-12, equal_nan=True)
        audit[cleaned] = {'rows': len(raw), 'finite': int(value.notna().sum()),
                          'mismatches': int((~equal).sum())}
    return audit


def track_s1_peer(data, raw, filtered):
    grid = data.read('processed/segments_grid')
    anchors = joint.joint_anchors(raw, joint.PRIOR_DATE, S1_CUTOFF, joint.PEERS)
    baseline, _, folds = joint.evaluate(data, raw, grid, anchors)
    # Only the historical thickness inputs change. Even on this retrospective,
    # the anchor map, holdout labels and fallback contract are fixed.
    candidate, _, candidate_folds = joint.evaluate(data, filtered, grid, anchors)
    if folds != candidate_folds:
        raise ValueError('S1 peer fold or anchor support changed')
    pd.testing.assert_frame_equal(baseline[joint.KEYS + ['fold', 'supported', 'next_mm']],
                                  candidate[joint.KEYS + ['fold', 'supported', 'next_mm']])
    baseline = baseline.assign(prediction_date=S1_CUTOFF)
    candidate = candidate.assign(prediction_date=S1_CUTOFF)
    return (prediction(baseline[baseline.supported], 'joint_blend'),
            prediction(candidate[candidate.supported], 'joint_blend'),
            baseline[KEY + ['fold', 'supported']], folds)


def track_s2_retro(prepared, train, model):
    intervals = prepared['interval_map']
    artifact = fit_candidate_artifacts(train, intervals)[model]
    parts = []
    for key, labels in prepared['post'].groupby(['id', 'prediction_date'], sort=True):
        full = prepared['full_frames'][key]
        pframe = full[KEY].assign(prediction=_predict_candidate(artifact, full, intervals)['h_clipped'])
        part = labels[KEY + ['next_mm']].merge(pframe, on=KEY, how='left', validate='one_to_one')
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def archive_frame(spec, replay, folds, tolerance):
    archive = pd.read_csv(spec['source'])
    if 'model' in archive:
        archive = archive[archive.model.eq(spec['model'])]
        column = 'prediction'
    else:
        column = spec['model']
    if 'prediction_date' not in archive:
        archive['prediction_date'] = spec['date']
    archive['prediction_date'] = pd.to_datetime(archive.prediction_date).astype('datetime64[ns]')
    split_column = 'block' if 'block' in archive else 'fold'
    if split_column in archive:
        extra = ['supported'] if 'supported' in archive else []
        old = archive.set_index(KEY)[[split_column, *extra]].sort_index()
        new = folds[folds.role.eq('test')].set_index(KEY)[[split_column, *extra]].sort_index()
        pd.testing.assert_frame_equal(old, new, check_dtype=False)
    archive = archive.loc[archive.supported] if 'supported' in archive else archive
    old, new = (frame.set_index(KEY).sort_index()
                for frame in (prediction(archive, column), replay))
    pd.testing.assert_index_equal(old.index, new.index)
    np.testing.assert_allclose(old.next_mm, new.next_mm, atol=5e-10, rtol=0.)
    np.testing.assert_allclose(old.prediction, new.prediction, atol=tolerance, rtol=0.)
    return float((old.prediction - new.prediction).abs().max())


def score_comparison(base, candidate, spec, parent_count, output):
    pair = checked_comparison(base, candidate)
    by_profile = scores(pair, spec['case'])
    diagnostics = {}
    for label in ('baseline', 'candidate'):
        measured = metrics(pair.next_mm, pair[label])
        diagnostics[label] = {key: measured[key] for key in (
            'pearson_r', 'pearson_squared', 'mae_mm', 'rmse_mm', 'bias_mm', 'pearson_reason')}
        diagnostics[label]['prediction_std'] = float(pair[label].std(ddof=0))
    base_r2 = diagnostics['baseline']['pearson_squared']
    candidate_r2 = diagnostics['candidate']['pearson_squared']
    np.testing.assert_allclose(base_r2, spec['r2'], rtol=0., atol=1e-8)
    if len(pair) != spec['n']:
        raise ValueError(f"Expected {spec['n']} rows, received {len(pair)}")
    design = np.column_stack([np.ones(len(pair)), pair.baseline])
    affine = design @ np.linalg.lstsq(design, pair.candidate, rcond=None)[0]
    summary = {'n': len(pair), 'profile_count': len(by_profile),
               'independent_parent_count': parent_count, **diagnostics,
               'delta_pearson_squared': None if candidate_r2 is None else candidate_r2 - base_r2,
               'max_abs_affine_residual_mm': float(np.max(np.abs(pair.candidate - affine))),
               'changed_predictions': int((pair.baseline.sub(pair.candidate).abs() > 1e-10).sum())}
    write_csv(pair, output / 'validation_predictions.csv', float_format='%.15g')
    write_csv(pd.DataFrame(by_profile), output / 'metrics_by_profile.csv')
    write_json(output / 'summary.json', summary)
    return summary


def split_keys(fit, scheme):
    parts = []
    column = 'block' if 'block_m' in scheme else 'fold'
    for _, fold, train, test in split_frames(fit, scheme):
        parts.extend([train[KEY].assign(**{column: fold}, role='train'),
                      test[KEY].assign(**{column: fold}, role='test')])
    return pd.concat(parts, ignore_index=True)


def list_uzt(data):
    uzt = data.read('data/cleaned/uzt')
    rows = uzt[uzt.id.isin((S1_ID, *joint.PEERS)) & uzt.date.lt(S1_CUTOFF)]
    return {str(pid): {'rows': len(part), 'eligible': int(part.measurement_eligible.sum())}
            for pid, part in [(pid, rows[rows.id.eq(pid)]) for pid in (S1_ID, *joint.PEERS)]}


def run(protocol_path, output):
    start = perf_counter()
    protocol = json.loads(protocol_path.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    snapshot = freeze_checkout(output)
    write_json(output / 'source_snapshot.json', snapshot)
    data = RepositoryData(Path(protocol['build']))
    raw = data.read('data/cleaned/vtd')
    overlay, remapped, removed = filter_audit(data, raw)
    filtered = overlay.read('data/cleaned/vtd')
    audited = parse_audit(data, raw)
    known = target_key_grid(data)
    if known.groupby('id').size().to_dict() != {S1_ID: 525, S2_ID: 1475}:
        raise ValueError('Unexpected target key counts')

    prepared = prepare_full_grid(data)
    strict = prepared['strict']
    if not strict.prediction_date.lt(S2_CUTOFF).all() or strict.id.eq(S2_ID).any():
        raise ValueError('S2 fit includes forbidden labels')
    adjusted = mapped_labels(remapped)
    current = mapped_labels(data.read('processed/vtd_by_segment'))
    paired = strict[KEY + ['next_mm']].merge(current, on=KEY, how='left',
                                            validate='one_to_one', suffixes=('', '_mapped'))
    np.testing.assert_array_equal(paired.next_mm, paired.next_mm_mapped)
    post_keys = prepared['post'][KEY]
    if post_keys.id.eq(S2_ID).any():
        raise ValueError('Retrospective includes target diagnostic labels')
    reduced = strict.drop(columns='next_mm').merge(adjusted, on=KEY, validate='one_to_one')
    profiles_before = strict[KEY + ['next_mm']]
    changes = profiles_before.merge(reduced[KEY + ['next_mm']], on=KEY, how='left',
                                    validate='one_to_one', suffixes=('', '_filtered'))
    changes = changes[changes.next_mm.ne(changes.next_mm_filtered)]
    audits = {
        'H01': {'verified_target_anchors': 0, 'peer_geometry_available_on': str(S1_CUTOFF.date())},
        'H02': audited,
        'H03': {'wall_nominal_mismatch_rows': int(raw.wall_nominal_mismatch_flag.sum()),
                'verified_dated_local_nominal_map': False},
        'H04': {'mentor_policy': str(MENTOR), 'raw_records_excluded_in_sensitivity': removed,
                'strict_fit_rows': len(strict), 'sensitivity_fit_rows': len(reduced),
                's2_training_changes': changes.astype({'prediction_date': str}).to_dict('records')},
        'H05': list_uzt(data),
    }

    parent = data.parent_groups()
    s1_fit = s1_fit_rows(s1_training_table(data))
    s1_alt = s1_fit_rows(s1_training_table(overlay))
    pd.testing.assert_frame_equal(s1_fit[KEY + ['next_mm']], s1_alt[KEY + ['next_mm']])
    s1_fit = attach_defect_stats(s1_fit, _s1_stats(data, [s1_fit]))
    s1_alt = attach_defect_stats(s1_alt, _s1_stats(overlay, [s1_alt]))
    summary = {}
    for name, spec in protocol['schemes'].items():
        model = spec['model']
        if name == 'S1_peer_2023':
            base, cand, folds, anchor_folds = track_s1_peer(data, raw, filtered)
            folds = folds.assign(role='test')
            write_json(output / 'baseline' / name / 'anchor_folds.json', anchor_folds)
        elif name == 'S2_retro_2023':
            base = track_s2_retro(prepared, strict, model)
            cand = track_s2_retro(prepared, reduced, model)
            folds = pd.concat([strict[KEY].assign(role='train'), base[KEY].assign(role='test')],
                              ignore_index=True)
        elif spec['case'] == 'S1':
            base = fit_s1_scheme(s1_fit, spec, model)
            cand = fit_s1_scheme(s1_alt, spec, model)
            folds = split_keys(s1_fit, spec)
        else:
            grids = {key: terrain_features(value) for key, value in prepared['full_frames'].items()
                     if key[1] < S2_CUTOFF}
            added = sorted(set(VARIANTS[model]) - set(strict.columns))
            features = pd.concat([frame[KEY + added] for frame in grids.values()], ignore_index=True)
            fit = strict.merge(features, on=KEY, how='left', validate='one_to_one')
            intervals = prepared['interval_map']
            base = fit_s2_scheme(fit, grids, intervals, spec, model)
            cand = fit_s2_scheme(fit, grids, intervals, spec, model, adjusted)
            folds = split_keys(fit, spec)
        for frame in (base, cand, folds):
            frame['id'] = frame.id.astype('int64')
            frame['segment_id'] = frame.segment_id.astype('int64')
            frame['prediction_date'] = frame.prediction_date.astype('datetime64[ns]')
        diff = archive_frame(spec, base, folds, protocol['archive_tolerance_mm'])
        if spec['case'] == 'S2':
            folds['h04_included'] = folds.role.eq('test') | pd.MultiIndex.from_frame(folds[KEY]).isin(
                pd.MultiIndex.from_frame(reduced[KEY]))
        location = output / 'baseline' / name
        write_csv(folds, location / 'split_keys.csv')
        write_csv(base, location / 'validation_predictions.csv', float_format='%.15g')
        metadata = score_comparison(
            base, cand, spec, base.id.map(parent).nunique(),
            output / 'hypotheses' / 'H04' / 'seam_sensitivity' / name)
        summary[name] = {**spec, **metadata, 'archive_max_abs_replay_mm': diff}
        print(f"{name}: n={metadata['n']}, delta={metadata['delta_pearson_squared']}", flush=True)

    states = {'H01': 'blocked_data', 'H02': 'no_confirmed_correction',
              'H03': 'blocked_data', 'H04': 'exploratory_sensitivity_only', 'H05': 'blocked_data'}
    for name in IDS:
        location = output / 'hypotheses' / name
        write_json(location / 'config.json', {'id': name, 'specification': protocol['batch_01'][name]})
        write_json(location / 'summary.json', json_safe({'status': states[name], 'audit': audits[name]}))
    write_json(output / 'stage_results.json', json_safe({'schemes': summary, 'audits': audits,
                                                        'states': states}))
    inputs = {spec['source'] for spec in protocol['schemes'].values()} | {str(MENTOR)}
    write_json(output / 'run_manifest.json', {
        'protocol_sha256': sha256(protocol_path), 'code_sha256': sha256(Path(__file__)),
        'data_manifest_sha256': sha256(data.root / 'reports/run_manifest.json'),
        'input_sha256': {**data.hashes, **{path: sha256(Path(path)) for path in sorted(inputs)}},
        'environment': versions(), 'seed': protocol['seed'], 'seconds': perf_counter() - start,
        'command': f'"{sys.executable}" -B -m src.experiments.r2_pool_stage1_photo '
                   f'--protocol "{protocol_path}" --output "{output}"',
        'status': 'completed',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)

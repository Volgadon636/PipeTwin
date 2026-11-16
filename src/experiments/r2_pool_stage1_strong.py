"""Recheck H01–H05 against the highest reproducible comparable Pearson r²."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from src.experiments.r2_pool_stage1 import IDS, KEY, WELD, checked_comparison, scores, summarize, units
from src.ml.data import RepositoryData
from src.ml.incremental import _fit_s1, _fit_s2, _s1_stats
from src.ml.incremental_features import DEFECT_STATS, attach_defect_stats
from src.ml.next_common import S1_CORE_FEATURES, predict_ridge, s1_fit_rows, s1_training_table, versions, write_csv
from src.ml.s1_contract import S1_ID
from src.ml.s1_next import _ridge_oof
from src.ml.s2_terrain import VARIANTS, join_predictions, terrain_features
from src.ml.targeted_s1 import CORE_FEATURES, _block_folds, _physical_block_frame
from src.ml.targeted_s2 import prepare_full_grid
from src.segment_mapper import map_vtd_to_segments
from src.storage import sha256, write_json

ROOT = Path('experiments/r2_pool')
DEFAULT_PROTOCOL = ROOT / 'protocol_strong_v2.json'
DEFAULT_OUTPUT = ROOT / 'revisions/r2-pool-v2'


class FilteredVTD:
    """Override exact historical records without modifying the frozen repository."""

    def __init__(self, data, raw, mapped):
        self.data = data
        self.overrides = {'processed/vtd_by_segment': mapped, 'data/cleaned/vtd': raw}

    def __getattr__(self, name):
        return getattr(self.data, name)

    def read(self, name):
        return self.overrides[name].copy() if name in self.overrides else self.data.read(name)


def prediction(frame, column='prediction'):
    return frame[KEY + ['next_mm', column]].rename(columns={column: 'prediction'})


def archive_prediction(path, column, expected, date):
    source = pd.read_csv(path)
    if 'prediction_date' not in source:
        source['prediction_date'] = date
    source['prediction_date'] = pd.to_datetime(source.prediction_date).astype('datetime64[ns]')
    current, archived = (frame.set_index(KEY).sort_index()
                         for frame in (expected, prediction(source, column)))
    pd.testing.assert_index_equal(current.index, archived.index)
    # Old archives export 12 significant digits. Evaluate on the frozen full-precision labels.
    np.testing.assert_allclose(current.next_mm, archived.next_mm, rtol=0., atol=5e-11)
    archived['next_mm'] = current.next_mm
    checked_comparison(expected, archived.reset_index())
    return archived.reset_index()


def reference(frame):
    return frame[KEY + ['next_mm']].assign(prediction=0.)


def filter_audit(data, raw):
    match = raw.feature_description.fillna('').str.strip().str.casefold().eq(WELD)
    fields = ['id', 'date', 'source_row', 'distance', 'residual_for_target_mm']
    found = raw.loc[match, fields].copy()
    found['date'] = found.date.dt.strftime('%Y-%m-%d')
    filtered = raw.loc[~match].copy()
    mapped = map_vtd_to_segments(filtered, data.read('processed/segments_grid'))
    return FilteredVTD(data, filtered, mapped), mapped, found.to_dict('records')


def fit_s1_scheme(fit, scheme, model):
    if 'block_m' not in scheme:
        if model != 'mix_025':
            raise ValueError(f'No exact replay for S1 LOSO winner {model}')
        pred, _ = _ridge_oof(fit, S1_CORE_FEATURES, 'target_delta_mm', 'row_uniform',
                             baseline='last_mm', model_name='ridge_geometry')
        pred['prediction'] = pred.last_mm + .25 * (pred.prediction - pred.last_mm)
        pred['prediction_date'] = pd.Timestamp(scheme['date'])
        return prediction(pred)
    if model != 'defect_stats_support':
        raise ValueError(f'No exact replay for S1 block winner {model}')
    width, buffer = scheme['block_m'], scheme['buffer_m']
    target = _physical_block_frame(fit[fit.id.eq(S1_ID)], width)
    parts = []
    for fold in _block_folds(fit, target, width, buffer):
        artifact = _fit_s1(fold['train_global'], CORE_FEATURES + DEFECT_STATS['support'], model)
        test = fold['test']
        parts.append(test[KEY + ['next_mm']].assign(prediction=predict_ridge(artifact, test)))
    return pd.concat(parts, ignore_index=True)


def fit_s2_scheme(strict, grids, intervals, scheme, model, adjusted=None):
    if model not in VARIANTS:
        raise ValueError(f'No exact replay for S2 winner {model}')
    parts = []
    for pid, fold, train, held in split_frames(strict, scheme):
        if adjusted is not None:
            train = train.drop(columns='next_mm').merge(adjusted, on=KEY, validate='one_to_one')
        artifacts = _fit_s2(train, intervals, {model: VARIANTS[model]})
        day, = held.prediction_date.drop_duplicates()
        part = join_predictions(held, grids[pid, day], intervals, artifacts)
        parts.append(prediction(part, model))
    return pd.concat(parts, ignore_index=True)


def split_frames(frame, scheme):
    if 'block_m' in scheme:
        width, buffer = scheme['block_m'], scheme['buffer_m']
        target = _physical_block_frame(frame[frame.id.eq(S1_ID)], width)
        return [(S1_ID, x['block'], x['train_global'], x['test'])
                for x in _block_folds(frame, target, width, buffer)]
    return [(int(pid), int(pid), frame[~frame.id.eq(pid)], held)
            for pid, held in frame.groupby('id', sort=True)]


def mapped_labels(mapped):
    return mapped.loc[mapped.target_eligible].rename(columns={
        'date': 'prediction_date', 'min_residual_thickness_mm': 'next_mm'})[KEY + ['next_mm']]


def ranking(scheme, expected):
    candidates = []
    for path, names in scheme['sources'].items():
        for model in names.split():
            frame = archive_prediction(Path(path), model, expected, scheme['date'])
            paired = checked_comparison(frame, frame)
            profile = scores(paired, scheme['case'])
            aggregate = summarize(profile)
            pooled = paired.next_mm.corr(paired.candidate) if paired.candidate.nunique() > 1 else np.nan
            candidates.append({
                'model': model, 'source': path, 'source_sha256': sha256(Path(path)),
                'n': len(frame), 'profile_r2': aggregate['candidate_r2'],
                'pooled_r2': None if pd.isna(pooled) else float(pooled ** 2),
                'profiles': profile, 'frame': frame,
            })
    metric = scheme.get('rank_by', 'profile_r2')
    return sorted(candidates, key=lambda c: -1. if c[metric] is None else c[metric], reverse=True)


def actual_cases(hypothesis):
    if hypothesis in ('H01', 'H05'):
        return {'S1': 'blocked_data', 'S2': 'not_applicable'}
    status = 'implemented' if hypothesis in ('H02', 'H04') else 'blocked_data'
    return {'S1': status, 'S2': status}


def freeze_checkout(output):
    def git(*args):
        return subprocess.check_output(['git', '-c', 'safe.directory=C:/Program Files/PipeTwin', *args])
    changes = git('diff', '--binary', 'HEAD')
    (output / 'tracked_at_start.patch').write_bytes(changes)
    status = git('status', '--porcelain', '--untracked-files=all').decode('utf-8').splitlines()
    untracked = [Path(path) for path in git('ls-files', '--others', '--exclude-standard', '-z')
                 .decode('utf-8').split('\0') if path and not Path(path).resolve().is_relative_to(output.resolve())]
    dirty_code = output / 'untracked_source_at_start.zip'
    with ZipFile(dirty_code, 'w', compression=ZIP_DEFLATED) as archive:
        for path in sorted(untracked):
            if path.is_file() and path.parts[0] in ('src', 'tests', 'configs'):
                archive.write(path, path.as_posix())
    return {'head': git('rev-parse', 'HEAD').decode().strip(),
            'origin_momo': git('rev-parse', 'origin/momo').decode().strip(),
            'status': status, 'tracked_patch_sha256': hashlib.sha256(changes).hexdigest(),
            'untracked_source_zip_sha256': sha256(dirty_code),
            'untracked_files_sha256': {path.as_posix(): sha256(path)
                                       for path in untracked if path.is_file()}}


def run(protocol_path, output):
    tick = perf_counter()
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    schemes = {scheme['name']: scheme for scheme in protocol['schemes']}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'source_snapshot.json', freeze_checkout(output))
    data = RepositoryData(Path(protocol['build']))
    if 'vtd_quantiles' in data.config or any('quantile' in key for key in data.manifest['outputs']):
        raise ValueError('P05 artifacts cannot be inputs to this experiment')
    raw = data.read('data/cleaned/vtd')
    overlay, mapped, removed = filter_audit(data, raw)
    fit = s1_fit_rows(s1_training_table(data))
    filtered_fit = s1_fit_rows(s1_training_table(overlay))
    prepared = prepare_full_grid(data)
    intervals = prepared['interval_map']
    grids = {key: terrain_features(value) for key, value in prepared['full_frames'].items()
             if key[1] < pd.Timestamp('2022-07-01')}
    additions = sorted(set(sum(VARIANTS.values(), [])) - set(prepared['strict'].columns))
    features = pd.concat([frame[KEY + additions] for frame in grids.values()], ignore_index=True)
    strict = prepared['strict'].merge(features, on=KEY, how='left', validate='one_to_one')
    adjusted = mapped_labels(mapped)
    control_labels = mapped_labels(data.read('processed/vtd_by_segment'))
    labels = strict[KEY + ['next_mm']]
    check = labels.merge(control_labels, on=KEY, validate='one_to_one', suffixes=('', '_mapped'))
    if len(check) != len(labels):
        raise ValueError('Frozen S2 labels are missing from the unmodified VTD map')
    np.testing.assert_array_equal(check.next_mm, check.next_mm_mapped)
    replaced = labels.merge(adjusted, on=KEY, how='left', validate='one_to_one', suffixes=('', '_filtered'))
    changed_s2 = replaced.loc[replaced.next_mm.ne(replaced.next_mm_filtered)]
    if len(changed_s2) > len(removed):
        raise ValueError('H04 changed more S2 labels than the number of records removed')
    changed_s2 = changed_s2.astype({'prediction_date': str}).astype(object).where(changed_s2.notna(), None)
    pd.testing.assert_frame_equal(fit[KEY + ['next_mm']], filtered_fit[KEY + ['next_mm']])
    audits = {'H01': {'target_reference_rows': int((raw.id.eq(S1_ID) & raw.date.eq('2023-12-01')).sum()),
                      'verified_pre_cutoff_map': False},
              'H02': units(data, raw),
              'H03': {'observed_wall_mismatch': int(raw.wall_nominal_mismatch_flag.sum()),
                      'verified_dated_joint_nominal_map': False},
              'H04': {'rule': WELD, 'removed': removed, 's2_affected_training_labels': changed_s2.to_dict('records')},
              'H05': {'eligible_s1_uzt': int(data.read('data/cleaned/uzt').query(
                  'id == @S1_ID and date < "2023-12-01"').measurement_eligible.sum())}}
    if any(field['mismatches'] for field in audits['H02'].values()):
        raise ValueError('Unresolved H02 parsing mismatch; inspect before scoring')
    baseline_info, all_results = {}, {}
    for name, scheme in schemes.items():
        if 'same_sources_as' in scheme:
            scheme = {**schemes[scheme['same_sources_as']], **scheme}
        name, case = scheme['name'], scheme['case']
        expected = fit if case == 'S1' else strict
        if 'block_m' in scheme:
            expected = expected[expected.id.eq(S1_ID)]
        expected = reference(expected)
        expected['prediction_date'] = expected.prediction_date.astype('datetime64[ns]')
        ordered = ranking(scheme, expected)
        champion = ordered[0]
        model = champion['model']
        baseline = champion['frame']
        if case == 'S1':
            control, candidate = fit, filtered_fit
            if 'block_m' in scheme:
                control = attach_defect_stats(fit, _s1_stats(data, [fit]))
                candidate = attach_defect_stats(filtered_fit, _s1_stats(overlay, [filtered_fit]))
            replay = fit_s1_scheme(control, scheme, model)
            variant = fit_s1_scheme(candidate, scheme, model)
        else:
            replay = fit_s2_scheme(strict, grids, intervals, scheme, model)
            variant = fit_s2_scheme(strict, grids, intervals, scheme, model, adjusted)
        for frame in (replay, variant):
            frame['prediction_date'] = frame.prediction_date.astype('datetime64[ns]')
        reproducible = checked_comparison(baseline, replay)
        difference = float((reproducible.baseline - reproducible.candidate).abs().max())
        if difference > 1e-8:
            raise ValueError(f'{name}/{model} archive replay differs by {difference}')
        baseline_path = output / 'baseline' / name
        write_csv(baseline, baseline_path / 'validation_predictions.csv', float_format='%.12g')
        split_audit = []
        for _, fold, train, test in split_frames(fit if case == 'S1' else strict, scheme):
            training = train[KEY].assign(fold=fold, role='train', h04_included=True)
            if case == 'S2':
                training['h04_included'] = pd.MultiIndex.from_frame(training[KEY]).isin(
                    pd.MultiIndex.from_frame(adjusted[KEY]))
            split_audit.extend([training, test[KEY].assign(fold=fold, role='test', h04_included=True)])
        write_csv(pd.concat(split_audit, ignore_index=True), baseline_path / 'split_keys.csv')
        write_json(baseline_path / 'leaderboard.json', [
            {key: value for key, value in item.items() if key != 'frame'} for item in ordered])
        baseline_info[name] = {'case': case, 'model': model, 'source': champion['source'],
                               'n': champion['n'], 'profile_r2': champion['profile_r2'],
                               'pooled_r2': champion['pooled_r2'], 'max_abs_replay_mm': difference,
                               'validation_predictions_sha256': sha256(baseline_path/'validation_predictions.csv')}
        all_results[name] = {}
        for hypothesis, pred in [('H02', baseline), ('H04', variant)]:
            pair = checked_comparison(baseline, pred)
            by_profile = scores(pair, case)
            summary = summarize(by_profile)
            summary['baseline_pooled_r2'] = float(pair.next_mm.corr(pair.baseline) ** 2)
            summary['candidate_pooled_r2'] = float(pair.next_mm.corr(pair.candidate) ** 2)
            summary['delta_pooled_r2'] = summary['candidate_pooled_r2'] - summary['baseline_pooled_r2']
            summary['ranking_metric'] = scheme.get('rank_by', 'profile_r2')
            delta = summary['delta_pooled_r2' if summary['ranking_metric'] == 'pooled_r2'
                            else 'delta_pearson_squared']
            summary['status'] = ('no_increment' if abs(delta) < 1e-12 else
                                 'worse' if delta < 0 else 'local_gain_unconfirmed')
            all_results[name][hypothesis] = summary
            folder = output / 'hypotheses' / hypothesis / name
            write_csv(pair, folder / 'validation_predictions.csv', float_format='%.12g')
            write_csv(pd.DataFrame(by_profile), folder / 'metrics_by_profile.csv')
            write_json(folder / 'summary.json', summary)
    write_json(output / 'baseline' / 'summary.json', baseline_info)
    for hypothesis in IDS:
        folder = output / 'hypotheses' / hypothesis
        write_json(folder / 'config.json', {'id': hypothesis, 'specification': protocol['batch_01'][hypothesis]})
        write_json(folder / 'summary.json', {'cases': actual_cases(hypothesis), 'audit': audits[hypothesis],
                                            'protocol': protocol['protocol_version']})
    write_json(output / 'stage_results.json', {'baseline': baseline_info, 'results': all_results,
                                               'audits': audits, 'seconds': perf_counter() - tick})
    write_json(output / 'run_manifest.json', {
        'protocol_sha256': sha256(protocol_path), 'code_sha256': sha256(Path(__file__)),
        'data_manifest_sha256': sha256(data.root / 'reports/run_manifest.json'),
        'input_sha256': data.hashes, 'environment': versions(), 'seed': protocol['seed'],
        'command': f'python -B -m src.experiments.r2_pool_stage1_strong --protocol {protocol_path} --output {output}',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.protocol, args.output)

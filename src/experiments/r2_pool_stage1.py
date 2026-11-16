"""Isolated execution of H01–H05. Run from the repository root."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.ml.data import RepositoryData
from src.ml.next_common import s1_fit_rows, s1_training_table, versions, write_csv
from src.ml.s1_contract import S1_ID, S2_ID, target_key_grid
from src.ml.targeted_s2 import S2_CUTOFF, _fit_physics, _predict_physics, prepare_full_grid
from src.segment_mapper import map_vtd_to_segments
from src.storage import sha256, write_json

OUT = Path('experiments/r2_pool')
KEY = ['id', 'segment_id', 'prediction_date']
IDS = [f'H{i:02d}' for i in range(1, 6)]
WELD = 'аномалия поперечного шва (несплошность плоскостного типа)'


def s1_persistence(data):
    eligible = s1_fit_rows(s1_training_table(data))
    return eligible[KEY + ['next_mm', 'last_mm']].rename(columns={'last_mm': 'prediction'})


def s2_loso(strict, intervals, training_labels=None):
    parts = []
    for pid, test in strict.groupby('id', sort=True):
        fit = strict[~strict.id.eq(pid)].copy()
        if training_labels is not None:
            fit = fit.drop(columns='next_mm').merge(training_labels, on=KEY, validate='one_to_one')
        model = _fit_physics(fit, intervals)
        parts.append(test[KEY + ['next_mm']].assign(
            prediction=_predict_physics(model, test, intervals), training_rows=len(fit)))
    return pd.concat(parts, ignore_index=True)


def checked_comparison(baseline, candidate):
    for frame in (baseline, candidate):
        if frame.duplicated(KEY).any() or not np.isfinite(frame[['next_mm', 'prediction']]).all().all():
            raise ValueError('Duplicate validation keys or nonfinite prediction/label')
    a, b = (frame.set_index(KEY).sort_index() for frame in (baseline, candidate))
    pd.testing.assert_index_equal(a.index, b.index)
    np.testing.assert_array_equal(a.next_mm, b.next_mm)
    return a.reset_index().rename(columns={'prediction': 'baseline'}).assign(
        candidate=b.prediction.to_numpy())


def scores(pair, case):
    rows = []
    for (pid, date), profile in pair.groupby(['id', 'prediction_date'], sort=True):
        y = profile.next_mm.to_numpy()
        row = {'case': case, 'id': int(pid), 'prediction_date': date.date().isoformat(), 'n': len(y),
               'weak_profile': len(y) < 10, 'independent_parent_count': 1}
        for name in ('baseline', 'candidate'):
            p = profile[name].to_numpy()
            status = ('too_few_rows' if len(y) < 3 else 'constant_truth' if np.ptp(y) <= 1e-12
                      else 'constant_prediction' if np.ptp(p) <= 1e-12 else 'defined')
            r = float(np.corrcoef(y, p)[0, 1]) if status == 'defined' else None
            error = p - y
            row.update({f'{name}_pearson_r': r, f'{name}_r2': None if r is None else r * r,
                        f'{name}_status': status, f'{name}_mae': float(np.abs(error).mean()),
                        f'{name}_rmse': float(np.sqrt((error**2).mean())),
                        f'{name}_bias': float(error.mean()), f'{name}_std': float(p.std())})
        a, b = row['baseline_r2'], row['candidate_r2']
        row['delta_pearson_squared'] = None if a is None or b is None else b - a
        row['changed_predictions'] = int((np.abs(profile.candidate - profile.baseline) > 1e-10).sum())
        rows.append(row)
    return rows


def summarize(rows):
    def mean(column):
        values = [row[column] for row in rows]
        return None if any(x is None for x in values) else float(np.mean(values))
    keys = ('baseline_r2', 'candidate_r2', 'delta_pearson_squared', 'candidate_pearson_r')
    return {**{k: mean(k) for k in keys}, 'n': sum(row['n'] for row in rows),
            'independent_parent_count': 1, 'profile_count': len(rows),
            'changed_predictions': sum(row['changed_predictions'] for row in rows)}


def units(data, vtd):
    raw = Path('data/vtd.csv')
    if sha256(raw) != data.manifest['inputs']['vtd.csv']['sha256']:
        raise ValueError('Raw input hash differs from frozen preprocessing manifest')
    with raw.open(encoding='utf-8-sig', newline='') as fh:
        records = list(csv.DictReader(fh))
    paired = pd.DataFrame([records[i - 1] for i in vtd.source_record_number], index=vtd.index)
    np.testing.assert_array_equal(pd.to_numeric(paired.row_id), vtd.row_id)
    result = {}
    for original, cleaned in [('Distance', 'distance'), ('Residual Wall Thickness', 'residual_wall_thickness'),
                              ('Defect Depth in %', 'defect_depth_pct')]:
        parsed = pd.to_numeric(paired[original].str.strip().str.replace(',', '.', regex=False),
                               errors='coerce').astype(float)
        if cleaned == 'defect_depth_pct':
            fractional = vtd.date.dt.strftime('%Y-%m-%d').isin(data.config['vtd_fraction_dates'])
            parsed = parsed.where(~fractional, 100 * parsed)
        expected = vtd[cleaned].astype(float)
        same = np.isclose(parsed, expected, atol=1e-9, rtol=1e-12, equal_nan=True)
        result[cleaned] = {'rows': len(vtd), 'finite': int(parsed.notna().sum()),
                           'mismatches': int((~same).sum())}
    return result


def main():
    start = time.perf_counter()
    protocol_path = OUT / 'protocol.json'
    setup = json.loads(protocol_path.read_text(encoding='utf-8'))
    data = RepositoryData(Path(setup['build']))
    vtd, grid = data.read('data/cleaned/vtd'), data.read('processed/segments_grid')
    uzt = data.read('data/cleaned/uzt')
    prepared = prepare_full_grid(data)
    cold, intervals = prepared['strict'], prepared['interval_map']
    baseline = {'S1': s1_persistence(data), 'S2': s2_loso(cold, intervals)}
    base_dir = OUT / 'baseline'
    source = Path(setup['baseline']['result'])
    result = pd.read_csv(source)
    expected = target_key_grid(data)
    pd.testing.assert_frame_equal(result[['id', 'segment_id']], expected)
    if not np.isfinite(result.thickness_pred).all():
        raise ValueError('Nonfinite baseline production outputs')
    parent = data.parent_groups()
    for case, frame in baseline.items():
        sizes = {str(pid): n for pid, n in frame.groupby('id').size().items()}
        if sizes != setup[case]['profiles'] or frame.id.map(parent).nunique() != 1:
            raise ValueError('Frozen validation roster changed')
        if not frame.prediction_date.eq(pd.Timestamp(setup[case]['validation_date'])).all():
            raise ValueError('Frozen validation date changed')
        write_csv(frame, base_dir / f'{case}_validation_predictions.csv', float_format='%.12g')
    write_csv(result, base_dir / 'result.csv', float_format='%.12g')
    inputs = dict(data.hashes)
    shared_manifest = {'protocol_version': setup['protocol_version'], 'protocol_sha256': sha256(protocol_path),
                       'source_snapshot_sha256': sha256(base_dir/'source_snapshot.json'),
                       'python': versions(), 'seed': setup['seed'],
                       'validation_split': {k: setup[k]['profiles'] for k in baseline},
                       'data_root': str(data.root), 'data_manifest_sha256': sha256(data.root/'reports/run_manifest.json'),
                       'data_hashes': inputs, 'code_sha256': sha256(Path(__file__)),
                       'baseline_result_sha256': sha256(source),
                       'command': 'python -B -m src.experiments.r2_pool_stage1'}
    frozen_scores = [row for case, p in baseline.items()
                     for row in scores(checked_comparison(p, p), case)]
    write_csv(pd.DataFrame(frozen_scores), base_dir/'metrics_by_profile.csv')
    write_json(base_dir/'run_manifest.json', shared_manifest)

    stage = {}
    audits = {
        'H01': {'target_2023_vtd_rows': int((vtd.id.eq(S1_ID) & vtd.date.eq('2023-12-01')).sum()),
                'verified_pre_cutoff_reference_map': False,
                'existing_implementation': 'src/ml/s1_joint_alignment.py and src/odo_align.py'},
        'H02': units(data, vtd),
        'H03': {'nominal_mismatch_rows_on_s1_target': int(vtd.loc[vtd.id.eq(S1_ID), 'wall_nominal_mismatch_flag'].sum()),
                'verified_dated_construction_map': False},
        'H05': {'all_uzt_rows': len(uzt), 'above_nominal': int(uzt.above_nominal_flag.sum()),
                's1_target_rows': int(uzt.id.eq(S1_ID).sum()),
                's1_eligible_before_cutoff': int((uzt.id.eq(S1_ID) & uzt.date.lt('2023-12-01')
                                                  & uzt.measurement_eligible).sum())},
    }
    raw_rule = vtd.feature_description.fillna('').str.strip().str.casefold().eq(WELD)
    mapped = map_vtd_to_segments(vtd.loc[~raw_rule], grid)
    audits['H04'] = {'removed_vtd_rows': int(raw_rule.sum()), 'rule': WELD,
                     'labels_held_fixed': True}
    for hypothesis in IDS:
        tick = time.perf_counter()
        folder = OUT/'hypotheses'/hypothesis/'run_01'
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder/'config.json', {'id': hypothesis, 'method': setup['batch_01'][hypothesis]})
        if hypothesis == 'H04':
            class VtdOverlay:
                def __getattr__(self, name):
                    return getattr(data, name)

                def read(self, stem):
                    return mapped.copy() if stem == 'processed/vtd_by_segment' else data.read(stem)

            s1 = s1_persistence(VtdOverlay()).drop(columns='next_mm')
            s1 = baseline['S1'][KEY + ['next_mm']].merge(s1, on=KEY, how='left', validate='one_to_one')
            if s1.prediction.isna().any():
                raise ValueError('H04 removed support for a frozen S1 validation key')
            training = mapped.loc[mapped.target_eligible, ['id', 'date', 'segment_id', 'min_residual_thickness_mm']]
            s2 = s2_loso(cold, intervals, training.rename(columns={
                'date': 'prediction_date', 'min_residual_thickness_mm': 'next_mm'}))
            variants = {'S1': s1, 'S2': s2}
        elif hypothesis == 'H02' and all(x['mismatches'] == 0 for x in audits[hypothesis].values()):
            variants = baseline
        else:
            variants = {}
        metrics = []
        case_results = {}
        for case in baseline:
            if case in variants:
                pair = checked_comparison(baseline[case], variants[case])
                rows = scores(pair, case)
                metrics.extend(rows)
                info = summarize(rows)
                delta = info['delta_pearson_squared']
                status = ('undefined_metric' if delta is None else 'no_increment' if abs(delta) < 1e-12
                          else 'worse' if delta < 0 else 'local_gain_unconfirmed')
                write_csv(pair, folder/f'{case}_validation_predictions.csv', float_format='%.12g')
                case_results[case] = {'implementation': 'implemented', 'result': status, **info}
            else:
                applicable = case == 'S1' or hypothesis not in ('H01', 'H05')
                case_results[case] = {'implementation': 'blocked_data' if applicable else 'not_applicable',
                                      'result': 'not_evaluated'}
        if metrics:
            write_csv(pd.DataFrame(metrics), folder/'metrics_by_profile.csv')
        stage[hypothesis] = {'cases': case_results, 'audit': audits[hypothesis]}
        write_json(folder/'summary.json', stage[hypothesis])
        write_json(folder/'run_manifest.json', {**shared_manifest, 'hypothesis': hypothesis,
                                                 'duration_seconds': time.perf_counter() - tick})
    write_json(OUT/'batches/batch_01/stage_results.json', {
        'hypotheses': stage, 'duration_seconds': time.perf_counter() - start})


if __name__ == '__main__':
    main()

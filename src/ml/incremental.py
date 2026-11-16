"""Evaluate opt-in PipeTwin features and export predictions from saved models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.storage import sha256, write_json
from .data import RepositoryData
from .incremental_features import (
    DEFECT_STATS, SCALES_M, attach_defect_stats, elevation_columns,
    multiscale_elevation, prior_inspection_stats,
)
from .next_common import (
    evaluate_predictions, fit_ridge, json_safe, load_artifact, make_weights,
    predict_ridge, s1_fit_rows, s1_production_table, s1_training_table,
    save_artifact, utcnow, write_csv, write_manifest,
)
from .s1_contract import S1_CUTOFF, S1_ID, nominal_frame, target_key_grid
from .s2_next import fit_spatial_residual
from .submission import COLUMNS, export_result
from .targeted_audit import tail_metrics
from .targeted_s1 import CORE_FEATURES, _block_folds, _physical_block_frame
from .targeted_s1_stage4 import SCHEMES
from .targeted_s2 import (
    S2_CUTOFF, SPATIAL_FEATURES, _fit_physics, _predict_candidate,
    _predict_physics, prepare_full_grid, prepare_target_full,
)

KEYS = ['id', 'segment_id']
FORECAST_KEYS = KEYS + ['prediction_date']
S1_BASE = 'global_ridge_geometry'
S2_BASE = 'spatial_no_temperature_centered'
UNSUPPORTED = {
    's1_enable_alignment': 'DEC-0002 requires Distance. No verified independent '
                           'pre-cutoff section anchor map is provided.',
    's1_enable_residual_correction': 'No independent forward OOF base forecasts: '
                                     'all eligible S1 fit labels are one campaign.',
}


def read_config(path):
    config = json.loads(path.read_text(encoding='utf-8'))
    flags = ['s1_enable_defect_stats', 's2_enable_multiscale_elevation', *UNSUPPORTED]
    for flag in flags:
        if type(config.get(flag)) is not bool:
            raise ValueError(f'{flag} must be true or false')
        if config[flag] and flag in UNSUPPORTED:
            raise ValueError(UNSUPPORTED[flag])
    variant = config['s1_defect_stats_variant']
    widths = config['s2_elevation_scales_m']
    if variant not in DEFECT_STATS:
        raise ValueError(f'Unknown S1 feature set: {variant}')
    if (not isinstance(widths, list) or len(widths) not in (1, len(SCALES_M))
            or any(type(width) is not int or width not in SCALES_M for width in widths)
            or len(set(widths)) != len(widths)):
        raise ValueError(f'Expected one elevation scale or all three from {SCALES_M}')
    config['s2_elevation_scales_m'] = sorted(widths)
    return config


def s2_variant(scales):
    return 'multiscale_' + '_'.join(str(width) for width in scales)


def _s1_stats(data, frames):
    return prior_inspection_stats(data.read('data/cleaned/vtd'),
                                  data.read('processed/segments_grid'),
                                  pd.concat(frames, ignore_index=True))


def _fit_s1(frame, features, name):
    return fit_ridge(frame, features, 'target_delta_mm',
                     make_weights(frame, 'row_uniform'), alpha=10.0,
                     baseline='last_mm', model_name=name)


def _fit_s2(train, intervals, variants):
    physics = _fit_physics(train, intervals)
    level = _predict_physics(physics, train, intervals)
    artifacts = {}
    for name, features in variants.items():
        spatial = fit_spatial_residual(train, level, features, 'ridge')
        artifacts[name] = {'physics': physics, 'centered': True, 'spatial': spatial,
                           'metadata': {**spatial['metadata'], 'candidate': name,
                                        'centering': 'length-weighted full grid',
                                        'physics_metadata': physics.metadata()}}
    return artifacts


def _score(frame, prediction, case, scheme, experiment, fold_count):
    evaluation = evaluate_predictions(frame, prediction)
    values = evaluation['pooled']
    tail = tail_metrics(frame, prediction, truth='next_mm', section='id')
    return {
        'case': case, 'scheme': scheme, 'experiment': experiment,
        'candidate': prediction, 'folds': fold_count, 'n': values['n'],
        **{key: values[key] for key in ('mae_mm', 'rmse_mm', 'bias_mm',
                                       'pearson_r', 'pearson_squared', 'r2')},
        'within_section_r': evaluation['within_section_centered']['pearson_r'],
        'top10_recall': tail['global_top10_recall'],
        'within_section_top10_recall': tail['lower_tail_recall'],
        'lower_tail_mae_mm': tail['global_lower_tail_mae_mm'],
    }


def _compare(actual, expected, keys, columns, atol=1e-9):
    if 'prediction_date' in keys:
        actual = actual.assign(prediction_date=actual.prediction_date.astype('datetime64[ns]'))
        expected = expected.assign(prediction_date=expected.prediction_date.astype('datetime64[ns]'))
    left = actual.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    pd.testing.assert_index_equal(left.index, right.index)
    pd.testing.assert_frame_equal(left[columns], right[columns], check_dtype=False,
                                  rtol=0, atol=atol)
    return float(np.abs(left[columns].to_numpy() - right[columns].to_numpy()).max())


def evaluate_s1(fit, variants, baseline_root, output):
    rows, checks = [], {}
    first = {'name': '100m_20m', 'block_m': 100.0, 'buffer_m': 20.0}
    for scheme in (first, *SCHEMES):
        name, width, buffer = scheme['name'], scheme['block_m'], scheme['buffer_m']
        target = _physical_block_frame(fit[fit.id.eq(S1_ID)], width)
        folds = _block_folds(fit, target, width, buffer)
        parts = []
        for fold in folds:
            test, train = fold['test'], fold['train_global']
            part = test[FORECAST_KEYS + ['block', 'next_mm']].copy()
            part['s1a'] = test.last_mm
            for candidate, features in variants.items():
                part[candidate] = predict_ridge(_fit_s1(train, features, candidate), test)
            parts.append(part)
        oof = pd.concat(parts, ignore_index=True)
        reference = Path('s1') if name == first['name'] else Path('s1_robust') / name
        verified = pd.read_csv(baseline_root / reference / 'oof_predictions.csv',
                               parse_dates=['prediction_date'])
        checks[name] = _compare(oof, verified, FORECAST_KEYS + ['block'],
                                ['next_mm', 's1a', S1_BASE])
        write_csv(oof, output / f's1_{name}_oof.csv', float_format='%.10f')
        rows.extend(_score(oof, column, 'S1', name,
                           'reference' if column == 's1a' else
                           'E0' if column == S1_BASE else 'E2', len(folds))
                    for column in ['s1a', *variants])
    return rows, checks


def evaluate_s2(strict, full_grids, intervals, variants, baseline_root, output):
    parts = []
    for pid, test in strict.groupby('id', sort=True):
        (date,) = test.prediction_date.drop_duplicates()
        full = full_grids[(int(pid), date)]
        if not full.segment_id.is_unique:
            raise ValueError(f'Duplicate full-grid S2 keys for {pid}')
        train = strict[~strict.id.eq(pid)]
        artifacts = _fit_s2(train, intervals, variants)
        level = _predict_physics(artifacts[S2_BASE]['physics'], full, intervals)
        values = full[KEYS].assign(physics_priors=level)
        for candidate, artifact in artifacts.items():
            values[candidate] = _predict_candidate(artifact, full, intervals)['h_clipped']
        parts.append(test[FORECAST_KEYS + ['next_mm']].merge(
            values, on=KEYS, validate='one_to_one'))
    oof = pd.concat(parts, ignore_index=True)
    verified = pd.read_csv(baseline_root / 's2/oof_predictions.csv',
                           parse_dates=['prediction_date'])
    check = _compare(oof, verified, FORECAST_KEYS, ['next_mm', 'physics_priors', S2_BASE])
    write_csv(oof, output / 's2_section_oof.csv', float_format='%.10f')
    columns = ['physics_priors', *variants]
    overall = [_score(oof, name, 'S2', 'section_LOO',
                      'reference' if name == 'physics_priors' else
                      'E0' if name == S2_BASE else 'E5', len(parts)) for name in columns]
    sections = [_score(part, name, 'S2', f'section_{pid}', 'section', 1)
                for pid, part in oof.groupby('id', sort=True) for name in columns]
    return overall, sections, check


def select_candidates(s1, s2, sections):
    """Use positive Pearson r² first and preserve pooled worst-decile recall."""
    s1 = pd.DataFrame(s1)
    reference = s1[s1.candidate.eq(S1_BASE)].set_index('scheme')
    accepted_s1 = []
    for name, group in s1[s1.experiment.eq('E2')].groupby('candidate'):
        rows = group.set_index('scheme').loc[reference.index]
        if ((rows.pearson_r > 0) & (rows.pearson_squared > reference.pearson_squared)
                & (rows.top10_recall >= reference.top10_recall)).all():
            accepted_s1.append((float(rows.pearson_squared.mean()), name))
    s2 = pd.DataFrame(s2).set_index('candidate')
    sections = pd.DataFrame(sections)
    baseline = s2.loc[S2_BASE]
    reference_sections = sections[sections.candidate.eq(S2_BASE)].set_index('scheme')
    accepted_s2 = []
    for name, row in s2[s2.experiment.eq('E5')].iterrows():
        by_section = sections[sections.candidate.eq(name)].set_index('scheme').loc[
            reference_sections.index]
        improved = ((by_section.pearson_r > 0)
                    & (by_section.pearson_r > reference_sections.pearson_r)).sum()
        if (row.pearson_r > 0 and row.pearson_squared > baseline.pearson_squared
                and row.top10_recall >= baseline.top10_recall
                and improved >= len(reference_sections) // 2 + 1):
            accepted_s2.append((float(row.pearson_squared), name))
    return (max(accepted_s1, default=(0, S1_BASE))[1],
            max(accepted_s2, default=(0, S2_BASE))[1])


def _s1_challenger(rows):
    scores = pd.DataFrame(rows)
    baseline = scores[scores.candidate.eq(S1_BASE)].set_index('scheme')
    alternatives = []
    for name, group in scores[scores.experiment.eq('E2')].groupby('candidate'):
        candidate = group.set_index('scheme').loc[baseline.index]
        if ((candidate.pearson_r > 0)
                & (candidate.pearson_squared > baseline.pearson_squared)).all():
            alternatives.append((float(candidate.pearson_squared.mean()), name))
    return max(alternatives, default=(0, None))[1]


def _prior_coverage(fit, prod, stats):
    own_prod = attach_defect_stats(prod, stats).loc[prod.has_own_prior_vtd]
    train_counts = np.rint(np.expm1(fit.prior_measured_count_log))
    prod_counts = np.rint(np.expm1(own_prod.prior_measured_count_log))

    def describe(counts):
        return {'rows': len(counts), 'single_measurement': int(counts.eq(1).sum()),
                'median': float(counts.median()), 'max': int(counts.max())}

    return {'fit': describe(train_counts), 'target_with_own_prior': describe(prod_counts),
            'target_above_fit_max': int(prod_counts.gt(train_counts.max()).sum())}


def _s1_predict(data, artifact, stats_variant=None):
    prod = s1_production_table(data)
    if stats_variant:
        prod = attach_defect_stats(prod, _s1_stats(data, [prod]))
    own = prod.has_own_prior_vtd
    predictions = prod.s1b_prediction.to_numpy(float).copy()
    predictions[own] = predict_ridge(artifact, prod.loc[own])
    return prod[KEYS].assign(thickness_pred=predictions)


def _s2_predict(data, artifact, scales=()):
    grid, interval = prepare_target_full(data)
    if scales:
        geometry = multiscale_elevation(data.read('processed/profile_by_segment'), scales)
        grid = grid.merge(geometry, on=KEYS, validate='one_to_one')
    intervals = {(int(grid.id.iloc[0]), pd.Timestamp(S2_CUTOFF)): interval}
    return grid[KEYS].assign(thickness_pred=_predict_candidate(
        artifact, grid, intervals)['h_clipped'])


def _artifact_path(output, name):
    return output / 'models' / name / 'model_artifact.joblib'


def predict(data, output, result_path, config, root=BASE_DIR, check_baseline=True):
    s1_variant = config['s1_defect_stats_variant'] if config['s1_enable_defect_stats'] else None
    scales = config['s2_elevation_scales_m'] if config['s2_enable_multiscale_elevation'] else ()
    s1_name = f'defect_stats_{s1_variant}' if s1_variant else S1_BASE
    s2_name = s2_variant(scales) if scales else S2_BASE
    s1 = _s1_predict(data, load_artifact(_artifact_path(output, s1_name)), s1_variant)
    s2 = _s2_predict(data, load_artifact(_artifact_path(output, s2_name)), scales)
    bounds = nominal_frame(data).rename(columns={'nominal_thickness_mm': 'h0_mm'})
    result = pd.concat([s1, s2], ignore_index=True)[COLUMNS]
    validation, _ = export_result(result, result_path, target_key_grid(data), bounds)
    validation['sha256'] = sha256(result_path)
    if check_baseline and not (s1_variant or scales):
        freeze = root / 'experiments/targeted_cycle/frozen_candidates'
        manifest = json.loads((freeze / 'candidate_freeze.json').read_text(encoding='utf-8'))
        baselines = [('S1', 'global_ridge_geometry_correction'), ('S2', S2_BASE)]
        written = pd.read_csv(result_path)
        for case, name in baselines:
            path = freeze / 'predictions' / case / f'{name}.csv'
            if sha256(path) != manifest['candidates'][name]['frozen_prediction_sha256']:
                raise ValueError(f'Frozen prediction hash mismatch: {name}')
            frozen = pd.read_csv(path)
            rows = written[written.id.isin(frozen.id)]
            validation[f'all_off_{case}_max_abs_mm'] = _compare(
                rows, frozen, KEYS, ['thickness_pred'])
    return validation


def evaluate(root, data, baseline_root, output, config):
    if output.exists():
        raise FileExistsError(f'Choose a new experiment output directory: {output}')
    if any(config[name] for name in ('s1_enable_defect_stats', 's2_enable_multiscale_elevation')):
        raise ValueError('Evaluate from all-off config; predict activates selected switches')
    started = utcnow()
    s1_variants = {S1_BASE: CORE_FEATURES,
                   **{f'defect_stats_{name}': CORE_FEATURES + features
                      for name, features in DEFECT_STATS.items()}}
    s2_variants = {S2_BASE: SPATIAL_FEATURES}
    for widths in [(width,) for width in SCALES_M] + [SCALES_M]:
        s2_variants[s2_variant(widths)] = SPATIAL_FEATURES + elevation_columns(widths)
    paths = {
        'S1_100': baseline_root / 's1/oof_predictions.csv',
        'S1_200': baseline_root / 's1_robust/200m_50m/oof_predictions.csv',
        'S1_500': baseline_root / 's1_robust/500m_100m/oof_predictions.csv',
        'S2': baseline_root / 's2/oof_predictions.csv',
    }
    write_json(output / 'protocol.json', {
        'config': config, 'primary': 'positive signed Pearson r, then Pearson r²',
        'gate': 'S1 improves r² on all three schemes with no pooled top-decile recall loss; '
                'S2 improves pooled r² and signed r in a majority of sections with no '
                'pooled top-decile recall loss. MAE is reported as secondary.',
        's1_variants': s1_variants, 's2_variants': s2_variants,
        'baseline_oof_sha256': {name: sha256(path) for name, path in paths.items()},
        'S1_cutoff': str(S1_CUTOFF.date()), 'S2_cutoff': str(S2_CUTOFF.date()),
        'ridge_alpha': 10.0, 'random_seed': None,
    })
    fit = s1_fit_rows(s1_training_table(data))
    prod = s1_production_table(data)
    stats = _s1_stats(data, [fit, prod])
    fit = attach_defect_stats(fit, stats)
    s1_rows, s1_checks = evaluate_s1(fit, s1_variants, baseline_root, output)
    for name, features in s1_variants.items():
        save_artifact(_fit_s1(fit, features, name), _artifact_path(output, name).parent)

    prepared = prepare_full_grid(data)
    geometry = multiscale_elevation(data.read('processed/profile_by_segment'), SCALES_M)
    strict = prepared['strict'].merge(geometry, on=KEYS, validate='one_to_one')
    grids = {key: frame.merge(geometry, on=KEYS, validate='one_to_one')
             for key, frame in prepared['full_frames'].items() if key[1] < S2_CUTOFF}
    intervals = prepared['interval_map']
    s2_rows, sections, s2_check = evaluate_s2(
        strict, grids, intervals, s2_variants, baseline_root, output)
    for name, artifact in _fit_s2(strict, intervals, s2_variants).items():
        save_artifact(artifact, _artifact_path(output, name).parent)

    s1_choice, s2_choice = select_candidates(s1_rows, s2_rows, sections)
    selected = dict(config)
    if s1_choice != S1_BASE:
        selected['s1_enable_defect_stats'] = True
        selected['s1_defect_stats_variant'] = s1_choice.removeprefix('defect_stats_')
    if s2_choice != S2_BASE:
        selected['s2_enable_multiscale_elevation'] = True
        selected['s2_elevation_scales_m'] = [int(width) for width in s2_choice.split('_')[1:]]
    e6 = [{**row, 'experiment': 'E6'} for row in s1_rows + s2_rows
          if row['candidate'] in (s1_choice, s2_choice)]
    write_csv(pd.DataFrame(s1_rows + s2_rows + e6), output / 'metrics.csv',
              float_format='%.10f')
    write_csv(pd.DataFrame(sections), output / 's2_section_metrics.csv', float_format='%.10f')
    write_json(output / 'e6_config.json', selected)
    checks = predict(data, output, output / 'e0/result.csv', config, root)
    proposal = predict(data, output, output / 'e6/result.csv', selected, root)
    challenger = _s1_challenger(s1_rows)
    challenger_run = None
    if challenger and challenger != s1_choice:
        trial = {**config, 's1_enable_defect_stats': True,
                 's1_defect_stats_variant': challenger.removeprefix('defect_stats_')}
        directory = output / 's1_challenger'
        write_json(directory / 'config.json', trial)
        challenger_run = {
            's1': challenger, 's2': S2_BASE, 'status': 'candidate for external evaluation',
            'validation': predict(data, output, directory / 'result.csv', trial, root),
        }
    write_json(output / 'decision.json', json_safe({
        'E0': {'baseline_oof_max_abs_mm': {**s1_checks, 'S2': s2_check}, **checks},
        'E1': {'status': 'unavailable', 'reason': UNSUPPORTED['s1_enable_alignment']},
        'E3': {'status': 'unavailable', 'reason': 'E1 has no admissible alignment; see E2 for stats'},
        'E4': {'status': 'unavailable', 'reason': UNSUPPORTED['s1_enable_residual_correction']},
        'E6': {'s1': s1_choice, 's2': s2_choice, 'validation': proposal,
               'status': 'historical diagnostic selection; external evaluation pending'},
        's1_challenger': challenger_run,
        'historical_prior_coverage': _prior_coverage(fit, prod, stats),
        'S1_fit_dates': sorted(str(day.date()) for day in fit.prediction_date.drop_duplicates()),
        'S2_fit_dates': sorted(str(day.date()) for day in strict.prediction_date.drop_duplicates()),
    }))
    audit_files = ['configs/incremental.json', 'src/ml/incremental.py',
                   'src/ml/incremental_features.py', 'src/ml/targeted_s1.py',
                   'src/ml/targeted_s1_stage4.py', 'src/ml/targeted_s2.py']
    artifacts = [*output.glob('models/*/*'), *output.glob('*/result.csv')]
    write_manifest(output / 'run_manifest.json', root=root, data=data, output=output,
                   stage='incremental', started=started,
                   extra={'experiment_code_sha256': {name: sha256(root / name)
                                                     for name in audit_files},
                          'artifact_sha256': {path.relative_to(output).as_posix(): sha256(path)
                                              for path in sorted(artifacts)}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate', 'predict'])
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, default=Path('scratch/incremental_baseline'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    config = read_config(args.config or args.root / 'configs/incremental.json')
    data = RepositoryData(args.data_root)
    if args.command == 'evaluate':
        evaluate(args.root, data, args.baseline_root, args.output_root, config)
    else:
        if args.result is None or args.result.exists():
            parser.error('predict requires --result pointing to a new file')
        validation = predict(data, args.output_root, args.result, config, args.root)
        print(json.dumps(validation, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Train and export the selected PipeTwin production forecast."""
from __future__ import annotations

import argparse
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from src.config import BASE_DIR
from src.storage import sha256, write_json
from .data import RepositoryData
from .experiment_v2 import profile_spatial
from .next_common import json_safe, s1_production_table, versions, write_csv
from .production_models import (
    FORECAST_KEYS, H18_FEATURES, H31_PARAMETERS, cold_grid, cold_training, fit_s1_models,
    fit_s2_models, predict_s1_models, predict_s2_models, s1_prediction_frame, s1_training,
)
from .s1_contract import S1_CUTOFF, S1_ID, S2_ID, nominal_frame, target_key_grid
from .s1_joint_alignment import verified_target_anchors
from .submission import COLUMNS, export_result, validate_submission
from .targeted_s2 import S2_CUTOFF

DEFAULT_CONFIG = BASE_DIR / 'configs/production_forecast.json'
DEFAULT_DATA = BASE_DIR / 'scratch/odo_alignment_verified'
BUNDLE_FILE = 'forecast_models.joblib'
BUNDLE_VERSION = 'pipetwin-production-v1'
METHODS = {'S1': ('baseline', 'H18'), 'S2': ('baseline', 'H31')}


def load_config(path):
    config = json.loads(Path(path).read_bytes())
    if set(config) != {'s1_method', 's2_method', 'previous_result', 'h31'}:
        raise ValueError('Invalid production forecast configuration fields')
    if config['s1_method'] not in METHODS['S1'] or config['s2_method'] not in METHODS['S2']:
        raise ValueError('Invalid method for S1 or S2')
    parameters = config['h31']
    if (set(parameters) != set(H31_PARAMETERS)
            or not all(type(value) in (int, float) and np.isfinite(value) and value > 0
                       for value in parameters.values())):
        raise ValueError('Invalid H31 covariance parameters')
    if not isinstance(config['previous_result'], str) or not config['previous_result']:
        raise ValueError('Expected an explicit previous result path')
    return config


@contextmanager
def new_output(path):
    """Publish a complete new output folder after successful validation."""
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f'Choose a new output directory: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.pipetwin-', dir=path.parent) as temp:
        intermediate = Path(temp).resolve() / 'output'
        if intermediate.parent.parent != path.parent:
            raise ValueError('Output staging must stay in the requested directory parent')
        intermediate.mkdir()
        yield intermediate
        if path.exists():
            raise FileExistsError(f'Output appeared while running: {path}')
        intermediate.rename(path)


def _provenance(data):
    paths = sorted((BASE_DIR / 'src').rglob('*.py'))
    return {
        'environment': {**versions(), 'scipy': scipy.__version__},
        'preprocessing_manifest_sha256': sha256(data.root / 'reports/run_manifest.json'),
        'input_sha256': dict(data.hashes),
        'code_sha256': {path.relative_to(BASE_DIR).as_posix(): sha256(path) for path in paths},
    }


def train(data, config, output):
    raw = data.read('data/cleaned/vtd')
    grid = data.read('processed/segments_grid')
    history = s1_training(data, raw, grid)
    historical_s2, _, intervals = cold_training(data)
    first = fit_s1_models(history)
    second = fit_s2_models(historical_s2, intervals, config['h31'])
    bundle = {'version': BUNDLE_VERSION, 'config': config, 'S1': first, 'S2': second}
    with new_output(output) as staging:
        checksum_path = staging / BUNDLE_FILE
        joblib.dump(bundle, checksum_path, compress=3)
        s1_keys = history[FORECAST_KEYS + ['last_vtd_date']].assign(case='S1')
        s2_keys = historical_s2[FORECAST_KEYS].assign(case='S2')
        write_csv(pd.concat([s1_keys, s2_keys], ignore_index=True), staging / 'training_keys.csv')
        write_json(staging / 'manifest.json', json_safe({
            'version': BUNDLE_VERSION,
            'bundle_sha256': sha256(checksum_path),
            'config': config,
            'S1': {'cutoff': str(S1_CUTOFF.date()), 'training_rows': len(history),
                   'H18_features': H18_FEATURES,
                   'models': {name: model['metadata'] for name, model in first.items()}},
            'S2': {'cutoff': str(S2_CUTOFF.date()), 'training_rows': len(historical_s2),
                   'training_ids': sorted(int(n) for n in historical_s2.id.unique()),
                   'training_dates': sorted(str(date.date()) for date in historical_s2.prediction_date.unique()),
                   'covariance': second['H31']['parameters'],
                   'physics': second['physics'].metadata(), 'baseline': second['baseline']['metadata']},
            **_provenance(data),
        }))


def load_bundle(path):
    manifest = json.loads((path / 'manifest.json').read_bytes())
    model_path = path / BUNDLE_FILE
    if manifest['version'] != BUNDLE_VERSION or sha256(model_path) != manifest['bundle_sha256']:
        raise ValueError('Forecast bundle version or checksum mismatch')
    model = joblib.load(model_path)
    if model['version'] != BUNDLE_VERSION or model['config'] != manifest['config']:
        raise ValueError('Forecast bundle does not match its configuration')
    return model, manifest['bundle_sha256']


def assemble(s1, s2, s1_method, s2_method):
    if s1_method not in METHODS['S1'] or s2_method not in METHODS['S2']:
        raise ValueError('Requested method is not available for that case')
    if (not s1.id.eq(S1_ID).all() or not s1.prediction_date.eq(S1_CUTOFF).all()
            or not s2.id.eq(S2_ID).all() or not s2.prediction_date.eq(S2_CUTOFF).all()):
        raise ValueError('Unexpected case ID or prediction date in export')
    pieces = [frame[['id', 'segment_id', method]].rename(columns={method: 'thickness_pred'})
              for frame, method in [(s1, s1_method), (s2, s2_method)]]
    return pd.concat(pieces, ignore_index=True)[COLUMNS]


def predict(data, bundle_path, output, *, anchors_path=None, template_path=None,
            s1_method=None, s2_method=None):
    bundle, bundle_sha = load_bundle(bundle_path)
    config = bundle['config']
    selected = (s1_method or config['s1_method'], s2_method or config['s2_method'])
    geometry = data.read('processed/segments_grid')
    raw = data.read('data/cleaned/vtd')
    s1, audit = s1_prediction_frame(
        s1_production_table(data), raw, geometry.loc[geometry.id.eq(S1_ID)],
        verified_target_anchors(anchors_path) if anchors_path is not None else None)
    if anchors_path and audit['prior_source_date'] != '2022-07-06':
        raise ValueError('Verified S1 anchors belong to the 2022-07-06 inspection')
    s1_prediction = predict_s1_models(bundle['S1'], s1)
    spatial = profile_spatial(data.read('processed/profile_by_segment'))
    s2, exposure = cold_grid(data, S2_ID, S2_CUTOFF, spatial)
    s2_prediction = predict_s2_models(bundle['S2'], s2, {(S2_ID, S2_CUTOFF): exposure})
    keys = target_key_grid(data)
    bounds = nominal_frame(data).rename(columns={'nominal_thickness_mm': 'h0_mm'})
    previous_path = BASE_DIR / config['previous_result']
    previous = pd.read_csv(previous_path)
    validate_submission(previous, keys, bounds)
    combinations = {'baseline': ('baseline', 'baseline'), 'h18_only': ('H18', 'baseline'),
                    'h31_only': ('baseline', 'H31'), 'h18_h31': ('H18', 'H31')}
    with new_output(output) as staging:
        selected_result = assemble(s1_prediction, s2_prediction, *selected)
        validation, _ = export_result(selected_result, staging / 'result.csv', keys, bounds, template_path)
        outputs = {'result.csv': sha256(staging / 'result.csv')}
        for name, methods in combinations.items():
            relative = f'ablation/{name}.csv'
            export_result(assemble(s1_prediction, s2_prediction, *methods), staging / relative,
                          keys, bounds, template_path)
            outputs[relative] = sha256(staging / relative)
        shutil.copyfile(previous_path, staging / 'previous_result.csv')
        outputs['previous_result.csv'] = sha256(staging / 'previous_result.csv')
        baseline = pd.read_csv(staging / 'ablation/baseline.csv')
        aligned = previous.merge(baseline, on=['id', 'segment_id'], validate='one_to_one')
        differences = (aligned.thickness_pred_x - aligned.thickness_pred_y).abs()
        case_differences = {case: float(differences[aligned.id.eq(pid)].max())
                            for case, pid in [('S1', S1_ID), ('S2', S2_ID)]}
        write_csv(s1_prediction, staging / 'predictions_s1.csv', float_format='%.15g')
        write_csv(s2_prediction, staging / 'predictions_s2.csv', float_format='%.15g')
        write_json(staging / 'manifest.json', json_safe({
            'version': BUNDLE_VERSION, 'bundle_sha256': bundle_sha,
            'methods': dict(zip(('S1', 'S2'), selected)),
            'S1': {**audit, 'applied_h18_rows': int(s1_prediction.h18_correction_mm.abs().gt(1e-10).sum()),
                   'anchors_sha256': sha256(anchors_path) if anchors_path is not None else None},
            'S2': {'scenario': 'cold_start', 'rows': len(s2_prediction),
                   'h31_clipped_rows': int(s2_prediction.h31_clipped.sum())},
            'previous_result_source': config['previous_result'], 'baseline_replay_max_abs_mm': case_differences,
            'submission_validation': validation, 'outputs_sha256': outputs, **_provenance(data),
        }))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train')
    infer = commands.add_parser('predict')
    for command in (fit, infer):
        command.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
        command.add_argument('--output-root', type=Path, required=True)
    fit.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    infer.add_argument('--model-root', type=Path, required=True)
    infer.add_argument('--verified-s1-anchors', type=Path)
    infer.add_argument('--template', type=Path)
    infer.add_argument('--s1', choices=METHODS['S1'])
    infer.add_argument('--s2', choices=METHODS['S2'])
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        data = RepositoryData(args.data_root)
        if args.command == 'train':
            train(data, load_config(args.config), args.output_root)
        else:
            predict(data, args.model_root, args.output_root, anchors_path=args.verified_s1_anchors,
                    template_path=args.template, s1_method=args.s1, s2_method=args.s2)


if __name__ == '__main__':
    main()

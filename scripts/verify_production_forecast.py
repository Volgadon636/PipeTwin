"""Check saved-model inference, exact exports, rollback copy and case isolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from threadpoolctl import threadpool_limits

from src.ml.data import RepositoryData
from src.ml.production import DEFAULT_DATA, load_bundle, predict
from src.ml.s1_contract import S1_CUTOFF, S1_ID, S2_ID
from src.ml.targeted_s2 import S2_CUTOFF
from src.storage import sha256, write_json


def verify(data_root, model_root, forecast_root, output_root):
    bundle, digest = load_bundle(model_root)
    if (bundle['config']['s1_method'], bundle['config']['s2_method']) != ('H18', 'H31'):
        raise ValueError('Expected the selected H18/S1 and H31/S2 bundle')
    reader = RepositoryData.read

    def prediction_inputs_only(data, stem):
        if stem == 'processed/training_targets':
            raise AssertionError('Saved inference must not read training labels')
        return reader(data, stem)

    fail = AssertionError('Saved inference must not fit models')
    with (patch.object(RepositoryData, 'read', prediction_inputs_only),
          patch('src.ml.production.train', side_effect=fail),
          patch('src.ml.production.fit_s1_models', side_effect=fail),
          patch('src.ml.production.fit_s2_models', side_effect=fail),
          patch('src.ml.next_common.fit_ridge', side_effect=fail),
          patch('src.ml.s1_model.fit_s1_ridge', side_effect=fail)):
        predict(RepositoryData(data_root), model_root, output_root)
    if sha256(forecast_root / 'result.csv') != sha256(output_root / 'result.csv'):
        raise AssertionError('Fresh-process saved inference changed result.csv')
    frames = {name: pd.read_csv(output_root / 'ablation' / f'{name}.csv')
              for name in ['baseline', 'h18_only', 'h31_only', 'h18_h31']}
    for name, other, pid in [('h18_only', 'baseline', S2_ID), ('h31_only', 'baseline', S1_ID),
                             ('h18_h31', 'h18_only', S1_ID), ('h18_h31', 'h31_only', S2_ID)]:
        pd.testing.assert_frame_equal(frames[name].loc[lambda f: f.id.eq(pid)].reset_index(drop=True),
                                      frames[other].loc[lambda f: f.id.eq(pid)].reset_index(drop=True))
    previous = pd.read_csv(output_root / 'previous_result.csv')
    pd.testing.assert_frame_equal(frames['baseline'], previous)
    if frames['h18_h31'].groupby('id').size().to_dict() != {S1_ID: 525, S2_ID: 1475}:
        raise AssertionError('Unexpected target row counts')
    keys = pd.read_csv(model_root / 'training_keys.csv', parse_dates=['prediction_date', 'last_vtd_date'])
    s1, s2 = keys.loc[keys.case.eq('S1')], keys.loc[keys.case.eq('S2')]
    if (not s1.prediction_date.lt(S1_CUTOFF).all()
            or not s1.last_vtd_date.lt(s1.prediction_date).all()
            or s1.id.eq(S2_ID).any() or s2.id.eq(S2_ID).any()
            or not s2.prediction_date.lt(S2_CUTOFF).all()):
        raise AssertionError('Invalid training cutoff or target-pipe diagnostic')
    source = json.loads((forecast_root / 'manifest.json').read_bytes())
    for relative, checksum in source['outputs_sha256'].items():
        if sha256(forecast_root / relative) != checksum:
            raise AssertionError(f'Forecast checksum mismatch: {relative}')
    result = {'saved_inference_without_fit_or_training_labels': True,
              'fresh_process_result_byte_identical': True, 'case_isolation_all_ablations': True,
              'previous_baseline_exact': True, 'training_cutoffs_and_s2_exclusion': True,
              'forecast_output_checksums': True, 'target_rows': {'S1': 525, 'S2': 1475},
              'bundle_sha256': digest, 'result_sha256': sha256(output_root / 'result.csv'),
              'S1_training_rows': len(s1), 'S2_training_rows': len(s2)}
    write_json(output_root / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--forecast-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        verify(args.data_root, args.model_root, args.forecast_root, args.output_root)


if __name__ == '__main__':
    main()

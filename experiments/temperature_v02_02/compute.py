"""Reproduce Stage 02 feature evidence, without fitting or exporting predictions.

Run from the repository root:
  .venv/bin/python -m experiments.temperature_v02_02.compute
"""
import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.ml.data import RepositoryData
from src.ml.exposure import exposure_intervals
from src.ml.temperature import (
    EA_CANDIDATES, REFERENCE_TEMPERATURE_K, R_J_MOL_K,
    THERMAL_FEATURE_COLUMNS, THERMAL_METADATA_COLUMNS,
    TemperatureFeatureRegistry, attach_temperature_features, temperature_features,
    temperature_intervals,
)
from src.ml.windows import build_training_windows, make_snapshot
from src.storage import sha256

STAGE_ROOT = BASE_DIR / 'experiments/temperature_v02_02'
KEYS = ['main_pipeline_id', 'id', 'segment_id', 'case', 'start_date', 'prediction_date']
GROUP_KEYS = ['main_pipeline_id', 'id', 'case', 'start_date', 'prediction_date']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def generate(output):
    output = Path(output).resolve()
    # Evidence must stay in this stage; prohibit overwriting a protected result.
    if output != STAGE_ROOT and STAGE_ROOT not in output.parents:
        raise ValueError('Output must be inside experiments/temperature_v02_02')
    output.mkdir(parents=True, exist_ok=True)
    data = RepositoryData()
    parameters = dict(ea_kj_mol=40.0, gap_policy='reference', max_hold_days=1.0)
    windows, histories = build_training_windows(data)
    historical, historical_reports = attach_temperature_features(windows, histories, **parameters)
    targets = []
    for request in data.config['targets']:
        frame, rows = make_snapshot(data, request)
        histories[(int(request['id']), pd.Timestamp(request['prediction_date']))] = rows
        targets.append(frame)
    production, production_reports = attach_temperature_features(pd.concat(targets, ignore_index=True), histories, **parameters)
    columns = KEYS + list(THERMAL_FEATURE_COLUMNS) + list(THERMAL_METADATA_COLUMNS)
    for name, frame in [('historical_features', historical), ('production_features', production)]:
        frame[columns].to_csv(output / f'{name}.csv', index=False, float_format='%.17g')

    audit_rows, all_intervals = [], []
    for dataset, frame in [('historical', historical), ('production', production)]:
        for key, group in frame.groupby(GROUP_KEYS, sort=True):
            parent, pid, case, start, cutoff = key
            rows = histories[(int(pid), cutoff)]
            intervals, _ = temperature_intervals(rows, start, cutoff)
            _, joint = exposure_intervals(rows, start, cutoff)
            record = dict(zip(GROUP_KEYS, key))
            record.update(dataset=dataset, segment_count=len(group),
                          **group.iloc[0][list(THERMAL_FEATURE_COLUMNS)+list(THERMAL_METADATA_COLUMNS)].to_dict(),
                          joint_observed_years=joint['observed_years'],
                          joint_coverage_fraction=joint['coverage_fraction'],
                          thermal_usable_rows=len(intervals),
                          thermal_valid_days=int(intervals.date.dt.normalize().nunique()),
                          thermal_temp_min_C=intervals.t_mix_usable.min(),
                          thermal_temp_max_C=intervals.t_mix_usable.max(),
                          thermal_source_date_min=intervals.date.min(),
                          thermal_source_date_max=intervals.date.max(),
                          thermal_time_span=(f'{intervals.date.min().isoformat()}/{intervals.date.max().isoformat()}'
                                             if len(intervals) else None),
                          thermal_sources='cutoff-specific hydraulic_asof.t_mix_usable',
                          thermal_observed_temperature_fallback_fraction=0.0)
            audit_rows.append(record)
            all_intervals.append(intervals.assign(dataset=dataset, main_pipeline_id=parent, case=case,
                                                  start_date=start, prediction_date=cutoff))
    pd.DataFrame(audit_rows).to_csv(output/'coverage_by_window.csv', index=False, float_format='%.17g')
    pd.concat(all_intervals, ignore_index=True).to_csv(output/'temperature_intervals.csv', index=False, float_format='%.17g')

    # Deterministic small examples include both policies and all predeclared Ea.
    examples = []
    for scenario, temperatures in [('measured_20C', [20.0, 20.0]),
                                   ('invalid_next', [0.0, np.nan]), ('no_history', [])]:
        dates = ['2020-01-01 12:00', '2020-01-02 00:00'][:len(temperatures)]
        rows = pd.DataFrame({'id': pd.Series([1]*len(dates), dtype=int),
                             'date': pd.to_datetime(dates), 't_mix_usable': pd.Series(temperatures, dtype=float)})
        for candidate, ea in EA_CANDIDATES.items():
            for policy in ['observed_only', 'reference']:
                examples.append(dict(scenario=scenario, id=1, start_date='2020-01-01', prediction_date='2020-01-04',
                                     **temperature_features(rows, '2020-01-01', '2020-01-04',
                                                            ea_kj_mol=ea, gap_policy=policy)))
    pd.DataFrame(examples).to_csv(output/'example_rows.csv', index=False, float_format='%.17g')
    # DataFrame JSON renders missing means/reference values as JSON null.
    (output/'example_rows.json').write_text(pd.DataFrame(examples).to_json(orient='records', indent=2, double_precision=15)+'\n')

    registry = TemperatureFeatureRegistry(histories, **parameters)
    catalog = []
    for name in registry.list_features():
        descriptor = registry.get(name)
        catalog.append({key: value for key, value in vars(descriptor).items() if key != 'compute_fn'})
    write_json(output/'feature_catalog.json', dict(
        numeric_features=catalog, metadata_columns=list(THERMAL_METADATA_COLUMNS),
        metadata_excluded_from_learnable_features=True,
        parameters=dict(**parameters, thermal_reference_temperature_K=REFERENCE_TEMPERATURE_K,
                        gas_constant_J_mol_K=R_J_MOL_K, candidates=dict(EA_CANDIDATES)),
        source_type='team_hypothesis', evidence_status='diagnostic_only',
        interpretation='EA40_mid is provisional; reference gaps are assumptions, never measured history'))
    write_json(output/'feature_audit.json', dict(
        historical=[r.to_dict() for r in historical_reports], production=[r.to_dict() for r in production_reports]))
    manifest = dict(
        git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE_DIR, text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=BASE_DIR, text=True).strip(),
        command='.venv/bin/python -m experiments.temperature_v02_02.compute',
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        parameters=parameters, historical_rows=len(historical), production_rows=len(production),
        input_sha256=data.hashes,
        code_sha256={str(p.relative_to(BASE_DIR)): sha256(p) for p in
                     [BASE_DIR/'src/ml/temperature.py', Path(__file__).resolve()]},
        output_sha256={p.name: sha256(p) for p in sorted(output.iterdir())
                       if p.name in ['historical_features.csv', 'production_features.csv', 'coverage_by_window.csv',
                                     'temperature_intervals.csv', 'example_rows.csv', 'example_rows.json',
                                     'feature_catalog.json', 'feature_audit.json']},
    )
    if any(sha256(BASE_DIR/name) != digest for name, digest in data.hashes.items()):
        raise ValueError('An input artifact changed during feature generation')
    write_json(output/'run_manifest.json', manifest)
    print(json.dumps(dict(historical_rows=len(historical), production_rows=len(production),
                         window_groups=len(audit_rows), output=str(output)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=STAGE_ROOT)
    generate(parser.parse_args().output_root)

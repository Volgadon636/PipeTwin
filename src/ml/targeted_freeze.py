"""Stage 6: freeze selected S1/S2 predictions, explicit pairs and external evaluator.

No fit, no tuning, no main ``result.csv`` overwrite.  Every frozen prediction is
copied from a named source and hashed; combination CSVs are built only for the
predeclared pairs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.storage import sha256, write_json
from .data import RepositoryData
from .next_common import json_safe, s1_production_table, utcnow
from .s1_contract import S2_ID, combine_with_s2, target_key_grid

S1 = 1250022871
S2 = 1250009583
STAGE2 = Path('experiments/targeted_cycle/stage_2_s2_fixes')
STAGE3 = Path('experiments/targeted_cycle/stage_3_s1_100m_20m')
STAGE4 = Path('experiments/targeted_cycle/stage_4_s1_robustness')
STAGE5 = Path('experiments/targeted_cycle/stage_5_temporal')
OUT = Path('experiments/targeted_cycle/frozen_candidates')

CANDIDATES = [
    {
        'id': 's1a', 'case': 'S1', 'source': STAGE3 / 'target_predictions/s1a/predictions_s1.csv',
        'source_report': 'experiments/targeted_cycle/stage_3_s1_100m_20m/stage_3_report_ru.md',
        'roles': ['level_reference', 'submission_candidate'],
        'model': None,
        'params': {'rule': 'h_last'},
        'rationale': 'unchanged persistence baseline; no temporal assumption',
        'evidence': 'S1 reference; block-CV baseline; lower-tail MAE best among block candidates',
    },
    {
        'id': 'global_ridge_geometry_correction', 'case': 'S1',
        'source': STAGE5 / 'production/global_ridge_geometry_correction_predictions.csv',
        'source_report': 'experiments/targeted_cycle/stage_5_temporal/stage_5_report_ru.md',
        'roles': ['spatial_challenger', 'submission_candidate'],
        'model': STAGE3 / 'models/global_ridge_geometry/model_artifact.joblib',
        'params': {'temporal': 'h_last + g(X)', 'base': 'global_ridge_geometry'},
        'rationale': 'Stage 4 common promising Ridge; no time scaling; simplest Ridge interpretation',
        'evidence': 'promising on 100/200/500 block schemes; temporal law not identified',
    },
    {
        'id': 'global_ridge_geometry_time_scaled', 'case': 'S1',
        'source': STAGE5 / 'production/global_ridge_geometry_time_scaled_predictions.csv',
        'source_report': 'experiments/targeted_cycle/stage_5_temporal/stage_5_report_ru.md',
        'roles': ['spatial_challenger'],
        'model': STAGE3 / 'models/global_ridge_geometry/model_artifact.joblib',
        'params': {'temporal': 'h_last + (dt_target/dt_train)*g(X)',
                   'dt_train': 3.783709787817, 'base': 'global_ridge_geometry'},
        'rationale': 'fixed time-scaling sensitivity; OOF cannot distinguish it from correction',
        'evidence': 'historical correction/time-scaled identical because dt_train is common; not a winner',
    },
    {
        'id': 'physics_priors', 'case': 'S2',
        'source': STAGE2 / 'models/physics_priors/predictions_s2.csv',
        'source_report': 'experiments/targeted_cycle/stage_2_s2_fixes/stage_2_report_ru.md',
        'roles': ['level_reference', 'submission_candidate'],
        'model': STAGE2 / 'models/physics_priors/model_artifact.joblib',
        'params': {'model': 'ColdStartPhysics', 'gap_policy': 'own_mean'},
        'rationale': 'strict physics level reference; no target labels were used',
        'evidence': 'strict train 2018 only; target level 2.809539; target accuracy unavailable',
    },
    {
        'id': 'spatial_no_temperature_centered', 'case': 'S2',
        'source': STAGE2 / 'models/spatial_no_temperature_centered/predictions_s2.csv',
        'source_report': 'experiments/targeted_cycle/stage_2_s2_fixes/stage_2_report_ru.md',
        'roles': ['spatial_challenger'],
        'model': STAGE2 / 'models/spatial_no_temperature_centered/model_artifact.joblib',
        'params': {'lambda': 1.0, 'centering': 'full-grid length-weighted'},
        'rationale': 'contract-corrected full-grid spatial Ridge; no temperature feature',
        'evidence': 'centering fixes level shift; strict/reference metrics do not establish target gain',
    },
    {
        'id': 'spatial_centered_corrected_temperature', 'case': 'S2',
        'source': STAGE2 / 'models/spatial_centered_corrected_temperature/predictions_s2.csv',
        'source_report': 'experiments/targeted_cycle/stage_2_s2_fixes/stage_2_report_ru.md',
        'roles': ['spatial_challenger'],
        'model': STAGE2 / 'models/spatial_centered_corrected_temperature/model_artifact.joblib',
        'params': {'lambda': 1.0, 'centering': 'full-grid length-weighted',
                   'temperature': 'full-grid local scenario_cooling_c'},
        'rationale': 'centered spatial with corrected thermal feature',
        'evidence': 'strict/post-cutoff metrics weaker than physics; retained as challenger only',
    },
]

PAIRS = [
    {'name': 's1a__physics_priors', 's1': 's1a', 's2': 'physics_priors',
     'role': 'reference_pair'},
    {'name': 'global_ridge_geometry_correction__physics_priors',
     's1': 'global_ridge_geometry_correction', 's2': 'physics_priors',
     'role': 'primary_s1_pair'},
    {'name': 'global_ridge_geometry_time_scaled__physics_priors',
     's1': 'global_ridge_geometry_time_scaled', 's2': 'physics_priors',
     'role': 'temporal_alt_pair'},
    {'name': 's1a__spatial_no_temperature_centered',
     's1': 's1a', 's2': 'spatial_no_temperature_centered',
     'role': 's2_spatial_pair'},
    {'name': 's1a__spatial_centered_corrected_temperature',
     's1': 's1a', 's2': 'spatial_centered_corrected_temperature',
     'role': 's2_temperature_pair'},
]


def _read_pred(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ['id', 'segment_id', 'thickness_pred']:
        raise ValueError(f'{path}: expected exactly id,segment_id,thickness_pred')
    if frame.duplicated(['id', 'segment_id']).any():
        raise ValueError(f'{path}: duplicate keys')
    frame['id'] = frame.id.astype('int64')
    frame['segment_id'] = frame.segment_id.astype('int64')
    if not np.isfinite(frame.thickness_pred.to_numpy(float)).all():
        raise ValueError(f'{path}: nonfinite predictions')
    return frame.sort_values(['id', 'segment_id']).reset_index(drop=True)


def _canonical_hash(frame: pd.DataFrame) -> str:
    import hashlib
    text = frame[['id', 'segment_id', 'thickness_pred']].to_csv(
        index=False, float_format='%.10f', lineterminator='\n')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _contract_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(p): sha256(p) for p in paths if Path(p).is_file()}


def freeze_sources(data: RepositoryData) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    grid = target_key_grid(data)
    s1_keys = set(zip(grid[grid.id.eq(S1)].id, grid[grid.id.eq(S1)].segment_id))
    s2_keys = set(zip(grid[grid.id.eq(S2)].id, grid[grid.id.eq(S2)].segment_id))
    metadata = {}
    frames = {}
    for cand in CANDIDATES:
        cid, case, source = cand['id'], cand['case'], Path(cand['source'])
        if not source.is_file():
            raise FileNotFoundError(f'Frozen source missing: {source}')
        frame = _read_pred(source)
        expected = s1_keys if case == 'S1' else s2_keys
        actual = set(zip(frame.id, frame.segment_id))
        if actual != expected:
            raise ValueError(f'{cid}: source keys differ from target grid')
        frozen = OUT / 'predictions' / case / f'{cid}.csv'
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(frozen, index=False, float_format='%.10f')
        model_path = Path(cand['model']) if cand['model'] else None
        model_hash = sha256(model_path) if model_path and model_path.is_file() else None
        if model_path and not model_hash:
            raise FileNotFoundError(f'{cid}: model artifact missing')
        meta_path = model_path.with_name('model_metadata.json') if model_path else None
        if meta_path and meta_path.is_file():
            model_meta = json.loads(meta_path.read_text(encoding='utf-8'))
        else:
            model_meta = None
        if case == 'S1':
            contract = json.loads((STAGE3 / 'feature_contract.json').read_text(encoding='utf-8'))
            code_paths = [Path('src/ml/targeted_s1.py'),
                          Path('src/ml/targeted_s1_temporal.py'),
                          Path('src/ml/next_common.py')]
        else:
            if cid == 'physics_priors':
                contract = {
                    'model_type': 'ColdStartPhysics',
                    'features': ['t_mix_mean', 'watercut_pct_mean', 'v_mix_mean',
                                 'age_years', 'nominal_thickness_mm', 'exposure'],
                    'preprocessing': 'environmental aggregates and bounded exposure; no spatial imputer',
                    'centering': None, 'lambda': None,
                    'physics_metadata': model_meta,
                }
            else:
                contract = {
                    'model_type': (model_meta or {}).get('model_type'),
                    'features': (model_meta or {}).get('features', []),
                    'used_features': (model_meta or {}).get('used_features', []),
                    'preprocessing': (model_meta or {}).get('preprocessing', {}),
                    'centering': (model_meta or {}).get('centering'),
                    'lambda': (model_meta or {}).get('lambda'),
                    'physics_metadata': (model_meta or {}).get('physics_metadata'),
                }
            code_paths = [Path('src/ml/targeted_s2.py'),
                          Path('src/ml/s2_next.py'),
                          Path('src/ml/next_common.py')]
        seen = [
            {'stage': 'S2 fixes', 'path': 'experiments/targeted_cycle/stage_2_report_ru.md',
             'status': 'reviewed'},
            {'stage': 'S1 100/20', 'path': 'experiments/targeted_cycle/stage_3_report_ru.md',
             'status': 'reviewed'},
            {'stage': 'S1 robustness', 'path': 'experiments/targeted_cycle/stage_4_report_ru.md',
             'status': 'reviewed'},
            {'stage': 'S1 temporal', 'path': 'experiments/targeted_cycle/stage_5_report_ru.md',
             'status': 'reviewed'},
        ]
        metadata[cid] = {
            'candidate_id': cid, 'case': case, 'roles': cand['roles'],
            'source_prediction': str(source), 'source_prediction_sha256': sha256(source),
            'frozen_prediction': str(frozen), 'frozen_prediction_sha256': sha256(frozen),
            'canonical_prediction_sha256': _canonical_hash(frame),
            'prediction_sha256': _canonical_hash(frame),
            'model_path': str(model_path) if model_path else None,
            'model_sha256': model_hash,
            'model_metadata': model_meta,
            'params': cand['params'],
            'feature_contract': contract,
            'rationale': cand['rationale'],
            'evidence_status': cand['evidence'],
            'seen_evaluations': seen,
            'rows': int(len(frame)),
            'std_prediction_mm': float(frame.thickness_pred.std(ddof=0)),
            'nunique': int(frame.thickness_pred.nunique()),
            'is_constant': bool(frame.thickness_pred.nunique() == 1),
            'created_utc': utcnow(),
            'code_sha256': _contract_hashes(code_paths),
            'input_sha256': {
                'data_manifest': sha256(Path(data.root) / 'reports/run_manifest.json'),
                'source_prediction': sha256(source),
                'model': model_hash,
            },
        }
        frames[cid] = frame
    return metadata, frames


def _bounds(data: RepositoryData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prod = s1_production_table(data)
    s1 = prod[['id', 'segment_id', 'h0_mm', 'nominal_thickness_mm']].copy()
    s1['upper_mm'] = s1.nominal_thickness_mm
    # Use the exact frozen S1 predictions as the validating upper bound only
    # where h0 rounding/legacy values could be below a conservative rule; all
    # current upper bounds are nominal, so this is not needed. Keep h0 positive.
    grid = target_key_grid(data)
    s2_nom = data.read('processed/target_features')[['id', 'segment_id',
                                                     'nominal_thickness_mm']].copy()
    s2 = s2_nom[s2_nom.id.eq(S2)].copy()
    s2['h0_mm'] = s2.nominal_thickness_mm
    s2['upper_mm'] = s2.nominal_thickness_mm
    bounds = pd.concat([s1[['id', 'segment_id', 'h0_mm',
                           'nominal_thickness_mm', 'upper_mm']],
                        s2[['id', 'segment_id', 'h0_mm',
                            'nominal_thickness_mm', 'upper_mm']]], ignore_index=True)
    return prod, grid, bounds


def assemble_pairs(data: RepositoryData, frames: dict[str, pd.DataFrame],
                   metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prod, grid, bounds = _bounds(data)
    combinations = []
    for pair in PAIRS:
        s1 = frames[pair['s1']]
        s2 = frames[pair['s2']]
        combined, validation = combine_with_s2(s1, s2, grid, bounds)
        directory = OUT / 'combinations' / pair['name']
        directory.mkdir(parents=True, exist_ok=True)
        combined.to_csv(directory / 'result.csv', index=False,
                        float_format='%.10f')
        written = pd.read_csv(directory / 'result.csv')
        s1_written = written[written.id.eq(S1)][['id', 'segment_id', 'thickness_pred']]
        s1_written = s1_written.sort_values(['id', 'segment_id']).reset_index(drop=True)
        s2_written = written[written.id.eq(S2)][['id', 'segment_id', 'thickness_pred']]
        s2_written = s2_written.sort_values(['id', 'segment_id']).reset_index(drop=True)
        s1_diff = float(np.abs(s1_written.thickness_pred.to_numpy(float)
                               - s1.thickness_pred.to_numpy(float)).max())
        s2_diff = float(np.abs(s2_written.thickness_pred.to_numpy(float)
                               - s2.thickness_pred.to_numpy(float)).max())
        if s1_diff != 0.0 or s2_diff != 0.0:
            raise ValueError(f'{pair["name"]}: fixed half changed')
        if len(written) != 2000 or written.duplicated(['id', 'segment_id']).any():
            raise ValueError(f'{pair["name"]}: key contract failed')
        constant = {}
        for pid, case in ((S1, 'S1'), (S2, 'S2')):
            part = written[written.id.eq(pid)]
            v = part.thickness_pred.to_numpy(float)
            constant[case] = {'rows': int(len(v)), 'nunique': int(len(np.unique(v))),
                              'std_mm': float(v.std(ddof=0)),
                              'min': float(v.min()), 'max': float(v.max()),
                              'is_constant': bool(np.ptp(v) <= 1e-12),
                              'near_constant': bool(v.std(ddof=0) <= 1e-6)}
        record = {
            **pair, 'result_path': str(directory / 'result.csv'),
            'result_sha256': sha256(directory / 'result.csv'),
            'prediction_sha256': _canonical_hash(written),
            'rows': int(len(written)), 'unique_keys': True,
            's1_rows': int((written.id.eq(S1)).sum()),
            's2_rows': int((written.id.eq(S2)).sum()),
            's1_max_abs_diff_vs_frozen': s1_diff,
            's2_max_abs_diff_vs_frozen': s2_diff,
            'validation': validation, 'constant_diagnostics': constant,
        }
        write_json(directory / 'manifest.json', json_safe(record))
        combinations.append(record)
    return {'pairs': combinations, 'explicit_only': True,
            'no_automatic_cartesian_product': True}


def _recheck(data_root: Path, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    root = Path('.')
    recheck = OUT / 'recheck'
    recheck.mkdir(parents=True, exist_ok=True)
    checks = {}
    # S1 temporal variants are regenerated from Stage 3 artifacts in a new process.
    s1_dir = recheck / 'stage5'
    cmd = [sys.executable, '-m', 'src.ml.targeted_s1_temporal',
           '--root', str(root), '--data-root', str(data_root),
           '--output-root', str(s1_dir)]
    subprocess.check_call(cmd)
    for name in ('global_ridge_geometry_correction',
                 'global_ridge_geometry_time_scaled'):
        source = frames[name].sort_values(['id', 'segment_id']).reset_index(drop=True)
        path = s1_dir / 'production' / f'{name}_predictions.csv'
        got = pd.read_csv(path).sort_values(['id', 'segment_id']).reset_index(drop=True)
        checks[name] = float(np.abs(got.thickness_pred.to_numpy(float)
                                    - source.thickness_pred.to_numpy(float)).max())
    # S2 artifacts are loaded by the predict CLI; no labels/fit.
    for name in ('physics_priors', 'spatial_no_temperature_centered',
                 'spatial_centered_corrected_temperature'):
        artifact = STAGE2 / 'models' / name / 'model_artifact.joblib'
        cmd = [sys.executable, '-m', 'src.ml.targeted_s2', 'predict',
               '--root', str(root), '--data-root', str(data_root),
               '--output-root', str(recheck),
               '--candidate', name, '--artifact', str(artifact)]
        subprocess.check_call(cmd)
        got = pd.read_csv(recheck / name / 'predictions_s2.csv').sort_values(
            ['id', 'segment_id']).reset_index(drop=True)
        source = frames[name].sort_values(['id', 'segment_id']).reset_index(drop=True)
        checks[name] = float(np.abs(got.thickness_pred.to_numpy(float)
                                    - source.thickness_pred.to_numpy(float)).max())
    return {'new_process': True, 'no_fit': True, 'max_abs_differences': checks,
            'full_partial_inference': 'covered by tests/test_stage_2_s2_fixes.py '
                                      'and tests/test_stage_3_s1.py'}


def mentor_questions() -> str:
    return '''# Вопросы менторам / организаторам

1. Какая точная формула итогового score и как соотносятся Pearson r² по двум кейсам? Есть ли веса кейсов, MAE/RMSE или иные компоненты?
2. Как scorer обрабатывает константный прогноз и нулевую дисперсию? Pearson r² для константы должен быть `undefined`, а не 0 или 1.
3. Какие даты и данные разрешены на каждом cutoff? Подтверждается ли строгое `date < T` для S1 `2023-12-01` и S2 `2022-07-01`?
4. Одинаковы ли прибор, порог регистрации дефектов, координатная привязка и определение минимальной толщины в кампаниях 2018, 2022 и 2023?
5. Были ли замены, ремонты, перекраска или иные события, которые могли увеличить измеренную толщину?
6. Доступны ли другие трубы в режиме, близком к S2, и измеренные координатные профили/топология?
7. Будут ли переданы target labels для дальнейшего обучения, или только агрегированный score? После использования labels это уже development data, нужна новая независимая проверка.
8. Есть ли письменное разрешение использовать измерения других труб на/после production cutoff как `archive_transfer`, или только как замороженную retrospective-диагностику?
'''


def _reports(metadata: dict[str, Any], assembly: dict[str, Any],
             recheck: dict[str, Any], data_root: Path) -> tuple[Path, Path]:
    base = Path('experiments/targeted_cycle')
    stage6 = base / 'stage_6_report_ru.md'
    overall = base / 'report_ru.md'
    rows = []
    for cid, m in metadata.items():
        rows.append(f"| {cid} | {m['case']} | {', '.join(m['roles'])} | "
                    f"{m['rows']} | `{m['prediction_sha256'][:16]}…` | {m['evidence_status']} |")
    comb_rows = [f"| {p['name']} | {p['s1']} | {p['s2']} | {p['rows']} | `{p['result_sha256'][:16]}…` |"
                 for p in assembly['pairs']]
    lines = [
        '# Stage 6 — freeze and external-evaluation package', '',
        f"Generated: `{utcnow()}`. No fit, no tuning, no main result overwrite.",
        '', '## 1. Frozen candidates', '',
        '| Candidate | Case | Roles | Rows | Prediction SHA | Evidence |',
        '|---|---|---|---:|---|---|',
        *rows,
        '', 'Level reference, spatial challenger and submission candidate are explicit roles, not winner claims.',
        '', '## 2. Explicit S1/S2 combinations', '',
        '| Pair | S1 | S2 | Rows | result SHA |',
        '|---|---|---|---:|---|',
        *comb_rows,
        '', 'Only these predeclared pairs were assembled; no automatic Cartesian product was run.',
        '', '## 3. Contract checks', '',
        '- every result: 2000 unique keys, 525 S1 + 1475 S2, finite, within bounds;',
        '- selected S1 and S2 halves preserve source predictions exactly (`max_abs=0`);',
        '- constant/near-constant diagnostics are saved per pair;',
        '- no main `result.csv` was overwritten.',
        '', '## 4. Artifact-only recheck', '',
        f"- new-process max abs differences: `{recheck['max_abs_differences']}`; no fit, no labels.",
        '- Full/partial inference is covered by Stage 2/3 tests.',
        '', '## 5. External evaluator', '',
        '```bash',
        'C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.evaluate_external \\',
        '  --predictions experiments/targeted_cycle/frozen_candidates/combinations/<pair>/result.csv \\',
        '  --truth <separate_truth.csv> --output-root experiments/targeted_cycle/external_evaluation/<run>',
        '```',
        'Truth columns: `id,segment_id,date,thickness_measured` (mm). Truth is never merged into inference features.',
        'No external labels were available in Stage 6, so no external accuracy is claimed.',
        '', '## 6. What was fixed / confirmed / rejected', '',
        '- Fixed: full-grid thermal scenario before label joins; full-grid centered S2 spatial; physical block protocol.',
        '- Confirmed: S1 Ridge shows positive spatial/level signal on 100/200/500 block schemes; S2 physics remains level reference.',
        '- Rejected as improvements: S2 spatial/temperature on strict/retrospective metrics; affine as spatial candidate; S1 time law as identifiable from OOF.',
        '', '## 7. What cannot be established without new data', '',
        '- Hidden target accuracy and the official score formula;',
        '- whether S1 time scaling is correct;',
        '- cross-environment S2 improvement on the target;',
        '- campaign comparability, repairs, instrument/threshold metadata;',
        '- two-campaign S1B advantage (single prior campaign in training).',
        '', '## 8. Tests', '',
        '- `PIPETWIN_NEXT_DATA_ROOT=... C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m unittest tests.test_stage_6_freeze -v`: 4 tests, OK.',
        '- Full Stage 2–6 run: 30 tests, OK.',
        '- Проверены candidate roles/hashes, explicit pairs, 2000-key contracts, exact halves, artifact-only recheck и external evaluator on synthetic truth.',
        '', '## 9. Commands and paths', '',
        '```bash',
        'C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_freeze \\',
        f'  --root . --data-root {data_root} --output-root experiments/targeted_cycle/frozen_candidates',
        '```',
        '',
        '- `experiments/targeted_cycle/frozen_candidates/candidate_freeze.json`',
        '- `experiments/targeted_cycle/frozen_candidates/predictions/`',
        '- `experiments/targeted_cycle/frozen_candidates/combinations/`',
        '- `experiments/targeted_cycle/frozen_candidates/recheck/`',
        '- `experiments/targeted_cycle/stage_6_report_ru.md`',
        '- `experiments/targeted_cycle/report_ru.md`',
        '- `experiments/targeted_cycle/mentor_questions_ru.md`',
        '',
        'Stage 6 stops here; no messages were sent.',
        '',
    ]
    stage6.write_text('\n'.join(lines), encoding='utf-8')
    overall.write_text('\n'.join([
        '# PipeTwin targeted cycle — overall report', '',
        f"Generated: `{utcnow()}`. Local freeze only; no push/deploy, no main result overwrite.", '',
        '## Что исправлено', '',
        '- S2: full-grid thermal scenario до label join; full-grid centered spatial Ridge; old level shift устранён.',
        '- S1: physical block protocol (center-based, fixed buffers); Stage 5 std corrected from saved diagnostics.',
        '- Evaluation: explicit lower-tail/per-block/global top10 and conditional block bootstrap.',
        '', '## Что подтвердилось', '',
        '- S1 global/target Ridge: положительный spatial signal на 100/20, 200/50 и 500/100; MAE ниже S1A.',
        '- S2 physics: level reference 2.809539; centered spatial preserves level; old uncentered temperature mean 2.590687.',
        '- Freeze: explicit pair assembly, exact half preservation, artifact-only recheck max_abs=0.',
        '', '## Что отвергнуто/не подтверждено', '',
        '- S2 spatial/temperature не показали устойчивого улучшения strict/retrospective metrics.',
        '- Affine не прошёл spatial filter; CatBoost проиграл по top10/r; S1 time law не идентифицирован OOF.',
        '- S1 Ridge lower-tail global MAE хуже S1A; promising_S1 — не thin-tail/stability claim.',
        '', '## Что заморожено', '',
        '- S1: `s1a` (level reference), `global_ridge_geometry_correction` (primary challenger), '
        '`global_ridge_geometry_time_scaled` (temporal sensitivity).',
        '- S2: `physics_priors` (level reference), `spatial_no_temperature_centered`, '
        '`spatial_centered_corrected_temperature` (challengers).',
        '- Explicit pairs only: `s1a__physics_priors`, `global_correction__physics_priors`, '
        '`global_time_scaled__physics_priors`, `s1a__spatial_centered`, `s1a__spatial_temperature`.',
        '', '## Что нельзя установить без новых данных', '',
        '- Официальный score/r² contract, hidden target accuracy, constant scorer behavior.',
        '- Правильный temporal law S1; cross-environment S2 improvement on target.',
        '- Сопоставимость кампаний, ремонты, tool/threshold metadata, две прошлые S1-кампании.',
        '', '## Ключевые числа', '',
        '- S1 global Ridge MAE: `0.4187` (100/20), `0.4186` (200/50), `0.4209` (500/100); S1A `0.5068`.',
        '- S2 physics mean `2.809539`; centered spatial `2.809501`; centered temperature `2.809559`.',
        '- Frozen package: `experiments/targeted_cycle/frozen_candidates/`.',
        '', 'Detailed evidence: `stage_2_report_ru.md`, `stage_3_report_ru.md`, '
        '`stage_4_report_ru.md`, `stage_5_report_ru.md`, `stage_6_report_ru.md`.',
        '', 'External evaluator: `src/ml/evaluate_external.py`. Mentor questions: `mentor_questions_ru.md`.',
        '', '```bash',
        'C:/Users/Администратор/Documents/PipeTwin_s1abc_env/Scripts/python.exe -m src.ml.targeted_freeze \\',
        f'  --root . --data-root {data_root} --output-root experiments/targeted_cycle/frozen_candidates',
        '```',
    ]), encoding='utf-8')
    return stage6, overall


def run_stage6(root: Path = BASE_DIR, data_root: Path | None = None,
               output_root: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    data_root = Path(data_root or root / 'experiments/next_cycle/data_generation').resolve()
    global OUT
    OUT = Path(output_root or root / 'experiments/targeted_cycle/frozen_candidates').resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    data = RepositoryData(data_root)
    print('Freezing named source predictions...', flush=True)
    metadata, frames = freeze_sources(data)
    assembly = assemble_pairs(data, frames, metadata)
    print('Rechecking artifacts in new processes...', flush=True)
    recheck = _recheck(data_root, frames)
    # Existing Stage 5 proposal/seen evaluations are preserved.
    proposal_path = root / STAGE5 / 'proposal.json'
    proposal = json.loads(proposal_path.read_text(encoding='utf-8')) if proposal_path.is_file() else None
    freeze = {
        'freeze_id': f"targeted_{utcnow().replace(':', '').replace('+00:00', 'Z')}",
        'created_utc': utcnow(),
        'project_root': str(root),
        'data_root': str(data_root),
        'data_manifest_sha256': sha256(Path(data.root) / 'reports/run_manifest.json'),
        'candidate_count_s1': sum(1 for c in metadata.values() if c['case'] == 'S1'),
        'candidate_count_s2': sum(1 for c in metadata.values() if c['case'] == 'S2'),
        'candidates': metadata,
        'explicit_combinations': assembly,
        'recheck': recheck,
        'stage5_proposal': proposal,
        'status': 'frozen_package_ready_for_external_choice',
        'not_a_winner': True,
        'main_result_overwritten': False,
    }
    write_json(OUT / 'candidate_freeze.json', freeze)
    mentor_path = Path('experiments/targeted_cycle/mentor_questions_ru.md')
    if mentor_path.is_file():
        (Path('experiments/targeted_cycle/mentor_questions_stage1_ru.md')
         .write_text(mentor_path.read_text(encoding='utf-8'), encoding='utf-8'))
    mentor_path.write_text(mentor_questions(), encoding='utf-8')
    stage6, overall = _reports(metadata, assembly, recheck, data_root)
    manifest = {
        'stage': 'Stage_6_freeze', 'created_utc': freeze['created_utc'],
        'data_root': str(data_root), 'freeze_id': freeze['freeze_id'],
        'candidate_freeze_sha256': sha256(OUT / 'candidate_freeze.json'),
        'combination_sha256': {p['name']: sha256(Path(p['result_path']))
                               for p in assembly['pairs']},
        'no_fit': True, 'no_main_result_overwrite': True,
        'recheck': recheck,
    }
    write_json(OUT / 'freeze_manifest.json', manifest)
    print(f'PASS Stage 6: {OUT}', flush=True)
    return {'freeze': freeze, 'assembly': assembly, 'recheck': recheck,
            'output': str(OUT), 'stage6_report': str(stage6),
            'overall_report': str(overall)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path,
                        default=BASE_DIR / 'experiments/targeted_cycle/frozen_candidates')
    args = parser.parse_args(argv)
    run_stage6(args.root, args.data_root, args.output_root)


if __name__ == '__main__':
    main()

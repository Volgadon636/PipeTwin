"""Compare transferable terrain features for the S2 cold-start forecast."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.storage import sha256, write_json
from .data import RepositoryData
from .incremental import FORECAST_KEYS, KEYS, S2_BASE, _compare, _fit_s2, _score
from .next_common import load_artifact, save_artifact, utcnow, write_csv
from .s1_contract import nominal_frame, target_key_grid
from .submission import COLUMNS, export_result
from .targeted_s1 import _block_folds, _physical_block_frame
from .targeted_s1_stage4 import SCHEMES
from .targeted_s2 import S2_CUTOFF, SPATIAL_FEATURES, _predict_candidate, prepare_full_grid, prepare_target_full

BASIN_WIDTHS = (30, 100)
BASINS = [f'basin_depth_{width}m' for width in BASIN_WIDTHS]
RANKS = [f'{feature}_rank' for feature in SPATIAL_FEATURES]
VARIANTS = {
    S2_BASE: SPATIAL_FEATURES,
    'profile_ranks': RANKS,
    'basin_depth': SPATIAL_FEATURES + BASINS,
    'basin_ranks': RANKS + [f'{feature}_rank' for feature in BASINS],
}
SOURCE = Path('experiments/incremental_cycle')


def terrain_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Build covariates on a complete section. No inspection data is required.

    The basin depth is the rise to the lower of the two local barriers.
    A one-sided rise and a section endpoint therefore have depth zero.
    """
    result = grid.sort_values('start_m').copy()
    if result.id.nunique() != 1 or result.duplicated(KEYS).any():
        raise ValueError('Expected one complete section with unique keys')
    center = (result.start_m + result.end_m).to_numpy(float) / 2
    elevation = result.z_center.to_numpy(float)
    if not (np.isfinite(elevation).all() and (np.diff(center) > 0).all()):
        raise ValueError('Invalid section geometry')
    for width, name in zip(BASIN_WIDTHS, BASINS):
        left = np.searchsorted(center, center - width, side='left')
        right = np.searchsorted(center, center + width, side='right')
        result[name] = [
            min(elevation[lo:i + 1].max(), elevation[i:hi].max()) - elevation[i]
            for i, (lo, hi) in enumerate(zip(left, right))
        ]
    for feature in SPATIAL_FEATURES + BASINS:
        result[f'{feature}_rank'] = (
            result[feature].rank(method='average') - 0.5) / len(result)
    return result


def join_predictions(test, full, intervals, artifacts):
    values = full[KEYS].copy()
    for name, artifact in artifacts.items():
        values[name] = _predict_candidate(artifact, full, intervals)['h_clipped']
    joined = test[FORECAST_KEYS + ['next_mm']].merge(
        values, on=KEYS, how='left', validate='one_to_one')
    if not np.isfinite(joined[list(artifacts)].to_numpy()).all():
        raise ValueError('Missing or nonfinite prediction')
    return joined


def section_folds(strict, grids, intervals):
    folds = []
    for pid, test in strict.groupby('id', sort=True):
        (date,) = test.prediction_date.drop_duplicates()
        train = strict[~strict.id.eq(pid)]
        artifacts = _fit_s2(train, intervals, VARIANTS)
        preds = join_predictions(test, grids[(int(pid), date)], intervals, artifacts)
        folds.append(preds.assign(fold=str(pid), train_n=len(train)))
    return pd.concat(folds, ignore_index=True)


def block_folds(strict, grids, intervals, largest, scheme):
    width, buffer = scheme['block_m'], scheme['buffer_m']
    target = _physical_block_frame(strict[strict.id.eq(largest)], width)
    folds = _block_folds(strict, target, width, buffer)
    (date,) = target.prediction_date.drop_duplicates()
    grid = grids[(largest, date)]
    return pd.concat([
        join_predictions(fold['test'], grid, intervals,
                         _fit_s2(fold['train_global'], intervals, VARIANTS)).assign(
                             fold=fold['block'], train_n=len(fold['train_global']))
        for fold in folds
    ], ignore_index=True)


def score(predictions, scheme):
    return [_score(predictions, name, 'S2', scheme, 'terrain', predictions.fold.nunique())
            for name in VARIANTS]


def select(metrics, largest):
    """Require positive relative gain on the pooled and dominant held-out tests."""
    table = pd.DataFrame(metrics).set_index(['scheme', 'candidate'])
    checks = ['section_LOO', f'section_{largest}']
    baseline = table.xs(S2_BASE, level='candidate').loc[checks]
    eligible = []
    for name in VARIANTS.keys() - {S2_BASE}:
        variant = table.xs(name, level='candidate').loc[checks]
        if ((variant.pearson_r > 0) & (variant.pearson_squared > baseline.pearson_squared)).all():
            eligible.append((float(variant.loc['section_LOO', 'pearson_squared']), name))
    return max(eligible, default=(0.0, None))[1]


def export_candidate(data, root, artifact, result_path):
    """Keep the S1 half of the incremental challenger unchanged."""
    full, interval = prepare_target_full(data)
    if not full.prediction_date.eq(S2_CUTOFF).all():
        raise ValueError('Unexpected S2 prediction cutoff')
    grid = terrain_features(full)
    intervals = {(int(grid.id.iloc[0]), S2_CUTOFF): interval}
    prediction = _predict_candidate(artifact, grid, intervals)
    s2 = grid[KEYS].assign(thickness_pred=prediction['h_clipped'])
    prior = pd.read_csv(root / SOURCE / 's1_challenger/result.csv')
    s1 = prior[~prior.id.isin(s2.id)]
    bounds = nominal_frame(data).rename(columns={'nominal_thickness_mm': 'h0_mm'})
    validation, _ = export_result(pd.concat([s1, s2], ignore_index=True)[COLUMNS],
                                  result_path, target_key_grid(data), bounds)
    reloaded = pd.read_csv(result_path)
    validation['s1_max_abs_difference_mm'] = _compare(
        reloaded[reloaded.id.isin(s1.id)], s1, KEYS, ['thickness_pred'], atol=0)
    return {**validation, 'sha256': sha256(result_path), 's2_rows': len(s2),
            's2_clipped': int(prediction['clipping'].sum()),
            's2_mean_mm': float(s2.thickness_pred.mean()),
            's2_std_mm': float(s2.thickness_pred.std(ddof=0))}


def run(data, root, output):
    output.mkdir(parents=True, exist_ok=False)
    started = utcnow()
    write_json(output / 'protocol.json', {
        'variants': VARIANTS, 'radius_m': BASIN_WIDTHS,
        'cutoff': str(S2_CUTOFF.date()), 'ridge_alpha': 10.0,
        'selection': 'Highest pooled Pearson r² among models improving over baseline '
                     'with positive signed r both pooled and on the largest held-out section.',
        'blocked_diagnostic': SCHEMES,
        'post_cutoff_policy': 'Previously viewed retrospective, scored after selection, never fit.',
    })
    prepared = prepare_full_grid(data)
    strict = prepared['strict']
    if not strict.prediction_date.lt(S2_CUTOFF).all():
        raise ValueError('S2 fit includes labels at or after cutoff')
    intervals = prepared['interval_map']
    grids = {key: terrain_features(grid) for key, grid in prepared['full_frames'].items()}
    new_columns = BASINS + RANKS + [f'{name}_rank' for name in BASINS]
    lookup = pd.concat([grid[FORECAST_KEYS + new_columns] for grid in grids.values()],
                       ignore_index=True)
    strict = strict.merge(lookup, on=FORECAST_KEYS, how='left', validate='one_to_one')
    if not np.isfinite(strict[new_columns].to_numpy()).all():
        raise ValueError('Strict training keys must have complete terrain features')
    largest = int(strict.groupby('id').size().idxmax())
    oof = section_folds(strict, grids, intervals)
    baseline_file = root / SOURCE / 's2_section_oof.csv'
    baseline = pd.read_csv(baseline_file, parse_dates=['prediction_date'])
    max_diff = _compare(oof, baseline, FORECAST_KEYS, ['next_mm', S2_BASE])
    metrics = score(oof, 'section_LOO')
    for pid, section in oof.groupby('id', sort=True):
        metrics.extend(score(section, f'section_{pid}'))
    chosen = select(metrics, largest)
    write_csv(oof, output / 'section_oof.csv')
    for scheme in SCHEMES:
        blocked = block_folds(strict, grids, intervals, largest, scheme)
        metrics.extend(score(blocked, scheme['name']))
        write_csv(blocked, output / f'{scheme["name"]}_oof.csv')
    artifacts = _fit_s2(strict, intervals, {name: VARIANTS[name]
                         for name in (S2_BASE, chosen) if name is not None})
    retrospective = []
    for key, observed in prepared['post'].groupby(['id', 'prediction_date'], sort=True):
        retrospective.append(join_predictions(observed, grids[key], intervals,
                                              artifacts).assign(fold=str(key[0])))
    post = pd.concat(retrospective, ignore_index=True)
    write_csv(post, output / 'retrospective_predictions.csv')
    metrics.extend(_score(post, name, 'S2', 'seen_retrospective', 'diagnostic',
                          post.fold.nunique()) for name in artifacts)
    write_csv(pd.DataFrame(metrics), output / 'metrics.csv', float_format='%.10f')
    manifest = {'started_utc': started, 'chosen': chosen,
                'baseline_oof_max_abs_mm': max_diff, 'baseline_oof_sha256': sha256(baseline_file),
                'strict_rows': len(strict), 'strict_parent_count': strict.main_pipeline_id.nunique(),
                'strict_dates': sorted(str(day.date()) for day in strict.prediction_date.unique()),
                'retrospective_rows': len(post), 'code_sha256': sha256(Path(__file__)),
                'input_sha256': data.hashes}
    if chosen:
        directory = output / chosen
        saved = save_artifact(artifacts[chosen], directory)
        manifest['artifact_sha256'] = saved['artifact_sha256']
        manifest['export'] = export_candidate(data, root, artifacts[chosen], directory / 'result.csv')
    write_json(output / 'decision.json', manifest)
    print(json.dumps({'candidate': chosen, 'strict_rows': len(strict)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE_DIR)
    parser.add_argument('--data-root', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output-root', type=Path)
    group.add_argument('--artifact', type=Path)
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    data = RepositoryData(args.data_root)
    if args.output_root:
        run(data, args.root, args.output_root)
    else:
        if args.result is None or args.result.exists():
            parser.error('--artifact requires --result pointing to a new file')
        print(json.dumps(export_candidate(data, args.root, load_artifact(args.artifact), args.result)))


if __name__ == '__main__':
    main()

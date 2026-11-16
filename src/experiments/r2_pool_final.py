"""Compile exact-key rankings and pairwise diagnostics after all eight batches."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.r2_pool_stage1 import KEY
from src.experiments.r2_pool_stage2 import read_baseline
from src.experiments.r2_pool_stage4 import KEY_DTYPES, assert_replay
from src.ml.evaluation import metrics
from src.ml.next_common import json_safe, write_csv
from src.storage import sha256, write_json

ROOT = Path('experiments/r2_pool')
PEER = 'S1_peer_2023'
RETRO = 'S2_retro_2023'


def candidate_paths(root, hypothesis, batch, scheme):
    if batch == 1:
        home = root / 'revisions/r2-pool-v3/hypotheses' / hypothesis
        path = home / 'seam_sensitivity' / scheme / 'validation_predictions.csv'
        return [('seam_sensitivity', path)] if path.is_file() else []
    home = root / 'batches' / f'batch_{batch:02d}' / 'hypotheses' / hypothesis / scheme
    if batch == 2:
        path = home / 'validation_predictions.csv'
        name = {'H09': 'historical_minimum_mix', 'H10': 'local_rate_shrinkage'}.get(hypothesis)
        return [(name, path)] if path.is_file() else []
    return [(path.parent.name, path) for path in sorted(home.glob('*/validation_predictions.csv'))]


def profiles_affine(frame, tolerance):
    differences = []
    for _, rows in frame.groupby(['id', 'prediction_date'], sort=True):
        x, y = rows.baseline.to_numpy(float), rows.candidate.to_numpy(float)
        if len(rows) < 3 or min(np.ptp(x), np.ptp(y)) <= 1e-12:
            continue
        design = np.column_stack([np.ones(len(x)), x])
        fit = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        differences.append(float(np.max(np.abs(fit - y))))
    if not differences:
        return None, None
    return max(differences) <= tolerance, max(differences)


def ranking_for_scheme(root, baseline_root, scheme, spec, rows, scopes, tolerance):
    base_path = baseline_root / scheme / 'validation_predictions.csv'
    base = read_baseline(base_path.parent, spec).sort_values(KEY)
    b = metrics(base.next_mm, base.prediction)
    n = len(base)
    base_r2 = b['pearson_squared']
    seen = {'baseline': base.prediction.to_numpy(float)}
    frames = {'baseline': base.prediction.to_numpy(float)}
    results = [{
        'case': spec['case'], 'scheme': scheme, 'hypothesis': 'baseline', 'variant': spec['model'],
        'role': 'baseline', 'n': n, 'pearson_r_squared': base_r2, 'signed_pearson_r': b['pearson_r'],
        'delta_r_squared': 0., 'mae_mm': b['mae_mm'], 'rmse_mm': b['rmse_mm'],
        'bias_mm': b['bias_mm'], 'rankable': True, 'result': 'baseline',
        'predictions': base_path.as_posix(), 'summary': '',
    }]
    for row in rows:
        batch = int(row['batch'])
        h = row['hypothesis_id']
        paths = candidate_paths(root, h, batch, scheme)
        if not paths:
            continue
        primary = ('seam_sensitivity' if batch == 1 else paths[0][0] if batch == 2 else
                   json.loads((paths[0][1].parents[2] / 'summary.json').read_bytes())['primary'])
        for variant, path in sorted(paths, key=lambda item: item[0] != primary):
            pair = pd.read_csv(path, parse_dates=['prediction_date']).astype(KEY_DTYPES).sort_values(KEY)
            assert_replay(base, pair.rename(columns={'baseline': 'prediction'}), tolerance)
            c = metrics(pair.next_mm, pair.candidate)
            r2 = c['pearson_squared']
            affine, affine_residual = profiles_affine(pair, tolerance)
            role = 'primary' if variant == primary else 'control'
            part = pair.candidate.to_numpy(float)
            key = f'{h}/{variant}'
            frames[key] = part
            duplicate_of = next((name for name, earlier in seen.items()
                                 if np.max(np.abs(earlier - part)) <= tolerance), None)
            gaps = row['candidate_admissibility']
            scope = scopes.get(h, {}).get('allowed_cases', ['S1', 'S2'])
            reasons = []
            if role != 'primary':
                reasons.append('diagnostic_control')
            if gaps:
                reasons.append(gaps)
            if spec['case'] not in scope:
                reasons.append('historical_result_outside_user_candidate_scope')
            if duplicate_of:
                reasons.append('duplicate_prediction')
            elif affine:
                reasons.append('affine_within_profiles')
            if r2 is None:
                reasons.append('undefined_pearson')
            if role == 'primary' and not duplicate_of and not gaps and spec['case'] in scope and not affine:
                seen[key] = part
            delta = None if r2 is None else r2 - base_r2
            status = ('undefined_metric' if delta is None else 'no_increment' if abs(delta) <= 1e-12
                      else 'exploratory_gain' if delta > 0. else 'worse')
            results.append({
                'case': spec['case'], 'scheme': scheme, 'hypothesis': h, 'variant': variant, 'role': role,
                'n': n, 'pearson_r_squared': r2, 'signed_pearson_r': c['pearson_r'],
                'delta_r_squared': delta, 'mae_mm': c['mae_mm'], 'rmse_mm': c['rmse_mm'], 'bias_mm': c['bias_mm'],
                'result': status, 'rankable': not reasons, 'rank_exclusion': '|'.join(reasons),
                'duplicate_of': duplicate_of, 'affine_within_profiles': affine,
                'max_abs_within_profile_affine_residual_mm': affine_residual,
                'changed_predictions': int((np.abs(part - base.prediction.to_numpy(float)) > 1e-10).sum()),
                'recorded_qualification': row['qualification'],
                'predictions': path.as_posix(), 'summary': (path.parent / 'summary.json').as_posix(),
            })
    entries = sorted(results, key=lambda row: row['pearson_r_squared']
                     if row['pearson_r_squared'] is not None else -1., reverse=True)
    rank = 0
    for row in entries:
        if row['rankable']:
            rank += 1
            row['rank'] = rank
    return base, entries, frames


def correlation(a, b):
    if len(a) < 3 or min(np.ptp(a), np.ptp(b)) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def compare_leaders(scheme, base, frames, selected):
    truth = base.next_mm.to_numpy(float)
    groups = [('pooled', np.arange(len(base)))]
    groups.extend((f'{pid}@{date.date()}', rows)
                  for (pid, date), rows in base.groupby(['id', 'prediction_date']).indices.items())
    comparisons = []
    for a, b in combinations(dict.fromkeys(selected), 2):
        for profile, idx in groups:
            one, two, actual = frames[a][idx], frames[b][idx], truth[idx]
            first, second = one - actual, two - actual
            comparisons.append({
                'scheme': scheme, 'profile': profile, 'n': len(idx), 'A': a, 'B': b,
                'prediction_pearson_r': correlation(one, two),
                'error_pearson_r': correlation(first, second),
                'mean_absolute_prediction_difference_mm': float(np.mean(np.abs(one - two))),
                'n_A_closer': int(np.sum(np.abs(first) + 1e-10 < np.abs(second))),
                'n_B_closer': int(np.sum(np.abs(second) + 1e-10 < np.abs(first))),
                'n_tied': int(np.sum(np.abs(np.abs(first) - np.abs(second)) <= 1e-10)),
            })
    return comparisons


def run(root, output):
    state = json.loads((root / 'workflow_state.json').read_bytes())
    protocol = json.loads((root / 'protocol_photo_v3.json').read_bytes())
    registry_file = root / 'registry.csv'
    with registry_file.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = {(f'H{i:02d}', case) for i in range(1, 41) for case in ('S1', 'S2')}
    if (len(rows) != 80 or {(row['hypothesis_id'], row['case']) for row in rows} != expected
            or state['reviewed_count'] != 40 or any(row['implementation'] == 'not_started' for row in rows)):
        raise ValueError('The final comparison requires all 40 hypotheses to be reviewed')
    scopes = state['candidate_scope']
    baseline_root = root / 'revisions/r2-pool-v3/baseline'
    by_scheme, pairwise, sources = {}, [], {registry_file.as_posix(): sha256(registry_file)}
    for scheme, spec in protocol['schemes'].items():
        case = spec['case']
        base, entries, frames = ranking_for_scheme(
            root, baseline_root, scheme, spec, [row for row in rows if row['case'] == case],
            scopes, protocol['archive_tolerance_mm'])
        by_scheme[scheme] = entries
        sources.update({entry['predictions']: sha256(Path(entry['predictions'])) for entry in entries})
        if scheme in (PEER, RETRO):
            best = [f"{row['hypothesis']}/{row['variant']}" for row in entries
                    if row['rankable'] and row['hypothesis'] != 'baseline'][:3]
            named = (['H12/selected_lambda', 'H18/geometry_type_distribution', 'H16/squared_loss_matched']
                     if scheme == PEER else ['H31/spatial_gaussian_process', 'H30/shallow_catboost',
                                             'H22/terrain_times_log_liquid_velocity'])
            pairwise.extend(compare_leaders(scheme, base, frames, ['baseline', *best, *named]))
    combined = pd.DataFrame([row for entries in by_scheme.values() for row in entries])
    write_csv(combined, output / 'all_variants.csv', float_format='%.15g')
    write_csv(pd.DataFrame(rows), output / 'registry_H01-H40_by_case.csv')
    for case in ('S1', 'S2'):
        write_csv(combined.loc[combined.case.eq(case) & combined.rankable],
                  output / f'ranking_{case}.csv', float_format='%.15g')
    write_csv(pd.DataFrame(pairwise), output / 'leader_pairwise.csv', float_format='%.15g')
    write_csv(combined.loc[combined.result.eq('worse')], output / 'negative_results.csv', float_format='%.15g')
    write_csv(combined.loc[combined.duplicate_of.notna() | combined.affine_within_profiles.eq(True)],
              output / 'duplicates_and_affine.csv', float_format='%.15g')
    leaders = {scheme: {'best_ranked': next(row for row in entries if row['rankable']),
                        'best_observed': entries[0]} for scheme, entries in by_scheme.items()}
    write_json(output / 'comparison_manifest.json', json_safe({
        'protocol_version': protocol['protocol_version'], 'sources_sha256': sources,
        'code_sha256': sha256(Path(__file__)), 'schemes': leaders, 'scopes': scopes,
        'rank_rule': 'Compare within scheme. Rank only primary variants or baseline, applying the user scope, '
                     'admissibility, duplicate and within-profile affine checks.',
        'reviewed': state['reviewed_count'], 'measured': state['evaluated_count'],
        'blocked': state['blocked_count'], 'observed_comparisons': len(combined),
    }))
    print(f'{len(rows)} case registry rows, {len(combined)} comparisons, {len(pairwise)} pairwise entries')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)

"""Verify protected bytes, baseline reproduction and deterministic feature outputs."""
import json
import tempfile
from pathlib import Path

from src.config import BASE_DIR
from src.storage import sha256
from .compute import STAGE_ROOT, generate, write_json


def verify():
    before = json.loads((STAGE_ROOT/'protected_before.json').read_text())
    changed = [name for name, digest in before['sha256'].items()
               if not (BASE_DIR/name).is_file() or sha256(BASE_DIR/name) != digest]
    if changed:
        raise ValueError(f'Protected files changed: {changed}')
    baseline_names = ['experiments/ml_baseline/result.csv',
                      'experiments/m0_baseline_reference/result.csv',
                      'experiments/temperature_v02_02/baseline_control/result.csv']
    baseline_hashes = {name: sha256(BASE_DIR/name) for name in baseline_names}
    if len(set(baseline_hashes.values())) != 1:
        raise ValueError('Baseline predictions differ')
    manifest = json.loads((STAGE_ROOT/'run_manifest.json').read_text())
    with tempfile.TemporaryDirectory(prefix='reproduction_', dir=STAGE_ROOT) as temp:
        generate(Path(temp))
        reproduced = {name: sha256(Path(temp)/name) for name in manifest['output_sha256']}
    if reproduced != manifest['output_sha256']:
        raise ValueError('Feature evidence is not deterministic')
    current = {name: sha256(STAGE_ROOT/name) for name in manifest['output_sha256']}
    if current != reproduced:
        raise ValueError('Saved feature evidence changed')
    result = dict(protected_files_checked=len(before['sha256']), protected_files_changed=changed,
                  baseline_prediction_sha256=baseline_hashes,
                  deterministic_outputs_checked=len(reproduced), deterministic_outputs_match=True,
                  feature_output_sha256=reproduced)
    write_json(STAGE_ROOT/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()

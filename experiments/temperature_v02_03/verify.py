"""Preservation and independent rerun checks for Stage 03, including both graphs."""
import json
import tempfile
from pathlib import Path

from src.config import BASE_DIR
from src.storage import sha256
from .run import STAGE_ROOT, generate, write_json


def verify():
    before = json.loads((STAGE_ROOT/'protected_before.json').read_text())
    changed = [name for name, digest in before['sha256'].items()
               if not (BASE_DIR/name).is_file() or sha256(BASE_DIR/name) != digest]
    if changed:
        raise ValueError(f'Protected files changed: {changed}')
    manifest = json.loads((STAGE_ROOT/'run_manifest.json').read_text())
    if sha256(STAGE_ROOT/'preregistration.json') != manifest['preregistration_sha256']:
        raise ValueError('Preregistration changed after execution')
    with tempfile.TemporaryDirectory(prefix='reproduction_', dir=STAGE_ROOT) as temp:
        generate(Path(temp))
        reproduced = {name: sha256(Path(temp)/name) for name in manifest['outputs_sha256']}
    if reproduced != manifest['outputs_sha256']:
        raise ValueError('Reproduction differs: '+str([name for name in reproduced if reproduced[name]!=manifest['outputs_sha256'][name]]))
    if any(sha256(STAGE_ROOT/name) != digest for name, digest in reproduced.items()):
        raise ValueError('Saved results differ from the reproduced artifacts')
    result = dict(protected_files_checked=len(before['sha256']), protected_files_changed=changed,
                  deterministic_artifacts_checked=len(reproduced), deterministic_artifacts_match=True,
                  baseline_prediction_sha256=sha256(BASE_DIR/'experiments/ml_baseline/result.csv'),
                  preregistration_sha256=manifest['preregistration_sha256'], output_sha256=reproduced)
    write_json(STAGE_ROOT/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()

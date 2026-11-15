"""Run focused regressions and finalize Stage 04 provenance without old-stage edits."""
import json
import os
import subprocess
import sys

from src.storage import sha256 as repository_sha256
from .run import ROOT, OUT, digest, write_json


def main():
    before=json.loads((OUT/'protected_before.json').read_text())
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    suites=[('tests_stage04.txt',['-m','unittest','experiments.temperature_v02_04.test_model','-v']),
            ('tests_ml.txt',['-m','unittest','discover','-s','tests','-p','test_ml.py','-v']),
            ('tests_temperature_regression.txt',['-m','unittest','experiments.temperature_v02_02.test_temperature',
                'experiments.temperature_v02_03.test_candidates','experiments.temperature_v02_03_1.test_experiment','-v'])]
    test_runs=[]
    for logfile,args in suites:
        command=[sys.executable,'-X','utf8','-B',*args]
        if '--reuse-regressions' in sys.argv and logfile!='tests_stage04.txt':
            previous=json.loads((OUT/'verification.json').read_text())
            exit_code=next(r['exit_code'] for r in previous['tests'] if r['log']==logfile)
            reused=True
        else:
            with (OUT/logfile).open('w',encoding='utf-8') as stream:
                completed=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1'))
            exit_code=completed.returncode; reused=False
        lines=(OUT/logfile).read_text(encoding='utf-8').splitlines()
        summary=[s for s in lines if s.startswith(('Ran ','OK','FAIL:','ERROR:','FAILED'))]
        test_runs.append(dict(log=logfile,command=command,exit_code=exit_code,summary=summary,reused_existing_run=reused))
        print(logfile,exit_code,summary,flush=True)
    changed=[n for n,h in before['sha256'].items() if not (ROOT/n).is_file() or digest(ROOT/n)!=h]
    mismatches=[n for n,h in manifest['input_sha256'].items() if repository_sha256(ROOT/n)!=h]
    protocol_ok=digest(OUT/'protocol.json')==before['protocol_sha256']
    verification=dict(protected_files_checked=len(before['sha256']),changed_protected_files=changed,
        input_files_checked=len(manifest['input_sha256']),input_mismatches=mismatches,protocol_unchanged=protocol_ok,
        tests=test_runs,stage04_checks_passed=not changed and not mismatches and protocol_ok and test_runs[0]['exit_code']==0,
        full_requested_regressions_passed=all(r['exit_code']==0 for r in test_runs),
        baseline_byte_copy_identical=digest(OUT/'B0_original_M0.csv')==manifest['immutable_baseline_raw_sha256'])
    write_json('verification.json',verification)
    lines=['# Stage 04 QA','',f"Branch `{manifest['branch']}`, HEAD `{manifest['head']}`.",'',
        f"Fixed Ea40 development provenance: {manifest['parameter_provenance']}.",'',
        f"Protected files checked: {len(before['sha256'])}; changed: {len(changed)}. Published input mismatches: {len(mismatches)}. Protocol unchanged: {protocol_ok}.",'',
        'The initial `.gitignore`, `.mcp.json` and uncommitted Stage 03.1 work were included in the protected inventory. No existing source, test, data, configuration, prior experiment or official result was edited.','',
        '## Actual test runs','']
    for r in test_runs: lines.append(f"- `{r['log']}`: exit {r['exit_code']}; {'; '.join(r['summary'])}")
    lines.extend(['',f"Stage 04 checks passed: {verification['stage04_checks_passed']}. All requested regressions passed: {verification['full_requested_regressions_passed']}.",'',
        'Known repository limitation: the unchanged Stage 02 preservation test hashes raw CRLF baseline bytes against a historical LF hash. Stage 04 copies B0 byte-for-byte and separately verifies exact reconstructed M0/EA00 identity at the original exporter\'s ten decimal places. The saved Stage 03.1 full regression log also records a preprocessing nullable-boolean setup error; preprocessing was not rerun for this isolated model change and that historical failure is not represented as a current pass.','',
        'The first Stage 04 run stopped at an overstrict 1e-12 comparison against B0 before candidate fitting. Source inspection confirmed src/ml/experiment.py exports ten decimal places; the corrected check verifies that exact original serialization. The failed log is retained as run_attempt01_B0_precision.txt. No B0 values or training rules changed.','',
        'EA40 remains a user-chosen development hypothesis. No independent folds, no target-label evaluation and no production promotion. Export schema and physical bounds are tested for all 30 candidate files. EA40 result.csv is the reference-gap assumption; observed-only is explicitly marked as an incomplete descriptor.','',
        'GO for exploratory Stage 05 ablations; NO-GO for independently validated-performance claims or production promotion.'])
    (OUT/'qa_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest['verification']=verification
    manifest['verification_command']='.venv/Scripts/python.exe -X utf8 -B -m experiments.temperature_v02_04.verify'
    manifest['stage_source_sha256']={p.name:digest(p) for p in OUT.glob('*.py')}
    manifest['outputs_sha256']={p.relative_to(OUT).as_posix():digest(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name not in ('run_manifest.json','verify.txt') and '__pycache__' not in p.parts}
    manifest['hash_note']='Raw SHA256 for preservation/outputs; RepositoryData newline-normalized hashes for input contract. Manifest and live verify.txt excluded from output self-hashing.'
    write_json('run_manifest.json',manifest)
    if not verification['stage04_checks_passed'] or not verification['full_requested_regressions_passed']:
        raise SystemExit('QA has failures; see verification.json and saved test logs')


if __name__=='__main__':
    main()

"""Execute tests, check preservation and finalize a non-self-referential manifest."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

from experiments.temperature_v02_03_1.run import ROOT, OUT, digest, write_json
from src.storage import sha256 as repository_sha256


def main():
    commands=[
        ('tests_stage03_1.txt',['-m','unittest','experiments.temperature_v02_03_1.test_experiment','-v']),
        ('tests_regression.txt',['-m','unittest','discover','-s','tests','-v']),
    ]
    results=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    for logfile,args in commands:
        command=[sys.executable,'-X','utf8','-B',*args]
        reuse=('--reuse-tests' in sys.argv or
               (logfile=='tests_regression.txt' and '--reuse-regression' in sys.argv))
        if reuse:
            previous=json.loads((OUT/'verification.json').read_text())
            exit_code=next(x['exit_code'] for x in previous['actual_test_runs'] if x['log']==logfile)
        else:
            with (OUT/logfile).open('w',encoding='utf-8') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env)
            exit_code=result.returncode
        lines=(OUT/logfile).read_text(encoding='utf-8').splitlines()
        summary=[x for x in lines if x.startswith(('Ran ','OK','FAILED','ERROR:','FAIL:'))]
        results.append(dict(command=command,exit_code=exit_code,log=logfile,summary=summary,
                            reused_existing_run=reuse))
        print(logfile,exit_code,summary,flush=True)
    before=json.loads((OUT/'protected_before.json').read_text())
    changed=[name for name,h in before['sha256'].items() if not (ROOT/name).is_file() or digest(ROOT/name)!=h]
    # Preserve a concurrently observed ancillary edit; never rewrite the initial inventory.
    ancillary_changes=[name for name in changed if name=='.gitignore']
    scientific_changes=[name for name in changed if name not in ancillary_changes]
    unchanged_protocol=digest(OUT/'preregistration.json')==before['preregistration_sha256']
    input_hashes=json.loads((OUT/'run_manifest.json').read_text())['input_sha256']
    changed_inputs=[name for name,h in input_hashes.items() if repository_sha256(ROOT/name)!=h]
    baseline_evidence=[]
    expected='ef1c4e4e33217a5c821a94739f43288942f99e351c05ee454305eae9e3644394'
    for name in ['experiments/ml_baseline/result.csv','experiments/m0_baseline_reference/result.csv']:
        baseline_evidence.append(dict(path=name,raw_sha256=digest(ROOT/name),
            unchanged_since_before_experiment=digest(ROOT/name)==before['sha256'][name],
            normalized_sha256=repository_sha256(ROOT/name),test_expected_sha256=expected,
            normalized_matches_legacy_test=repository_sha256(ROOT/name)==expected))
    verification=dict(protected_files_checked=len(before['sha256']),changed_protected_files=changed,
                      changed_scientific_protected_files=scientific_changes,
                      concurrent_ancillary_changes=ancillary_changes,
                      input_files_checked=len(input_hashes),changed_inputs=changed_inputs,
                      preregistration_unchanged=unchanged_protocol,
                      input_hash_algorithm='src.storage.sha256; CRLF normalized to LF for published text artifacts',
                      preservation_hash_algorithm='Exact raw-byte SHA256',baseline_hash_evidence=baseline_evidence,
                      actual_test_runs=results,
                      stage03_1_checks_passed=not scientific_changes and not changed_inputs and unchanged_protocol and results[0]['exit_code']==0,
                      passed=not changed and not changed_inputs and unchanged_protocol and all(x['exit_code']==0 for x in results))
    write_json('verification.json',verification)
    write_json('protected_verification.json',dict(phase='final_after_tests',
        protected_files_checked=len(before['sha256']),changed=changed,
        changed_scientific_protected_files=scientific_changes,
        concurrent_ancillary_changes=ancillary_changes))
    decision=json.loads((OUT/'decision.json').read_text())
    cohorts=json.loads((OUT/'cohort_summary.json').read_text())
    fit=pd.read_csv(OUT/'candidate_fits.csv'); folds=pd.read_csv(OUT/'fold_audit.csv')
    stability=pd.read_csv(OUT/'robustness_summary.csv')
    lines=['# Stage 03.1 QA and decision', '',
           f"Branch `{before['branch']}`, HEAD `{before['head']}`. Initial user changes: `.mcp.json`, untracked `.github/`; preserved.", '',
           f"Protected files checked: {len(before['sha256'])}; changed: {len(changed)}; scientific/source/data/baseline changes: {len(scientific_changes)}. Inputs checked: {len(input_hashes)}; mismatches: {len(changed_inputs)}. Frozen preregistration unchanged: {unchanged_protocol}.", '',
           'A `.gitignore` addition (`.github`) appeared during execution after the initial clean inventory. No Stage 03.1 command edits that file. It is preserved and disclosed as a concurrent ancillary change; the original before-hash inventory is not rewritten to hide it. Consequently the blanket all-protected-files-unchanged gate remains false.', '',
           'The first execution failed before data loading because Windows default cp1252 could not read the existing UTF-8 manifest. Reproduction uses `-X utf8`; original source and loader were not changed. See run_attempt_01_encoding_failure.txt.', '',
           'The initial verifier incorrectly compared raw-byte input hashes against the loader\'s CRLF-normalized hashes. The corrected verifier calls src.storage.sha256 for that contract and separately checks exact bytes for all protected files. No input checksum was bypassed.', '']
    for c in cohorts:
        case=c['case']; d=decision['cases'][case]; f=fit[fit['case'].eq(case)&fit.hold_days.eq(1)]
        independent=folds[folds['case'].eq(case)&folds.hold_days.eq(1)&folds.evidence_status.eq('forward_time_held_parent_selection_diagnostic')]
        sections=stability[stability['case'].eq(case)&stability.hold_days.eq(1)].usable_section_count.min()
        lines.extend([f"## {case}: {d['outcome']}", '',
            f"{c['rows']} eligible rows; {len(c['parents'])} parent; {len(c['campaigns'])} campaign; {c['sections']} sections; {c['history_windows']} section/windows. Latest label: {c['latest_label']}. Negative measured thinning: {c['negative_observed_loss_rows']} rows.", '',
            f"Admissible independent folds: {int(independent.admissible.sum())}; usable section holdouts: {sections}. One-day support: {f.observed_support_fraction.min():.6%}; unknown share: {f.unknown_share.max():.6%}. Missing-history rows: {int(f.missing_temperature_history_rows.max())}.", '',
            'Failed selection gates: '+', '.join(d['failed_rules'])+'.', '',
            'Retained: '+', '.join(d['retained_candidates'])+'.', ''])
    lines.extend(['## Interpretation and Stage 04', '',
        'The experiment does not justify a unique selection under the frozen engineering tolerances; it cannot rule out a physical candidate merely from its numerical rank. All retained hypotheses remain compatible with this limited evidence. Cold-start support is sparse; between-VTD exposure shape differences are small. Practical held-out ties and shared parent/campaign histories prevent identification. All 60 sensitivity matrices have effective rank 2; rank deficiency is not the reason selection was withheld. See findings.md for the numerical interpretation.', '',
        'GO for isolated Stage 04 development with the retained candidate grid and explicit gap assumptions. NO-GO for claiming a scientifically selected Ea or promoting production forecasts. EA40 can be software continuity only. Obtain independent campaigns/parents spanning varied, densely measured temperatures, paired W/v, reliable baseline and repeated thickness labels before making a transfer claim.', '',
        '## Actual tests', ''])
    for r in results:
        lines.append(f"- `{r['log']}`: exit {r['exit_code']}; {'; '.join(r['summary'])}")
    lines.extend(['',f"Stage 03.1 tests/input/preservation checks passed: {verification['stage03_1_checks_passed']}. Full regression suite passed: {verification['passed']}.",'',
                  'Existing regression limitations: the Stage 02 baseline test hashes raw CRLF bytes against an LF hash; both baseline files retain their exact pre-experiment bytes and their normalized hashes match the expected value. Preprocessing DatasetTests setup raises `TypeError: boolean value of NA is ambiguous` at src/data_cleaner.py:132 with this installed pandas/Arrow environment. Those unchanged source paths do not import Stage 03.1. No prior tests, parser, data or baselines were edited to make them pass. Full failure traces are retained in tests_regression.txt.', '',
                  'The 13.3086% prior cold-start production support remains cited handoff context; no production reconstruction was required for historical selection. Production segment counts are regression-tested, not counted as independent experiments.'])
    (OUT/'qa_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                    command='.venv/Scripts/python.exe -X utf8 -B -m experiments.temperature_v02_03_1.run',
                    verification_command='.venv/Scripts/python.exe -X utf8 -B -m experiments.temperature_v02_03_1.verify',
                    input_raw_bytes_sha256={name:digest(ROOT/name) for name in input_hashes},
                    reporting_command='.venv/Scripts/python.exe -X utf8 -B -m experiments.temperature_v02_03_1.run --summarize-only',
                    verification=verification,
                    stage_source_sha256={p.name:digest(p) for p in OUT.glob('*.py')})
    manifest['outputs_sha256']={p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ('run_manifest.json','verify.txt')}
    manifest['hash_inventory_note']='Manifest excludes itself and live verify.txt to avoid recursive/unstable hashes; test logs, run log, source, preregistration and all scientific artifacts included.'
    write_json('run_manifest.json',manifest)
    if not verification['passed']:
        raise SystemExit('Verification failed; inspect saved logs and mismatch lists')


if __name__=='__main__':
    main()

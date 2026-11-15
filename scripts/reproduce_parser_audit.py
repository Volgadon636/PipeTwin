"""Reproduce the two isolated builds and audit, without changing the checkout.
Run with the project environment: .venv/bin/python scripts/reproduce_parser_audit.py
Preserves published root outputs; replaces only generated audit build directories.
"""
import io,json,subprocess,sys,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/parser_audit'
sha=json.loads((OUT/'before_manifest.json').read_text())['head']
def command(args,cwd=ROOT):subprocess.run([sys.executable,*map(str,args)],cwd=cwd,check=True)
with tempfile.TemporaryDirectory(prefix='pipetwin-audit-source-') as t:
    archive=subprocess.check_output(['git','archive',sha,'src','configs'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(t,filter='data')
    command(['-m','src.data_prep','--data-dir',ROOT/'data','--config',Path(t)/'configs/preprocessing.json','--output-root',OUT/'baseline'],cwd=t)
for name in ['corrected','repeated']:
    command(['-m','src.data_prep','--output-root',OUT/name])
for name in ['parser_audit.py','parser_audit_extended.py','parser_ml_impact.py']:
    command([ROOT/'scripts'/name])
command(['-m','unittest','discover','-s','tests','-v'])
# Refresh portable manifests and check original root preservation / exact repeat bytes.
import hashlib,shutil
import pandas as pd
for version in ['baseline','corrected']:
    for filename in ['run_manifest.json','validation.json']:
        shutil.copyfile(OUT/version/'reports'/filename,OUT/(version+'_'+filename))
manifest=json.loads((OUT/'corrected_run_manifest.json').read_text())
repeat=[]
for stem in manifest['outputs']:
    for extension in ['csv','parquet']:
        path=f'{stem}.{extension}'
        same=(OUT/'corrected'/path).read_bytes()==(OUT/'repeated'/path).read_bytes()
        repeat.append(dict(path=path,byte_equal=same))
pd.DataFrame(repeat).to_csv(OUT/'REPRODUCIBILITY.csv',index=False)
original=json.loads((OUT/'before_manifest.json').read_text())['sha256']
changed=[name for name,digest in original.items() if name.startswith(('data/','processed/','reports/','configs/')) and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
(OUT/'IMMUTABILITY.json').write_text(json.dumps(dict(changed_original_artifacts=changed,repeat_equal=sum(r['byte_equal'] for r in repeat),repeat_files=len(repeat)),indent=2)+'\n')
if changed or not all(r['byte_equal'] for r in repeat):raise SystemExit('Immutability/repeat check failed')

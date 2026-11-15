"""Compare M0 with baseline code/data versus corrected code/data.
The baseline source is extracted from the recorded commit without switching branches.
"""
import io,json,shutil,subprocess,sys,tarfile,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/parser_audit'
BASE_SHA=json.loads((OUT/'before_manifest.json').read_text())['head']

def evaluate(source,version):
    sys.path.insert(0,str(source))
    import pandas as pd
    from src.ml.data import RepositoryData
    from src.ml.windows import build_training_windows,make_snapshot,eligible_training
    from src.ml.models import fit_constant_rate
    root=OUT/version;(root/'configs').mkdir(exist_ok=True)
    shutil.copyfile(ROOT/'configs/preprocessing.json',root/'configs/preprocessing.json')
    data=RepositoryData(root);windows,_=build_training_windows(data)
    windows.to_csv(OUT/(version+'_training_windows.csv'),index=False)
    outputs=[];models=[]
    for request in data.config['targets']:
        parent=data.parent_groups().loc[request['id']]
        train=eligible_training(windows,request['prediction_date'],request['case'],parent if request['case']=='cold_start' else None)
        model=fit_constant_rate(train,'W1');features,_=make_snapshot(data,request)
        prediction=features[['id','segment_id']].copy();prediction['thickness_pred']=model.predict(features);outputs.append(prediction)
        models.append(dict(case=request['case'],rows=len(train),rate=model.rate_mm_year))
    pd.concat(outputs,ignore_index=True).to_csv(OUT/(version+'_m0_predictions.csv'),index=False)
    (OUT/(version+'_m0_fit.json')).write_text(json.dumps(models,indent=2)+'\n')

if len(sys.argv)>1 and sys.argv[1]=='--child':
    evaluate(Path(sys.argv[2]),sys.argv[3])
else:
    import pandas as pd
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='pipetwin-baseline-') as t:
        archive=subprocess.check_output(['git','archive',BASE_SHA,'src'],cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(t,filter='data')
        subprocess.run([sys.executable,__file__,'--child',t,'baseline'],check=True)
    subprocess.run([sys.executable,__file__,'--child',str(ROOT),'corrected'],check=True)
    a,b=[pd.read_csv(OUT/(v+'_m0_predictions.csv')) for v in ['baseline','corrected']]
    pd.testing.assert_frame_equal(a[['id','segment_id']],b[['id','segment_id']])
    wa,wb=[pd.read_csv(OUT/(v+'_training_windows.csv')) for v in ['baseline','corrected']]
    pd.testing.assert_frame_equal(wa,wb)
    meta={v:json.loads((OUT/(v+'_m0_fit.json')).read_text()) for v in ['baseline','corrected']}
    meta.update(baseline_source_commit=BASE_SHA,training_windows_equal=True,training_window_rows=len(wa),max_absolute_prediction_change_mm=float(np.abs(a.thickness_pred-b.thickness_pred).max()),changed_prediction_count=int((a.thickness_pred!=b.thickness_pred).sum()))
    (OUT/'ML_IMPACT.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))

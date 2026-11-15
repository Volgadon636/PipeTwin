"""Independent rules/joins verification; reads the isolated corrected build.
Production helpers are used ONLY to obtain historical snapshots to be checked.
"""
import csv,json,sys,math,re,statistics
from datetime import datetime,timedelta
from pathlib import Path
from decimal import Decimal,InvalidOperation
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import TECH_FILE,NUMERIC_COLUMNS,DATE_COLUMNS,HYDRAULIC_FEATURES
OUT=ROOT/'reports/parser_audit';NEW=OUT/'corrected'
config=json.loads((ROOT/'configs/preprocessing.json').read_text())
manifest=json.loads((NEW/'reports/run_manifest.json').read_text())
checks=[]
def check(label,value,detail=''):
    checks.append({'check':label,'passed':bool(value),'detail':detail})
def equal(a,b):
    if pd.isna(a) or pd.isna(b):return pd.isna(a) and pd.isna(b)
    if isinstance(a,(int,float,np.number)) and not isinstance(a,(bool,np.bool_)):
        return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-10)
    return a==b
def read(name):return pd.read_parquet(NEW/(name+'.parquet'))
clean={name:read('data/cleaned/'+Path(name).stem) for name in manifest['inputs']}
passport=clean['passport.csv'].set_index('id');tol=config['vtd_thickness_tolerance_mm']
issues=read('processed/quality_issues');failures=[];date_errors=[];archival_errors=[]
missing={'','#N/A','NaN','nan','NULL'}
for name,meta in manifest['inputs'].items():
    with (ROOT/'data'/meta['source_filename']).open(encoding=meta['encoding'],newline='') as h:
        raw=list(csv.DictReader(h,delimiter=meta['delimiter'],strict=True))
    allout=clean[name]
    rejected=NEW/'processed'/(Path(name).stem+'_rejected.parquet')
    if rejected.exists():allout=pd.concat([allout,pd.read_parquet(rejected)])
    allout=allout.set_index('source_row')
    for sr,row in enumerate(raw,2):
        if sr not in allout.index:continue
        for original,col in meta['column_mapping'].items():
            value=row[original];actual=allout.at[sr,col]
            if col in DATE_COLUMNS[name]:
                expected=None
                if value not in missing:
                    s=value.strip()
                    try:
                        if re.match(r'^\d{4}-\d{2}-\d{2}',s):expected=datetime.fromisoformat(s)
                        elif re.match(r'^\d{2}\.\d{2}\.\d{4}$',s):expected=datetime.strptime(s,'%d.%m.%Y')
                    except ValueError:pass
                if not equal(pd.NaT if expected is None else expected,actual):date_errors.append((name,sr,col,value,str(actual)))
                invalid=expected is None and value not in missing
            elif col in NUMERIC_COLUMNS[name]:
                try:n=Decimal(value.strip().replace('\u00a0','').replace(',','.'));invalid=not n.is_finite()
                except InvalidOperation:invalid=True
                invalid=invalid and value not in missing
                if col=='id' and not invalid and value not in missing:check(f'exact ID {name}/{sr}',int(n)==actual)
            else:
                invalid=False
                if not equal(pd.NA if value in missing else value,actual):archival_errors.append((name,sr,col))
            if invalid:failures.append(dict(source_file=name,source_row=sr,field=col,original_value=value))
check('all raw dates independently parsed',not date_errors,str(date_errors[:5]))
check('all archival strings preserved except documented NA sentinels',not archival_errors,str(archival_errors[:5]))
logged={(r.file,r.source_row,r.field,r.original_value) for r in issues[issues.rule.eq('parse_failed')].itertuples()}
check('every invalid conversion has exact source evidence',all((r['source_file'],r['source_row'],r['field'],r['original_value']) in logged for r in failures))
pd.DataFrame(failures,columns=['source_file','source_row','field','original_value']).to_csv(OUT/'FAILED_CONVERSIONS.csv',index=False)
# Independent per-row rules; assertions count all records, not selected examples.
v=clean['vtd.csv'];expected=[]
for r in v.itertuples():
    nominal=passport.loc[r.id,'nominal_thickness_mm'];commission=passport.loc[r.id,'commissioning_date']
    normalized=str(r.date.date()) in config['vtd_fraction_dates'] and pd.notna(r.defect_depth_pct_raw)
    depth=r.defect_depth_pct_raw*(100 if normalized else 1)
    invalid=pd.notna(r.residual_wall_thickness) and (r.residual_wall_thickness<=0 or (pd.notna(r.wall_thickness) and r.residual_wall_thickness>r.wall_thickness+tol))
    before=pd.notna(commission) and r.date<commission
    held=any(r.id==t['id'] and r.date>=pd.Timestamp(t['prediction_date']) for t in config['targets'])
    values={'depth_fraction_normalized':normalized,'defect_depth_pct':depth,'residual_invalid_flag':invalid,'residual_missing_flag':pd.isna(r.residual_wall_thickness),'residual_for_target_mm':np.nan if invalid else r.residual_wall_thickness,'before_commissioning_flag':before,'commissioning_unknown_flag':pd.isna(commission),'heldout_or_future_flag':held,'wall_nominal_mismatch_flag':abs(r.wall_thickness-nominal)>tol,'depth_residual_mismatch_flag':abs(r.residual_wall_thickness-r.wall_thickness*(1-depth/100))>tol,'target_eligible':pd.notna(r.residual_wall_thickness) and not invalid and not before and pd.notna(commission) and not held}
    expected.append(all(equal(getattr(r,k),val) for k,val in values.items()))
check('all VTD normalization/flags/eligibility independently reconstructed',all(expected),f'{len(expected)} records')
v.groupby(['id','date','feature_description','defect_type'],dropna=False).agg(rows=('source_row','size'),measured=('residual_wall_thickness','count'),depth_mismatches=('depth_residual_mismatch_flag','sum'),min_residual=('residual_wall_thickness','min')).to_csv(OUT/'VTD_FEATURE_COHORTS.csv')
v[v.depth_residual_mismatch_flag].to_csv(OUT/'VTD_DEPTH_DISAGREEMENTS.csv',index=False)
anomalies=[]
for name,flags in {'vtd':['before_commissioning_flag','wall_nominal_mismatch_flag','depth_residual_mismatch_flag','residual_missing_flag'],'uzt':['above_nominal_flag'],'profile':['steep_step_flag'],'regime':['direction_conflict_flag','exposure_invalid_flag','exposure_mismatch_flag','density_suspect_flag','negative_protective_effect_flag']}.items():
    d=clean[name+'.csv']
    for flag in flags:
        g=d[d[flag]]
        anomalies.append(dict(source=name+'.csv',flag=flag,count=len(g),example_source_rows=';'.join(map(str,g.source_row.head(5))),ids=';'.join(map(str,g.id.dropna().unique())),interpretation='Source anomaly / unresolved measurement interpretation; not automatically a parser defect'))
pd.DataFrame(anomalies).to_csv(OUT/'SOURCE_ANOMALIES.csv',index=False)
# All aggregates, using independently assigned segments and independent validity.
for name,stem in [('vtd.csv','vtd_by_segment'),('uzt.csv','uzt_by_segment')]:
    c=clean[name].copy();c['sid']=[min(int(Decimal(str(r.distance))//10),math.ceil(float(passport.loc[r.id,'length_km'])*1000/10)-1) for r in c.itertuples()]
    actual=read('processed/'+stem).set_index(['id','date','segment_id']);ok=True
    for key,g in c.groupby(['id','date','sid']):
        if name=='vtd.csv':
            values={'observation_count':len(g),'thickness_measurement_count':g.residual_for_target_mm.count(),'min_residual_thickness_mm':g.residual_for_target_mm.min(),'max_defect_depth_pct':g.defect_depth_pct.max()}
            values.update({k:g[k].any() for k in ['before_commissioning_flag','commissioning_unknown_flag','wall_nominal_mismatch_flag','heldout_or_future_flag']})
            values['target_eligible']=values['thickness_measurement_count']>0 and not any(values[k] for k in ['before_commissioning_flag','commissioning_unknown_flag','heldout_or_future_flag'])
        else:
            eligible=[]
            for r in g.itertuples():
                commission=passport.loc[r.id,'commissioning_date'];nom=passport.loc[r.id,'nominal_thickness_mm']
                valid=r.residual_wall_thickness>0 and r.residual_wall_thickness<=nom+tol and pd.notna(commission) and r.date>=commission
                ok &= equal(r.measurement_eligible,valid)
                if valid:eligible.append(r.residual_wall_thickness)
            values={'measurement_count':g.residual_wall_thickness.count(),'min_residual_thickness_mm':g.residual_wall_thickness.min(),'mean_residual_thickness_mm':g.residual_wall_thickness.mean(),'eligible_measurement_count':len(eligible),'eligible_min_thickness_mm':min(eligible) if eligible else np.nan,'above_nominal_flag':g.above_nominal_flag.any()}
        ok &= all(equal(actual.loc[key,k],val) for k,val in values.items())
    check('complete aggregates '+name,ok)
a=read('processed/vtd_by_segment');pd.testing.assert_frame_equal(a[a.target_eligible].reset_index(drop=True),read('processed/training_targets').reset_index(drop=True))
check('training targets exact eligible observed subset',True)
tech=clean[TECH_FILE];diameter_ok=True;record_ok=True
for r in tech.itertuples():
    known=r.id in passport.index;d=passport.loc[r.id,'diameter_mm'] if known else np.nan
    meters=pd.notna(r.diameter_mm_raw) and abs(r.diameter_mm_raw*1000-d)<=.1
    millimeters=pd.notna(r.diameter_mm_raw) and abs(r.diameter_mm_raw-d)<=.1
    diameter_ok &= equal(r.diameter_mm,r.diameter_mm_raw*(1000 if meters else 1)) and equal(r.diameter_mismatch_flag,not(meters or millimeters))
    commissioned=passport.loc[r.id,'commissioning_date'] if known else pd.NaT
    record_ok &= equal(r.record_eligible,known and pd.notna(r.date) and pd.notna(commissioned) and r.date>=commissioned)
check('all hydraulic diameter conversions vs passport',diameter_ok)
check('all hydraulic record eligibility',record_ok)
def direction(s):return '' if pd.isna(s) else ''.join(c for c in s.lower() if c.isalnum() or c=='_')
monitor=clean['regime.csv'];ok=True
for r in monitor.itertuples():
    known=r.id in passport.index;matched=known and pd.notna(r.direction_name) and direction(r.direction_name)==direction(passport.loc[r.id,'direction_name'])
    duration=(r.sample_removal_date-r.sample_installation_date).total_seconds()/86400 if pd.notna(r.sample_removal_date) and pd.notna(r.sample_installation_date) else np.nan
    invalid=pd.isna(duration) or duration<=0;mismatch=abs(duration-r.exposure_days)>1
    ok &= equal(r.observation_eligible,matched and pd.notna(r.sample_removal_date) and not invalid and not mismatch)
    ok &= equal(r.mapping_eligible,matched) and equal(r.exposure_days_from_dates,duration)
check('all monitoring ID/direction/exposure eligibility',ok)
# Hydraulic mask reference: Python sets of calendar dates and explicit runs.
def expected_hydraulic(pid,cutoff):
    rows=tech[tech.id.eq(pid)&tech.record_eligible&tech.date.lt(cutoff)].copy()
    bad={f:set() for f in HYDRAULIC_FEATURES};limits=config['soft_limits']
    rules=[]
    def add(field,predicate,hard=False):rules.append((field,predicate,hard))
    for f in ['q_liquid','q_oil','v_liquid_true','v_mix','density_liquid_work','density_oil']:add(f,lambda r,f=f:getattr(r,f)<0,True)
    add('watercut_pct',lambda r:pd.notna(r.watercut_pct) and not 0<=r.watercut_pct<=100,True)
    for f in ['p_start','p_end']:
        add(f,lambda r,f=f:r.q_liquid>0 and getattr(r,f)==0);add(f,lambda r,f=f:getattr(r,f)>limits['pressure_high'])
    for f in ['q_liquid','q_oil','v_liquid_true','v_mix']:add(f,lambda r:r.q_liquid>limits['liquid_flow_high'])
    for f in ['v_liquid_true','v_mix']:add(f,lambda r,f=f:getattr(r,f)>limits['velocity_high'])
    for f in ['t_mix','t_start','t_end']:
        add(f,lambda r,f=f:getattr(r,f)<=-273.15,True);add(f,lambda r,f=f:r.q_liquid>0 and getattr(r,f)==0)
        add(f,lambda r,f=f:getattr(r,f)<limits['temperature_low']);add(f,lambda r,f=f:getattr(r,f)>limits['temperature_high'])
    for f in ['density_liquid_work','density_oil']:
        add(f,lambda r,f=f:r.q_liquid>0 and getattr(r,f)==0,True)
        add(f,lambda r,f=f:r.q_liquid>0 and 0<getattr(r,f)<limits['density_low']);add(f,lambda r,f=f:getattr(r,f)>limits['density_high'])
    rr=list(rows.itertuples())
    for field,predicate,hard in rules:
        candidates=[r for r in rr if predicate(r)];days=sorted({r.date.date() for r in candidates});persistent=set();run=[]
        for day in days+[None]:
            if run and (day is None or day-run[-1]!=timedelta(days=1)):
                if len(run)>=config['persistent_days']:persistent.update(run)
                run=[]
            if day is not None:run.append(day)
        bad[field].update(r.source_row for r in candidates if hard or r.date.date() not in persistent)
    for f in HYDRAULIC_FEATURES:rows[f+'_usable']=[np.nan if r.source_row in bad[f] else getattr(r,f) for r in rr]
    return rows
from src.segment_mapper import build_features_asof
vtd=read('processed/vtd_by_segment');uzt=read('processed/uzt_by_segment');grid=read('processed/segments_grid');profile=read('processed/profile_by_segment')
requests=list(config['targets'])
# Three historical cutoffs for every pipe and both scenarios, including inspection boundary days.
for pid,g in vtd.groupby('id'):
    dates=sorted(g.date.unique())
    for cutoff in sorted(set([dates[0],dates[len(dates)//2],dates[-1]+pd.Timedelta(days=1)])):
        for case in ['between_vtd','cold_start']:requests.append(dict(id=int(pid),prediction_date=str(cutoff),case=case))
for i,req in enumerate(requests):
    pid=req['id'];cutoff=pd.Timestamp(req['prediction_date']);case=req['case']
    result=build_features_asof(clean,grid,profile,vtd,uzt,[req],config)
    features=result['target_features'];ref=expected_hydraulic(pid,cutoff);actual=result['hydraulic_asof'].set_index('source_row')
    ok=set(ref.source_row)==set(actual.index)
    for r in ref.itertuples():
        ok &= all(equal(getattr(r,f+'_usable'),actual.loc[r.source_row,f+'_usable']) for f in HYDRAULIC_FEATURES)
    check(f'snapshot {i} hydraulic cutoff and every usable value',ok,str(req))
    expected_profile=profile[profile.id.eq(pid)].set_index('segment_id');ok=True
    dates=sorted({d.date() for d in ref.date});count=len(dates);span=(dates[-1]-dates[0]).days+1 if dates else 0
    basic={'hydraulic_records_count':len(ref),'hydraulic_days_count':count,'hydraulic_coverage_fraction':count/span if span else np.nan,'hydraulic_max_gap_days':max(((b-a).days for a,b in zip(dates,dates[1:])),default=np.nan),'hydraulic_source_date_min':ref.date.min(),'hydraulic_source_date_max':ref.date.max()}
    for f in HYDRAULIC_FEATURES:
        daily={}
        for r in ref.itertuples():
            value=getattr(r,f+'_usable')
            if pd.notna(value):daily.setdefault(r.date.date(),[]).append(value)
        vals=[statistics.median(v) for v in daily.values()]
        basic[f+'_mean']=statistics.mean(vals) if vals else np.nan;basic[f+'_std']=statistics.stdev(vals) if len(vals)>1 else np.nan;basic[f+'_valid_days']=len(vals)
    usable=monitor[monitor.id.eq(pid)&monitor.observation_eligible&monitor.sample_removal_date.lt(cutoff)]
    candidates=monitor[monitor.id.eq(pid)&monitor.sample_removal_date.lt(cutoff)]
    basic.update(monitoring_candidate_records=len(candidates),monitoring_records_count=len(usable),monitoring_source_date_max=usable.sample_removal_date.max())
    for f in ['ph','co2_water','total_mineralization']:basic[f+'_median']=usable[f].median()
    for r in features.itertuples():
        ok &= all(equal(getattr(r,k),val) for k,val in basic.items())
        ok &= all(equal(getattr(r,k),val) for k,val in expected_profile.loc[r.segment_id].items() if k!='id')
        for kind,hist,col,out in [('vtd',vtd,'min_residual_thickness_mm','last_vtd_thickness_mm'),('uzt',uzt,'eligible_min_thickness_mm','last_uzt_thickness_mm')]:
            g=hist[hist.id.eq(pid)&hist.segment_id.eq(r.segment_id)&hist.date.lt(cutoff)]
            g=g[g.target_eligible] if kind=='vtd' else g[g.eligible_measurement_count.gt(0)]
            if case=='cold_start':g=g.iloc[:0]
            latest=g.sort_values('date').iloc[-1] if len(g) else None
            ok &= equal(getattr(r,out),latest[col] if latest is not None else np.nan)
            ok &= equal(getattr(r,'last_'+kind+'_date'),latest.date if latest is not None else pd.NaT)
        conflict=case!='cold_start' and any(x.date<cutoff and x.before_commissioning_flag for x in vtd[vtd.id.eq(pid)].itertuples())
        age=(cutoff-passport.loc[pid,'commissioning_date']).total_seconds()/(86400*365.25)
        ok &= equal(r.history_date_conflict_flag,conflict) and equal(r.age_years,np.nan if conflict or age<0 else age)
    check(f'snapshot {i} independent final joins and daily statistics',ok,str(req))
pd.DataFrame(checks).to_csv(OUT/'EXTENDED_CHECKS.csv',index=False)
summary={'checks':len(checks),'failures':[r for r in checks if not r['passed']],'historical_and_target_requests':len(requests),'failed_conversions':len(failures)}
(OUT/'EXTENDED_SUMMARY.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
print(json.dumps(summary,default=str))
if summary['failures']:raise SystemExit(1)

"""Independent source/aggregate reconciliation and complete baseline/corrected diff.
Run: .venv/bin/python scripts/parser_audit.py
"""
import sys,csv,json,hashlib,subprocess,math,bisect
from pathlib import Path
from decimal import Decimal,InvalidOperation
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import NUMERIC_COLUMNS,DATE_COLUMNS,TECH_FILE
OUT=ROOT/'reports/parser_audit'; BASE=OUT/'baseline'; NEW=OUT/'corrected'

def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')
def frame(name,rows):
    columns={'NUMERIC_CONVERSION_DISAGREEMENTS.csv':['file','source_row','field','raw','actual']}
    pd.DataFrame(rows,columns=columns.get(name)).to_csv(OUT/name,index=False)
def numeric(v):
    try: return float(Decimal(v.strip().replace('\u00a0','').replace(',','.')))
    except (InvalidOperation,ValueError): return np.nan

def run():
    manifest=json.loads((NEW/'reports/run_manifest.json').read_text());published=json.loads((ROOT/'reports/run_manifest.json').read_text())
    checks=[]
    def check(name,ok,detail=''):
        checks.append(dict(check=name,passed=bool(ok),detail=str(detail)))
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    inventory=[]
    for directory in ['data','processed','reports']:
        for p in sorted((ROOT/directory).rglob('*')):
            if p.is_file() and OUT not in p.parents:
                inventory.append(dict(path=str(p.relative_to(ROOT)),tracked=str(p.relative_to(ROOT)) in tracked,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    frame('FILE_INVENTORY.csv',inventory)
    raw={};mapping=[];account=[];units=[];conversion=[]
    for name,meta in manifest['inputs'].items():
        path=ROOT/'data'/meta['source_filename']
        with path.open(encoding=meta['encoding'],newline='') as f:
            records=list(csv.reader(f,delimiter=meta['delimiter'],strict=True))
        head,records=records[0],records[1:]
        check(name+': logical record widths',all(len(r)==len(head) for r in records))
        check(name+': raw count',len(records)==meta['rows'])
        table=pd.DataFrame(records,columns=head).rename(columns=meta['column_mapping']);table['source_row']=range(2,len(table)+2);raw[name]=table
        clean=pd.read_parquet(NEW/'data/cleaned'/Path(name).with_suffix('.parquet'))
        rejectpath=NEW/'processed'/(Path(name).stem+'_rejected.parquet')
        rejected=pd.read_parquet(rejectpath) if rejectpath.exists() else pd.DataFrame(columns=['source_row'])
        retained=set(clean.source_row);quarantine=set(rejected.source_row)
        check(name+': disjoint source partitions',not retained&quarantine)
        allout=pd.concat([clean,rejected],ignore_index=True).set_index('source_row')
        for original,normalized in meta['column_mapping'].items():
            kind='numeric' if normalized in NUMERIC_COLUMNS[name] else 'date' if normalized in DATE_COLUMNS[name] else 'archival text'
            mapping.append(dict(source_file=name,original_column=original,normalized_column=normalized,source_dtype='CSV text',target_dtype=str(clean[normalized].dtype),transformation=kind,status='preserved or typed; exclusions in row reconciliation'))
            if kind=='numeric':
                x=table[normalized].map(numeric)
                target='defect_depth_pct_raw' if name=='vtd.csv' and normalized=='defect_depth_pct' else 'diameter_mm_raw' if name==TECH_FILE and normalized=='diameter_mm' else normalized
                for sr,value,original_value in zip(table.source_row,x,table[normalized]):
                    if sr in allout.index:
                        actual=allout.at[sr,target]
                        good=(pd.isna(actual) and np.isnan(value)) or (pd.notna(actual) and np.isclose(float(actual),value,rtol=1e-12,atol=1e-10))
                        if not good: conversion.append(dict(file=name,source_row=sr,field=normalized,raw=original_value,actual=actual))
                unit=('identifier (no physical dimension)' if normalized in ['id','row_id'] else 'mm' if normalized in ['diameter_mm','nominal_thickness_mm','wall_thickness','residual_wall_thickness','scrap_wall_thickness_mm'] else 'Celsius' if normalized in ['t_mix','t_start','t_end'] else 'percent' if normalized in ['watercut_pct','defect_depth_pct'] else 'km' if normalized=='length_km' else 'm' if normalized in ['distance','total_distance','length_accounting_m','z'] else 'days' if normalized=='exposure_days' else 'calendar year' if normalized=='year' else 'unresolved / see source documentation')
                units.append(dict(file=name,field=normalized,source_units=('mixed scale; retained raw' if target!=normalized else unit),target_units=unit,conversion=('cohort x100' if target=='defect_depth_pct_raw' else 'passport-matched x1000' if target=='diameter_mm_raw' else 'none'),min=x.min(),max=x.max(),missing=int(x.isna().sum()),negative=int(x.lt(0).sum()),zero=int(x.eq(0).sum()),reversible='raw source immutable',evidence='original header: '+original+'; documented pipeline convention only where unit specified; diameter subtype unresolved',dimensional_consistency='unverified' if unit.startswith('unresolved') else 'consistent with declared convention; physical semantics require source confirmation'))
        for row in table.itertuples():
            state='retained' if row.source_row in retained else 'quarantined' if row.source_row in quarantine else 'excluded_exact_profile_duplicate' if name=='profile.csv' else 'UNEXPLAINED'
            reason=state;eligible=None
            if row.source_row in allout.index:
                r=allout.loc[row.source_row]
                flags=[c for c in r.index if c.endswith('_flag') and pd.notna(r[c]) and bool(r[c])]
                reason=';'.join(flags) or state
                for field in ['target_eligible','measurement_eligible','observation_eligible','record_eligible']:
                    if field in r: eligible=bool(r[field])
            account.append(dict(source_file=name,source_row=row.source_row,id=getattr(row,'id',None),partition=state,analytical_eligible=eligible,reason=reason))
        check(name+': all rows accounted',all(r['partition']!='UNEXPLAINED' for r in account if r['source_file']==name))
        if name=='profile.csv':
            omitted=table[~table.source_row.isin(retained|quarantine)]
            check('profile omitted rows are exact duplicates',set(table.loc[table.drop(columns='source_row').duplicated(),'source_row'])==set(omitted.source_row))
    frame('COLUMN_TRANSFORMATION_AUDIT.csv',mapping);frame('SOURCE_TO_OUTPUT_RECONCILIATION.csv',account);frame('UNITS_AUDIT.csv',units)
    frame('NUMERIC_CONVERSION_DISAGREEMENTS.csv',conversion);check('independent Decimal conversions',not conversion)
    passport=pd.read_parquet(NEW/'data/cleaned/passport.parquet').set_index('id')
    grid=pd.read_parquet(NEW/'processed/segments_grid.parquet')
    for pid,g in grid.groupby('id'):
        length=round(float(passport.loc[pid,'length_km'])*1000,6)
        expected=[(i,i*10,min((i+1)*10,length)) for i in range(math.ceil(length/10))]
        check(f'independent grid {pid}',list(g[['segment_id','start_m','end_m']].itertuples(index=False,name=None))==expected)
    # Independent row assignment + measured minima, without assign_segments.
    lineage=[]
    for source,stem in [('vtd.csv','vtd_by_segment'),('uzt.csv','uzt_by_segment')]:
        c=pd.read_parquet(NEW/'data/cleaned'/Path(source).with_suffix('.parquet'))
        rows=[]
        for r in c.itertuples():
            length=round(float(passport.loc[r.id,'length_km'])*1000,6)
            sid=min(int(Decimal(str(r.distance))//Decimal(10)),math.ceil(length/10)-1)
            val=r.residual_wall_thickness
            valid=pd.notna(val) and val>0
            if source=='vtd.csv': valid=valid and (pd.isna(r.wall_thickness) or val<=r.wall_thickness+.02)
            else: valid=valid and not r.above_nominal_flag and not r.before_commissioning_flag
            rows.append(dict(id=r.id,date=r.date,segment_id=sid,value=val if valid else np.nan))
            lineage.append(dict(source_file=source,source_row=r.source_row,id=r.id,date=r.date,segment_id=sid))
        expected=pd.DataFrame(rows).groupby(['id','date','segment_id']).value.min()
        actual=pd.read_parquet(NEW/'processed'/(stem+'.parquet')).set_index(['id','date','segment_id'])
        col='min_residual_thickness_mm' if source=='vtd.csv' else 'eligible_min_thickness_mm'
        same_keys = set(expected.index) == set(actual.index)
        check('independent minima '+source,same_keys and np.allclose(expected,actual[col].reindex(expected.index),equal_nan=True))
    frame('MEASUREMENT_LINEAGE.csv',lineage)
    c=pd.read_parquet(NEW/'data/cleaned/vtd.parquet')
    residual=[]
    for (pid,date),g in c.groupby(['id','date']):
        valid=g.residual_wall_thickness.notna() & g.wall_thickness.gt(0) & g.defect_depth_pct_raw.notna()
        q=g[valid];e=(q.residual_wall_thickness-q.wall_thickness*(1-q.defect_depth_pct_raw)).abs();e2=(q.residual_wall_thickness-q.wall_thickness*(1-q.defect_depth_pct_raw/100)).abs()
        residual.append(dict(id=pid,date=date,paired=len(q),fraction_error_median=e.median(),percent_error_median=e2.median(),mismatch_rows=int(g.depth_residual_mismatch_flag.sum()),before_commissioning=int(g.before_commissioning_flag.sum()),nominal_mismatch=int(g.wall_nominal_mismatch_flag.sum())))
    frame('VTD_COHORT_AUDIT.csv',residual)
    # Independent piecewise linear function and integral (bisect; no np.interp).
    profile=pd.read_parquet(NEW/'data/cleaned/profile.parquet');mapped=pd.read_parquet(NEW/'processed/profile_by_segment.parquet')
    profile_errors=[]
    for pid,g in mapped.groupby('id'):
        p=profile[profile.id.eq(pid)].groupby('total_distance').z.mean().sort_index();xs=list(p.index);ys=list(p)
        pipe_end=float(g.end_m.max())
        def height(x):
            j=bisect.bisect_right(xs,x)
            if j==0:return ys[0]
            if j==len(xs):return ys[-1]
            return ys[j-1]+(ys[j]-ys[j-1])*(x-xs[j-1])/(xs[j]-xs[j-1])
        for r in g.itertuples():
            nodes=[r.start_m]+[x for x in xs if r.start_m<x<r.end_m]+[r.end_m]
            integral=sum((b-a)*(height(a)+height(b))/2 for a,b in zip(nodes,nodes[1:]))
            if not np.isclose(integral/r.length_m,r.z_mean,atol=1e-9,rtol=0): profile_errors.append((pid,r.segment_id))
            check(f'profile slope {pid}/{r.segment_id}',np.isclose((height(r.end_m)-height(r.start_m))/r.length_m,r.slope_m_per_m,atol=1e-9))
            heights=[height(x) for x in nodes]
            values=[(r.z_start,height(r.start_m)),(r.z_end,height(r.end_m)),(r.z_center,height((r.start_m+r.end_m)/2)),(r.z_min,min(heights)),(r.z_max,max(heights))]
            check(f'profile heights {pid}/{r.segment_id}',all(np.isclose(a,b,atol=1e-9,rtol=0) for a,b in values))
            check(f'profile extrapolation {pid}/{r.segment_id}',r.profile_extrapolated_flag==(r.start_m<xs[0] or r.end_m>xs[-1]))
            count=sum(x>=r.start_m and (x<r.end_m or (r.end_m==pipe_end and x<=r.end_m+.05)) for x in xs)
            support=xs[max(0,bisect.bisect_left(xs,r.start_m)-1):min(len(xs),bisect.bisect_left(xs,r.end_m)+1)]
            gap=max((b-a for a,b in zip(support,support[1:])),default=np.nan)
            check(f'profile support {pid}/{r.segment_id}',count==r.points_count and np.isclose(gap,r.max_profile_gap_m,equal_nan=True))
    check('independent profile integrals',not profile_errors)
    hyd=pd.read_parquet(NEW/'processed/hydraulic_asof.parquet');features=pd.read_parquet(NEW/'processed/target_features.parquet')
    for (pid,cutoff),g in hyd.groupby(['id','prediction_date']):
        check(f'hydraulic cutoff {pid}',g.date.lt(cutoff).all())
        for field in [c for c in g if c.endswith('_usable')]:
            buckets={}
            for date,val in zip(g.date,g[field]):
                if pd.notna(val):buckets.setdefault(date.date(),[]).append(float(val))
            val=np.mean([sorted(v)[len(v)//2] if len(v)%2 else sum(sorted(v)[len(v)//2-1:len(v)//2+1])/2 for v in buckets.values()]) if buckets else np.nan
            actual=features[features.id.eq(pid)][field.removesuffix('_usable')+'_mean'].iloc[0]
            check(f'independent daily aggregate {pid} {field}',np.isclose(val,actual,equal_nan=True))
    # All cells and serialization pairs, plus pre-existing published outputs.
    comparisons=[];cells=[];row_changes=[]
    for stem,meta in manifest['outputs'].items():
        a=pd.read_parquet(BASE/(stem+'.parquet'));b=pd.read_parquet(NEW/(stem+'.parquet'))
        common=[c for c in a if c in b]; changed=set();missing={}
        keys=([c for c in ['file','source_row','rule','field'] if c in a] if stem=='processed/quality_issues' else ['source_row']) if 'source_row' in a else [c for c in ['id','segment_id','date','prediction_date'] if c in a]
        def keyed(df):
            q=df.copy();q['_occurrence']=q.groupby(keys,dropna=False).cumcount() if keys else np.arange(len(q));return q.set_index(keys+['_occurrence'])
        aa,bb=keyed(a),keyed(b);overlap=aa.index.intersection(bb.index)
        for kind,source,indexes in [('added',bb,bb.index.difference(aa.index)),('removed',aa,aa.index.difference(bb.index))]:
            for key in indexes:
                row_changes.append(dict(table=stem,key=str(key),change=kind,values=json.dumps(source.loc[key].to_dict(),default=str),classification='expected downstream consequence' if stem=='processed/quality_issues' else 'requires review'))
        for col in [c for c in common if c not in keys]:
            left,right=aa.loc[overlap,col],bb.loc[overlap,col]
            if pd.api.types.is_numeric_dtype(left) and not pd.api.types.is_bool_dtype(left):
                eq=np.isclose(left.to_numpy(float,na_value=np.nan),right.to_numpy(float,na_value=np.nan),equal_nan=True,atol=1e-10,rtol=1e-12)
            else:eq=(left.astype('string').fillna('<NULL>')==right.astype('string').fillna('<NULL>')).to_numpy()
            for idx in np.flatnonzero(~eq):
                key=overlap[idx];changed.add(str(key));cells.append(dict(table=stem,key=str(key),column=col,old=left.iloc[idx],new=right.iloc[idx],classification='expected correction: missing exposure dates invalid' if col=='exposure_invalid_flag' else 'requires review'))
            missing[col]=int(b[col].isna().sum()-a[col].isna().sum())
        published_equal=None
        if (ROOT/(stem+'.parquet')).exists():
            try: pd.testing.assert_frame_equal(a,pd.read_parquet(ROOT/(stem+'.parquet')),check_dtype=False,atol=1e-10,rtol=1e-12);published_equal=True
            except AssertionError:published_equal=False
        read=pd.read_csv(NEW/(stem+'.csv'),dtype='string',keep_default_na=False,na_values=[''])
        from src.storage import validate_roundtrip
        validate_roundtrip(b,read,stem) # production check; labelled separately from independent reconstructions
        comparisons.append(dict(table=stem,old_rows=len(a),new_rows=len(b),added_rows=len(bb.index.difference(aa.index)),removed_rows=len(aa.index.difference(bb.index)),modified_rows=len(changed),added_columns=str(list(b.columns.difference(a.columns))),removed_columns=str(list(a.columns.difference(b.columns))),dtype_changes=str({c:(str(a[c].dtype),str(b[c].dtype)) for c in common if a[c].dtype!=b[c].dtype}),missingness_changes=json.dumps({c:v for c,v in missing.items() if v}),baseline_matches_published=published_equal,csv_bytes_changed=(BASE/(stem+'.csv')).read_bytes()!=(NEW/(stem+'.csv')).read_bytes()))
    frame('TABLE_COMPARISON.csv',comparisons);frame('CELL_DIFFERENCES.csv',cells);frame('ADDED_REMOVED_ROWS.csv',row_changes)
    published_issues=pd.read_parquet(ROOT/'processed/quality_issues.parquet')
    baseline_issues=pd.read_parquet(BASE/'processed/quality_issues.parquet')
    published_diff=baseline_issues.merge(published_issues,on=['file','source_row','rule','field'],suffixes=('_baseline','_published'))
    published_diff[published_diff.original_value_baseline!=published_diff.original_value_published].to_csv(OUT/'PUBLISHED_BASELINE_DIFFERENCES.csv',index=False)
    for stem in manifest['outputs']:
        subset=[r for r in cells if r['table']==stem]+[dict(table=stem,key=r['key'],column='<row>',old=r['values'] if r['change']=='removed' else '',new=r['values'] if r['change']=='added' else '',classification=r['classification']) for r in row_changes if r['table']==stem]
        folder=OUT/'table_diffs';folder.mkdir(exist_ok=True);pd.DataFrame(subset,columns=['table','key','column','old','new','classification']).to_csv(folder/(stem.replace('/','__')+'.csv'),index=False)
    frame('CHECKS.csv',checks)
    dump('AUDIT_SUMMARY.json',dict(checks=len(checks),failed=[c for c in checks if not c['passed']],raw_rows={k:len(v) for k,v in raw.items()},tables=len(comparisons),modified_cells=len(cells),published_code_match={name:published.get('code_sha256',{}).get(name)==digest for name,digest in json.loads((BASE/'reports/run_manifest.json').read_text())['code_sha256'].items()}))
    print(json.dumps(dict(checks=len(checks),failed=sum(not c['passed'] for c in checks),raw_rows=sum(map(len,raw.values())),modified_cells=len(cells))))
    if any(not c['passed'] for c in checks): raise SystemExit(1)
if __name__=='__main__':run()

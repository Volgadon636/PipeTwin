"""Reproduce Stage 04 in its own experiment directory only."""
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd

from src.ml.data import RepositoryData
from src.ml.models import fit_constant_rate
from src.ml.submission import validate_submission
from src.ml.temperature import EA_CANDIDATES
from src.ml.windows import build_training_windows, make_snapshot, eligible_training
from src.storage import sha256 as repository_sha256
from experiments.temperature_v02_03_1.run import thickness_metrics
from .model import ExposureCache, eligible_cohort, fit_temperature_rate

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
KEYS = ['id','segment_id','prediction_date','case']
PROVENANCE = ['main_pipeline_id','campaign_id','start_date','h0_mm','horizon_years',
              'baseline_source','baseline_missing','baseline_neighbor_segment_id','baseline_neighbor_distance_m']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,(np.integer,np.bool_)): return value.item()
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,(pd.Timestamp,Path)): return str(value)
    return value


def write_json(name,value):
    (OUT/name).write_text(json.dumps(json_safe(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def save(name,records):
    frame=records if isinstance(records,pd.DataFrame) else pd.DataFrame(records)
    (OUT/name).parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUT/name,index=False,float_format='%.17g',lineterminator='\n')
    return frame


def model_id(candidate,policy,hold):
    return f'{candidate}__{policy}__hold_{hold:g}d'


def independent_folds(cohort):
    audits,folds=[],[]
    for (parent,date),test in cohort.groupby(['main_pipeline_id','prediction_date']):
        case=test['case'].iloc[0]
        train=eligible_training(cohort,date,case,excluded_parent=parent)
        identifier=f'{case}:{parent}:{date.date()}'
        audits.append(dict(case=case,fold_id=identifier,heldout_parent=parent,test_date=date,
            train_rows=len(train),test_rows=len(test),train_parent_count=train.main_pipeline_id.nunique(),
            train_date_max=train.prediction_date.max(),admissible=not train.empty,
            reason='admissible' if len(train) else 'no_earlier_independent_training_group'))
        if len(train): folds.append((identifier,train,test))
    return audits,folds


def prediction_rows(frame,model,fit,mid,candidate,split):
    fields=KEYS+[x for x in PROVENANCE if x in frame]+[x for x in frame if x.startswith('thermal_')]+['exposure_years','interval_sha256']
    out=frame[list(dict.fromkeys(fields))].copy()
    out['model_id']=mid; out['candidate_id']=candidate
    out['gap_policy']=model.gap_policy; out['hold_days']=model.hold_days
    out['r_ref_mm_year']=model.rate_mm_year; out['split']=split
    out['thickness_pred']=model.predict(frame)
    out['predicted_loss_unclipped_mm']=model.rate_mm_year*frame.exposure_years.to_numpy(float)
    out['predicted_loss_clipped_mm']=frame.h0_mm.to_numpy(float)-out.thickness_pred
    out['zero_exposure_flag']=frame.exposure_years.eq(0).to_numpy()
    out['evidence_status']='diagnostic_only'
    out['forecast_status']=('experimental_reference_gap_assumption' if model.gap_policy=='reference_gap'
                            else 'incomplete_history_descriptor_not_full_forecast')
    out['ea_provenance']='user_fixed_development_choice' if candidate=='EA40_mid' else 'predeclared_comparator'
    if split=='historical_fit':
        out['min_residual_thickness_mm']=frame.min_residual_thickness_mm.to_numpy()
        out['observed_loss_mm']=(frame.h0_mm-frame.min_residual_thickness_mm).to_numpy()
        out['negative_observed_loss_quality_flag']=out.observed_loss_mm.lt(0)
    return out


def generate():
    protocol=json.loads((OUT/'protocol.json').read_text())
    before=json.loads((OUT/'protected_before.json').read_text())
    if digest(OUT/'protocol.json')!=before['protocol_sha256']:
        raise ValueError('Protocol changed')
    data=RepositoryData(ROOT)
    windows,histories=build_training_windows(data)
    parents=data.parent_groups()
    cohorts={r['case']:eligible_cohort(windows,r,parents) for r in protocol['targets']}
    targets={}
    for request in protocol['targets']:
        target,rows=make_snapshot(data,request)
        targets[request['case']]=target
        histories[(request['id'],pd.Timestamp(request['prediction_date']))]=rows
    cache=ExposureCache(histories)
    grid=data.read('processed/target_segments_grid')
    bounds=pd.concat(targets.values(),ignore_index=True)
    baseline_path=ROOT/'experiments/m0_baseline_reference/result.csv'
    baseline=pd.read_csv(baseline_path,float_precision='round_trip')
    validate_submission(baseline,grid,bounds)
    shutil.copyfile(baseline_path,OUT/'B0_original_M0.csv')
    baseline_models=json.loads((ROOT/'experiments/m0_baseline_reference/models.json').read_text())
    fits,history_predictions,predictions,evaluation,fold_audit=[],[],[],[],[]
    cohorts_out,cohort_audit,baseline_recon,baseline_deltas,coverage=[],[],[],[],[]
    b0_predictions=[]; independent_predictions=[]
    for request in protocol['targets']:
        case=request['case']; cutoff=pd.Timestamp(request['prediction_date'])
        train=cohorts[case]; target=targets[case]
        if not train.prediction_date.lt(cutoff).all(): raise ValueError('Future training label')
        if case=='cold_start':
            if train.main_pipeline_id.eq(parents.loc[request['id']]).any(): raise ValueError('Cold target parent leakage')
            if target[['last_vtd_thickness_mm','last_uzt_thickness_mm']].notna().any().any(): raise ValueError('Cold diagnostic leakage')
        cohorts_out.append(train[KEYS+PROVENANCE+['min_residual_thickness_mm']].assign(
            target_id=request['id'],production_cutoff=cutoff,
            negative_observed_loss_quality_flag=train.h0_mm.lt(train.min_residual_thickness_mm)))
        raw=windows[KEYS+['main_pipeline_id','campaign_id','baseline_missing','baseline_source','h0_mm','min_residual_thickness_mm']].copy()
        raw['target_case']=case; raw['production_cutoff']=cutoff
        raw['exclusion_reason']=np.select([
            ~raw['case'].eq(case),~raw.prediction_date.lt(cutoff),
            raw.main_pipeline_id.eq(parents.loc[request['id']]) & (case=='cold_start'),
            raw.baseline_missing & (case=='between_vtd')],
            ['different_case','label_at_or_after_cutoff','cold_target_parent','missing_prior_baseline'],default='included')
        cohort_audit.append(raw)
        audit,folds=independent_folds(train)
        fold_audit.extend(dict(**a,production_cutoff=cutoff) for a in audit)
        m0=fit_constant_rate(train,'W1')
        b0=m0.predict(target)
        comparison=target[['id','segment_id']].assign(recomputed=b0).merge(baseline,on=['id','segment_id'],validate='one_to_one')
        maxdelta=float(np.max(np.abs(comparison.recomputed-comparison.thickness_pred)))
        # Existing exporter serializes result.csv with float_format='%.10f'.
        # Verify that exact serialization, not an invented sub-rounding tolerance.
        exported=np.array([float(f'{value:.10f}') for value in comparison.recomputed])
        np.testing.assert_array_equal(exported,comparison.thickness_pred.to_numpy())
        if abs(m0.rate_mm_year-baseline_models[case]['training']['rate_mm_year'])>1e-14:
            raise ValueError('M0 fitted rate differs from saved comparator')
        b0_predictions.append(target[KEYS+[x for x in PROVENANCE if x in target]].assign(
            model_id='B0_M0_reconstructed',thickness_pred=b0,r_ref_mm_year=m0.rate_mm_year,
            forecast_status='baseline_comparator_not_independently_validated'))
        baseline_recon.append(dict(case=case,r_ref_mm_year=m0.rate_mm_year,training_rows=len(train),
            saved_rate_mm_year=baseline_models[case]['training']['rate_mm_year'],max_abs_prediction_difference_mm=maxdelta,
            saved_export_decimal_places=10,exact_saved_precision_identity=True,
            evidence_status='diagnostic_only',**thickness_metrics(train,m0.predict(train))))
        print(f'{case}: {len(train)} eligible training rows, {len(target)} target segments, {len(folds)} independent folds',flush=True)
        for hold in protocol['holds_days']:
            for policy in protocol['gap_policies']:
                for candidate,ea in EA_CANDIDATES.items():
                    mid=model_id(candidate,policy,hold)
                    model,features,fit=fit_temperature_rate(train,cache,ea_kj_mol=ea,gap_policy=policy,hold_days=hold)
                    target_features=cache.attach(target,ea_kj_mol=ea,gap_policy=policy,hold_days=hold)
                    metadata=dict(model_id=mid,case=case,target_id=request['id'],production_cutoff=cutoff,
                        candidate_id=candidate,ea_kJ_mol=ea,theta_K=1000*ea/protocol['R_J_mol_K'],
                        reference_temperature_K=protocol['reference_temperature_K'],gap_policy=policy,hold_days=hold,
                        parameter_provenance='user_fixed_development_choice' if ea==40 else 'fixed_grid_comparator',
                        training_rows=len(train),training_parent_count=train.main_pipeline_id.nunique(),
                        training_campaign_count=train.campaign_id.nunique(),training_section_count=train.id.nunique(),
                        training_parents=';'.join(sorted(train.main_pipeline_id.unique())),
                        training_campaigns=';'.join(sorted(train.campaign_id.unique())),
                        training_label_date_max=train.prediction_date.max(),
                        negative_observed_loss_rows=int(train.h0_mm.lt(train.min_residual_thickness_mm).sum()),
                        training_zero_exposure_rows=int(features.exposure_years.eq(0).sum()),
                        training_coverage_fraction=float(features.thermal_coverage_fraction.mean()),
                        independent_validation_fold_count=len(folds),evidence_status='diagnostic_only')
                    fits.append(dict(**metadata,**fit,**(thickness_metrics(features,model.predict(features)) if model.rate_mm_year is not None else {})))
                    if model.rate_mm_year is None:
                        evaluation.append(dict(**metadata,split='target_prediction',status='unavailable_zero_exposure_denominator'))
                        continue
                    hp=prediction_rows(features,model,fit,mid,candidate,'historical_fit')
                    pp=prediction_rows(target_features,model,fit,mid,candidate,'production_candidate')
                    for part in (hp,pp):
                        part['production_cutoff']=cutoff; part['training_label_date_max']=train.prediction_date.max()
                        part['training_rows']=len(train); part['training_parent_count']=train.main_pipeline_id.nunique()
                        part['training_campaign_count']=train.campaign_id.nunique()
                    history_predictions.append(hp); predictions.append(pp)
                    for label,frame in [('training',features),('production',target_features)]:
                        unique=frame.drop_duplicates(['id','start_date','prediction_date'])
                        coverage.append(dict(**metadata,dataset=label,rows=len(frame),
                            distinct_section_windows=len(unique),
                            coverage_fraction_row_mean=float(frame.thermal_coverage_fraction.mean()),
                            coverage_fraction_window_pooled=float(unique.thermal_observed_years.sum()/unique.horizon_years.sum()),
                            unknown_share_row_mean=float((frame.thermal_unobserved_years/frame.horizon_years).mean()),
                            reference_gap_exposure_share=float(frame.thermal_unobserved_years.sum()/frame.exposure_years.sum()) if policy=='reference_gap' else None,
                            zero_exposure_rows=int(frame.exposure_years.eq(0).sum()),
                            history_missing_rows=int(frame.thermal_history_missing.sum()),
                            baseline_missing_rows=int(frame.baseline_missing.sum()),
                            max_gap_days=float(frame.thermal_longest_gap_days.max()),
                            observed_source_date_max=frame.thermal_source_date_max.max()))
                    if candidate=='EA00_null' and policy=='reference_gap':
                        np.testing.assert_array_equal(features.exposure_years,train.horizon_years)
                        np.testing.assert_array_equal(target_features.exposure_years,target.horizon_years)
                        if model.rate_mm_year!=m0.rate_mm_year: raise ValueError('EA00/reference rate not M0')
                        np.testing.assert_array_equal(pp.thickness_pred,b0)
                    joined=pp.merge(baseline,on=['id','segment_id'],suffixes=('','_B0'),validate='one_to_one')
                    delta=joined.thickness_pred-joined.thickness_pred_B0
                    baseline_deltas.append(dict(**metadata,target_rows=len(pp),
                        mean_prediction_mm=float(pp.thickness_pred.mean()),min_prediction_mm=float(pp.thickness_pred.min()),
                        max_prediction_mm=float(pp.thickness_pred.max()),unique_prediction_count=pp.thickness_pred.nunique(),
                        mean_delta_vs_B0_mm=float(delta.mean()),mean_abs_delta_vs_B0_mm=float(delta.abs().mean()),
                        max_abs_delta_vs_B0_mm=float(delta.abs().max()),
                        comparison_status='prediction_difference_not_accuracy_metric'))
                    evaluation.append(dict(**metadata,split='historical_fit',status='in_sample_diagnostic',
                                           **thickness_metrics(features,model.predict(features))))
                    if not folds:
                        evaluation.append(dict(**metadata,split='independent_validation',status='unavailable',
                            reason='no_earlier_independent_training_group',mae_mm=None,rmse_mm=None,bias_mm=None,pearson_squared=None))
                    for fold_id,tr,te in folds:
                        fm,_,ff=fit_temperature_rate(tr,cache,ea_kj_mol=ea,gap_policy=policy,hold_days=hold)
                        tf=cache.attach(te,ea_kj_mol=ea,gap_policy=policy,hold_days=hold)
                        if fm.rate_mm_year is None:
                            evaluation.append(dict(**metadata,split='independent_validation',fold_id=fold_id,status='unavailable_zero_exposure_denominator'))
                        else:
                            evaluation.append(dict(**metadata,split='independent_validation',fold_id=fold_id,
                                status='held_parent_forward_diagnostic',fold_training_rows=len(tr),
                                fold_training_date_max=tr.prediction_date.max(),fold_rate_mm_year=fm.rate_mm_year,
                                **thickness_metrics(tf,fm.predict(tf))))
                            ip=prediction_rows(tf,fm,ff,mid,candidate,'independent_held_out')
                            ip['fold_id']=fold_id; independent_predictions.append(ip)
    all_predictions=pd.concat(predictions,ignore_index=True)
    save('prediction_diagnostics.csv',all_predictions)
    save('historical_fit_predictions.csv',pd.concat(history_predictions,ignore_index=True))
    save('training_windows.csv',pd.concat(cohorts_out,ignore_index=True))
    save('cohort_audit.csv',pd.concat(cohort_audit,ignore_index=True))
    fit_table=save('candidate_fits.csv',fits)
    save('fold_audit.csv',fold_audit); save('evaluation.csv',evaluation)
    save('independent_predictions.csv',pd.concat(independent_predictions,ignore_index=True) if independent_predictions else pd.DataFrame(columns=KEYS+['model_id','thickness_pred','fold_id']))
    summary=save('candidate_prediction_summary.csv',baseline_deltas); save('coverage_summary.csv',coverage)
    save('B0_reconstructed_diagnostics.csv',pd.concat(b0_predictions,ignore_index=True))
    write_json('baseline_reconstruction.json',baseline_recon)
    contracts=[]
    for mid,part in all_predictions.groupby('model_id'):
        result=part[['id','segment_id','thickness_pred']].sort_values(['id','segment_id']).reset_index(drop=True)
        check=validate_submission(result,grid,bounds)
        counts=result.groupby('id').size().to_dict()
        if counts!={1250009583:1475,1250022871:525}: raise ValueError('Unexpected scenario row counts')
        save(f'candidates/{mid}.csv',result)
        contracts.append(dict(model_id=mid,**check,between_vtd_rows=525,cold_start_rows=1475,
                              candidate_path=f'candidates/{mid}.csv',evidence_status='diagnostic_only'))
        if mid==model_id('EA40_mid','reference_gap',1.): save('result.csv',result)
        if mid==model_id('EA40_mid','observed_only',1.): save('result_ea40_observed_only_descriptor.csv',result)
    save('export_contracts.csv',contracts)
    interval_rows,interval_tables=[],[]
    for key,ints in cache.intervals.items():
        pid,start,end,hold=key
        interval_rows.append(dict(id=pid,start_date=start,prediction_date=end,hold_days=hold,**cache.summaries[key]))
        interval_tables.append(ints.assign(start_date=start,prediction_date=end,hold_days=hold))
    save('interval_audit.csv',interval_rows); save('intervals.csv',pd.concat(interval_tables,ignore_index=True))
    feature_examples=all_predictions[(all_predictions.candidate_id=='EA40_mid') & all_predictions.hold_days.eq(1) & all_predictions.gap_policy.eq('reference_gap')].groupby('case',group_keys=False).head(2)
    save('feature_examples.csv',feature_examples)
    robustness=[]
    for (case,policy),part in all_predictions[all_predictions.candidate_id.eq('EA40_mid')].groupby(['case','gap_policy']):
        primary=part[part.hold_days.eq(1)][['id','segment_id','thickness_pred']]
        for hold,g in part.groupby('hold_days'):
            merged=g.merge(primary,on=['id','segment_id'],suffixes=('','_primary'),validate='one_to_one')
            delta=merged.thickness_pred-merged.thickness_pred_primary
            robustness.append(dict(case=case,gap_policy=policy,hold_days=hold,
                r_ref_mm_year=float(g.r_ref_mm_year.iloc[0]),mean_prediction_mm=float(g.thickness_pred.mean()),
                mean_delta_vs_primary_hold_mm=float(delta.mean()),max_abs_delta_vs_primary_hold_mm=float(delta.abs().max()),
                interpretation='Assumption sensitivity, not confidence bounds'))
    save('ea40_robustness.csv',robustness)
    # Cross-check independently executed Stage 03.1 rates on identical cohorts.
    previous=pd.read_csv(ROOT/'experiments/temperature_v02_03_1/candidate_fits.csv',float_precision='round_trip')
    matching=fit_table.copy(); matching['gap_policy']=matching.gap_policy.replace({'reference_gap':'reference'})
    matched=matching.merge(previous,on=['case','candidate_id','gap_policy','hold_days'],suffixes=('','_stage03_1'),validate='one_to_one')
    if len(matched)!=60: raise ValueError('Stage 03.1 comparison incomplete')
    np.testing.assert_allclose(matched.r_ref_mm_year,matched.r_ref_mm_year_stage03_1,rtol=1e-13,atol=1e-14)
    write_json('stage03_1_consistency.json',dict(comparisons=60,all_rates_match=True,
        max_abs_rate_difference_mm_year=float(np.max(np.abs(matched.r_ref_mm_year-matched.r_ref_mm_year_stage03_1))),
        prior_decision='candidate_set_retained; no scientific Ea selection',user_development_ea_kJ_mol=40.))
    write_report(fit_table,summary,pd.DataFrame(coverage),pd.DataFrame(fold_audit))
    write_json('run_manifest.json',dict(branch=before['branch'],head=before['head'],initial_status_short=before['initial_status_short'],
        command='.venv/Scripts/python.exe -X utf8 -B -m experiments.temperature_v02_04.run',
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        protocol_sha256=digest(OUT/'protocol.json'),parameter_provenance=protocol['parameter_provenance'],
        development_ea_kJ_mol=40.,production_promoted=False,input_sha256=data.hashes,
        input_hash_contract='src.storage.sha256 text newline normalization',
        input_raw_bytes_sha256={n:digest(ROOT/n) for n in data.hashes},
        immutable_baseline_raw_sha256=digest(baseline_path),document_sha256=before['document_sha256'],
        source_sha256={n:digest(ROOT/n) for n in before['sha256'] if n.startswith('src/') or n=='experiments/temperature_v02_03_1/run.py'},
        spec_substitution=before['spec_substitution']))
    changed=[n for n,h in before['sha256'].items() if not (ROOT/n).is_file() or digest(ROOT/n)!=h]
    write_json('preservation.json',dict(files_checked=len(before['sha256']),changed=changed))
    if changed: raise ValueError(f'Protected files changed: {changed}')
    if any(repository_sha256(ROOT/n)!=h for n,h in data.hashes.items()): raise ValueError('Input mutated')
    print('Completed 60 rate fits, 30 candidate exports, EA40 primary export; all protected files unchanged.',flush=True)


def write_report(fits,summary,coverage,folds):
    lines=['# Stage 04: fixed Ea40 development model','',
        'Ea = **40 kJ/mol** is fixed by the user for development. Stage 03.1 did not scientifically select it. All other grid values are comparators; no new Ea selection occurs. No production promotion occurred.','',
        '`h_pred = clip(h0_mm - r_ref_mm_year * E_T, 0, h0_mm)`; `r_ref = max(0, sum(W1*E_T*(h0-h_obs))/sum(W1*E_T^2))`. The objective uses unclipped measured losses, including negative thinning. Rates are fit separately for each case and assumption. E_T uses only cutoff-specific t_mix_usable in Kelvin, R=8.314462618, Tref=293.15; no legacy Arrhenius/spatial term is multiplied in.','',
        '`result.csv` is EA40 with a one-day hold and F=1 assumed in unsupported history. It is an experimental candidate, not an official result. The observed-only export is an incomplete-history descriptor and must not be treated as a full-history physical forecast.','',
        '## Training rates (primary one-day hold)','',
        '| Case | Policy | Ea kJ/mol | Rate mm/year | Rows | Parents | Campaigns |','|---|---|---:|---:|---:|---:|---:|']
    for r in fits[fits.hold_days.eq(1)].itertuples():
        lines.append(f'| {r.case} | {r.gap_policy} | {r.ea_kJ_mol:g} | {r.r_ref_mm_year:.9g} | {r.training_rows} | {r.training_parent_count} | {r.training_campaign_count} |')
    lines.extend(['','## Ea40 target candidate summaries','',
        'Prediction differences are comparisons with B0, not errors against target labels.','',
        '| Case | Policy | Mean prediction mm | Mean delta vs B0 mm | Maximum absolute delta mm | Coverage | Unknown share |','|---|---|---:|---:|---:|---:|---:|'])
    for r in summary[summary.candidate_id.eq('EA40_mid')&summary.hold_days.eq(1)].itertuples():
        c=coverage[coverage['case'].eq(r.case)&coverage.candidate_id.eq('EA40_mid')&coverage.hold_days.eq(1)&coverage.gap_policy.eq(r.gap_policy)&coverage.dataset.eq('production')].iloc[0]
        lines.append(f'| {r.case} | {r.gap_policy} | {r.mean_prediction_mm:.6f} | {r.mean_delta_vs_B0_mm:.6f} | {r.max_abs_delta_vs_B0_mm:.6f} | {c.coverage_fraction_row_mean:.4%} | {c.unknown_share_row_mean:.4%} |')
    lines.extend(['','## Why E_a is presently unidentifiable','',
        f'The current audit found {int(folds.admissible.sum())} independent forward-time held-parent folds. Both eligible cohorts contain one parent and one campaign. Stage 03.1 showed practical ties and sensitivity to reference-gap filling. There is no independent accuracy claim, fitted material constant, or population confidence interval. Full local sensitivity rank does not establish practical identification.','',
        'Within-parent same-campaign section diagnostics remain documented in Stage 03.1; they are not relabelled as validation here. Temperature, time and operating histories are confounded, and chemistry is unmeasured. Independent parents/campaigns, denser temperature records, paired W/v and reliable repeated thickness measurements are needed.','',
        'Cold-start section-level exposure cannot create spatial risk rankings when h0 and the time window are shared. Constant predictions yield undefined Pearson squared, not zero. Production metrics are deliberately unavailable because production labels are not used. Four between-VTD target baselines use the existing nearest-prior assumption and remain flagged.','',
        '## Provenance and reproduction','',
        'The attached Stage 04 prompt defines the experiment scope; the user fixes Ea40. The English specification was absent; the supplied Russian v1.1 specification was read in full. Existing source wins over historical descriptions. The original M0 saved result is copied byte-for-byte into B0_original_M0.csv and checked against a reconstruction on the identical eligible rows. EA00/reference reproduces that reconstruction; EA00/observed-only need not.','',
        'Historical windows are built once, then each target snapshot is built once through make_snapshot. Existing RepositoryData verifies published input hashes. Intervals are frozen once per section/start/cutoff/hold and reused for all candidates. W/v missingness does not gate temperature; invalid next records end prior holds. Baseline provenance, interval hashes, support and all observed-loss quality flags are exported.','',
        'Run from repository root:','',
        '```powershell',
        '.venv/Scripts/python.exe -X utf8 -B -m experiments.temperature_v02_04.run',
        '.venv/Scripts/python.exe -X utf8 -B -m experiments.temperature_v02_04.verify',
        '```','',
        'Tests, exact output hashes and preservation checks are finalized by verify.py. Candidate CSVs have exactly id,segment_id,thickness_pred and 2000 keys (525 + 1475). prediction_diagnostics.csv retains the richer contract and evidence status. All outputs are isolated in this directory.','',
        '## Next-stage gate','',
        'GO for Stage 05 exploratory ablation/development with Ea40 fixed and limitations carried forward. NO-GO for claiming improved independent performance, a validated Ea, or production promotion. Known repository regression failures remain documented separately; this stage does not modify preprocessing or old tests to mask them.'])
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    generate()

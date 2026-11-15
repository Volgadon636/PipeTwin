"""Numerical counterexamples and persisted historical-evidence contracts."""
import json
import unittest

import numpy as np
import pandas as pd

from src.ml.models import fit_constant_rate, sample_weights
from src.ml.temperature import EA_CANDIDATES, temperature_features, temperature_intervals
from experiments.temperature_v02_03_1.run import (
    OUT, COMPARE, decision_for_case, exposure_and_derivative, fit_rate, fit_ties,
    interval_hash, mae_ties, make_folds, sensitivity, shape, target_cohort,
)


def history(temperatures, dates=None):
    return pd.DataFrame(dict(id=[1]*len(temperatures),
        date=pd.to_datetime(dates if dates is not None else pd.date_range('2020-01-01',periods=len(temperatures)),format='mixed'),
        t_mix_usable=temperatures))


def training():
    return pd.DataFrame(dict(id=[1,1,2,3],segment_id=[0,1,0,0],main_pipeline_id=['A']*4,
        campaign_id=['A:2020']*4,case=['cold_start']*4,
        start_date=pd.to_datetime(['2018-01-01']*4),
        prediction_date=pd.to_datetime(['2020-01-01']*4),baseline_missing=[False]*4,
        h0_mm=[10.]*4,horizon_years=[1.,2.,3.,4.],min_residual_thickness_mm=[9.8,9.6,9.4,9.2]))


class NumericalTests(unittest.TestCase):
    def test_analytic_derivative_against_central_difference_both_policies(self):
        ints,_=temperature_intervals(history([-10.,20.,55.]),'2020-01-01','2020-01-06')
        for ea in [25.,40.,55.,70.]:
            for policy in ['reference','observed_only']:
                analytic=exposure_and_derivative(ints,5/365.25,ea,policy)[1]
                plus=exposure_and_derivative(ints,5/365.25,ea+1e-3,policy)[0]
                minus=exposure_and_derivative(ints,5/365.25,ea-1e-3,policy)[0]
                self.assertAlmostEqual(analytic,(plus-minus)/.002,delta=abs(analytic)*1e-8)

    def test_fixed_intervals_match_existing_api_without_wv_gate(self):
        rows=history([40.,np.nan,0.,60.],['2019-12-31 18:00','2020-01-01 06:00','2020-01-02','2020-01-03'])
        for hold in [1.,.5,.25]:
            ints,_=temperature_intervals(rows,'2020-01-01','2020-01-04',hold)
            before=interval_hash(ints)
            for ea in EA_CANDIDATES.values():
                for policy in ['reference','observed_only']:
                    actual=exposure_and_derivative(ints,3/365.25,ea,policy)[0]
                    expected=temperature_features(rows,'2020-01-01','2020-01-04',ea_kj_mol=ea,gap_policy=policy,max_hold_days=hold)
                    key='thermal_exposure_reference_years' if policy=='reference' else 'thermal_exposure_observed_years'
                    self.assertAlmostEqual(actual,expected[key],places=15)
            self.assertEqual(before,interval_hash(ints))

    def test_constant_rate_recovery_and_negative_losses_remain(self):
        e=np.array([0.,1.,2.,4.]); w=np.array([1.,2.,1.,.5])
        fit=fit_rate(.23*e,e,w)
        self.assertAlmostEqual(fit['r_ref_mm_year'],.23)
        self.assertLess(fit['weighted_sse_mm2'],1e-28)
        fit=fit_rate([-1.,1.],[1.,1.],[1.,1.])
        self.assertEqual(fit['r_ref_mm_year'],0)
        self.assertEqual(fit['weighted_sse_mm2'],2)

    def test_zero_exposure_retains_residual_zero_denominator_unavailable(self):
        fit=fit_rate([3.,2.],[0.,1.],[2.,1.])
        self.assertEqual(fit['r_ref_mm_year'],2.)
        self.assertEqual(fit['weighted_sse_mm2'],18.)
        empty=fit_rate([3.,2.],[0.,0.],[2.,1.])
        self.assertIsNone(empty['r_ref_mm_year'])
        self.assertEqual(empty['zero_rate_sse_mm2'],22.)
        ints,_=temperature_intervals(history([]),'2020-01-01','2020-01-04')
        self.assertEqual(exposure_and_derivative(ints,3/365.25,70,'observed_only')[:2],(0.,0.))
        self.assertEqual(exposure_and_derivative(ints,3/365.25,70,'reference')[:2],(3/365.25,0.))
        self.assertEqual(sensitivity([0,1],[0,1],np.ones(2),0)['effective_normalized_rank'],0)

    def test_nonfinite_or_negative_exposure_rejected(self):
        for e in [[np.nan,1],[-1,1],[np.inf,1]]:
            with self.assertRaises(ValueError): fit_rate([1,1],e,[1,1])

    def test_finite_inputs_cannot_overflow_rate_fit_sums(self):
        for loss,exposure in [([1.,1.],[1e308,1e308]),([1e308,1e308],[1.,1.])]:
            with self.assertRaisesRegex(ValueError,'Nonfinite rate-fit sums'):
                fit_rate(loss,exposure,[1.,1.])

    def test_proportional_exposure_is_unidentifiable(self):
        e=np.array([1.,2.,4.,5.]); w=np.array([1.,2.,3.,1.]); y=np.array([.3,.4,.9,.8])
        original=fit_rate(y,e,w)
        for scale in [1.,2.,10.,100.]:
            f=fit_rate(y,scale*e,w)
            self.assertAlmostEqual(f['weighted_sse_mm2'],original['weighted_sse_mm2'])
            self.assertAlmostEqual(f['r_ref_mm_year']*scale,original['r_ref_mm_year'])
            self.assertLess(shape(scale*e,e,w)['proportionality_residual'],1e-14)
        self.assertEqual(sensitivity(e,.03*e,w,.2)['effective_normalized_rank'],1)

    def test_heterogeneous_synthetic_histories_recover_fixed_ea(self):
        vectors={ea:[] for ea in EA_CANDIDATES.values()}; derivatives=[]
        for temps in [[5.,10.,15.],[20.,25.,30.],[40.,50.,60.],[10.,60.,20.]]:
            ints,_=temperature_intervals(history(temps),'2020-01-01','2020-01-04')
            for ea in vectors:
                e,de,*_=exposure_and_derivative(ints,3/365.25,ea,'reference')
                vectors[ea].append(e)
                if ea==40: derivatives.append(de)
        loss=.4*np.array(vectors[40]); weights=np.ones(4)
        results={ea:fit_rate(loss,e,weights) for ea,e in vectors.items()}
        self.assertEqual(min(results,key=lambda ea:results[ea]['weighted_sse_mm2']),40.)
        self.assertAlmostEqual(results[40]['r_ref_mm_year'],.4)
        self.assertGreater(shape(vectors[40],vectors[0],weights)['proportionality_residual'],.1)
        self.assertEqual(sensitivity(vectors[40],derivatives,weights,.4)['effective_normalized_rank'],2)

    def test_ea00_reference_exact_m0_identity(self):
        frame=training(); weights=sample_weights(frame,'W1'); loss=frame.h0_mm-frame.min_residual_thickness_mm
        model=fit_constant_rate(frame,'W1')
        fit=fit_rate(loss,frame.horizon_years,weights)
        self.assertEqual(model.rate_mm_year,fit['r_ref_mm_year'])
        np.testing.assert_array_equal(model.predict(frame),np.clip(frame.h0_mm-fit['r_ref_mm_year']*frame.horizon_years,0,frame.h0_mm))

    def test_section_splits_isolate_all_segments_and_reject_identical_trajectories(self):
        frame=training(); request=dict(case='cold_start')
        fingerprints=pd.Series(['x','x','y','z'])
        audits,folds=make_folds(frame,request,fingerprints)
        self.assertEqual(len(folds),3)
        for _,status,tr,te in folds:
            self.assertFalse(set(frame.loc[tr].id)&set(frame.loc[te].id))
            self.assertEqual(status,'within_parent_same_campaign_diagnostic')
        audits,folds=make_folds(frame,request,pd.Series(['x']*4))
        self.assertEqual(len(folds),0)
        self.assertIn('identical_trajectories_cross_split',[a['reason'] for a in audits])

    def test_strict_cutoff_and_target_parent_exclusion(self):
        frame=training()
        frame.loc[0,'prediction_date']=pd.Timestamp('2021-01-01')
        frame.loc[1,'prediction_date']=pd.Timestamp('2022-01-01')
        frame.loc[2,'main_pipeline_id']='TARGET'
        actual=target_cohort(frame,dict(id=9,case='cold_start',prediction_date='2021-01-01'),pd.Series({9:'TARGET'}))
        self.assertEqual(actual.id.tolist(),[3])

    def test_independent_fold_requires_earlier_date_and_different_parent(self):
        frame=training()
        frame['main_pipeline_id']=['A','B','C','TARGET']
        frame['prediction_date']=pd.to_datetime(['2018-01-01','2019-01-01','2019-01-01','2017-01-01'])
        cohort=target_cohort(frame,dict(id=9,case='cold_start',prediction_date='2021-01-01'),pd.Series({9:'TARGET'}))
        _,folds=make_folds(cohort,dict(case='cold_start'),pd.Series(['a','b','c']))
        self.assertEqual(len(folds),2)
        for _,_,tr,te in folds:
            self.assertLess(cohort.loc[tr].prediction_date.max(),cohort.loc[te].prediction_date.min())
            self.assertFalse(set(cohort.loc[tr].main_pipeline_id)&set(cohort.loc[te].main_pipeline_id))

    def test_practical_ties_and_zero_objective(self):
        self.assertEqual(fit_ties({'a':100.,'b':100.5,'c':103.}),['a','b'])
        self.assertEqual(fit_ties({'a':0.,'b':0.,'c':.01}),['a','b'])
        self.assertEqual(mae_ties({'a':.2,'b':.205,'c':.3}),['a','b'])

    def test_decision_refuses_training_winner_without_transfer_or_when_tied(self):
        for sections in [0,3]:
            fits=[]; stability=[]; shapes=[]
            for hold in [1.,.5,.25]:
                for policy in ['observed_only','reference']:
                    for candidate in EA_CANDIDATES:
                        base=dict(case='cold_start',hold_days=hold,gap_policy=policy,candidate_id=candidate)
                        fits.append(dict(**base,weighted_sse_mm2=1. if candidate=='EA40_mid' else 100.))
                        stability.append(dict(**base,equal_section_mae_mm=.2 if sections else np.nan,usable_section_count=sections))
                        shapes.append(dict(**base,reference_candidate='EA00_null',proportionality_residual=.1))
            result=decision_for_case('cold_start',pd.DataFrame(fits),pd.DataFrame(stability),pd.DataFrame(),pd.DataFrame(shapes),pd.DataFrame())
            self.assertIsNone(result['selected_candidate'])
            self.assertIn('EA00_null',result['retained_candidates'])
            self.assertFalse(result['rule_results']['unique_primary_best_both_policies'])


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fits=pd.read_csv(OUT/'candidate_fits.csv')
        cls.pred=pd.read_csv(OUT/'diagnostic_predictions.csv')

    def test_all_sixty_comparisons_and_interval_identity(self):
        self.assertEqual(len(self.fits),60)
        self.assertTrue(self.fits.groupby(COMPARE).candidate_id.nunique().eq(5).all())
        e=pd.read_csv(OUT/'candidate_exposures.csv')
        g=e.groupby(['case','history_id','hold_days'])
        self.assertTrue(g.interval_sha256.nunique().eq(1).all())
        self.assertTrue(g.size().eq(10).all())
        self.assertTrue(pd.read_csv(OUT/'m0_identity.csv')[['rate_equal','exposure_equal','predictions_equal']].all().all())

    def test_predictions_only_eligible_historical_rows_and_split_provenance(self):
        p=self.pred
        self.assertTrue((pd.to_datetime(p.prediction_date)<pd.to_datetime(p.production_cutoff)).all())
        self.assertFalse(p.id.eq(1250009583).any())
        self.assertFalse((p['case'].eq('cold_start') & p.main_pipeline_id.eq('MAIN-4')).any())
        self.assertTrue(p.h_pred_mm.between(0,p.h0_mm).all())
        self.assertTrue(np.allclose(p.observed_loss_mm,p.h0_mm-p.min_residual_thickness_mm))
        self.assertTrue(p.negative_observed_loss.eq(p.observed_loss_mm<0).all())
        for row in p[p.split.eq('held_out')].drop_duplicates(['case','fold_id']).itertuples():
            self.assertNotIn(str(row.id),str(row.calibration_section_ids).split(';'))
        for _,g in p[p.split.eq('historical_fit')].groupby(COMPARE):
            signatures=[set(zip(h.id,h.segment_id,h.prediction_date)) for _,h in g.groupby('candidate_id')]
            self.assertTrue(all(s==signatures[0] for s in signatures))

    def test_saved_fit_losses_and_grouped_calibration_recompute(self):
        p=self.pred
        for row in self.fits.itertuples():
            g=p[p['case'].eq(row.case)&p.hold_days.eq(row.hold_days)&p.gap_policy.eq(row.gap_policy)&p.candidate_id.eq(row.candidate_id)&p.split.eq('historical_fit')]
            f=fit_rate(g.observed_loss_mm,g.exposure_years,sample_weights(g,'W1'))
            self.assertAlmostEqual(f['r_ref_mm_year'],row.r_ref_mm_year,places=10)
            self.assertAlmostEqual(f['weighted_sse_mm2'],row.weighted_sse_mm2,places=9)
        for key,g in p[p.split.eq('held_out')].groupby(COMPARE+['candidate_id','fold_id']):
            case,hold,policy,candidate,fold=key
            full=p[p['case'].eq(case)&p.hold_days.eq(hold)&p.gap_policy.eq(policy)&p.candidate_id.eq(candidate)&p.split.eq('historical_fit')]
            allowed={int(x) for x in str(g.calibration_section_ids.iloc[0]).split(';')}
            tr=full[full.id.isin(allowed)]
            f=fit_rate(tr.observed_loss_mm,tr.exposure_years,sample_weights(tr,'W1'))
            self.assertAlmostEqual(f['r_ref_mm_year'],g.r_ref_mm_year.iloc[0],places=10)
            np.testing.assert_allclose(g.h_pred_mm,np.clip(g.h0_mm-f['r_ref_mm_year']*g.exposure_years,0,g.h0_mm),atol=1e-12)

    def test_null_control_retained_unless_all_selection_gates_pass(self):
        decision=json.loads((OUT/'decision.json').read_text())
        for result in decision['cases'].values():
            if result['selected_candidate'] is not None:
                self.assertTrue(all(result['rule_results'].values()))
            else:
                self.assertEqual(set(result['retained_candidates']),set(EA_CANDIDATES))


if __name__=='__main__':
    unittest.main()

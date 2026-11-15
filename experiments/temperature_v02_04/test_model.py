"""Opt-in model contracts, leakage counterexamples and actual export checks."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ml.models import fit_constant_rate, sample_weights
from src.ml.temperature import attach_temperature_features
from src.ml.windows import YEAR_SECONDS
from .model import ExposureCache, TemperatureRateModel, eligible_cohort, fit_temperature_rate
from .run import OUT, ROOT, digest, independent_folds


def fixture():
    start=pd.Timestamp('2020-01-01'); end=pd.Timestamp('2020-01-04')
    frame=pd.DataFrame(dict(id=[1,1,2],segment_id=[0,1,0],case=['cold_start']*3,
        main_pipeline_id=['A','A','B'],campaign_id=['A:1','A:1','B:1'],
        start_date=[start]*3,prediction_date=[end]*3,horizon_years=[3/365.25]*3,
        h0_mm=[8.]*3,min_residual_thickness_mm=[8.-.2*3/365.25]*3,baseline_missing=[False]*3))
    histories={(pid,end):pd.DataFrame(dict(id=[pid]*3,date=pd.date_range(start,periods=3),
        prediction_date=[end]*3,t_mix_usable=[20.]*3,
        watercut_pct_usable=[np.nan]*3,v_liquid_true_usable=[np.nan]*3)) for pid in [1,2]}
    return frame,histories


class ModelTests(unittest.TestCase):
    def test_ea40_defaults_and_synthetic_recovery(self):
        frame,histories=fixture()
        model,features,fit=fit_temperature_rate(frame,ExposureCache(histories))
        self.assertEqual(model.ea_kj_mol,40.)
        self.assertEqual(model.gap_policy,'reference_gap')
        self.assertAlmostEqual(model.rate_mm_year,.2,places=10)
        np.testing.assert_allclose(model.predict(features),frame.min_residual_thickness_mm,atol=1e-14)

    def test_w1_parent_weighting_is_used(self):
        frame,histories=fixture()
        frame['min_residual_thickness_mm']=[7.,7.,6.]
        model,features,_=fit_temperature_rate(frame,ExposureCache(histories))
        e=features.exposure_years.to_numpy(); loss=(frame.h0_mm-frame.min_residual_thickness_mm).to_numpy()
        w=sample_weights(frame,'W1')
        self.assertAlmostEqual(model.rate_mm_year,np.sum(w*e*loss)/np.sum(w*e*e))
        self.assertNotAlmostEqual(model.rate_mm_year,np.sum(e*loss)/np.sum(e*e))

    def test_negative_measured_loss_is_retained_and_rate_nonnegative(self):
        frame,histories=fixture(); frame['min_residual_thickness_mm']=[9.,8.,8.]
        model,features,fit=fit_temperature_rate(frame,ExposureCache(histories))
        self.assertEqual(len(features),3)
        self.assertEqual(model.rate_mm_year,0.)
        self.assertGreater(fit['weighted_sse_mm2'],0)
        np.testing.assert_array_equal(model.predict(features),frame.h0_mm)

    def test_zero_exposure_is_unavailable_and_not_silently_filled(self):
        frame,histories=fixture()
        empty={key:rows.iloc[:0] for key,rows in histories.items()}
        cache=ExposureCache(empty)
        model,features,fit=fit_temperature_rate(frame,cache,gap_policy='observed_only')
        self.assertIsNone(model.rate_mm_year)
        self.assertTrue(features.thermal_history_missing.all())
        self.assertTrue(features.thermal_factor_observed_mean.isna().all())
        self.assertTrue(features.exposure_years.eq(0).all())
        with self.assertRaisesRegex(ValueError,'Unavailable fit'): model.predict(features)
        reference,features,_=fit_temperature_rate(frame,cache)
        self.assertAlmostEqual(reference.rate_mm_year,.2,places=10)
        self.assertTrue(features.thermal_coverage_fraction.eq(0).all())

    def test_zero_exposure_row_keeps_its_residual(self):
        frame,histories=fixture(); key=next(k for k in histories if k[0]==2)
        histories[key]=histories[key].iloc[:0]
        frame.loc[2,'min_residual_thickness_mm']=7.
        model,features,fit=fit_temperature_rate(frame,ExposureCache(histories),gap_policy='observed_only')
        self.assertEqual(len(features),3)
        self.assertGreater(fit['weighted_sse_mm2'],1.)
        self.assertEqual(model.predict(features)[2],8.)
        self.assertTrue(features.thermal_history_missing.iloc[2])

    def test_parameters_and_prediction_metadata_fail_closed(self):
        frame,histories=fixture()
        for kwargs in [dict(ea_kj_mol=20.79),dict(hold_days=2.),dict(hold_days=0.),dict(gap_policy='implicit')]:
            with self.assertRaises(ValueError): fit_temperature_rate(frame,ExposureCache(histories),**kwargs)
        model,features,_=fit_temperature_rate(frame,ExposureCache(histories))
        for column,value in [('case','between_vtd'),('thermal_ea_kJ_mol',70.),('thermal_gap_policy','observed_only'),('thermal_exposure_policy','unbounded')]:
            with self.assertRaisesRegex(ValueError,'do not match'): model.predict(features.assign(**{column:value}))
        with self.assertRaises(ValueError): TemperatureRateModel('cold_start',rate_mm_year=-1.)
        with self.assertRaises(ValueError): TemperatureRateModel('cold_start',rate_mm_year=np.inf)

    def test_interval_cache_shared_by_all_candidates_and_policies(self):
        frame,histories=fixture(); cache=ExposureCache(histories)
        from . import model as module
        with patch.object(module,'temperature_intervals',wraps=module.temperature_intervals) as calls:
            for ea in [0.,25.,40.,55.,70.]:
                for policy in ['observed_only','reference_gap']:
                    enriched=cache.attach(frame,ea_kj_mol=ea,gap_policy=policy)
                    self.assertTrue(enriched.thermal_coverage_fraction.eq(1).all())
            self.assertEqual(calls.call_count,2)

    def test_stage02_feature_contract_and_no_wv_requirement(self):
        frame,histories=fixture()
        histories[(1,pd.Timestamp('2020-01-04'))]['t_mix_usable']=[0.,np.nan,55.]
        cache=ExposureCache(histories)
        for hold in [1.,.5,.25]:
            for policy,alias in [('observed_only','observed_only'),('reference_gap','reference')]:
                for ea in [0.,40.,70.]:
                    actual=cache.attach(frame,ea_kj_mol=ea,gap_policy=policy,hold_days=hold)
                    expected,_=attach_temperature_features(frame,histories,ea_kj_mol=ea,gap_policy=alias,max_hold_days=hold)
                    for column in ['thermal_exposure_observed_years','thermal_exposure_reference_years','thermal_observed_years','thermal_unobserved_years','thermal_coverage_fraction','thermal_factor_observed_mean']:
                        np.testing.assert_allclose(actual[column],expected[column],rtol=1e-14,atol=1e-14,equal_nan=True)

    def test_history_cutoff_identity_and_window_horizon(self):
        frame,histories=fixture(); key=next(iter(histories))
        for column,value in [('id',99),('prediction_date',pd.Timestamp('2021-01-01')),('date',pd.Timestamp('2020-01-04'))]:
            poisoned=dict(histories); poisoned[key]=histories[key].assign(**{column:value})
            with self.assertRaisesRegex(ValueError,'cutoff mismatch'):
                ExposureCache(poisoned).attach(frame)
        with self.assertRaisesRegex(ValueError,'Horizon'): ExposureCache(histories).attach(frame.assign(horizon_years=999.))

    def test_future_and_cold_target_parent_labels_cannot_change_fit(self):
        frame,histories=fixture(); request=dict(id=99,prediction_date='2020-02-01',case='cold_start')
        forbidden=frame.iloc[[0]].assign(id=99,main_pipeline_id='TARGET',campaign_id='TARGET:1')
        future=frame.iloc[[0]].assign(prediction_date=pd.Timestamp('2020-02-01'))
        pool=pd.concat([frame,forbidden,future],ignore_index=True)
        parents=pd.Series({99:'TARGET'})
        clean=eligible_cohort(pool,request,parents)
        pool.loc[pool.id.eq(99)|pool.prediction_date.ge(pd.Timestamp('2020-02-01')),'min_residual_thickness_mm']=-1e20
        poisoned=eligible_cohort(pool,request,parents)
        pd.testing.assert_frame_equal(clean,poisoned)
        a,_,_=fit_temperature_rate(clean,ExposureCache(histories))
        b,_,_=fit_temperature_rate(poisoned,ExposureCache(histories))
        self.assertEqual(a,b)

    def test_ea00_reference_identity_and_physical_clipping(self):
        frame,histories=fixture(); cache=ExposureCache(histories)
        model,features,_=fit_temperature_rate(frame,cache,ea_kj_mol=0.)
        original=fit_constant_rate(frame,'W1')
        self.assertEqual(model.rate_mm_year,original.rate_mm_year)
        np.testing.assert_array_equal(model.predict(features),original.predict(frame))
        fast=TemperatureRateModel('cold_start',0.,'reference_gap',1.,10000.)
        self.assertTrue((fast.predict(features)==0.).all())

    def test_independent_folds_exclude_same_parent_and_same_day(self):
        frame,_=fixture()
        frame['prediction_date']=pd.to_datetime(['2018-01-01','2019-01-01','2019-01-01'])
        audits,folds=independent_folds(frame)
        self.assertEqual(len(folds),1)
        for _,tr,te in folds:
            self.assertLess(tr.prediction_date.max(),te.prediction_date.min())
            self.assertFalse(set(tr.main_pipeline_id)&set(te.main_pipeline_id))


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred=pd.read_csv(OUT/'prediction_diagnostics.csv',float_precision='round_trip')
        cls.fits=pd.read_csv(OUT/'candidate_fits.csv',float_precision='round_trip')

    def test_all_exports_have_exact_contract_and_ea40_primary(self):
        contracts=pd.read_csv(OUT/'export_contracts.csv')
        self.assertEqual(len(contracts),30)
        primary=pd.read_csv(OUT/'result.csv',float_precision='round_trip')
        for record in contracts.itertuples():
            data=pd.read_csv(OUT/record.candidate_path,float_precision='round_trip')
            self.assertEqual(list(data),['id','segment_id','thickness_pred'])
            self.assertEqual(data.groupby('id').size().to_dict(),{1250009583:1475,1250022871:525})
            self.assertFalse(data.duplicated(['id','segment_id']).any())
            self.assertTrue(np.isfinite(data.thickness_pred).all())
        expected=self.pred[self.pred.candidate_id.eq('EA40_mid')&self.pred.gap_policy.eq('reference_gap')&self.pred.hold_days.eq(1)][['id','segment_id','thickness_pred']].sort_values(['id','segment_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(primary,expected)

    def test_all_rates_match_previous_experiment_and_negative_losses_kept(self):
        self.assertEqual(len(self.fits),60)
        check=json.loads((OUT/'stage03_1_consistency.json').read_text())
        self.assertTrue(check['all_rates_match'])
        training=pd.read_csv(OUT/'training_windows.csv')
        self.assertEqual(training.groupby('case').size().to_dict(),{'between_vtd':357,'cold_start':372})
        self.assertEqual(int(training.negative_observed_loss_quality_flag.sum()),177)
        self.assertTrue((pd.to_datetime(training.prediction_date)<pd.to_datetime(training.production_cutoff)).all())
        self.assertFalse((training['case'].eq('cold_start')&training.main_pipeline_id.eq('MAIN-4')).any())

    def test_m0_comparator_preserved_and_null_identity_for_targets(self):
        self.assertEqual(digest(OUT/'B0_original_M0.csv'),digest(ROOT/'experiments/m0_baseline_reference/result.csv'))
        baseline=pd.read_csv(OUT/'B0_original_M0.csv',float_precision='round_trip')
        for _,group in self.pred[self.pred.candidate_id.eq('EA00_null')&self.pred.gap_policy.eq('reference_gap')].groupby('hold_days'):
            merged=group.merge(baseline,on=['id','segment_id'],suffixes=('','_B0'),validate='one_to_one')
            serialized=np.array([float(f'{value:.10f}') for value in merged.thickness_pred])
            np.testing.assert_array_equal(serialized,merged.thickness_pred_B0.to_numpy())

    def test_forecasts_have_bounds_honest_status_and_no_target_labels(self):
        p=self.pred
        self.assertTrue(p.thickness_pred.between(0,p.h0_mm).all())
        self.assertTrue(p.predicted_loss_unclipped_mm.ge(0).all())
        self.assertNotIn('min_residual_thickness_mm',p)
        self.assertTrue(p.evidence_status.eq('diagnostic_only').all())
        self.assertTrue(p[p.gap_policy.eq('observed_only')].forecast_status.eq('incomplete_history_descriptor_not_full_forecast').all())
        self.assertTrue((pd.to_datetime(p.training_label_date_max)<pd.to_datetime(p.production_cutoff)).all())
        self.assertTrue((pd.to_datetime(p.thermal_source_date_max)<pd.to_datetime(p.prediction_date)).all())
        self.assertTrue(p.thermal_fallback_fraction.eq(0).all())
        flags=p[p.candidate_id.eq('EA40_mid')&p.gap_policy.eq('reference_gap')&p.hold_days.eq(1)]
        self.assertEqual(int(flags.baseline_missing.sum()),4)

    def test_same_support_across_candidates_and_independent_metrics_unavailable(self):
        group=self.pred.groupby(KEYS+['hold_days'])
        self.assertTrue(group.interval_sha256.nunique().eq(1).all())
        self.assertTrue(group.size().eq(10).all())
        audit=pd.read_csv(OUT/'fold_audit.csv')
        self.assertFalse(audit.admissible.any())
        evaluation=pd.read_csv(OUT/'evaluation.csv')
        independent=evaluation[evaluation.split.eq('independent_validation')]
        self.assertEqual(len(independent),60)
        self.assertTrue(independent[['mae_mm','rmse_mm','bias_mm','pearson_squared']].isna().all().all())


KEYS=['id','segment_id','prediction_date','case']

if __name__=='__main__':
    unittest.main()

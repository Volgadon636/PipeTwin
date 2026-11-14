"""Regression tests for actual failures and independent data invariants."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR, FILE_ALIASES, TECH_FILE
from src.format_detector import parse_numeric, robust_parse_dates, resolve_sources, build_column_mapping
from src.grid_builder import build_segments_grid, assign_segments
from src.segment_mapper import map_vtd_to_segments, map_profile_to_segments, build_features_asof
from src.temporal import persistent_calendar_days, hydraulic_asof
from src.validator import require
from src.data_prep import run_pipeline
from src.storage import publish_file

CONFIG = json.loads((BASE_DIR/'configs/preprocessing.json').read_text(encoding='utf-8'))


class UnitTests(unittest.TestCase):
    def test_windows_rename_denied_falls_back_to_verified_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/'new.csv', Path(tmp)/'old.csv'
            source.write_bytes(b'id,value\n1,5.53\n')
            dest.write_bytes(b'id,value\n1,7.975\n')
            with patch.object(Path, 'replace', side_effect=PermissionError('rename denied')):
                publish_file(source,dest)
            self.assertEqual(dest.read_bytes(),source.read_bytes())

    def test_mixed_numeric_separators(self):
        np.testing.assert_allclose(parse_numeric(pd.Series(['0.219','4251,17','-0.25','0','bad'])),
                                   [.219,4251.17,-.25,0,np.nan],equal_nan=True)

    def test_dates_parsed_individually(self):
        result = robust_parse_dates(pd.Series(['23.09.2018','2022-07-06','30.09.2021 0:00:00','bad']))
        self.assertEqual(result.iloc[0],pd.Timestamp('2018-09-23'))
        self.assertEqual(result.iloc[1],pd.Timestamp('2022-07-06'))
        self.assertEqual(result.iloc[2],pd.Timestamp('2021-09-30'))
        self.assertTrue(pd.isna(result.iloc[3]))

    def test_archival_column_collision_preserved(self):
        mapping = build_column_mapping('passport.csv',['KCV +20','KCV -20'])
        self.assertEqual(len(set(mapping.values())),2)

    def test_original_aliases_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for canonical, aliases in FILE_ALIASES.items():
                (root/aliases[-1]).write_text('id,value\n1,2\n',encoding='utf-8')
            sources = resolve_sources(root)
            self.assertEqual(sources['passport.csv'].name,'pot_south.csv')
            (root/'passport.csv').write_text('id,value\n1,3\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Conflicting aliases'):
                resolve_sources(root)

    def test_missing_source_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                resolve_sources(tmp)

    def test_grid_and_exact_endpoint(self):
        passport = pd.DataFrame({'id':[1,2],'length_km':[.020,.025]})
        grid = build_segments_grid(passport)
        self.assertEqual(grid.groupby('id').size().to_dict(),{1:2,2:3})
        source = pd.DataFrame({'id':[1,1,1,2],'distance':[0,10,20,25]})
        self.assertEqual(assign_segments(source,'distance',grid).segment_id.tolist(),[0,1,1,2])
        for bad in [-.001,20.01]:
            with self.assertRaises(ValueError):
                assign_segments(pd.DataFrame({'id':[1],'distance':[bad]}),'distance',grid)

    def test_invalid_length_rejected(self):
        with self.assertRaises(ValueError):
            build_segments_grid(pd.DataFrame({'id':[1],'length_km':[0]}))

    def test_target_is_measured_residual_not_nominal_formula(self):
        grid = build_segments_grid(pd.DataFrame({'id':[1],'length_km':[.02]}))
        source = pd.DataFrame({'id':[1,1,1],'distance':[1.,2.,12.], 'date':pd.to_datetime(['2018-09-23']*3),
            'source_row':[2,3,4], 'residual_for_target_mm':[2.7,3.2,np.nan], 'defect_depth_pct':[40.,.15,20.],
            'before_commissioning_flag':[False]*3, 'commissioning_unknown_flag':[False]*3,
            'wall_nominal_mismatch_flag':[True]*3, 'heldout_or_future_flag':[False]*3})
        result = map_vtd_to_segments(source,grid)
        self.assertEqual(result.iloc[0].min_residual_thickness_mm,2.7)
        self.assertTrue(pd.isna(result.iloc[1].min_residual_thickness_mm))
        self.assertFalse(result.iloc[1].target_eligible)

    def test_profile_linear_ramp_and_short_last_segment(self):
        grid = build_segments_grid(pd.DataFrame({'id':[1],'length_km':[.022]}))
        profile = pd.DataFrame({'id':[1,1,1],'total_distance':[0.,10.,22.],'z':[0.,10.,22.]})
        result = map_profile_to_segments(profile,grid)
        np.testing.assert_allclose(result.slope_m_per_m,[1.,1.,1.])
        np.testing.assert_allclose(result.z_mean,[5.,15.,21.])
        self.assertEqual(result.length_m.iloc[-1],2.)

    def test_profile_mean_is_distance_weighted(self):
        grid = build_segments_grid(pd.DataFrame({'id':[1],'length_km':[.01]}))
        profile = pd.DataFrame({'id':[1,1,1],'total_distance':[0.,1.,10.],'z':[0.,1.,10.]})
        self.assertAlmostEqual(map_profile_to_segments(profile,grid).z_mean.iloc[0],5.)

    def test_three_calendar_days_not_three_rows(self):
        frame = pd.DataFrame({'id':[1]*3,'date':pd.to_datetime(['2020-01-01 01:00','2020-01-01 02:00','2020-01-01 03:00'])})
        self.assertFalse(persistent_calendar_days(frame,pd.Series([True]*3)).any())

    def test_calendar_gap_breaks_persistence(self):
        frame = pd.DataFrame({'id':[1]*3,'date':pd.to_datetime(['2020-01-01','2020-01-02','2020-01-04'])})
        self.assertFalse(persistent_calendar_days(frame,pd.Series([True]*3)).any())

    def test_persistent_run_retained_separately_per_pipe(self):
        frame = pd.DataFrame({'id':[1,1,1,2],'date':pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03','2020-01-03'])})
        self.assertEqual(persistent_calendar_days(frame,pd.Series([True]*4)).tolist(),[True,True,True,False])

    def test_good_day_breaks_anomaly_run(self):
        frame = pd.DataFrame({'id':[1]*3,'date':pd.date_range('2020-01-01',periods=3)})
        self.assertFalse(persistent_calendar_days(frame,pd.Series([True,False,True])).any())

    def test_checks_not_python_asserts(self):
        with self.assertRaises(ValueError):
            require(False,'must fail even under python -O')


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Path(os.environ.get('PIPETWIN_TEST_DATA', BASE_DIR/'data'))
        if not (cls.data/'passport.csv').exists() and not (cls.data/'pot_south.csv').exists():
            raise RuntimeError('Integration tests require the six supplied CSV files')
        cls.temp = tempfile.TemporaryDirectory()
        cls.result = run_pipeline(cls.data,Path(cls.temp.name),BASE_DIR/'configs/preprocessing.json')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_target_grid_exact_keys(self):
        grid = self.result['tables']['target_segments_grid']
        self.assertEqual(grid.groupby('id').size().to_dict(),{1250009583:1475,1250022871:525})
        self.assertEqual(len(grid),2000)

    def test_original_measured_minimum(self):
        # Independent calculation directly from the unmodified CSV, not cleaned columns.
        name = resolve_sources(self.data)['vtd.csv']
        raw = pd.read_csv(name,low_memory=False)
        rows = raw[(raw.id==1250009490)&(raw.date=='2018-09-23')&raw.Distance.between(0,10,inclusive='left')]
        expected = rows['Residual Wall Thickness'].min()
        out = self.result['tables']['vtd_by_segment']
        row = out[(out.id==1250009490)&(out.date==pd.Timestamp('2018-09-23'))&(out.segment_id==0)].iloc[0]
        self.assertEqual(expected,5.53)
        self.assertEqual(row.min_residual_thickness_mm,expected)

    def test_fraction_depth_normalized_without_altering_residual(self):
        clean = self.result['cleaned']['vtd.csv']
        part = clean[clean.date.eq(pd.Timestamp('2018-09-23'))&clean.defect_depth_pct_raw.notna()]
        np.testing.assert_allclose(part.defect_depth_pct,part.defect_depth_pct_raw*100)
        raw = pd.read_csv(resolve_sources(self.data)['vtd.csv'],low_memory=False).set_index('row_id')
        values = raw.loc[clean.row_id,'Residual Wall Thickness'].to_numpy()
        np.testing.assert_allclose(clean.residual_wall_thickness,values,equal_nan=True)

    def test_bad_coordinates_quarantined_not_clipped(self):
        clean = self.result['cleaned']['vtd.csv']
        rejects = self.result['quarantine']['vtd_rejected']
        self.assertEqual(len(rejects),251)
        self.assertEqual(int(rejects.distance.lt(0).sum()),4)
        self.assertEqual(len(clean),48040)
        self.assertFalse(clean.distance.lt(0).any())

    def test_questionable_uzt_kept_but_not_used(self):
        uzt = self.result['cleaned']['uzt.csv']
        self.assertEqual(len(uzt),67)
        self.assertTrue(uzt.above_nominal_flag.all())
        self.assertFalse(uzt.measurement_eligible.any())
        self.assertTrue(self.result['tables']['target_features'].last_uzt_thickness_mm.isna().all())

    def test_monitoring_conflict_not_joined(self):
        monitor = self.result['cleaned']['regime.csv']
        self.assertEqual(int(monitor.direction_conflict_flag.sum()),83)
        self.assertFalse(monitor.mapping_eligible.any())
        self.assertTrue(self.result['tables']['monitoring_mapped'].monitoring_records_count.eq(0).all())

    def test_diameter_in_millimeters(self):
        tech = self.result['cleaned'][TECH_FILE]
        self.assertEqual(set(tech.diameter_mm),{159.,219.})

    def test_invalid_chronology_not_training_target(self):
        target = self.result['tables']['training_targets']
        self.assertFalse(target.before_commissioning_flag.any())
        self.assertFalse(target.date.eq(pd.Timestamp('2014-12-28')).any())

    def test_future_injection_cannot_change_features(self):
        result = self.result
        clean = {k:v.copy() for k,v in result['cleaned'].items()}
        # Future records cannot establish a three-day run or change any feature.
        for request in CONFIG['targets']:
            tech = clean[TECH_FILE]
            mask = tech.id.eq(request['id'])&tech.date.ge(pd.Timestamp(request['prediction_date']))
            tech.loc[mask,'q_liquid'] = 99999999.
            tech.loc[mask,'p_start'] = 99999999.
        other = build_features_asof(clean,result['grid'],result['tables']['profile_by_segment'],
            result['tables']['vtd_by_segment'],result['tables']['uzt_by_segment'],CONFIG['targets'],CONFIG)
        pd.testing.assert_frame_equal(other['target_features'],result['tables']['target_features'])
        pd.testing.assert_frame_equal(other['hydraulic_quality'],result['tables']['hydraulic_quality'])

    def test_third_future_day_does_not_validate_two_day_anomaly(self):
        row = self.result['cleaned'][TECH_FILE].iloc[[0]].copy()
        frame = pd.concat([row]*3,ignore_index=True)
        frame['id']=1
        frame['date']=pd.date_range('2020-01-01',periods=3)
        frame['source_row']=[2,3,4]
        frame['record_eligible']=True
        frame['q_liquid']=20000.
        before,_ = hydraulic_asof(frame,[1],'2020-01-03',CONFIG)
        after,_ = hydraulic_asof(frame,[1],'2020-01-04',CONFIG)
        self.assertTrue(before.q_liquid_usable.isna().all())
        self.assertTrue(before.q_oil_usable.isna().all())
        self.assertTrue(after.q_liquid_usable.eq(20000).all())

    def test_cold_start_excludes_diagnostic_history(self):
        result=self.result
        req=[dict(id=1250022871,prediction_date='2023-12-01',case='cold_start')]
        feature=build_features_asof(result['cleaned'],result['grid'],result['tables']['profile_by_segment'],
            result['tables']['vtd_by_segment'],result['tables']['uzt_by_segment'],req,CONFIG)['target_features']
        self.assertTrue(feature.last_vtd_thickness_mm.isna().all())
        self.assertTrue(feature.last_vtd_date.isna().all())

    def test_empty_past_history_stays_missing(self):
        result=self.result
        req=[dict(id=1250022871,prediction_date='2001-01-01',case='between_vtd')]
        tables=build_features_asof(result['cleaned'],result['grid'],result['tables']['profile_by_segment'],
            result['tables']['vtd_by_segment'],result['tables']['uzt_by_segment'],req,CONFIG)
        self.assertTrue(tables['target_features'].hydraulic_records_count.eq(0).all())
        self.assertTrue(tables['target_features'].q_liquid_mean.isna().all())
        self.assertTrue(tables['target_features'].last_vtd_thickness_mm.isna().all())

    def test_impossible_quantities_not_rescued_by_persistence(self):
        row=self.result['cleaned'][TECH_FILE].iloc[[0]].copy()
        frame=pd.concat([row]*3,ignore_index=True)
        frame['id']=1
        frame['date']=pd.date_range('2020-01-01',periods=3)
        frame['source_row']=[2,3,4]
        frame['record_eligible']=True
        frame['q_liquid']=-100.
        frame['watercut_pct']=101.
        clean,_=hydraulic_asof(frame,[1],'2020-01-04',CONFIG)
        self.assertTrue(clean.q_liquid_usable.isna().all())
        self.assertTrue(clean.watercut_pct_usable.isna().all())

    def test_every_export_roundtrip_checked(self):
        self.assertGreaterEqual(self.result['validation']['csv_parquet_pairs_verified'],20)
        manifest=json.loads((Path(self.temp.name)/'reports/run_manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(all(x['csv_parquet_verified'] for x in manifest['outputs'].values()))


if __name__=='__main__':
    unittest.main()

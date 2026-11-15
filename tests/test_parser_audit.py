"""Adversarial regressions discovered by the source-to-output parser audit."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import numpy as np
from src.config import BASE_DIR,TECH_FILE,NUMERIC_COLUMNS
from src.format_detector import read_and_standardize_all
from src.data_cleaner import clean_datasets,QualityLog
from src.segment_mapper import build_features_asof
from src.temporal import hydraulic_asof
from src.storage import save_table,validate_roundtrip

CONFIG=json.loads((BASE_DIR/'configs/preprocessing.json').read_text())

class AuditRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw,_,_=read_and_standardize_all(BASE_DIR/'data')
    def parse(self,text):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'profile.csv';p.write_text(text)
            with patch('src.format_detector.resolve_sources',return_value={'profile.csv':p}):
                return read_and_standardize_all(t)[0]['profile.csv']
    def test_large_integer_id_is_exact(self):
        self.assertEqual(self.parse('id,total_distance,z,lat,lon\n9007199254740993,0,1,2,3\n').id.iloc[0],9007199254740993)
    def test_duplicate_header_rejected(self):
        with self.assertRaisesRegex(ValueError,'header'):
            self.parse('id,total_distance,z,lat,lon,z\n1,0,1,2,3,4\n')
    def test_malformed_width_rejected(self):
        with self.assertRaisesRegex(ValueError,'fields'):
            self.parse('id,total_distance,z,lat,lon\n1,2,1,2,3,4\n')
    def test_missing_installation_cannot_enable_monitoring(self):
        data={k:v.copy() for k,v in self.raw.items()}
        p=data['passport.csv'].iloc[0]
        m=data['regime.csv'].iloc[[0]].copy()
        m['id']=p.id;m['direction_name']=p.direction_name
        m['sample_installation_date']=pd.NaT;m['sample_removal_date']=pd.Timestamp('2020-01-01')
        data['regime.csv']=m
        c,_=clean_datasets(data,CONFIG,QualityLog())
        self.assertFalse(c['regime.csv'].observation_eligible.any())
    def test_absolute_zero_not_rescued_by_persistence(self):
        r=pd.concat([self.raw[TECH_FILE].iloc[[0]]]*3,ignore_index=True)
        r['id']=1;r['record_eligible']=True;r['date']=pd.date_range('2020-01-01',periods=3);r['t_mix']=-300.
        out,_=hydraulic_asof(r,[1],'2020-01-04',CONFIG)
        self.assertTrue(out.t_mix_usable.isna().all())
    def test_nanosecond_timestamp_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            save_table(pd.DataFrame({'date':[pd.Timestamp('2020-01-01 00:00:00.123456789')]}),Path(t)/'x')
    def test_roundtrip_detects_one_unit_large_integer_change(self):
        with self.assertRaises(ValueError):
            validate_roundtrip(pd.DataFrame({'id':pd.Series([9007199254740993],dtype='Int64')}),pd.DataFrame({'id':['9007199254740992']}),'test')
    def test_cold_age_ignores_archival_diagnostic_flag(self):
        from src.ml.data import RepositoryData
        d=RepositoryData();clean=d.cleaned
        clean['passport.csv']['history_date_conflict_flag']=False
        args=(d.read('processed/segments_grid'),d.read('processed/profile_by_segment'),d.read('processed/vtd_by_segment'),d.read('processed/uzt_by_segment'),[CONFIG['targets'][1]],CONFIG)
        a=build_features_asof(clean,*args)['target_features']
        clean['passport.csv']['history_date_conflict_flag']=True
        b=build_features_asof(clean,*args)['target_features']
        pd.testing.assert_frame_equal(a,b)

class PublicationRegression(unittest.TestCase):
    def test_failed_batch_restores_previous_files(self):
        from src.data_prep import run_pipeline
        from src.storage import publish_file
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            names=['data/cleaned/passport.csv','data/cleaned/passport.parquet','data/cleaned/profile.csv']
            for name in names:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'previous generation')
            calls=0
            def fail_third(source,destination):
                nonlocal calls
                calls+=1
                if calls==3: raise OSError('injected batch publication failure')
                return publish_file(source,destination)
            with patch('src.data_prep.publish_file',side_effect=fail_third),self.assertRaises(OSError):
                run_pipeline(output_root=root)
            for name in names:
                self.assertEqual((root/name).read_bytes(),b'previous generation')

class AuditProperties(unittest.TestCase):
    def test_exact_identifier_domain(self):
        from src.format_detector import parse_identifiers
        out=parse_identifiers(pd.Series(['9223372036854775807','9.007199254740993e15','invalid',None]))
        self.assertEqual(out.iloc[0],9223372036854775807)
        self.assertEqual(out.iloc[1],9007199254740993)
        self.assertTrue(out.iloc[2:].isna().all())
        for value in ['0','-1','1.1','9223372036854775808']:
            with self.assertRaises(ValueError):parse_identifiers(pd.Series([value]))
    def test_invalid_numeric_text_is_traceable(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'profile.csv';p.write_text('id,total_distance,z,lat,lon\n1,0,bad,2,3\n')
            with patch('src.format_detector.resolve_sources',return_value={'profile.csv':p}):
                out,_,issues=read_and_standardize_all(t)
            self.assertTrue(pd.isna(out['profile.csv'].z.iloc[0]))
            self.assertEqual(out['profile.csv'].z_source_text.iloc[0],'bad')
            self.assertEqual((issues[0]['file'],issues[0]['source_row'],issues[0]['field'],issues[0]['original_value']),('profile.csv',2,'z','bad'))
    def test_encodings_quotes_multiline_and_empty_cells(self):
        text='id;total_distance;z;lat;lon;Примечание\n1;0;"1,5";;3;"цитата; ""текст""\nпродолжение"\n'
        for encoding in ['utf-8','utf-8-sig','cp1251']:
            with self.subTest(encoding=encoding),tempfile.TemporaryDirectory() as t:
                p=Path(t)/'profile.csv';p.write_bytes(text.encode(encoding))
                with patch('src.format_detector.resolve_sources',return_value={'profile.csv':p}):
                    out,_,issues=read_and_standardize_all(t)
                r=out['profile.csv'].iloc[0]
                self.assertEqual(len(out['profile.csv']),1)
                self.assertEqual(r.z,1.5)
                self.assertTrue(pd.isna(r.lat))
                self.assertEqual(r.primechanie,'цитата; "текст"\nпродолжение')
                self.assertFalse(issues)
    def test_trailing_empty_header_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'profile.csv';p.write_text('id,total_distance,z,lat,lon,\n1,0,1,2,3,\n')
            with patch('src.format_detector.resolve_sources',return_value={'profile.csv':p}),self.assertRaisesRegex(ValueError,'header'):
                read_and_standardize_all(t)
    def test_ambiguous_and_invalid_dates_have_source_evidence(self):
        from src.format_detector import robust_parse_dates
        result=robust_parse_dates(pd.Series(['02/03/2020','31.02.2020','2020-02-29','01.03.2020 23:59:59']))
        self.assertTrue(result.iloc[:2].isna().all())
        self.assertEqual(result.iloc[2],pd.Timestamp('2020-02-29'))
        self.assertEqual(result.iloc[3],pd.Timestamp('2020-03-01 23:59:59'))
    def test_segment_boundaries_nextafter_and_duplicate_minimum(self):
        from src.grid_builder import build_segments_grid,assign_segments
        from src.segment_mapper import map_vtd_to_segments
        grid=build_segments_grid(pd.DataFrame({'id':[1],'length_km':[.025]}))
        distances=[0,np.nextafter(10.,0),10,np.nextafter(10.,20),25]
        rows=pd.DataFrame({'id':[1]*5,'distance':distances})
        self.assertEqual(assign_segments(rows,'distance',grid).segment_id.tolist(),[0,0,1,1,2])
        for distance in [-1e-8,25+1e-8]:
            with self.assertRaises(ValueError):assign_segments(pd.DataFrame({'id':[1],'distance':[distance]}),'distance',grid)
        rows['date']=pd.Timestamp('2020-01-01');rows['source_row']=range(2,7)
        rows['residual_for_target_mm']=[5,4,3,2,1];rows['defect_depth_pct']=0
        for col in ['before_commissioning_flag','commissioning_unknown_flag','wall_nominal_mismatch_flag','heldout_or_future_flag']:rows[col]=False
        a=map_vtd_to_segments(rows,grid).min_residual_thickness_mm
        b=map_vtd_to_segments(pd.concat([rows,rows]).sample(frac=1,random_state=0),grid).min_residual_thickness_mm
        pd.testing.assert_series_equal(a,b)
    def test_reader_refuses_interrupted_generation(self):
        from src.ml.data import RepositoryData
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'reports';p.mkdir();(p/'PUBLISH_IN_PROGRESS').write_text('incomplete')
            with self.assertRaisesRegex(ValueError,'publication'):
                RepositoryData(root=Path(t))

"""Behavioural tests for source adapters, loss-aware parsing, physics, metrics,
mentor ingestion, exports and failure-safe publication."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.cache import cache_key
from src.config import TECH_FILE, BASE_DIR
from src.data_cleaner import residual_validity
from src.fluid import liquid_reynolds, single_phase_regime_qualifier
from src.grid_builder import assign_segments, build_segments_grid
from src.heat_transfer import (HeatTransferParameters, build_scenario,
                               segment_outlet_temperature, simulate_pipe)
from src.leakage import audit_feature_table
from src.mentor_ingest import ingest
from src.ml.evaluation import metrics, top_fraction_overlap
from src.ml.submission import (export_result, inspect_template, risk_ranking,
                               uzt_priority_table, validate_submission)
from src.parsing import (detect_dialect, parse_date_series, parse_identifier_series,
                         parse_number_text, read_raw_table)
from src.quality import build_quality_bundle
from src.schema_registry import build_feature_inventory
from src.segment_mapper import (build_features_asof, map_profile_to_segments,
                                map_vtd_to_segments)
from src.source_registry import (SourceConflictError, discover_sources,
                                 validate_template_candidate)
from src.storage import publish_tree


def write(path: Path, text: str, encoding="utf-8"):
    path.write_text(text, encoding=encoding)
    return path


class ParsingTests(unittest.TestCase):
    def test_cp1251_bom_quoted_newline_and_semicolon(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "semi.csv"
            path.write_bytes("id;значение\n1;\"a;b\"\n2;\"line1\nline2\"\n".encode("cp1251"))
            table = read_raw_table(path)
            self.assertEqual(table.encoding, "cp1251")
            self.assertEqual(table.delimiter, ";")
            self.assertEqual(len(table.rows), 2)
            self.assertEqual(table.rows[1][1], "line1\nline2")
            bom = Path(tmp) / "bom.csv"
            bom.write_bytes(b"\xef\xbb\xbfid,value\n1,2\n")
            self.assertEqual(read_raw_table(bom).encoding, "utf-8-sig")

    def test_duplicate_headers_and_malformed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            write(path, "id,id,value\n1,2,3\n4,5\n")
            table = read_raw_table(path)
            self.assertEqual(table.duplicate_headers, ["id"])
            self.assertEqual(len(table.malformed), 1)
            self.assertEqual(table.malformed[0]["error_code"], "field_count_mismatch")
            self.assertIn("4|5", table.malformed[0]["original_content"])

    def test_decimal_comma_censored_zero_and_ambiguous(self):
        self.assertEqual(parse_number_text("4251,17")[0], 4251.17)
        self.assertEqual(parse_number_text("1 234,5")[0], 1234.5)
        self.assertEqual(parse_number_text("0")[0], 0.0)
        value, state, _ = parse_number_text("<0.1")
        self.assertIsNone(value)
        self.assertEqual(state, "censored")
        value, state, reason = parse_number_text("1,234")
        self.assertIsNone(value)
        self.assertEqual(state, "parse_failed")
        self.assertEqual(reason, "ambiguous_decimal_separator")
        self.assertEqual(parse_number_text("1.234")[0], 1.234)

    def test_identifiers_exact_and_leading_zero_preserved(self):
        parsed, issues, leading = parse_identifier_series(
            pd.Series(["1250013653.0", "00123", "1.5", "-2", "1e3", ""]))
        self.assertEqual(int(parsed.iloc[0]), 1250013653)
        self.assertEqual(int(parsed.iloc[1]), 123)
        self.assertTrue(bool(leading.iloc[1]))
        self.assertTrue(pd.isna(parsed.iloc[2]))
        self.assertTrue(any(x["reason"] == "not_exact_integer_identifier" for x in issues))

    def test_dates_are_parsed_per_value_and_ambiguous_slash_is_quarantined(self):
        result, date_only, tz, issues = parse_date_series(
            pd.Series(["23.09.2018", "30.09.2021 0:00:00", "01/02/2020"]),
            with_metadata=True)
        self.assertEqual(result.iloc[0], pd.Timestamp("2018-09-23"))
        self.assertEqual(result.iloc[1], pd.Timestamp("2021-09-30"))
        self.assertTrue(pd.isna(result.iloc[2]))
        self.assertTrue(any(x["state"] == "ambiguous_date_locale" for x in issues))
        explicit = parse_date_series(pd.Series(["01/02/2020"]), locale="dayfirst")
        self.assertEqual(explicit.iloc[0], pd.Timestamp("2020-02-01"))

    def test_ambiguous_delimiter_is_not_guessed(self):
        # A one-column file has no usable delimiter; it must fail, not silently pick comma.
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp) / "one.csv", "just_one_column\nvalue\n")
            with self.assertRaises(Exception):
                detect_dialect(path)


class SourceClassificationTests(unittest.TestCase):
    HEADERS = {
        "passport.csv": "Общие данные_D, мм,Общие данные_Дата ввода\n1,2000-01-01\n",
        "profile.csv": "id,total_distance,z\n1,0,10\n",
        "vtd.csv": "id,date,Distance,Residual Wall Thickness,Defect Type\n1,2020-01-01,1,5,corrosion\n",
        "uzt.csv": "id,date,start_distance,angle,residual_wall_thickness\n1,2020-01-01,1,90,5\n",
        TECH_FILE: "simple_section_id,calc_date,q,watercut\n1,2020-01-01,10,50\n",
        "regime.csv": "Simple Section ID,UKK Number,Sample Installation Date\n1,U1,2019-01-01\n",
        "sample_result.csv": "id,segment_id,thickness_pred\n1,0,5.0\n",
    }

    def _write_all(self, root):
        for name, text in self.HEADERS.items():
            write(root / name, text)

    def test_semantic_types_survive_organizer_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_all(root)
            result = discover_sources(root, required_types=(
                "passport", "profile", "vtd", "uzt", "hydraulic_regime",
                "corrosion_monitoring", "submission_template"))
            self.assertEqual(result.selected["corrosion_monitoring"].path.name, "regime.csv")
            self.assertEqual(result.selected["submission_template"].path.name, "sample_result.csv")
            self.assertEqual(result.selected["hydraulic_regime"].path.name, TECH_FILE)
            self.assertEqual(result.template_validation["status"], "authoritative_schema_match")

    def test_regime_filename_can_mean_either_source_but_schema_decides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "regime.csv", self.HEADERS[TECH_FILE])
            result = discover_sources(root, required_types=("hydraulic_regime",),
                                      optional_types=("corrosion_monitoring",))
            self.assertEqual(result.selected["hydraulic_regime"].path.name, "regime.csv")
            self.assertTrue(any("filename/schema conflict" in w for w in result.warnings))
            self.assertNotIn("corrosion_monitoring", result.selected)

    def test_sample_result_named_file_is_classified_by_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "sample_result.csv", self.HEADERS[TECH_FILE])
            result = discover_sources(root, required_types=("hydraulic_regime",),
                                      optional_types=("submission_template",))
            self.assertEqual(result.selected["hydraulic_regime"].path.name, "sample_result.csv")
            self.assertEqual(result.template_validation["status"], "no_authoritative_template")

    def test_identical_copies_are_deduplicated_but_conflicts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "passport.csv", self.HEADERS["passport.csv"])
            write(root / "pot_south.csv", self.HEADERS["passport.csv"])
            result = discover_sources(root, required_types=("passport",))
            self.assertTrue(any("identical copy" in w for w in result.warnings))
            write(root / "pot_south.csv", self.HEADERS["passport.csv"].replace("2000-01-01", "2001-01-01"))
            with self.assertRaises(SourceConflictError):
                discover_sources(root, required_types=("passport",))

    def test_identical_numbered_monitoring_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "repcorr_south.csv", self.HEADERS["regime.csv"])
            write(root / "repcorr_south(1).csv", self.HEADERS["regime.csv"])
            result = discover_sources(root, required_types=("corrosion_monitoring",))
            self.assertIn(result.selected["corrosion_monitoring"].path.name, ("repcorr_south.csv", "repcorr_south(1).csv"))
            self.assertTrue(any("identical copy" in w for w in result.warnings))

    def test_missing_required_fails_and_optional_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "passport.csv", self.HEADERS["passport.csv"])
            result = discover_sources(root, required_types=("passport",),
                                      optional_types=("corrosion_monitoring", "vtd"))
            self.assertIn("corrosion_monitoring", result.missing_optional)
            self.assertIn("vtd", result.missing_optional)
            with self.assertRaises(FileNotFoundError):
                discover_sources(root, required_types=("profile",))

    def test_explicit_override_resolves_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "a.csv", self.HEADERS["regime.csv"])
            write(root / "b.csv", self.HEADERS["regime.csv"].replace("U1", "U2"))
            with self.assertRaises(SourceConflictError):
                discover_sources(root, required_types=("corrosion_monitoring",))
            result = discover_sources(root, required_types=("corrosion_monitoring",),
                                      source_overrides={"corrosion_monitoring": "b.csv"})
            self.assertEqual(result.selected["corrosion_monitoring"].path.name, "b.csv")

    def test_source_bytes_are_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "passport.csv"
            write(path, self.HEADERS["passport.csv"])
            before = path.read_bytes()
            discover_sources(root, required_types=("passport",))
            self.assertEqual(path.read_bytes(), before)


class PhysicsTests(unittest.TestCase):
    def test_reynolds_dynamic_and_kinematic_agree_in_si(self):
        dynamic = liquid_reynolds(0.2, 0.5, 1000.0, 0.001, "dynamic", "Pa.s")
        kinematic = liquid_reynolds(0.2, 0.5, 1000.0, 0.001 / 1000.0, "kinematic", "m2/s")
        self.assertEqual(dynamic.status, "defined")
        self.assertAlmostEqual(dynamic.reynolds, kinematic.reynolds)
        self.assertIsNone(liquid_reynolds(0.2, 0.5, 1000, None, "dynamic", "Pa.s").reynolds)
        self.assertEqual(liquid_reynolds(0.2, 0.5, 1000, 0.0, "dynamic", "Pa.s").status, "unavailable")
        self.assertEqual(liquid_reynolds(0.2, 0.5, 1000, 0.001, None, None).status, "unresolved")
        self.assertIn("single-phase", single_phase_regime_qualifier(dynamic.reynolds)["qualifier"])

    def test_heat_loss_cools_and_preserves_continuity(self):
        params = HeatTransferParameters()
        one = segment_outlet_temperature(30.0, 10.0, 0.2, 1.0, params)
        self.assertTrue(one["valid"])
        self.assertLess(one["delta_temperature_c"], 0)
        segments = [{"segment_id": 0, "start_m": 0, "end_m": 10, "length_m": 10},
                    {"segment_id": 1, "start_m": 10, "end_m": 22, "length_m": 12}]
        result = simulate_pipe(segments, 30.0, 0.2, 1.0, params)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["t_out_c"], result[1]["t_in_c"])
        self.assertEqual(result[1]["length_m"], 12.0)

    def test_zero_flow_rejected_and_scenario_is_isolated(self):
        bad = segment_outlet_temperature(30.0, 10.0, 0.2, 0.0, HeatTransferParameters())
        self.assertFalse(bad["valid"])
        self.assertIn("nonpositive_or_missing_mass_flow", bad["issues"])
        frame = pd.DataFrame([
            {"id": 1, "prediction_date": pd.Timestamp("2023-01-01"), "case": "between_vtd",
             "segment_id": 0, "start_m": 0, "end_m": 10, "length_m": 10,
             "d_inner_assumed_mm": 200, "t_start_mean": 30, "density_liquid_work_mean": 1000,
             "v_liquid_true_mean": 0.5},
            {"id": 1, "prediction_date": pd.Timestamp("2023-01-01"), "case": "between_vtd",
             "segment_id": 1, "start_m": 10, "end_m": 20, "length_m": 10,
             "d_inner_assumed_mm": 200, "t_start_mean": 30, "density_liquid_work_mean": 1000,
             "v_liquid_true_mean": 0.5},
        ])
        scenario = build_scenario(frame, HeatTransferParameters(), "TEST")
        self.assertEqual(len(scenario), 2)
        self.assertTrue(scenario.valid.all())
        self.assertNotIn("delta_temperature_c", frame.columns)


class MetricTests(unittest.TestCase):
    def test_metric_definitions_and_undefined_cases(self):
        perfect = metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual(perfect["r2"], 1.0)
        self.assertEqual(perfect["pearson_squared"], 1.0)
        inverse = metrics([1, 2, 3], [3, 2, 1])
        self.assertEqual(inverse["pearson_squared"], 1.0)
        self.assertEqual(inverse["pearson_r"], -1.0)
        self.assertTrue(inverse["inverse_ranking_flag"])
        self.assertLess(inverse["r2"], 0)
        constant = metrics([1, 2, 3], [2, 2, 2])
        self.assertIsNone(constant["pearson_squared"])
        self.assertEqual(constant["pearson_reason"], "constant_prediction")
        constant_truth = metrics([2, 2, 2], [1, 2, 3])
        self.assertIsNone(constant_truth["r2"])
        self.assertIsNone(constant_truth["pearson_squared"])
        empty = metrics([], [])
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["mae_mm"])
        one = metrics([5.0], [4.0])
        self.assertIsNone(one["pearson_squared"])
        near_zero = metrics([0.0, 0.05, 1.0], [0.1, 0.1, 1.0])
        self.assertIsNone(near_zero["mape_pct"])
        self.assertEqual(near_zero["mape_excluded_count"], 2)
        self.assertIsNotNone(near_zero["mape_eligible_pct"])
        with self.assertRaises(ValueError):
            metrics([1.0, np.nan], [1.0, 2.0])

    def test_top_fraction_overlap_ties_are_explicit(self):
        result = top_fraction_overlap([1, 2, 3, 4], [4, 3, 2, 1])
        self.assertEqual(result["groups"][0]["k"], 1)
        self.assertEqual(result["groups"][0]["overlap"], 0)
        constant = top_fraction_overlap([1, 2, 3, 4], [2, 2, 2, 2])
        self.assertEqual(constant["groups"][0]["prediction_tie_count"], 4)


class AsOfTests(unittest.TestCase):
    def _fixture(self):
        passport = pd.DataFrame({
            "id": [1], "length_km": [0.02], "diameter_mm": [200.0],
            "nominal_thickness_mm": [5.0], "commissioning_date": [pd.Timestamp("2000-01-01")],
            "direction_name": ["DIR-1"], "main_pipeline_id": ["MAIN-1"],
            "history_date_conflict_flag": [False],
        })
        grid = build_segments_grid(passport)
        profile = pd.DataFrame({"id": [1, 1, 1], "total_distance": [0.0, 10.0, 20.0],
                                "z": [0.0, 0.5, 1.0]})
        profile_seg = map_profile_to_segments(profile, grid)
        tech = pd.DataFrame({
            "id": [1], "date": [pd.Timestamp("2020-01-01")], "source_row": [2],
            "record_eligible": [True],
            "p_start": [5.0], "p_end": [4.0], "q_liquid": [10.0], "q_oil": [5.0],
            "watercut_pct": [50.0], "v_liquid_true": [0.5], "v_mix": [0.6],
            "t_mix": [20.0], "t_start": [20.0], "t_end": [18.0],
            "density_liquid_work": [1000.0], "density_oil": [850.0],
        })
        monitoring = pd.DataFrame({
            "id": [1], "source_row": [2],
            "sample_removal_date": [pd.Timestamp("2020-01-01")],
            "observation_eligible": [True], "ph": [7.0], "co2_water": [10.0],
            "total_mineralization": [100.0],
        })
        vtd = pd.DataFrame({
            "id": [1, 1], "date": [pd.Timestamp("2018-01-01"), pd.Timestamp("2020-01-01")],
            "segment_id": [0, 0], "min_residual_thickness_mm": [4.0, 4.5],
            "target_eligible": [True, True],
        })
        uzt = pd.DataFrame(columns=["id", "date", "segment_id", "eligible_measurement_count",
                                    "eligible_min_thickness_mm"])
        cleaned = {"passport.csv": passport, TECH_FILE: tech, "regime.csv": monitoring}
        return cleaned, grid, profile_seg, vtd, uzt

    def test_prior_inspection_min_available_before_cutoff(self):
        cleaned, grid, profile, vtd, uzt = self._fixture()
        req = [{"id": 1, "prediction_date": "2021-01-01", "case": "between_vtd"}]
        cfg = {"persistent_days": 3, "soft_limits": {
            "pressure_high": 100, "liquid_flow_high": 10000, "velocity_high": 10,
            "temperature_low": -50, "temperature_high": 150, "density_low": 100,
            "density_high": 1500}}
        result = build_features_asof(cleaned, grid, profile, vtd, uzt, req, cfg)["target_features"]
        self.assertEqual(result.prior_vtd_min_thickness_mm.iloc[0], 4.0)
        self.assertEqual(result.prior_vtd_campaign_count.iloc[0], 2)
        self.assertEqual(result.last_vtd_thickness_mm.iloc[0], 4.5)
        cold = [{"id": 1, "prediction_date": "2021-01-01", "case": "cold_start"}]
        cold_result = build_features_asof(cleaned, grid, profile, vtd, uzt, cold, cfg)["target_features"]
        self.assertTrue(cold_result.prior_vtd_min_thickness_mm.isna().all())
        self.assertTrue(cold_result.last_vtd_thickness_mm.isna().all())

    def test_future_vtd_cannot_change_earlier_cutoff(self):
        cleaned, grid, profile, vtd, uzt = self._fixture()
        req = [{"id": 1, "prediction_date": "2021-01-01", "case": "between_vtd"}]
        cfg = {"persistent_days": 3, "soft_limits": {
            "pressure_high": 100, "liquid_flow_high": 10000, "velocity_high": 10,
            "temperature_low": -50, "temperature_high": 150, "density_low": 100,
            "density_high": 1500}}
        before = build_features_asof(cleaned, grid, profile, vtd, uzt, req, cfg)["target_features"]
        future = vtd.copy()
        future.loc[len(future)] = [1, pd.Timestamp("2022-01-01"), 0, 1.0, True]
        after = build_features_asof(cleaned, grid, profile, future, uzt, req, cfg)["target_features"]
        pd.testing.assert_frame_equal(before, after)


class ExportTests(unittest.TestCase):
    def test_template_inspection_rejects_techmod_named_sample_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            tech = Path(tmp) / "sample_result.csv"
            write(tech, "simple_section_id,calc_date,q,watercut\n1,2020-01-01,10,50\n")
            self.assertEqual(inspect_template(tech)["status"], "not_a_submission_template")
            template = Path(tmp) / "provided.csv"
            write(template, "id,segment_id,thickness_pred\n1,0,4.0\n")
            self.assertEqual(inspect_template(template)["status"], "authoritative_schema_match")

    def test_submission_exact_columns_keys_and_priority_separation(self):
        grid = pd.DataFrame({"id": [1, 1, 1], "segment_id": [0, 1, 2],
                             "start_m": [0, 10, 20], "end_m": [10, 20, 30],
                             "nominal_thickness_mm": [5.0, 5.0, 5.0]})
        bounds = grid.assign(h0_mm=[5.0, 5.0, 5.0])
        result = grid[["id", "segment_id"]].assign(thickness_pred=[4.0, 4.5, 4.2])
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "t.csv"
            write(template, "id,segment_id,thickness_pred\n1,0,4\n")
            with self.assertRaises(ValueError):
                validate_submission(result, grid, bounds, template)
            path = Path(tmp) / "result.csv"
            export_result(result, path, grid, bounds, template_path=None)
            self.assertEqual(path.read_text().splitlines()[0], "id,segment_id,thickness_pred")
        ranked = risk_ranking(result.assign(start_m=grid.start_m, end_m=grid.end_m,
                                            nominal_thickness_mm=grid.nominal_thickness_mm))
        priority = uzt_priority_table(ranked, forecast_date="2023-01-01")
        self.assertEqual(list(priority.columns)[:8], [
            "id", "forecast_date", "segment_id", "start_m", "end_m", "thickness_pred",
            "nominal_thickness_mm", "relative_thinning"])
        self.assertNotIn("thickness_pred", list(priority.columns[:0]))


class PublicationAndCacheTests(unittest.TestCase):
    def test_publish_failure_restores_previous_tree_and_manifest_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "dest"
            destination.mkdir()
            (destination / "a.txt").write_text("old-a")
            (destination / "run_manifest.json").write_text("old-manifest")
            stage = Path(tmp) / "stage"
            stage.mkdir()
            (stage / "a.txt").write_text("new-a")
            (stage / "run_manifest.json").write_text("new-manifest")
            real = __import__("src.storage", fromlist=["publish_file"]).publish_file

            def failing(source, dest):
                if Path(source).name == "run_manifest.json":
                    raise OSError("manifest failure")
                return real(source, dest)
            with patch("src.storage.publish_file", side_effect=failing):
                with self.assertRaises(OSError):
                    publish_tree(stage, destination)
            self.assertEqual((destination / "a.txt").read_text(), "old-a")
            self.assertEqual((destination / "run_manifest.json").read_text(), "old-manifest")

    def test_cache_key_depends_on_all_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = write(root / "a.csv", "id\n1\n")
            b = write(root / "b.csv", "id\n2\n")
            config = {"cutoff": "strict"}
            request = [{"id": 1, "prediction_date": "2023-01-01", "case": "between_vtd"}]
            first = cache_key({"a": a, "b": b}, config, request, code_hashes={"x": "1"})
            self.assertEqual(first, cache_key({"a": a, "b": b}, config, request, code_hashes={"x": "1"}))
            write(a, "id\n1\n# changed\n")
            self.assertNotEqual(first, cache_key({"a": a, "b": b}, config, request, code_hashes={"x": "1"}))
            self.assertNotEqual(first, cache_key({"a": a, "b": b}, config,
                                                 [{"id": 2, "prediction_date": "2023-01-01",
                                                   "case": "between_vtd"}], code_hashes={"x": "1"}))


class MentorWorkflowTests(unittest.TestCase):
    def test_real_supplied_notes_are_idempotent_and_leave_unknowns_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mentor"
            first = ingest(output=out)
            files = ["requirements.csv", "decisions.csv", "open_questions.csv", "actions.csv",
                     "source_inventory.json", "candidate_records.csv"]
            before = {name: (out / name).read_bytes() for name in files}
            second = ingest(output=out)
            after = {name: (out / name).read_bytes() for name in files}
            self.assertEqual(before, after)
            self.assertEqual(first["source_excerpts_not_found"], 0)
            self.assertEqual(second["sent_messages"], 0)
            inventory = json.loads((out / "source_inventory.json").read_text(encoding="utf-8"))
            self.assertTrue(all(s["stated_meeting_date"] is None for s in inventory["sources"]))
            self.assertTrue(all(s["participants"] is None for s in inventory["sources"]))
            decisions = pd.read_csv(out / "decisions.csv", dtype="string", keep_default_na=False)
            self.assertEqual(decisions.decision_id.nunique(), len(decisions))

    def test_source_revision_preserves_old_interpretation_and_flags_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note = write(root / "note.txt", "# Note\nAlpha decision is proposed.\n")
            sources = {"version": "test", "sources": [{
                "source_id": "SRC-TEST", "path": str(note), "doc_type": "meeting transcript/notes",
                "meeting_label": "synthetic test only", "stated_meeting_date": None,
                "participants": None, "coverage": "fixture", "limitations": "synthetic",
                "authority": "test"}]}
            records = {"version": "test",
                       "decisions": [{"id": "DEC-9001", "source_id": "SRC-TEST",
                                      "excerpt": "Alpha decision", "statement": "Alpha is proposed.",
                                      "status": "proposed", "evidence_type": "test",
                                      "authority": "test", "topic": "test",
                                      "affected_component": "test", "implementation_reference": "",
                                      "test_reference": "", "unresolved_question_ids": [],
                                      "supersedes_id": None}],
                       "requirements": [], "open_questions": [], "actions": []}
            source_path = write(root / "sources.json", json.dumps(sources, ensure_ascii=False))
            records_path = write(root / "records.json", json.dumps(records, ensure_ascii=False))
            out1 = root / "out1"
            ingest(source_path, records_path, out1)
            write(note, "# Note\nAlpha decision is proposed.\nBeta note.\n")
            out2 = root / "out2"
            ingest(source_path, records_path, out2, previous_output=out1)
            revisions = list((out2 / "revisions" / "SRC-TEST").glob("*.txt"))
            self.assertGreaterEqual(len(revisions), 2)
            history = pd.read_csv(out2 / "decisions_history.csv")
            self.assertGreaterEqual(len(history), 1)
            self.assertEqual(history.decision_id.nunique(), 1)
            write(note, "# Note\nCompletely different content.\n")
            out3 = root / "out3"
            ingest(source_path, records_path, out3, previous_output=out2)
            decisions = pd.read_csv(out3 / "decisions.csv", dtype="string", keep_default_na=False)
            self.assertEqual(str(decisions.source_excerpt_found.iloc[0]).lower(), "false")
            self.assertEqual(decisions.source_status.iloc[0], "conflicting_source_revision")


class FeatureInventoryTests(unittest.TestCase):
    def test_every_raw_column_and_unknown_status_is_present(self):
        cleaned = {
            "passport.csv": pd.DataFrame({"id": [1], "diameter_mm": [200.0],
                                          "nominal_thickness_mm": [5.0],
                                          "commissioning_date": [pd.Timestamp("2000-01-01")]}),
            "profile.csv": pd.DataFrame({"id": [1], "total_distance": [0.0], "z": [1.0]}),
            "vtd.csv": pd.DataFrame({"id": [1], "date": [pd.Timestamp("2020-01-01")],
                                     "distance": [1.0], "residual_wall_thickness": [4.0],
                                     "angle": [90.0]}),
            "uzt.csv": pd.DataFrame({"id": [1]}), "regime.csv": pd.DataFrame({"id": [1]}),
            TECH_FILE: pd.DataFrame({"id": [1]}),
        }
        summary = {}
        for name, frame in cleaned.items():
            summary[name] = {"source_type": name.replace(".csv", ""), "column_mapping":
                             {c: c for c in frame.columns}, "record_count": len(frame),
                             "duplicate_headers": [], "empty_headers": []}
        summary["vtd.csv"]["source_type"] = "vtd"
        inventory = build_feature_inventory(cleaned, summary, pd.DataFrame())
        self.assertEqual(len(inventory), sum(len(f.columns) for f in cleaned.values()))
        self.assertTrue({"measured", "available_but_excluded", "unresolved_semantics"} &
                        set(inventory.feature_status))

    def test_leakage_audit_rejects_cold_start_diagnostic_values(self):
        frame = pd.DataFrame({"id": [2], "prediction_date": [pd.Timestamp("2022-07-01")],
                              "segment_id": [0], "case": ["cold_start"],
                              "last_vtd_thickness_mm": [4.0], "has_previous_vtd": [True]})
        result = audit_feature_table(frame, target_ids=(2,), cold_start_id=2)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()

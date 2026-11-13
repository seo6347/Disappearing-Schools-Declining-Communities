"""Selection policy, frozen handoff, and distance classification tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


selector = module("scripts/04-5_select_final_retained_schools.py", "selector")
impacts = module("scripts/05-1_analyze_final_commute_impacts.py", "impacts")
runner = module("run_pipeline.py", "runner")
from final_scenario_io import load_final_scenario, MANIFEST_FILE, SCENARIO_ID


def options():
    return pd.DataFrame({
        "integrated_school_key": ["S1", "S2"], "option_analysis_status": ["READY", "READY"],
        "weighted_mean_additional_time_burden_min": [5.0, 2.0],
        "weighted_mean_duration_change_min": [-10.0, 1.0],
        "weighted_p90_duration_change_min": [10.0, 5.0], "max_duration_change_min": [12.0, 7.0],
    })


class FinalSelectionTests(unittest.TestCase):
    def test_additional_burden_not_net_change(self):
        self.assertEqual(selector.select_retained_school(options()).integrated_school_key, "S2")

    def test_ready_first_even_if_partial_has_smaller_score(self):
        frame = options()
        frame.loc[1, "option_analysis_status"] = "PARTIAL_ROUTE_EXCLUSION"
        selected = selector.select_retained_school(frame)
        self.assertEqual(selected.integrated_school_key, "S1")
        self.assertEqual(selected.selection_pool_status, "READY_ONLY")
        self.assertEqual(selected.available_option_count, 2)
        self.assertEqual(selected.compared_option_count, 1)

    def test_partial_only_fallback(self):
        frame = options().assign(option_analysis_status="PARTIAL_ROUTE_EXCLUSION")
        selected = selector.select_retained_school(frame)
        self.assertEqual(selected.integrated_school_key, "S2")
        self.assertEqual(selected.selection_pool_status, "PARTIAL_ONLY")

    def test_no_usable_option_fails(self):
        with self.assertRaises(ValueError):
            selector.select_retained_school(options().assign(option_analysis_status="EXCLUDED_NO_USABLE_DEMAND"))

    def test_missing_auxiliary_not_zero_imputed(self):
        frame = options()
        frame.loc[1, "weighted_p90_duration_change_min"] = np.nan
        with self.assertRaises(ValueError):
            selector.select_retained_school(frame)

    def test_all_tie_break_levels(self):
        for column in selector.RANK_COLUMNS[1:]:
            frame = options()
            for c in selector.RANK_COLUMNS[:-1]:
                frame[c] = 1.0
            frame["integrated_school_key"] = ["S2", "S1"]
            if column != "integrated_school_key":
                frame.loc[1, column] = 0.5
            with self.subTest(column=column):
                self.assertEqual(selector.select_retained_school(frame).integrated_school_key, "S1")

    def test_near_score_is_not_misreported_as_a_tie(self):
        frame = options()
        frame[selector.METRIC] = [1.0, 1.0 + 1e-10]
        frame["weighted_mean_duration_change_min"] = [100.0, -100.0]
        selected = selector.select_retained_school(frame)
        self.assertEqual(selected.integrated_school_key, "S1")
        self.assertEqual(selected.primary_metric_tie_count, 1)

    def test_row_order_invariance(self):
        for seed in range(10):
            self.assertEqual(selector.select_retained_school(options().sample(frac=1, random_state=seed)).integrated_school_key, "S2")

    def test_office_loop_and_single_assignment_handoff(self):
        option_rows, member_rows, detail_rows = [], [], []
        for i in range(720):
            cluster, option = f"FC_{i:05d}", f"OPT_{i}"
            office, key = f"OFFICE_{i % 176:03d}", f"S{i}_0"
            row = options().iloc[0].to_dict()
            row.update(final_cluster_id=cluster, integration_option_id=option,
                       integrated_school_key=key, integrated_school_name=key, school_edu_office=office)
            option_rows.append(row)
            for school in range(3):
                member_rows.append(dict(final_cluster_id=cluster, school_key=f"S{i}_{school}", school_edu_office=office))
                detail_rows.append(dict(final_cluster_id=cluster, integration_option_id=option,
                                        integrated_school_key=key, current_school_key=f"S{i}_{school}",
                                        demand_assignment_id=f"DA_{i}_{school}", census_id=f"C_{i}_{school}",
                                        student_weight="1", commute_analysis_eligible_flag="True"))
        scenario, schools, detail = selector.build_scenario(pd.DataFrame(option_rows), pd.DataFrame(member_rows), pd.DataFrame(detail_rows))
        self.assertEqual(len(scenario), 720)
        self.assertEqual(scenario.school_edu_office.nunique(), 176)
        self.assertEqual(schools.school_status.value_counts().to_dict(), {"CLOSED": 1440, "RETAINED": 720})
        self.assertEqual(len(detail), 2160)
        self.assertFalse(detail.demand_assignment_id.duplicated().any())

    def test_stage5_does_not_invoke_selection_or_require_raw_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "execute") as execute:
            with patch("sys.argv", ["runner", "--stage", "5", "--project-root", tmp]):
                runner.main()
            self.assertEqual([c.args[1] for c in execute.call_args_list], [
                "05-1_analyze_final_commute_impacts.py", "05-2_analyze_distance_change_distributions.py",
                "05-3_analyze_regional_distance_burden.py", "05-4_analyze_commute_time_thresholds.py"])

    def test_stage6_does_not_require_route_collection_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "execute") as execute:
            raw = Path(tmp) / "data/01_raw"
            raw.mkdir(parents=True)
            for name in ["elementary_master.csv", "census_demand_points.csv", "census_demand_points.gpkg"]:
                (raw / name).write_text("fixture")
            with patch("sys.argv", ["runner", "--stage", "6", "--project-root", tmp]):
                runner.main()
            self.assertEqual([c.args[1] for c in execute.call_args_list], ["06_analyze_school_voids.py"])


class FrozenInputTests(unittest.TestCase):
    def test_modified_stage4_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "data/03_outputs"
            folder.mkdir(parents=True)
            hashes = {}
            for filename in ["final_integration_scenario.csv", "final_integration_schools.csv", "final_integration_demand_assignments.csv"]:
                (folder / filename).write_text("original")
                hashes[filename] = hashlib.sha256(b"original").hexdigest()
            (folder / MANIFEST_FILE).write_text(json.dumps({"scenario_id": SCENARIO_ID, "outputs_sha256": hashes}))
            (folder / "final_integration_scenario.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Frozen Stage 4 output changed"):
                load_final_scenario(root)

    def test_distance_category_independent_of_school_change_and_exclusions(self):
        values = pd.Series([1.0, -1.0, 0.0, 5e-10, np.nan, 0.0])
        eligible = pd.Series([True, True, True, True, False, False])
        self.assertEqual(impacts.distance_categories(values, eligible).tolist(),
                         ["INCREASED", "DECREASED", "UNCHANGED", "UNCHANGED", "EXCLUDED", "EXCLUDED"])


if __name__ == "__main__":
    unittest.main()

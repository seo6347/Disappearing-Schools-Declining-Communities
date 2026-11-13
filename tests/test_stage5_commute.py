"""Weighted population, threshold, geography and output-isolation regression tests."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from stage5_commute_common import (OutputBundle, classify_region, output_path,
                                   population_groups, prepare_detail)


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


impacts = module("05-1_analyze_final_commute_impacts.py", "s5_impacts")
distributions = module("05-2_analyze_distance_change_distributions.py", "s5_distributions")
regional = module("05-3_analyze_regional_distance_burden.py", "s5_regional")
thresholds = module("05-4_analyze_commute_time_thresholds.py", "s5_thresholds")


def raw_fixture():
    before = np.array([30, 29, 31, 35, 10, 0, 20, 30, 9, np.nan])
    after = np.array([30, 31, 30, 40, 9, 10, 20, 30.000001, 20, np.nan])
    distance_change = np.array([1, -1, 0, 2, -2, 0, 3, -3, 1, np.nan])
    school = [f"S{i // 2 + 1}" for i in range(10)]
    return pd.DataFrame({
        "demand_assignment_id": [f"A{i:03}" for i in range(10)],
        "census_id": [f"P{i // 2:03}" for i in range(10)],
        "origin_lon": [127. + (i // 2) / 10 for i in range(10)],
        "origin_lat": [37. + (i // 2) / 10 for i in range(10)],
        "current_school_key": [f"CURRENT{i}" for i in range(10)],
        "integrated_school_key": school, "integrated_school_name": school,
        "final_cluster_id": [f"C{i // 2 + 1}" for i in range(10)],
        "current_school_edu_office": ["O1"] * 10, "integrated_school_edu_office": ["O1"] * 10,
        "current_school_population_decline_type": ["인구감소지역", "미해당", "인구감소지역", "인구감소관심지역", None,
                                                  "인구감소지역", "미해당", "미해당", "인구감소지역", "미해당"],
        "integrated_school_population_decline_type": [None, None, "미해당", "미해당", "인구감소지역", "인구감소지역",
                                                      "인구감소관심지역", "인구감소관심지역", "미해당", "미해당"],
        "student_weight": [2, 3, 5, 7, 11, 13, 17, .25, 19, 1000],
        "commute_analysis_eligible_flag": [True] * 9 + [False],
        "moved_student_flag": [False] * 10,
        "current_road_distance_km": [10.] * 9 + [np.nan],
        "integrated_road_distance_km": 10. + distance_change,
        "road_distance_change_km": distance_change,
        "additional_distance_burden_km": np.maximum(distance_change, 0),
        "current_analysis_road_duration_min": before,
        "integrated_analysis_road_duration_min": after,
        "road_duration_change_min": after - before,
        "additional_time_burden_min": np.maximum(after - before, 0),
        "current_over_30_min_flag": before > 30, "integrated_over_30_min_flag": after > 30,
        "newly_over_30_min_flag": (before <= 30) & (after > 30),
    })


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.work = prepare_detail(raw_fixture())

    def test_national_category_partition_and_exclusion(self):
        detail, table = impacts.analyze(raw_fixture())
        row = table.iloc[0]
        self.assertAlmostEqual(row.analysis_eligible_student_weight, 77.25)
        self.assertEqual(row.analysis_excluded_student_weight, 1000)
        self.assertAlmostEqual(sum(row[f"{c}_student_rate"] for c in ["increased", "decreased", "unchanged"]), 1)
        self.assertEqual(detail.iloc[-1].distance_change_category, "EXCLUDED")

    def test_changed_groups_have_independent_means_and_denominators(self):
        summary, distance, time, _ = distributions.analyze(self.work, "CHANGED")
        self.assertEqual(summary.distance_change_category.tolist(), ["INCREASED", "DECREASED"])
        for category, group in population_groups(self.work, "CHANGED"):
            row = summary.set_index("distance_change_category").loc[category]
            self.assertAlmostEqual(row.weighted_mean_distance_change_km,
                                   np.average(group.road_distance_change_km, weights=group.student_weight))
            for table in [distance, time]:
                selected = table.loc[table.distance_change_category.eq(category)]
                self.assertEqual(set(selected.denominator_student_weight), {group.student_weight.sum()})
                self.assertAlmostEqual(selected.student_rate.sum(), 1)
        self.assertGreater(summary.iloc[0].weighted_mean_distance_change_km, 0)
        self.assertLess(summary.iloc[1].weighted_mean_distance_change_km, 0)

    def test_all_population_options_and_no_movement_filter(self):
        expected = {"INCREASED": {"INCREASED"}, "DECREASED": {"DECREASED"},
                    "CHANGED": {"INCREASED", "DECREASED"},
                    "ALL_ELIGIBLE": {"INCREASED", "DECREASED", "UNCHANGED"}}
        for population, categories in expected.items():
            groups = list(population_groups(self.work, population))
            self.assertEqual({c for c, _ in groups}, categories)
            self.assertTrue(all(not g.empty for _, g in groups))
            self.assertFalse(any(g.demand_assignment_id.eq("A009").any() for _, g in groups))

    def test_changed_regional_statistics_are_separate(self):
        tables = regional.analyze(self.work, "CHANGED")
        national = tables["national_average.csv"]
        self.assertEqual(national.distance_change_category.tolist(), ["INCREASED", "DECREASED"])
        self.assertGreater(national.iloc[0].weighted_mean_distance_change_km, 0)
        self.assertLess(national.iloc[1].weighted_mean_distance_change_km, 0)
        self.assertTrue(tables["top5_above_national.csv"].empty)
        self.assertIn("UNKNOWN", tables["population_decline_metrics.csv"].current_school_region_group.values)

    def test_time_distribution_keeps_negative_zero_positive(self):
        group = self.work.iloc[:5].copy()
        group["road_duration_change_min"] = [-30., -2., 0., 2., 30.]
        table = distributions.make_distribution(group, "road_duration_change_min", distributions.TIME_CUTS)
        self.assertEqual(table.assignment_row_count.sum(), 5)
        self.assertEqual(table.loc[table.assignment_row_count.gt(0)].shape[0], 5)
        self.assertAlmostEqual(table.student_weight.sum(), group.student_weight.sum())

    def test_empty_groups_keep_schema_and_zero_weight(self):
        work = self.work.loc[self.work.distance_change_category.eq("UNCHANGED")]
        summary, distance, time, _ = distributions.analyze(work, "CHANGED")
        self.assertTrue(summary.student_weight.eq(0).all())
        self.assertTrue(summary.weighted_mean_distance_change_km.isna().all())
        self.assertTrue(distance.student_rate.isna().all())
        self.assertTrue(time.student_rate.isna().all())
        tables = regional.analyze(work, "CHANGED")
        self.assertTrue(tables["education_office_metrics.csv"].empty)

    def test_unknown_population_rejected(self):
        with self.assertRaises(ValueError):
            list(population_groups(self.work, "TIME_INCREASED"))


class InputValidationTests(unittest.TestCase):
    def test_unknown_region_never_becomes_non_decline(self):
        actual = classify_region(pd.Series(["인구감소지역", "인구감소관심지역", "미해당", None, "미확인", "", "UNKNOWN"]))
        self.assertEqual(actual.tolist(), ["DECLINE", "NON_DECLINE", "NON_DECLINE", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"])

    def test_duplicate_assignments_rejected(self):
        raw = raw_fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_detail(pd.concat([raw, raw.iloc[:1]]))

    def test_invalid_eligible_metric_rejected(self):
        raw = raw_fixture()
        raw.loc[0, "current_road_distance_km"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            prepare_detail(raw)

    def test_invalid_weight_rejected(self):
        for value in [0, -1, np.inf, np.nan]:
            raw = raw_fixture()
            raw.loc[0, "student_weight"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "student_weight"):
                prepare_detail(raw)


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.work = prepare_detail(raw_fixture())

    def test_exact_time_boundaries(self):
        values = pd.Series([0, 9.9999, 10, 19.9999, 20, 29.9999, 30, np.nextafter(30., np.inf)])
        self.assertEqual(thresholds.time_categories(values).tolist(),
                         ["[0,10)", "[0,10)", "[10,20)", "[10,20)", "[20,30]", "[20,30]", "[20,30]", "(30,inf)"])

    def test_all_eligible_threshold_counts_and_fractional_weight(self):
        tables = thresholds.analyze(self.work, expected_newly_schools=2)
        national = tables["thresholds_national.csv"].set_index("condition")
        self.assertEqual(national.loc["current_over_30", "student_weight"], 12)
        self.assertEqual(national.loc["integrated_over_30", "student_weight"], 10.25)
        self.assertEqual(national.loc["newly_over_30", "student_weight"], 3.25)
        self.assertEqual(national.loc["resolved_over_30", "student_weight"], 5)
        self.assertEqual(national.loc["newly_over_30", "retained_school_count"], 2)
        self.assertAlmostEqual(national.loc["newly_over_30", "student_rate"], 3.25 / 77.25)
        self.assertTrue(tables["validation_summary.csv"].status.eq("PASS").all())

    def test_current_region_overlap_and_disjoint_retained_locations(self):
        tables = thresholds.analyze(self.work)
        current = tables["thresholds_by_current_school_region.csv"]
        before = current.loc[current.condition.eq("current_over_30")]
        self.assertEqual(before.current_school_region_affected_retained_school_count.sum(), 2)
        national = tables["thresholds_national.csv"].set_index("condition")
        self.assertEqual(national.loc["current_over_30", "retained_school_count"], 1)
        locations = tables["thresholds_by_retained_school_location.csv"]
        new = locations.loc[locations.condition.eq("newly_over_30")]
        self.assertEqual(new.retained_location_unique_school_count.sum(), 2)
        self.assertEqual(new.loc[new.retained_school_location_region_group.eq("UNKNOWN"), "retained_location_unique_school_count"].iloc[0], 1)
        overlaps = tables["current_school_region_entity_overlaps.csv"]
        self.assertTrue(((overlaps.condition == "current_over_30") & (overlaps.entity_type == "retained_school")).any())

    def test_census_point_denominator_uses_all_assignments(self):
        table = thresholds.analyze(self.work)["census_newly_over_30.csv"].set_index("census_id")
        self.assertEqual(table.loc["P000", "eligible_student_weight"], 5)
        self.assertEqual(table.loc["P000", "newly_over_30_student_weight"], 3)
        self.assertAlmostEqual(table.loc["P000", "newly_over_30_student_rate"], .6)
        self.assertFalse(table.index.duplicated().any())

    def test_empty_threshold_outputs_have_schema(self):
        raw = raw_fixture()
        for column in ["current_analysis_road_duration_min", "integrated_analysis_road_duration_min"]:
            raw.loc[raw.commute_analysis_eligible_flag, column] = 5
        for column in ["road_duration_change_min", "additional_time_burden_min"]:
            raw.loc[raw.commute_analysis_eligible_flag, column] = 0
        for column in ["current_over_30_min_flag", "integrated_over_30_min_flag", "newly_over_30_min_flag"]:
            raw[column] = False
        tables = thresholds.analyze(prepare_detail(raw))
        self.assertTrue(tables["retained_school_threshold_details.csv"].empty)
        self.assertTrue(tables["census_newly_over_30.csv"].empty)
        self.assertTrue(tables["validation_summary.csv"].status.eq("PASS").all())

    def test_regression_difference_is_explicit_review(self):
        validation = thresholds.analyze(self.work, expected_newly_schools=68)["validation_summary.csv"]
        self.assertEqual(validation.iloc[-1].status, "REVIEW")
        self.assertEqual(validation.iloc[-1].actual, 2)

    def test_inconsistent_frozen_flag_rejected(self):
        self.work.loc[0, "integrated_over_30_min_flag"] = True
        with self.assertRaisesRegex(ValueError, "frozen_flag_mismatch"):
            thresholds.analyze(self.work)

    def test_multiple_coordinates_for_same_census_rejected(self):
        self.work.loc[0, "origin_lon"] += .1
        with self.assertRaisesRegex(ValueError, "Multiple representative"):
            thresholds.analyze(self.work)

    def test_conflicting_retained_location_rejected(self):
        self.work.loc[0, "retained_school_location_region_group"] = "DECLINE"
        with self.assertRaisesRegex(ValueError, "Inconsistent retained"):
            thresholds.analyze(self.work)


class OutputProtectionTests(unittest.TestCase):
    def test_csv_and_png_paths_are_namespaced_for_every_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for analysis in ["distance_change_distributions", "regional_distance_burden"]:
                for population in ["INCREASED", "DECREASED", "CHANGED", "ALL_ELIGIBLE"]:
                    args = argparse.Namespace(project_root=Path(tmp), output_dir=Path(tmp), population=population)
                    path = output_path(args, analysis)
                    paths.append(path)
                    with OutputBundle(path, False) as bundle:
                        bundle.csv("same.csv", pd.DataFrame({"value": [population]}))
                        (bundle.path / "same.png").write_bytes(population.encode())
            self.assertEqual(len(set(paths)), 8)
            for path in paths:
                self.assertEqual((path / "same.png").read_text(), path.name)
            selected = paths[0]
            with OutputBundle(selected, True) as bundle:
                bundle.csv("same.csv", pd.DataFrame({"value": ["replacement"]}))
                (bundle.path / "same.png").write_bytes(b"replacement")
            self.assertEqual((selected / "same.png").read_text(), "replacement")
            for path in paths[1:]:
                self.assertEqual((path / "same.png").read_text(), path.name)

    def test_overwrite_requires_flag_and_failed_bundle_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CHANGED"
            with OutputBundle(path, False) as bundle:
                bundle.csv("summary.csv", pd.DataFrame({"value": [1]}))
                (bundle.path / "plot.png").write_bytes(b"old image")
            csv_before = (path / "summary.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                with OutputBundle(path, False):
                    pass
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                with OutputBundle(path, True) as bundle:
                    bundle.csv("summary.csv", pd.DataFrame({"value": [2]}))
                    raise RuntimeError("render failed")
            self.assertEqual((path / "summary.csv").read_bytes(), csv_before)
            self.assertEqual((path / "plot.png").read_bytes(), b"old image")


if __name__ == "__main__":
    unittest.main()

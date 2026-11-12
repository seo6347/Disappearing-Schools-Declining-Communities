"""Stage 5: distance-first impact analysis of the already frozen Stage 4 scenario.

Distance is the main outcome; time remains supplementary. Neither outcome can
change the retained school. Directions distances and times describe the stored
car direct routes, not actual school-bus operation or observed arrival times.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from final_scenario_io import SCENARIO_ID, load_final_scenario, parse_bool, require_columns

DISTANCE_TOLERANCE_KM = 1e-9


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    return float(np.average(values.loc[valid], weights=weights.loc[valid])) if valid.any() else np.nan


def weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    value_array = values.loc[valid].to_numpy(dtype=float)
    weight_array = weights.loc[valid].to_numpy(dtype=float)
    order = np.argsort(value_array, kind="stable")
    value_array, weight_array = value_array[order], weight_array[order]
    position = int(np.searchsorted(np.cumsum(weight_array), quantile * weight_array.sum(), side="left"))
    return float(value_array[min(position, len(value_array) - 1)])


def distance_categories(change: pd.Series, eligible: pd.Series) -> pd.Series:
    if not np.isfinite(change.loc[eligible]).all():
        raise ValueError("Eligible distance change contains missing or non-finite values")
    return pd.Series(np.select([~eligible, change.gt(DISTANCE_TOLERANCE_KM),
                               change.lt(-DISTANCE_TOLERANCE_KM)],
                              ["EXCLUDED", "INCREASED", "DECREASED"], default="UNCHANGED"), index=change.index)


def analyze(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["student_weight", "moved_student_flag", "commute_analysis_eligible_flag",
                "current_road_distance_km", "integrated_road_distance_km", "road_distance_change_km",
                "additional_distance_burden_km", "current_analysis_road_duration_min",
                "integrated_analysis_road_duration_min", "road_duration_change_min", "additional_time_burden_min",
                "current_over_30_min_flag", "integrated_over_30_min_flag", "newly_over_30_min_flag",
                "time_increase_flag", "time_decrease_flag"]
    require_columns(detail, required, "final assignments")
    work = detail.copy()
    for column in [c for c in required if not c.endswith("_flag")]:
        work[column] = pd.to_numeric(work[column], errors="raise")
    for column in [c for c in required if c.endswith("_flag")]:
        work[column] = parse_bool(work[column])
    if not np.isfinite(work.student_weight).all() or work.student_weight.le(0).any():
        raise ValueError("Invalid student weight")
    eligible = work.commute_analysis_eligible_flag
    metric_columns = [c for c in required if not c.endswith("_flag")]
    if not np.isfinite(work.loc[eligible, metric_columns].to_numpy(float)).all():
        raise ValueError("Eligible analysis metrics must be finite")
    result = detail.copy()
    result["distance_change_category"] = distance_categories(work.road_distance_change_km, eligible)
    rows = []
    # These are evaluation populations, never different school-selection scenarios.
    for population_id in ["ALL_STUDENTS", "MOVED_STUDENTS"]:
        population = work if population_id == "ALL_STUDENTS" else work.loc[work.moved_student_flag]
        usable = population.loc[population.commute_analysis_eligible_flag]
        weights = usable.student_weight
        total = float(population.student_weight.sum())
        available = float(weights.sum())
        row = {"scenario_id": SCENARIO_ID, "evaluation_population": population_id,
               "selected_cluster_count": detail.final_cluster_id.nunique(),
               "demand_assignment_row_count": len(population), "unique_census_point_count": population.census_id.nunique(),
               "student_weight_total": total, "analysis_eligible_student_weight": available,
               "analysis_excluded_student_weight": float(population.loc[~population.commute_analysis_eligible_flag, "student_weight"].sum()),
               "analysis_student_coverage_rate": available / total if total > 0 else np.nan}
        for label, column in {
            "current_distance_km": "current_road_distance_km", "integrated_distance_km": "integrated_road_distance_km",
            "distance_change_km": "road_distance_change_km", "additional_distance_burden_km": "additional_distance_burden_km",
            "current_duration_min": "current_analysis_road_duration_min", "integrated_duration_min": "integrated_analysis_road_duration_min",
            "duration_change_min": "road_duration_change_min", "additional_time_burden_min": "additional_time_burden_min",
        }.items():
            row[f"weighted_mean_{label}"] = weighted_mean(usable[column], weights)
        for label, column in [("distance_change_km", "road_distance_change_km"), ("duration_change_min", "road_duration_change_min")]:
            row[f"weighted_p50_{label}"] = weighted_quantile(usable[column], weights, 0.5)
            row[f"weighted_p90_{label}"] = weighted_quantile(usable[column], weights, 0.9)
            row[f"max_{label}"] = float(usable[column].max()) if len(usable) else np.nan
            row[f"total_{label}_student_weight"] = float((usable[column] * weights).sum())
        for label, flag in [("current_over_30", "current_over_30_min_flag"),
                            ("integrated_over_30", "integrated_over_30_min_flag"),
                            ("newly_over_30", "newly_over_30_min_flag"),
                            ("time_increase", "time_increase_flag"), ("time_decrease", "time_decrease_flag")]:
            weight = float(usable.loc[usable[flag], "student_weight"].sum())
            row[f"{label}_student_weight"] = weight
            row[f"{label}_student_rate"] = weight / available if available > 0 else np.nan
        rows.append(row)
    return result, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().project_root.resolve()
    _, _, detail = load_final_scenario(root)
    result, summary = analyze(detail)
    output = root / "data/03_outputs"
    result.to_csv(output / "final_commute_demand_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "final_commute_national_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Fixed-scenario analysis: {len(result)} assignments, {len(summary)} evaluation populations")


if __name__ == "__main__":
    main()

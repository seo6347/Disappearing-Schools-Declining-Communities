"""Stage 5-4: all-eligible commute-time thresholds, without distance-change filtering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from final_scenario_io import parse_bool, require_columns
from stage5_commute_common import (REGIONS, OutputBundle, argument_parser, check,
    load_analysis, output_path, partition_checks, ratio, weighted_mean)

CONDITIONS = ("current_over_30", "integrated_over_30", "newly_over_30", "resolved_over_30")
ENTITIES = {"retained_school": "integrated_school_key", "cluster": "final_cluster_id",
            "education_office": "current_school_edu_office"}
TIME_BINS = ("[0,10)", "[10,20)", "[20,30]", "(30,inf)")


def threshold_masks(group: pd.DataFrame) -> dict[str, pd.Series]:
    before = group.current_analysis_road_duration_min
    after = group.integrated_analysis_road_duration_min
    return {"current_over_30": before.gt(30), "integrated_over_30": after.gt(30),
            "newly_over_30": before.le(30) & after.gt(30),
            "resolved_over_30": before.gt(30) & after.le(30)}


def time_categories(values: pd.Series) -> pd.Series:
    if not np.isfinite(values).all() or values.lt(0).any():
        raise ValueError("Time distribution requires finite, non-negative eligible durations")
    return pd.Series(np.select([values.lt(10), values.lt(20), values.le(30)],
                               TIME_BINS[:3], default=TIME_BINS[3]), index=values.index)


def entity_ids(group: pd.DataFrame, column: str) -> set:
    return set(group[column].dropna()) - ({"UNKNOWN"} if column.endswith("edu_office") else set())


def event_stats(group: pd.DataFrame, affected: pd.DataFrame) -> dict:
    denominator = float(group.student_weight.sum())
    weight = float(affected.student_weight.sum())
    return {"assignment_row_count": len(affected), "student_weight": weight,
            "denominator_student_weight": denominator, "student_rate": ratio(weight, denominator),
            **{f"{kind}_count": len(entity_ids(affected, column)) for kind, column in ENTITIES.items()}}


def analyze(work: pd.DataFrame, expected_newly_schools: int | None = None) -> dict[str, pd.DataFrame]:
    require_columns(work, ["integrated_school_name", "origin_lon", "origin_lat",
                          "current_over_30_min_flag", "integrated_over_30_min_flag", "newly_over_30_min_flag"],
                    "threshold assignments")
    eligible = work.loc[work.commute_analysis_eligible_flag].copy()
    for column in ["origin_lon", "origin_lat"]:
        eligible[column] = pd.to_numeric(eligible[column], errors="raise")
        if not np.isfinite(eligible[column]).all():
            raise ValueError(f"Invalid representative point coordinate: {column}")
    if not (eligible.origin_lon.between(-180, 180).all() and eligible.origin_lat.between(-90, 90).all()):
        raise ValueError("Representative point coordinates out of longitude/latitude range")
    coords = eligible.groupby("census_id")[["origin_lon", "origin_lat"]].nunique(dropna=False)
    if coords.gt(1).any().any():
        raise ValueError("Multiple representative coordinates for a census_id")
    metadata = ["final_cluster_id", "integrated_school_name", "integrated_school_edu_office",
                "retained_school_location_region_group"]
    if eligible.groupby("integrated_school_key")[metadata].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Inconsistent retained school metadata or location")
    if eligible.groupby("final_cluster_id").integrated_school_key.nunique().gt(1).any():
        raise ValueError("More than one retained school per frozen cluster")
    masks = threshold_masks(eligible)
    validation = partition_checks(work)
    for condition in CONDITIONS[:3]:
        validation.append(check(f"{condition}_frozen_flag_mismatch", int((masks[condition] != parse_bool(eligible[f"{condition}_min_flag"])).sum()), 0))

    national_rows, student_region_rows, location_rows, overlaps = [], [], [], []
    total_weight = float(eligible.student_weight.sum())
    # ALL_ELIGIBLE also exposes potential cross-region membership, even when a threshold has no overlap.
    for condition in ("ALL_ELIGIBLE", *CONDITIONS):
        affected = eligible if condition == "ALL_ELIGIBLE" else eligible.loc[masks[condition]]
        if condition != "ALL_ELIGIBLE":
            national_rows.append({"condition": condition, **event_stats(eligible, affected)})
        regional_weight = 0.
        for region in REGIONS:
            cohort = eligible.loc[eligible.current_school_region_group.eq(region)]
            selected = affected.loc[affected.current_school_region_group.eq(region)]
            stats = event_stats(cohort, selected)
            regional_weight += stats["student_weight"]
            if condition != "ALL_ELIGIBLE":
                counts = {f"current_school_region_affected_{kind}_count": stats.pop(f"{kind}_count") for kind in ENTITIES}
                student_region_rows.append({"condition": condition, "current_school_region_group": region,
                                            **stats, **counts})
        validation.append(check(f"{condition}_current_school_region_student_weight", regional_weight, affected.student_weight.sum()))
        for kind, column in ENTITIES.items():
            memberships = affected.loc[affected[column].isin(entity_ids(affected, column))].groupby(column).current_school_region_group.nunique()
            extra_memberships = int((memberships - 1).sum())
            region_count_sum = sum(len(entity_ids(affected.loc[affected.current_school_region_group.eq(r)], column)) for r in REGIONS)
            validation.append(check(f"{condition}_{kind}_regional_union", region_count_sum - extra_memberships, len(entity_ids(affected, column))))
            for identifier in memberships.loc[memberships.gt(1)].index:
                regions = sorted(set(affected.loc[affected[column].eq(identifier), "current_school_region_group"]))
                overlaps.append({"condition": condition, "entity_type": kind, "entity_id": identifier,
                                 "current_school_region_groups": "|".join(regions),
                                 "region_membership_count": len(regions), "extra_membership_count": len(regions) - 1})
        for region in REGIONS:
            cohort = eligible.loc[eligible.retained_school_location_region_group.eq(region)]
            selected = affected.loc[affected.retained_school_location_region_group.eq(region)]
            if condition != "ALL_ELIGIBLE":
                location_rows.append({"condition": condition, "retained_school_location_region_group": region,
                    "retained_location_unique_school_count": selected.integrated_school_key.nunique(),
                    "retained_location_unique_cluster_count": selected.final_cluster_id.nunique(),
                    "retained_location_education_office_count": len(entity_ids(selected, "integrated_school_edu_office")),
                    "retained_location_condition_student_weight": float(selected.student_weight.sum()),
                    "retained_location_eligible_student_weight": float(cohort.student_weight.sum())})
        for kind, column in [("school", "integrated_school_key"), ("cluster", "final_cluster_id")]:
            location_sets = [entity_ids(affected.loc[affected.retained_school_location_region_group.eq(r)], column) for r in REGIONS]
            validation.append(check(f"{condition}_retained_location_{kind}_additive", sum(map(len, location_sets)), len(entity_ids(affected, column))))
        location_office_sets = [entity_ids(affected.loc[affected.retained_school_location_region_group.eq(r)],
                                          "integrated_school_edu_office") for r in REGIONS]
        validation.append(check(f"{condition}_retained_location_office_union", len(set().union(*location_office_sets)),
                                len(entity_ids(affected, "integrated_school_edu_office"))))
        validation.append(check(f"{condition}_retained_location_student_weight",
            sum(affected.loc[affected.retained_school_location_region_group.eq(r), "student_weight"].sum() for r in REGIONS), affected.student_weight.sum()))

    national = pd.DataFrame(national_rows)
    weights = national.set_index("condition").student_weight
    validation.append(check("threshold_transition_weight_identity", weights.integrated_over_30,
                            weights.current_over_30 + weights.newly_over_30 - weights.resolved_over_30))

    school_rows, school_region_rows = [], []
    impacted_schools = set(eligible.loc[masks["current_over_30"] | masks["integrated_over_30"], "integrated_school_key"])
    school_columns = ["retained_school_key", "retained_school_name", "retained_school_edu_office", "final_cluster_id",
                      "retained_school_location_population_decline_type", "retained_school_location_region_group",
                      "eligible_student_weight", "assignment_row_count",
                      *[f"{c}_{suffix}" for c in CONDITIONS for suffix in ["student_weight", "student_rate"]]]
    school_region_columns = ["retained_school_key", "retained_school_name", "retained_school_edu_office", "final_cluster_id",
                            "retained_school_location_region_group", "current_school_region_group", "condition",
                            "student_weight", "denominator_student_weight", "student_rate", "assignment_row_count"]
    for key, group in eligible.loc[eligible.integrated_school_key.isin(impacted_schools)].groupby("integrated_school_key", sort=True):
        first = group.iloc[0]
        identity = {"retained_school_key": key, "retained_school_name": first.integrated_school_name,
                    "retained_school_edu_office": first.integrated_school_edu_office,
                    "final_cluster_id": first.final_cluster_id,
                    "retained_school_location_region_group": first.retained_school_location_region_group}
        denominator = float(group.student_weight.sum())
        row = {**identity, "retained_school_location_population_decline_type": first.integrated_school_population_decline_type,
               "eligible_student_weight": denominator, "assignment_row_count": len(group)}
        for condition, mask in threshold_masks(group).items():
            weight = float(group.loc[mask, "student_weight"].sum())
            row[f"{condition}_student_weight"] = weight
            row[f"{condition}_student_rate"] = ratio(weight, denominator)
        school_rows.append(row)
        for region, cohort in group.groupby("current_school_region_group", sort=True):
            for condition, mask in threshold_masks(cohort).items():
                selected = cohort.loc[mask]
                if not len(selected):
                    continue
                stats = event_stats(cohort, selected)
                school_region_rows.append({**identity, "current_school_region_group": region, "condition": condition,
                                          **{k: v for k, v in stats.items() if not k.endswith("_count") or k == "assignment_row_count"}})
    school_details = pd.DataFrame(school_rows, columns=school_columns)
    school_regions = pd.DataFrame(school_region_rows, columns=school_region_columns)
    for condition in CONDITIONS:
        expected = float(weights[condition])
        validation.append(check(f"{condition}_school_detail_student_weight", school_details[f"{condition}_student_weight"].sum(), expected))
        validation.append(check(f"{condition}_school_region_detail_student_weight", school_regions.loc[school_regions.condition.eq(condition), "student_weight"].sum(), expected))
        validation.append(check(f"{condition}_school_detail_unique_count", int(school_details[f"{condition}_student_weight"].gt(0).sum()),
                                int(national.loc[national.condition.eq(condition), "retained_school_count"].iloc[0])))
    validation.append(check("school_detail_duplicate_keys", school_details.retained_school_key.duplicated().sum(), 0))
    validation.append(check("school_region_detail_duplicate_keys", school_regions.duplicated(["retained_school_key", "current_school_region_group", "condition"]).sum(), 0))

    census_rows = []
    census_columns = ["census_id", "origin_lon", "origin_lat", "eligible_student_weight", "newly_over_30_student_weight",
                      "newly_over_30_student_rate", "newly_over_30_assignment_row_count", "newly_over_30_retained_school_count",
                      "newly_over_30_cluster_count", "newly_over_30_current_school_education_office_count",
                      "newly_over_30_weighted_mean_current_duration_min", "newly_over_30_weighted_mean_integrated_duration_min",
                      *[f"newly_over_30_current_school_region_{r}_student_weight" for r in REGIONS]]
    census_denominators = eligible.groupby("census_id").student_weight.sum()
    newly = eligible.loc[masks["newly_over_30"]]
    for census, group in newly.groupby("census_id", sort=True):
        first = group.iloc[0]
        weight = float(group.student_weight.sum())
        census_rows.append({"census_id": census, "origin_lon": first.origin_lon, "origin_lat": first.origin_lat,
            "eligible_student_weight": float(census_denominators[census]), "newly_over_30_student_weight": weight,
            "newly_over_30_student_rate": ratio(weight, census_denominators[census]),
            "newly_over_30_assignment_row_count": len(group), "newly_over_30_retained_school_count": group.integrated_school_key.nunique(),
            "newly_over_30_cluster_count": group.final_cluster_id.nunique(),
            "newly_over_30_current_school_education_office_count": len(entity_ids(group, "current_school_edu_office")),
            "newly_over_30_weighted_mean_current_duration_min": weighted_mean(group.current_analysis_road_duration_min, group.student_weight),
            "newly_over_30_weighted_mean_integrated_duration_min": weighted_mean(group.integrated_analysis_road_duration_min, group.student_weight),
            **{f"newly_over_30_current_school_region_{r}_student_weight": float(group.loc[group.current_school_region_group.eq(r), "student_weight"].sum()) for r in REGIONS}})
    census_table = pd.DataFrame(census_rows, columns=census_columns)
    validation.append(check("census_newly_over_30_student_weight", census_table.newly_over_30_student_weight.sum(), weights.newly_over_30))
    validation.append(check("census_duplicate_keys", census_table.census_id.duplicated().sum(), 0))
    validation.append(check("census_newly_over_30_region_weight", sum(census_table[f"newly_over_30_current_school_region_{r}_student_weight"].sum() for r in REGIONS), weights.newly_over_30))

    distribution_rows = []
    for scope, region, group in [("NATIONAL", "ALL", eligible), *[("CURRENT_SCHOOL_REGION", r, eligible.loc[eligible.current_school_region_group.eq(r)]) for r in REGIONS]]:
        bins = time_categories(group.integrated_analysis_road_duration_min)
        denominator = float(group.student_weight.sum())
        for order, label in enumerate(TIME_BINS):
            selected = group.loc[bins.eq(label)]
            distribution_rows.append({"scope": scope, "current_school_region_group": region, "bin_order": order,
                "bin_label": label, **event_stats(group, selected)})
        validation.append(check(f"post_time_distribution_{region}_weight", sum(r["student_weight"] for r in distribution_rows if r["current_school_region_group"] == region), denominator))
        validation.append(check(f"post_time_distribution_{region}_rows", sum(r["assignment_row_count"] for r in distribution_rows if r["current_school_region_group"] == region), len(group)))
    validation.append(check("post_time_distribution_over_30_weight", sum(r['student_weight'] for r in distribution_rows if r['scope'] == 'NATIONAL' and r['bin_label'] == TIME_BINS[3]), weights.integrated_over_30))
    if expected_newly_schools is not None:
        validation.append(check("regression_newly_over_30_retained_school_count",
            int(national.loc[national.condition.eq("newly_over_30"), "retained_school_count"].iloc[0]), expected_newly_schools, review=True))
    return {"thresholds_national.csv": national,
            "thresholds_by_current_school_region.csv": pd.DataFrame(student_region_rows),
            "thresholds_by_retained_school_location.csv": pd.DataFrame(location_rows),
            "retained_school_threshold_details.csv": school_details,
            "retained_school_current_region_details.csv": school_regions,
            "census_newly_over_30.csv": census_table,
            "post_commute_time_distribution.csv": pd.DataFrame(distribution_rows),
            "current_school_region_entity_overlaps.csv": pd.DataFrame(overlaps, columns=["condition", "entity_type", "entity_id", "current_school_region_groups", "region_membership_count", "extra_membership_count"]),
            "validation_summary.csv": pd.DataFrame(validation)}


def main() -> None:
    args = argument_parser(__doc__).parse_args()
    work, hashes = load_analysis(args)
    tables = analyze(work, expected_newly_schools=68)
    destination = output_path(args, "commute_time_thresholds")
    with OutputBundle(destination, args.overwrite) as output:
        for name, table in tables.items():
            output.csv(name, table)
        output.manifest(analysis="commute_time_thresholds", input_hashes=hashes, population="ALL_ELIGIBLE",
            aggregation_units={
                "thresholds_national.csv": ["condition"],
                "thresholds_by_current_school_region.csv": ["condition", "current_school_region_group"],
                "thresholds_by_retained_school_location.csv": ["condition", "retained_school_location_region_group"],
                "retained_school_threshold_details.csv": ["retained_school_key"],
                "retained_school_current_region_details.csv": ["retained_school_key", "current_school_region_group", "condition"],
                "census_newly_over_30.csv": ["census_id"],
                "post_commute_time_distribution.csv": ["scope", "current_school_region_group", "bin_order"],
                "current_school_region_entity_overlaps.csv": ["condition", "entity_type", "entity_id"]},
            threshold_minutes=30, threshold_comparison="strictly greater than 30; no rounding or tolerance",
            post_time_bins=list(TIME_BINS),
            condition_definitions={"current_over_30": "before > 30", "integrated_over_30": "after > 30",
                "newly_over_30": "before <= 30 and after > 30", "resolved_over_30": "before > 30 and after <= 30"},
            rate_denominators="All eligible students nationally or within current-school region; school and census rates use their own eligible cohorts.",
            unique_count_note="Current-school-region counts may overlap: reconcile ID unions, not sums. Retained-location school/cluster counts are disjoint, including UNKNOWN. Office counts across regions and across time bins may overlap.",
            retained_location_weight_note="Weights in the retained-location table describe students assigned to schools at that location; student regional classification remains current-school-based in student-region outputs.",
            map_note="Only census points with newly-over-30 students; one row per census_id using frozen origin coordinates. Denominator includes all eligible assignments at that point.",
            regression_expected_newly_over_30_retained_schools=68)
    review = tables["validation_summary.csv"].loc[lambda f: f.status.ne("PASS")]
    if len(review):
        print("[REVIEW] Regression differs; inspect frozen flags, eligibility, strict boundary and retained IDs:")
        print(review.to_string(index=False))
    print(tables["thresholds_national.csv"].to_string(index=False))
    print(f"[DONE] ALL_ELIGIBLE; {destination}")


if __name__ == "__main__":
    main()

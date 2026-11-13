"""Stage 5-1: distance-first impacts of the frozen scenario, including unchanged students."""
from __future__ import annotations

import pandas as pd

from stage5_commute_common import (CATEGORIES, DISTANCE_TOLERANCE_KM, OutputBundle,
    argument_parser, check, distance_categories, load_analysis, output_path,
    partition_checks, prepare_detail, ratio, summarize, weighted_mean, weighted_quantile)


def analyze(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = prepare_detail(detail)
    eligible = work.loc[work.commute_analysis_eligible_flag]
    total = float(work.student_weight.sum())
    available = float(eligible.student_weight.sum())
    row = {"population": "ALL_ELIGIBLE", **summarize(eligible),
           "student_weight_total": total, "analysis_eligible_student_weight": available,
           "analysis_excluded_student_weight": float(work.loc[~work.commute_analysis_eligible_flag, "student_weight"].sum()),
           "analysis_student_coverage_rate": ratio(available, total)}
    for category in CATEGORIES:
        weight = float(eligible.loc[eligible.distance_change_category.eq(category), "student_weight"].sum())
        row[f"{category.lower()}_student_weight"] = weight
        row[f"{category.lower()}_student_rate"] = ratio(weight, available)
    return work, pd.DataFrame([row])


def main() -> None:
    args = argument_parser(__doc__).parse_args()
    work, hashes = load_analysis(args)
    result, national = analyze(work)
    eligible_weight = float(result.loc[result.commute_analysis_eligible_flag, "student_weight"].sum())
    groups = []
    for category in CATEGORIES:
        group = result.loc[result.distance_change_category.eq(category)]
        groups.append({"population": "ALL_ELIGIBLE", "distance_change_category": category,
                       **summarize(group), "all_eligible_student_weight": eligible_weight,
                       "share_of_all_eligible_students": ratio(group.student_weight.sum(), eligible_weight)})
    validation = partition_checks(result)
    validation.append(check("category_shares_sum", sum(row["student_weight"] for row in groups), eligible_weight))
    destination = output_path(args, "commute_impacts")
    with OutputBundle(destination, args.overwrite) as output:
        output.csv("demand_results.csv", result.drop(columns=["scenario_id"], errors="ignore"))
        output.csv("national_summary.csv", national)
        output.csv("distance_group_summary.csv", pd.DataFrame(groups))
        output.csv("validation_summary.csv", pd.DataFrame(validation))
        output.manifest(analysis="commute_impacts", input_hashes=hashes, population="ALL_ELIGIBLE",
                        aggregation_units={"demand_results.csv": ["demand_assignment_id"],
                                           "national_summary.csv": ["population"],
                                           "distance_group_summary.csv": ["distance_change_category"]},
                        rate_denominators="Category shares: all eligible students; subgroup means: that subgroup only.")
    print(f"[DONE] {len(result)} frozen assignments; eligible weight={eligible_weight:.10f}; {destination}")


if __name__ == "__main__":
    main()

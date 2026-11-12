"""Stage 4: fix one retained school per cluster, explicitly by education office.

Reuses 04-4's ALL_STUDENTS weighted_mean_additional_time_burden_min:
sum(weight * max(after_time - before_time, 0)) / sum(eligible weight).
Students with non-increasing time remain in the denominator. No distance-based
classification, national impact aggregation, or lower/upper-bound selection occurs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from final_scenario_io import (SCENARIO_ID, SCENARIO_FILE, SCHOOL_FILE, DEMAND_FILE,
                               MANIFEST_FILE, parse_bool, require_columns, sha256_file,
                               validate_assignments)

FLOAT_TOLERANCE = 1e-9
WEIGHT_TOLERANCE = 1e-6
METRIC = "weighted_mean_additional_time_burden_min"
RANK_COLUMNS = [METRIC, "weighted_mean_duration_change_min",
                "weighted_p90_duration_change_min", "max_duration_change_min",
                "integrated_school_key"]
OPTION_FILE = "commute_time_option_summary.csv"
DETAIL_FILE = "commute_time_demand_option.csv"
VALIDATION_FILE = "final_integration_validation_summary.csv"


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result


def select_retained_school(options: pd.DataFrame) -> pd.Series:
    """Use the unchanged READY-first policy, then the exact declared ordering.

    A numerical tolerance is not an ordering equivalence (it is non-transitive).
    Exact stored numeric equality defines ties, in both sorting and tie counts.
    FLOAT_TOLERANCE is reserved for formula validation, not score rounding.
    """
    ready = options.option_analysis_status.eq("READY")
    if ready.any():
        pool = options.loc[ready].copy()
        pool_status = "READY_ONLY"
    else:
        pool = options.loc[options.option_analysis_status.eq("PARTIAL_ROUTE_EXCLUSION")].copy()
        pool_status = "PARTIAL_ONLY"
    if pool.empty or not np.isfinite(pool[RANK_COLUMNS[:-1]].to_numpy(float)).all():
        raise ValueError("No comparable option, or a missing/non-finite selection metric")
    if pool.integrated_school_key.isna().any() or pool.integrated_school_key.duplicated().any():
        raise ValueError("Invalid retained-school key in comparison pool")
    if (pool[METRIC] < 0).any():
        raise ValueError("Additional time burden cannot be negative")
    selected = pool.sort_values(RANK_COLUMNS, kind="stable").iloc[0].copy()
    selected["selection_pool_status"] = pool_status
    selected["available_option_count"] = len(options)
    selected["compared_option_count"] = len(pool)
    selected["primary_metric_tie_count"] = int(pool[METRIC].eq(selected[METRIC]).sum())
    selected["selection_metric"] = METRIC
    selected["selection_metric_value"] = selected[METRIC]
    selected["selection_basis"] = "ALL_STUDENTS"
    return selected


def validate_inputs(clusters: pd.DataFrame, stage3: pd.DataFrame, master: pd.DataFrame,
                    options: pd.DataFrame, detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_columns(clusters, ["selected_cluster_flag", "final_cluster_id", "candidate_id",
                              "capacity_scenario_id", "school_edu_office"] +
                    [f"school_{i}_{suffix}" for i in (1, 2, 3) for suffix in ("key", "name")], "Stage 3 clusters")
    require_columns(stage3, ["capacity_option_id", "candidate_id", "final_cluster_id", "school_edu_office",
                            "capacity_scenario_id", "retained_school_key", "final_road_eligible",
                            "closed_school_1_key", "closed_school_2_key"], "Stage 3 options")
    require_columns(master, ["학교고유ID", "학교명", "교육지원청", "시도교육청"], "school master")
    require_columns(options, ["final_cluster_id", "candidate_id", "capacity_scenario_id", "integration_option_id",
                             "integrated_school_key", "integrated_school_name", "integrated_school_edu_office",
                             "option_analysis_status", "student_weight_total", "analysis_eligible_student_weight",
                             "analysis_excluded_student_weight", "analysis_student_coverage_rate"] + RANK_COLUMNS[:-1], OPTION_FILE)
    require_columns(detail, ["integration_option_id", "final_cluster_id", "candidate_id", "capacity_scenario_id",
                            "demand_assignment_id", "current_school_key", "integrated_school_key", "student_weight",
                            "current_school_edu_office", "integrated_school_edu_office", "moved_student_flag",
                            "commute_analysis_eligible_flag"], DETAIL_FILE)
    selected = clusters.loc[parse_bool(clusters.selected_cluster_flag)].copy()
    if len(selected) != 720 or selected.final_cluster_id.isna().any() or selected.final_cluster_id.duplicated().any():
        raise ValueError("Stage 3 must contain exactly 720 selected unique clusters")
    if selected.candidate_id.duplicated().any():
        raise ValueError("Duplicate selected candidate")
    parts = []
    for i in (1, 2, 3):
        part = selected[["final_cluster_id", "candidate_id", "school_edu_office", f"school_{i}_key", f"school_{i}_name"]]
        parts.append(part.rename(columns={f"school_{i}_key": "school_key", f"school_{i}_name": "school_name"}))
    membership = pd.concat(parts, ignore_index=True)
    if membership.school_key.isna().any() or membership.school_key.duplicated().any() or len(membership) != 2160:
        raise ValueError("Clusters must contain three distinct schools, with no cross-cluster overlap")
    if master["학교고유ID"].isna().any() or master["학교고유ID"].duplicated().any():
        raise ValueError("School master keys are not unique")
    # The existing canonical master office name is the office identifier. It must
    # map to only one provincial education authority, never to homonymous offices.
    if master.groupby("교육지원청")["시도교육청"].nunique().gt(1).any():
        raise ValueError("Ambiguous education-office identifier in school master")
    office_by_school = master.set_index("학교고유ID")["교육지원청"]
    if membership.school_edu_office.isna().any() or not membership.school_key.map(office_by_school).eq(membership.school_edu_office).all():
        raise ValueError("A cluster contains a school from a different/unknown education office")
    if len(stage3) != 1679 or stage3.capacity_option_id.duplicated().any() or not parse_bool(stage3.final_road_eligible).all():
        raise ValueError("Expected the unchanged 1679 eligible Stage 3 options")
    for frame in (selected, stage3, options, detail):
        if frame.capacity_scenario_id.isna().any() or set(frame.capacity_scenario_id) != {"CAP28"}:
            raise ValueError("Only CAP28 is supported")
    if options.integration_option_id.isna().any() or options.integration_option_id.duplicated().any():
        raise ValueError("Option summary IDs must be unique")
    if set(options.integration_option_id) != set(stage3.capacity_option_id):
        raise ValueError("04-4 options differ from Stage 3 eligibility; no re-filtering is permitted")
    if set(options.final_cluster_id) != set(selected.final_cluster_id):
        raise ValueError("Missing or unexpected cluster in option summary")
    if detail.duplicated(["integration_option_id", "demand_assignment_id"]).any() or detail.demand_assignment_id.isna().any():
        raise ValueError("Duplicate or missing option-assignment keys")
    if set(detail.integration_option_id) != set(options.integration_option_id):
        raise ValueError("Detail and option ID sets differ")
    if options.assign(_total=pd.to_numeric(options.student_weight_total, errors="raise")).groupby("final_cluster_id")._total.agg(lambda s: s.max() - s.min()).gt(WEIGHT_TOLERANCE).any():
        raise ValueError("Student denominator differs between a cluster's options")
    for _, cluster_detail in detail.groupby("final_cluster_id", sort=False):
        assignment_sets = cluster_detail.groupby("integration_option_id").demand_assignment_id.agg(frozenset)
        if assignment_sets.nunique() != 1:
            raise ValueError("Options in a cluster do not cover identical demand assignments")
    if detail.groupby("demand_assignment_id")[["current_school_key", "census_id"]].nunique().gt(1).any().any():
        raise ValueError("Demand assignment identity changes between options")
    option_lookup = options.set_index("integration_option_id")
    stage3_lookup = stage3.set_index("capacity_option_id")
    cluster_lookup = selected.set_index("final_cluster_id")
    member_pairs = set(zip(membership.final_cluster_id, membership.school_key))
    for option in options.itertuples(index=False):
        source = stage3_lookup.loc[option.integration_option_id]
        if (option.final_cluster_id != source.final_cluster_id or option.candidate_id != source.candidate_id
                or option.integrated_school_key != source.retained_school_key
                or option.integrated_school_edu_office != source.school_edu_office
                or option.candidate_id != cluster_lookup.loc[option.final_cluster_id, "candidate_id"]
                or option.integrated_school_edu_office != cluster_lookup.loc[option.final_cluster_id, "school_edu_office"]):
            raise ValueError("Stage 3 / option / cluster identifiers disagree")
        if {source.retained_school_key, source.closed_school_1_key, source.closed_school_2_key} != set(
                membership.loc[membership.final_cluster_id.eq(option.final_cluster_id), "school_key"]):
            raise ValueError("Stage 3 option does not describe its three-school cluster")
    for column in ["final_cluster_id", "candidate_id", "integrated_school_key", "integrated_school_edu_office"]:
        if not detail.integration_option_id.map(option_lookup[column]).eq(detail[column]).all():
            raise ValueError(f"Detail-option mismatch: {column}")
    if any(pair not in member_pairs for pair in zip(detail.final_cluster_id, detail.current_school_key)):
        raise ValueError("Current school is outside its selected cluster")
    if not detail.current_school_key.map(office_by_school).eq(detail.current_school_edu_office).all():
        raise ValueError("Detail current-school education office mismatch")
    weights = pd.to_numeric(detail.student_weight, errors="raise")
    if not np.isfinite(weights).all() or weights.le(0).any():
        raise ValueError("Invalid student weights")
    if detail.assign(_weight=weights).groupby("demand_assignment_id")._weight.nunique().gt(1).any():
        raise ValueError("Student weight changes between options")
    if parse_bool(detail.moved_student_flag).ne(detail.current_school_key.ne(detail.integrated_school_key)).any():
        raise ValueError("Moved-student flag is inconsistent")
    eligible = parse_bool(detail.commute_analysis_eligible_flag)
    measured = detail.assign(_weight=weights, _eligible_weight=weights.where(eligible, 0), _eligible=eligible).groupby("integration_option_id").agg(
        total=("_weight", "sum"), eligible=("_eligible_weight", "sum"), eligible_rows=("_eligible", "sum"), rows=("_weight", "size"))
    # Validate the existing 04-4 denominator/coverage, without implementing a second scoring formula.
    checks = {"student_weight_total": measured.total, "analysis_eligible_student_weight": measured.eligible,
              "analysis_excluded_student_weight": measured.total - measured.eligible,
              "analysis_student_coverage_rate": measured.eligible / measured.total}
    for column, expected in checks.items():
        actual = pd.to_numeric(option_lookup[column], errors="raise").reindex(expected.index)
        if not np.isclose(actual, expected, atol=WEIGHT_TOLERANCE, rtol=0).all():
            raise ValueError(f"04-4 denominator/coverage mismatch: {column}")
    status = pd.Series(np.where(measured.eligible_rows.eq(measured.rows), "READY",
                                np.where(measured.eligible_rows.gt(0), "PARTIAL_ROUTE_EXCLUSION", "EXCLUDED_NO_USABLE_DEMAND")), index=measured.index)
    if not option_lookup.option_analysis_status.reindex(status.index).eq(status).all():
        raise ValueError("04-4 READY/PARTIAL state disagrees with detail")
    return selected, membership, options.assign(school_edu_office=options.integrated_school_edu_office)


def build_scenario(options: pd.DataFrame, membership: pd.DataFrame, detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    choices = []
    # With no shared inter-cluster constraints, the office's additive objective is
    # minimized by independently minimizing each cluster. Office processing is
    # explicit; no fictitious budget, facility, or bus constraints are introduced.
    for office, office_options in options.groupby("school_edu_office", sort=True):
        office_choices = []
        for cluster, candidates in office_options.groupby("final_cluster_id", sort=True):
            chosen = select_retained_school(candidates)
            chosen["scenario_id"] = SCENARIO_ID
            chosen["selection_id"] = f"{cluster}__{SCENARIO_ID}"
            office_choices.append(chosen)
        choices.extend(office_choices)
    scenario = pd.DataFrame(choices).sort_values(["school_edu_office", "final_cluster_id"], kind="stable").reset_index(drop=True)
    if len(scenario) != 720 or scenario.final_cluster_id.duplicated().any():
        raise ValueError("Incomplete final scenario")
    schools = membership.merge(scenario[["final_cluster_id", "scenario_id", "integrated_school_key", "integrated_school_name"]], on="final_cluster_id", validate="many_to_one")
    schools["school_status"] = np.where(schools.school_key.eq(schools.integrated_school_key), "RETAINED", "CLOSED")
    schools = schools.rename(columns={"integrated_school_key": "retained_school_key", "integrated_school_name": "retained_school_name"})
    if schools.groupby("final_cluster_id").school_status.apply(lambda s: s.eq("RETAINED").sum()).ne(1).any():
        raise ValueError("Each cluster must retain exactly one school")
    tags = scenario[["scenario_id", "selection_id", "final_cluster_id", "integration_option_id"]]
    assignments = detail.merge(tags, on=["final_cluster_id", "integration_option_id"], how="inner", validate="many_to_one")
    validate_assignments(assignments, scenario)
    if set(assignments.demand_assignment_id) != set(detail.demand_assignment_id):
        raise ValueError("Final selection lost demand assignments")
    return (scenario, schools.sort_values(["final_cluster_id", "school_key"]).reset_index(drop=True),
            assignments.sort_values(["final_cluster_id", "census_id", "current_school_key"], kind="stable").reset_index(drop=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().project_root.resolve()
    folder = root / "data/03_outputs"
    paths = {"clusters": root / "data/02_processed/nonoverlap_cluster_selection.csv",
             "stage3": root / "data/02_processed/final_selected_cluster_options.csv",
             "master": root / "data/01_raw/elementary_master.csv",
             "options": folder / OPTION_FILE, "detail": folder / DETAIL_FILE}
    frames = {key: pd.read_csv(path, dtype=str) for key, path in paths.items()}
    clusters, membership, options = validate_inputs(**frames)
    options = numeric(options, RANK_COLUMNS[:-1])
    scenario, schools, assignments = build_scenario(options, membership, frames["detail"])
    outputs = {SCENARIO_FILE: scenario, SCHOOL_FILE: schools, DEMAND_FILE: assignments}
    for filename, frame in outputs.items():
        frame.to_csv(folder / filename, index=False, encoding="utf-8-sig")
    # Only scope/quality diagnostics: national commute-impact aggregation starts in Stage 5.
    validation = {"selected_cluster_count": len(scenario), "school_count": len(schools),
                  "retained_school_count": int(schools.school_status.eq("RETAINED").sum()),
                  "closed_school_count": int(schools.school_status.eq("CLOSED").sum()),
                  "education_office_count": scenario.school_edu_office.nunique(),
                  "eligible_option_count": len(options), "assignment_count": len(assignments),
                  "student_weight_total": pd.to_numeric(assignments.student_weight).sum(),
                  "analysis_eligible_student_weight": pd.to_numeric(assignments.loc[parse_bool(assignments.commute_analysis_eligible_flag), "student_weight"]).sum(),
                  "analysis_excluded_student_weight": pd.to_numeric(assignments.loc[~parse_bool(assignments.commute_analysis_eligible_flag), "student_weight"]).sum(),
                  "partial_selected_cluster_count": int(scenario.selection_pool_status.eq("PARTIAL_ONLY").sum())}
    pd.DataFrame([{"metric": k, "value": v, "check_status": "PASS"} for k, v in validation.items()]).to_csv(folder / VALIDATION_FILE, index=False, encoding="utf-8-sig")
    manifest = {"scenario_id": SCENARIO_ID, "selection_metric": METRIC, "selection_basis": "ALL_STUDENTS",
                "ranking_columns": RANK_COLUMNS, "tie_definition": "exact numeric equality; no rounding",
                "input_sha256": {k: sha256_file(p) for k, p in paths.items()},
                "outputs_sha256": {name: sha256_file(folder / name) for name in outputs}}
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()

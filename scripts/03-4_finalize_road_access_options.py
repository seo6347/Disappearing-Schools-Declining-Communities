"""
통합 가능 시나리오 확정 4단계: 최종 도로 접근성 규칙 적용 (03-4)
=================================================================

목적
----
03-3 민감도 분석을 통해 확정한 극단경로 제외 규칙을 수용 가능한
후보–존치학교안에 적용하고, 다음 비중복 3개교 군집 선정에 사용할
후보별 최종 도로 접근성 상태를 생성한다.

최종 제외 규칙
--------------
다음 중 하나라도 충족하면 해당 '존치학교 선택안'을 제외한다.

1. 두 폐교예정학교→존치학교 경로 중 하나라도 페리를 이용
2. 두 경로의 최대 도로시간이 80분을 초과
3. 두 경로의 최대 도로시간이 40분을 초과하면서,
   최대 도로거리/직선거리 비율도 3배를 초과

경곗값은 모두 엄밀한 초과(>)로 처리한다. 따라서 정확히 40분,
정확히 3배, 정확히 80분은 이 규칙만으로 제외하지 않는다.

후보와 존치안의 관계
---------------------
- 최종 규칙은 후보 전체가 아니라 '존치학교 선택안' 단위로 적용한다.
- 한 3개교 후보에 최종 가능 존치안이 하나라도 남으면 후보를 유지한다.
- 수용 가능한 존치안이 모두 도로 규칙에서 제외될 때만 후보가 탈락한다.
- 제외된 행도 삭제하지 않고 상세 파일에 상태와 사유를 보존한다.

입력 파일
---------
data/02_processed/candidate_road_access_options.csv
    - 03-3에서 생성한 수용 가능 후보–존치학교안별 도로 접근성 상세표

data/02_processed/road_access_sensitivity_summary.csv
    - 03-3 민감도 분석표
    - S21_HYBRID_80_40_R3 결과와 이번 코드의 계산값을 교차검증하는 용도

출력 파일
---------
data/02_processed/final_road_access_options.csv
    - 모든 존치학교 선택안을 보존한 최종 상세표
    - final_road_eligible, final_road_status, final_exclusion_conditions 포함

data/02_processed/final_road_candidate_summary.csv
    - 수용력 시나리오–3개교 후보당 1행
    - 최종 가능 존치학교 수·목록과 후보 유지/탈락 상태 포함
    - 다음 비중복 군집 선정 단계의 후보 수준 입력자료

실행 방법
---------
프로젝트 루트에서:

    py scripts/03-4_finalize_road_access_options.py

주의
----
- 학교 간 도로시간은 일요일 새벽 비혼잡 시간대에 조회한 값이므로
  학생의 최종 통학시간이 아니라 극단적인 학교 간 접근성 검증용이다.
- 30분 초과는 자동 제외하지 않는다. 이후 집계구→존치학교 통학부담
  분석에서 결과 지표로 산출한다.
- 페리·극단경로 행은 재현성과 사례 검토를 위해 삭제하지 않는다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 분석 설정: 팀에서 확정한 최종 규칙
# -----------------------------------------------------------------------------

INPUT_OPTIONS = "candidate_road_access_options.csv"
INPUT_SENSITIVITY = "road_access_sensitivity_summary.csv"
OUTPUT_OPTIONS = "final_road_access_options.csv"
OUTPUT_CANDIDATES = "final_road_candidate_summary.csv"

FINAL_RULE_ID = "S21_HYBRID_80_40_R3"
EXTREME_ABSOLUTE_TIME_MIN = 80.0
STRUCTURAL_TIME_MIN = 40.0
STRUCTURAL_RATIO = 3.0
NUMERIC_TOLERANCE = 1e-9

OPTIONS_REQUIRED = {
    "capacity_option_id",
    "candidate_id",
    "capacity_scenario_id",
    "students_per_class",
    "school_edu_office",
    "component_id",
    "candidate_type",
    "total_demand",
    "retained_school_position",
    "retained_school_key",
    "retained_school_name",
    "closed_school_1_key",
    "closed_school_1_name",
    "closed_school_2_key",
    "closed_school_2_name",
    "capacity_feasible",
    "capacity_status",
    "route_1_status",
    "route_1_check",
    "route_2_status",
    "route_2_check",
    "max_school_road_duration_min",
    "max_road_straight_ratio",
    "any_ferry_route",
    "over_80min_flag",
    "structural_detour_over_40min_flag",
    "road_data_check",
}

SENSITIVITY_REQUIRED = {
    "capacity_scenario_id",
    "rule_id",
    "base_capacity_feasible_option_count",
    "base_capacity_feasible_candidate_count",
    "eligible_option_count",
    "eligible_candidate_count",
    "candidate_without_eligible_option_count",
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="확정된 페리·극단경로 규칙을 존치학교 선택안에 적용"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="final_project 폴더. 기본값은 scripts 폴더의 상위 폴더입니다.",
    )
    parser.add_argument("--options-file", type=Path, default=None)
    parser.add_argument("--sensitivity-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    project_root = args.project_root.resolve()
    processed_dir = project_root / "data" / "02_processed"
    options_path = (args.options_file or processed_dir / INPUT_OPTIONS).resolve()
    sensitivity_path = (
        args.sensitivity_file or processed_dir / INPUT_SENSITIVITY
    ).resolve()
    output_dir = (args.output_dir or processed_dir).resolve()
    return options_path, sensitivity_path, output_dir


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"[{label}] 필수 컬럼이 없습니다: {missing}")


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{label}에 결측값이 있습니다.")
        return series.astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "y": True,
        "yes": True,
        "false": False,
        "0": False,
        "n": False,
        "no": False,
    }
    result = normalized.map(mapping)
    if result.isna().any():
        bad = sorted(normalized.loc[result.isna()].dropna().unique())[:10]
        raise ValueError(f"{label}에 해석할 수 없는 값이 있습니다: {bad}")
    return result.astype(bool)


def load_inputs(
    options_path: Path,
    sensitivity_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [options_path, sensitivity_path]:
        if not path.exists():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    id_dtypes = {
        "capacity_option_id": "string",
        "candidate_id": "string",
        "capacity_scenario_id": "string",
        "retained_school_key": "string",
        "closed_school_1_key": "string",
        "closed_school_2_key": "string",
    }
    options = pd.read_csv(
        options_path,
        encoding="utf-8-sig",
        dtype=id_dtypes,
        low_memory=False,
    )
    sensitivity = pd.read_csv(
        sensitivity_path,
        encoding="utf-8-sig",
        dtype={"capacity_scenario_id": "string", "rule_id": "string"},
        low_memory=False,
    )
    require_columns(options, OPTIONS_REQUIRED, INPUT_OPTIONS)
    require_columns(sensitivity, SENSITIVITY_REQUIRED, INPUT_SENSITIVITY)

    for column in id_dtypes:
        options[column] = normalize_id(options[column])
    sensitivity["capacity_scenario_id"] = normalize_id(
        sensitivity["capacity_scenario_id"]
    )
    sensitivity["rule_id"] = normalize_id(sensitivity["rule_id"])

    return options, sensitivity


def validate_options(options: pd.DataFrame) -> pd.DataFrame:
    result = options.copy()

    if result.empty:
        raise ValueError("존치학교 선택안 파일이 비어 있습니다.")
    if result["capacity_option_id"].isna().any():
        raise ValueError("capacity_option_id에 결측값이 있습니다.")
    if result.duplicated("capacity_option_id").any():
        duplicated = result.loc[
            result.duplicated("capacity_option_id", keep=False), "capacity_option_id"
        ].head(10)
        raise ValueError(
            f"capacity_option_id가 중복되었습니다: {duplicated.tolist()}"
        )
    if result["candidate_id"].isna().any():
        raise ValueError("candidate_id에 결측값이 있습니다.")

    for column in [
        "capacity_feasible",
        "any_ferry_route",
        "over_80min_flag",
        "structural_detour_over_40min_flag",
    ]:
        result[column] = parse_bool(result[column], column)

    if not result["capacity_feasible"].all():
        raise ValueError("입력 상세표에 수용 불가능한 존치안이 포함되어 있습니다.")
    if not result["capacity_status"].eq("CAPACITY_FEASIBLE").all():
        raise ValueError("capacity_status가 CAPACITY_FEASIBLE이 아닌 행이 있습니다.")
    if not result["road_data_check"].eq("PASS").all():
        raise ValueError("road_data_check가 PASS가 아닌 행이 있습니다.")
    if not result["route_1_status"].eq("SUCCESS").all():
        raise ValueError("route_1_status가 SUCCESS가 아닌 행이 있습니다.")
    if not result["route_2_status"].eq("SUCCESS").all():
        raise ValueError("route_2_status가 SUCCESS가 아닌 행이 있습니다.")
    # REVIEW는 03-2에서 페리 또는 우회비율 3배 초과 경로를 사람이 확인할 수
    # 있도록 남긴 진단 상태다. API 실패가 아니므로 PASS와 함께 정상 입력으로
    # 허용하고, 실제 포함/제외 여부는 아래 최종 규칙으로 다시 계산한다.
    allowed_route_checks = {"PASS", "REVIEW"}
    if not result["route_1_check"].isin(allowed_route_checks).all():
        bad = result.loc[
            ~result["route_1_check"].isin(allowed_route_checks), "route_1_check"
        ].value_counts(dropna=False)
        raise ValueError(f"route_1_check에 허용되지 않은 상태가 있습니다: {bad.to_dict()}")
    if not result["route_2_check"].isin(allowed_route_checks).all():
        bad = result.loc[
            ~result["route_2_check"].isin(allowed_route_checks), "route_2_check"
        ].value_counts(dropna=False)
        raise ValueError(f"route_2_check에 허용되지 않은 상태가 있습니다: {bad.to_dict()}")

    numeric_columns = [
        "students_per_class",
        "total_demand",
        "retained_school_position",
        "max_school_road_duration_min",
        "max_road_straight_ratio",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"{column}에 숫자로 변환할 수 없는 값이 있습니다.")

    if result["max_school_road_duration_min"].lt(0).any():
        raise ValueError("최대 도로시간에 음수가 있습니다.")
    if result["max_road_straight_ratio"].lt(1 - NUMERIC_TOLERANCE).any():
        raise ValueError("도로거리/직선거리 비율이 1보다 작은 행이 있습니다.")

    recalculated_over_80 = (
        result["max_school_road_duration_min"]
        > EXTREME_ABSOLUTE_TIME_MIN + NUMERIC_TOLERANCE
    )
    recalculated_structural = (
        result["max_school_road_duration_min"]
        > STRUCTURAL_TIME_MIN + NUMERIC_TOLERANCE
    ) & (
        result["max_road_straight_ratio"]
        > STRUCTURAL_RATIO + NUMERIC_TOLERANCE
    )
    if not result["over_80min_flag"].eq(recalculated_over_80).all():
        raise ValueError("기존 over_80min_flag와 80분 초과 재계산 결과가 다릅니다.")
    if not result["structural_detour_over_40min_flag"].eq(
        recalculated_structural
    ).all():
        raise ValueError(
            "기존 structural_detour_over_40min_flag와 40분·3배 기준 재계산 결과가 다릅니다."
        )

    group_columns = ["capacity_scenario_id", "candidate_id"]
    scenario_consistency = result.groupby(group_columns, sort=False).agg(
        students_per_class_count=("students_per_class", "nunique"),
        office_count=("school_edu_office", "nunique"),
        candidate_type_count=("candidate_type", "nunique"),
        component_count=("component_id", "nunique"),
        demand_count=("total_demand", "nunique"),
        retained_key_count=("retained_school_key", "nunique"),
    )
    inconsistent = scenario_consistency[
        scenario_consistency[
            [
                "students_per_class_count",
                "office_count",
                "candidate_type_count",
                "component_count",
                "demand_count",
            ]
        ].ne(1).any(axis=1)
    ]
    if not inconsistent.empty:
        raise ValueError(
            "동일 후보–수용력 시나리오 안에서 기본 속성이 일치하지 않습니다: "
            f"{inconsistent.head(10).index.tolist()}"
        )
    if scenario_consistency["retained_key_count"].gt(3).any():
        raise ValueError("한 후보에 3개를 초과하는 존치학교가 있습니다.")

    return result


def apply_final_rule(options: pd.DataFrame) -> pd.DataFrame:
    result = options.copy()

    result["final_rule_id"] = FINAL_RULE_ID
    result["final_ferry_exclusion_flag"] = result["any_ferry_route"]
    result["final_extreme_time_over_80_flag"] = (
        result["max_school_road_duration_min"]
        > EXTREME_ABSOLUTE_TIME_MIN + NUMERIC_TOLERANCE
    )
    result["final_structural_detour_flag"] = (
        result["max_school_road_duration_min"]
        > STRUCTURAL_TIME_MIN + NUMERIC_TOLERANCE
    ) & (
        result["max_road_straight_ratio"]
        > STRUCTURAL_RATIO + NUMERIC_TOLERANCE
    )

    result["final_road_eligible"] = ~(
        result["final_ferry_exclusion_flag"]
        | result["final_extreme_time_over_80_flag"]
        | result["final_structural_detour_flag"]
    )

    # 상태는 해석이 명확하도록 우선순위를 둔다. 여러 조건의 동시 충족 여부는
    # final_exclusion_conditions에 모두 남긴다.
    result["final_road_status"] = np.select(
        [
            result["final_ferry_exclusion_flag"],
            result["final_extreme_time_over_80_flag"],
            result["final_structural_detour_flag"],
        ],
        [
            "EXCLUDED_FERRY",
            "EXCLUDED_EXTREME_TIME_OVER_80",
            "EXCLUDED_STRUCTURAL_DETOUR",
        ],
        default="ROAD_ELIGIBLE",
    )

    conditions = np.full(len(result), "", dtype=object)
    for flag_column, label in [
        ("final_ferry_exclusion_flag", "FERRY_REQUIRED"),
        ("final_extreme_time_over_80_flag", "TIME_OVER_80MIN"),
        ("final_structural_detour_flag", "TIME_OVER_40MIN_AND_RATIO_OVER_3"),
    ]:
        mask = result[flag_column].to_numpy(dtype=bool)
        conditions[mask] = np.where(
            conditions[mask] == "",
            label,
            conditions[mask] + "|" + label,
        )
    conditions[conditions == ""] = "NONE"
    result["final_exclusion_conditions"] = conditions
    result["final_rule_check"] = np.where(
        result["final_road_eligible"]
        == result["final_exclusion_conditions"].eq("NONE"),
        "PASS",
        "FAIL",
    )

    if not result["final_rule_check"].eq("PASS").all():
        raise AssertionError("최종 도로 규칙 상태와 제외 조건이 일치하지 않습니다.")

    return result


def join_sorted_unique(series: pd.Series) -> str:
    values = sorted(
        {
            str(value).strip()
            for value in series.dropna()
            if str(value).strip() != ""
        }
    )
    return "|".join(values)


def extract_candidate_schools(row: pd.Series) -> pd.Series:
    pairs = {
        str(row["retained_school_key"]): str(row["retained_school_name"]),
        str(row["closed_school_1_key"]): str(row["closed_school_1_name"]),
        str(row["closed_school_2_key"]): str(row["closed_school_2_name"]),
    }
    if len(pairs) != 3:
        raise ValueError(
            f"후보 {row['candidate_id']}의 고유 학교 수가 3개가 아닙니다: {pairs}"
        )
    ordered = sorted(pairs.items(), key=lambda item: item[0])
    values: dict[str, str] = {}
    for position, (school_key, school_name) in enumerate(ordered, start=1):
        values[f"school_{position}_key"] = school_key
        values[f"school_{position}_name"] = school_name
    return pd.Series(values)


def build_candidate_summary(options: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["capacity_scenario_id", "candidate_id"]
    grouped = options.groupby(group_columns, sort=False, dropna=False)

    summary = grouped.agg(
        students_per_class=("students_per_class", "first"),
        school_edu_office=("school_edu_office", "first"),
        component_id=("component_id", "first"),
        candidate_type=("candidate_type", "first"),
        total_demand=("total_demand", "first"),
        capacity_feasible_option_count=("capacity_option_id", "size"),
        final_eligible_option_count=("final_road_eligible", "sum"),
        ferry_excluded_option_count=("final_ferry_exclusion_flag", "sum"),
        extreme_time_over_80_option_count=(
            "final_extreme_time_over_80_flag",
            "sum",
        ),
        structural_detour_option_count=("final_structural_detour_flag", "sum"),
        min_option_max_road_duration_min=("max_school_road_duration_min", "min"),
        max_option_max_road_duration_min=("max_school_road_duration_min", "max"),
    ).reset_index()

    summary["final_excluded_option_count"] = (
        summary["capacity_feasible_option_count"]
        - summary["final_eligible_option_count"]
    )
    summary["candidate_road_eligible"] = summary[
        "final_eligible_option_count"
    ].gt(0)
    summary["candidate_road_status"] = np.where(
        summary["candidate_road_eligible"],
        "ROAD_FEASIBLE_CANDIDATE",
        "NO_ROAD_FEASIBLE_OPTION",
    )

    first_rows = options.drop_duplicates(group_columns, keep="first").copy()
    school_columns = first_rows.apply(extract_candidate_schools, axis=1)
    first_rows = pd.concat(
        [first_rows[group_columns].reset_index(drop=True), school_columns.reset_index(drop=True)],
        axis=1,
    )
    summary = summary.merge(
        first_rows,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    eligible = options.loc[options["final_road_eligible"]].copy()
    excluded = options.loc[~options["final_road_eligible"]].copy()

    eligible_lists = eligible.groupby(group_columns, sort=False).agg(
        eligible_capacity_option_ids=("capacity_option_id", join_sorted_unique),
        eligible_retained_school_keys=("retained_school_key", join_sorted_unique),
        eligible_retained_school_names=("retained_school_name", join_sorted_unique),
        min_eligible_max_road_duration_min=("max_school_road_duration_min", "min"),
        max_eligible_max_road_duration_min=("max_school_road_duration_min", "max"),
    ).reset_index()
    excluded_lists = excluded.groupby(group_columns, sort=False).agg(
        excluded_capacity_option_ids=("capacity_option_id", join_sorted_unique),
        excluded_retained_school_keys=("retained_school_key", join_sorted_unique),
        excluded_retained_school_names=("retained_school_name", join_sorted_unique),
        exclusion_conditions=("final_exclusion_conditions", join_sorted_unique),
    ).reset_index()

    summary = summary.merge(
        eligible_lists,
        on=group_columns,
        how="left",
        validate="one_to_one",
    ).merge(
        excluded_lists,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    list_columns = [
        "eligible_capacity_option_ids",
        "eligible_retained_school_keys",
        "eligible_retained_school_names",
        "excluded_capacity_option_ids",
        "excluded_retained_school_keys",
        "excluded_retained_school_names",
        "exclusion_conditions",
    ]
    summary[list_columns] = summary[list_columns].fillna("NONE")
    summary["candidate_exclusion_reason"] = np.where(
        summary["candidate_road_eligible"],
        "NONE",
        "ALL_OPTIONS_EXCLUDED:" + summary["exclusion_conditions"],
    )
    summary["candidate_road_check"] = np.where(
        (
            summary["capacity_feasible_option_count"]
            == summary["final_eligible_option_count"]
            + summary["final_excluded_option_count"]
        )
        & (
            summary["candidate_road_eligible"]
            == summary["final_eligible_option_count"].gt(0)
        ),
        "PASS",
        "FAIL",
    )

    ordered_columns = [
        "capacity_scenario_id",
        "candidate_id",
        "students_per_class",
        "school_edu_office",
        "component_id",
        "candidate_type",
        "total_demand",
        "school_1_key",
        "school_1_name",
        "school_2_key",
        "school_2_name",
        "school_3_key",
        "school_3_name",
        "capacity_feasible_option_count",
        "final_eligible_option_count",
        "final_excluded_option_count",
        "ferry_excluded_option_count",
        "extreme_time_over_80_option_count",
        "structural_detour_option_count",
        "eligible_capacity_option_ids",
        "eligible_retained_school_keys",
        "eligible_retained_school_names",
        "excluded_capacity_option_ids",
        "excluded_retained_school_keys",
        "excluded_retained_school_names",
        "exclusion_conditions",
        "min_option_max_road_duration_min",
        "max_option_max_road_duration_min",
        "min_eligible_max_road_duration_min",
        "max_eligible_max_road_duration_min",
        "candidate_road_eligible",
        "candidate_road_status",
        "candidate_exclusion_reason",
        "candidate_road_check",
    ]
    summary = summary[ordered_columns]

    if summary.duplicated(group_columns).any():
        raise AssertionError("후보 요약표의 후보–수용력 시나리오 키가 중복됩니다.")
    if not summary["candidate_road_check"].eq("PASS").all():
        raise AssertionError("후보 요약표 내부 검증에 실패한 행이 있습니다.")

    return summary


def validate_against_sensitivity(
    options: pd.DataFrame,
    candidates: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    expected = sensitivity.loc[sensitivity["rule_id"].eq(FINAL_RULE_ID)].copy()
    scenario_ids = sorted(options["capacity_scenario_id"].dropna().unique())

    if expected["capacity_scenario_id"].duplicated().any():
        raise ValueError(f"민감도 표의 {FINAL_RULE_ID} 행이 시나리오별로 중복됩니다.")
    if sorted(expected["capacity_scenario_id"].dropna().unique()) != scenario_ids:
        raise ValueError(
            f"민감도 표에 {FINAL_RULE_ID}의 모든 수용력 시나리오가 없습니다."
        )

    actual_rows = []
    for scenario_id, group in options.groupby("capacity_scenario_id", sort=False):
        candidate_group = candidates.loc[
            candidates["capacity_scenario_id"].eq(scenario_id)
        ]
        actual_rows.append(
            {
                "capacity_scenario_id": scenario_id,
                "actual_base_option_count": len(group),
                "actual_base_candidate_count": group["candidate_id"].nunique(),
                "actual_eligible_option_count": int(group["final_road_eligible"].sum()),
                "actual_eligible_candidate_count": int(
                    candidate_group["candidate_road_eligible"].sum()
                ),
                "actual_excluded_candidate_count": int(
                    (~candidate_group["candidate_road_eligible"]).sum()
                ),
            }
        )
    actual = pd.DataFrame(actual_rows)
    check = expected.merge(
        actual,
        on="capacity_scenario_id",
        how="inner",
        validate="one_to_one",
    )

    comparisons = {
        "base_capacity_feasible_option_count": "actual_base_option_count",
        "base_capacity_feasible_candidate_count": "actual_base_candidate_count",
        "eligible_option_count": "actual_eligible_option_count",
        "eligible_candidate_count": "actual_eligible_candidate_count",
        "candidate_without_eligible_option_count": "actual_excluded_candidate_count",
    }
    failed = pd.Series(False, index=check.index)
    for expected_column, actual_column in comparisons.items():
        failed |= pd.to_numeric(check[expected_column], errors="coerce").ne(
            check[actual_column]
        )
    if failed.any():
        diagnostic_columns = [
            "capacity_scenario_id",
            *comparisons.keys(),
            *comparisons.values(),
        ]
        raise AssertionError(
            "03-3 민감도 요약과 03-4 최종 계산이 일치하지 않습니다:\n"
            + check.loc[failed, diagnostic_columns].to_string(index=False)
        )


def log_summary(options: pd.DataFrame, candidates: pd.DataFrame) -> None:
    logging.info("최종 상세표: %s행 × %s열", *options.shape)
    logging.info("최종 후보표: %s행 × %s열", *candidates.shape)

    for scenario_id, group in options.groupby("capacity_scenario_id", sort=False):
        candidate_group = candidates.loc[
            candidates["capacity_scenario_id"].eq(scenario_id)
        ]
        logging.info(
            "[%s] 존치안 전체 %s / 가능 %s / 제외 %s",
            scenario_id,
            f"{len(group):,}",
            f"{int(group['final_road_eligible'].sum()):,}",
            f"{int((~group['final_road_eligible']).sum()):,}",
        )
        logging.info(
            "[%s] 후보 전체 %s / 유지 %s / 탈락 %s",
            scenario_id,
            f"{len(candidate_group):,}",
            f"{int(candidate_group['candidate_road_eligible'].sum()):,}",
            f"{int((~candidate_group['candidate_road_eligible']).sum()):,}",
        )
        logging.info(
            "[%s] 제외 존치안: 페리 %s / 80분 초과 %s / 40분·3배 초과 %s",
            scenario_id,
            f"{int(group['final_ferry_exclusion_flag'].sum()):,}",
            f"{int(group['final_extreme_time_over_80_flag'].sum()):,}",
            f"{int(group['final_structural_detour_flag'].sum()):,}",
        )


def main() -> int:
    args = parse_args()
    options_path, sensitivity_path, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("입력 상세표: %s", options_path)
    logging.info("입력 민감도표: %s", sensitivity_path)

    options, sensitivity = load_inputs(options_path, sensitivity_path)
    options = validate_options(options)
    final_options = apply_final_rule(options)
    final_candidates = build_candidate_summary(final_options)
    validate_against_sensitivity(final_options, final_candidates, sensitivity)

    options_output_path = output_dir / OUTPUT_OPTIONS
    candidates_output_path = output_dir / OUTPUT_CANDIDATES
    final_options.to_csv(options_output_path, index=False, encoding="utf-8-sig")
    final_candidates.to_csv(candidates_output_path, index=False, encoding="utf-8-sig")

    log_summary(final_options, final_candidates)
    logging.info("저장 완료: %s", options_output_path)
    logging.info("저장 완료: %s", candidates_output_path)
    logging.info("03-3 민감도 표의 %s 결과와 교차검증 PASS", FINAL_RULE_ID)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.exception("03-4 실행 실패: %s", error)
        sys.exit(1)

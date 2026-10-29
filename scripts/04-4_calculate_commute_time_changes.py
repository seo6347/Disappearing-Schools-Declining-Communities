"""
04-4. 통합 전후 통학시간·거리 변화 및 통합학교안별 결과 산출

이 스크립트가 하는 일
1) 04-1에서 만든 학생 수요점-통합학교안 행렬을 불러온다.
2) 04-3에서 확정한 도로 경로표를 현재 학교와 통합학교 경로에 각각 연결한다.
3) 수요점-현재학교 배정행별로 통합 전후 시간·거리, 증감, 30분 초과 여부를 계산한다.
4) 페리 경로가 필요한 행은 원자료를 보존하되 통학부담 계산에서 제외한다.
5) 통합학교안별 학생 가중 평균·가중 분위수·30분 초과 학생 가중치를 요약한다.
6) 요청키 누락, 중복 조인, 0분 처리, 페리 제외, 학생 가중치 보존을 검증한다.

중요한 분석 원칙
- 통합 전후 차이 = 통합 후 - 통합 전이다. 음수는 통합 후 더 가까워진 경우이므로 그대로 보존한다.
- 순수한 추가 부담은 max(통합 후 - 통합 전, 0)으로 별도 제공한다.
- '30분 초과'는 통합 후 직행 차량 시간이 30분보다 큰 경우다(30분 정확히는 초과 아님).
- 통합 후 30분 초과와 통합으로 인해 새롭게 30분을 넘은 경우를 구분한다.
- 학생 가중 평균과 분위수에는 student_weight를 사용한다.
- 계산 과정에서는 반올림하지 않는다. 표시에 필요한 반올림은 후속 기사·시각화 단계에서 한다.
- analysis_eligible_flag=TRUE인 경로만 계산에 사용한다.
- 출발지와 목적지가 같은 0분 보정 경로는 포함한다.
- 페리 경로는 제외하고, ROAD_TO_STRAIGHT_OVER_5 경로는 제외하지 않는다.
- 대표점->통합학교 직행 시간은 실제 다중 승하차 통학버스 시간보다 유리한 하한값이다.

입력 파일
- data/03_outputs/commute_demand_option_matrix.csv
- data/03_outputs/commute_road_routes.csv

핵심 산출물
- commute_time_demand_option.csv
  학생 수요점-현재학교 배정행 x 통합학교안별 통합 전후 상세 결과
- commute_time_option_summary.csv
  통합학교안 1개당 1행의 학생 가중 통학부담 요약
- commute_time_calculation_summary.csv
  입력·조인·제외·계산·출력 검증 결과

실행 예시
    python scripts/04-4_calculate_commute_time_changes.py

다른 프로젝트 경로를 직접 지정하는 경우
    python scripts/04-4_calculate_commute_time_changes.py --project-root "C:/.../final_project"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


MATRIX_FILE = "commute_demand_option_matrix.csv"
ROUTE_FILE = "commute_road_routes.csv"

DETAIL_OUTPUT_FILE = "commute_time_demand_option.csv"
OPTION_OUTPUT_FILE = "commute_time_option_summary.csv"
SUMMARY_OUTPUT_FILE = "commute_time_calculation_summary.csv"

THIRTY_MINUTE_THRESHOLD = 30.0
FLOAT_TOLERANCE = 1e-9
WEIGHT_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="04-4 통합 전후 통학시간 산출")
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="final_project 폴더. 기본값은 scripts 폴더의 상위 폴더",
    )
    return parser.parse_args()


def find_input(project_root: Path, filename: str) -> Path:
    search_dirs = [
        project_root / "data" / "03_outputs",
        project_root / "data" / "02_processed",
        project_root / "data" / "01_raw",
        project_root / "upload",
    ]
    matches = [folder / filename for folder in search_dirs if (folder / filename).exists()]
    if not matches:
        raise FileNotFoundError(
            f"입력 파일을 찾지 못했습니다: {filename}\n"
            f"확인한 폴더: {[str(path) for path in search_dirs]}"
        )
    if len(matches) > 1:
        print(f"[주의] {filename}이 여러 폴더에 있습니다. 우선순위가 가장 높은 파일 사용: {matches[0]}")
    return matches[0]


def read_csv_as_text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False)


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} 필수 칼럼 누락: {missing}")


def to_bool(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper().isin(
        {"TRUE", "1", "Y", "YES"}
    )


def to_numeric_strict(df: pd.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid = df[column].notna() & df[column].astype(str).str.strip().ne("") & converted.isna()
        if invalid.any():
            examples = df.loc[invalid, column].head(5).tolist()
            raise ValueError(
                f"{label}.{column} 숫자 변환 실패: {int(invalid.sum())}행, 예시={examples}"
            )
        df[column] = converted.astype(float)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values.loc[valid].to_numpy(), weights=weights.loc[valid].to_numpy()))


def weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    """양의 학생 가중치를 이용한 왼쪽 연속형 경험적 가중 분위수."""
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    value_array = values.loc[valid].to_numpy(dtype=float)
    weight_array = weights.loc[valid].to_numpy(dtype=float)
    order = np.argsort(value_array, kind="stable")
    value_array = value_array[order]
    weight_array = weight_array[order]
    cutoff = quantile * weight_array.sum()
    position = int(np.searchsorted(np.cumsum(weight_array), cutoff, side="left"))
    return float(value_array[min(position, len(value_array) - 1)])


def first_value(group: pd.DataFrame, column: str) -> object:
    values = group[column].dropna()
    return values.iloc[0] if len(values) else np.nan


def unique_non_null_count(group: pd.DataFrame, column: str) -> int:
    return int(group[column].dropna().nunique())


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = project_root / "data" / "03_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = find_input(project_root, MATRIX_FILE)
    route_path = find_input(project_root, ROUTE_FILE)

    print("[입력 파일]")
    print(f"- 학생-통합학교안 행렬: {matrix_path}")
    print(f"- 도로 경로 결과: {route_path}")

    matrix = read_csv_as_text(matrix_path)
    routes = read_csv_as_text(route_path)

    matrix_required = [
        "demand_assignment_id", "census_id", "student_weight", "final_cluster_id",
        "candidate_id", "capacity_scenario_id", "scenario_role",
        "integration_option_id", "current_school_key", "current_school_name",
        "integrated_school_key", "integrated_school_name", "moved_student_flag",
        "current_route_request_id", "integrated_route_request_id",
        "same_route_reuse_flag",
    ]
    route_required = [
        "route_request_id", "analysis_status", "analysis_eligible_flag",
        "analysis_exclusion_reason", "analysis_road_distance_m",
        "analysis_road_duration_min", "analysis_road_to_straight_ratio",
        "ferry_flag", "zero_route_imputed_flag", "merge_check",
    ]
    require_columns(matrix, matrix_required, MATRIX_FILE)
    require_columns(routes, route_required, ROUTE_FILE)

    if matrix.empty or routes.empty:
        raise ValueError("입력 파일이 비어 있습니다.")
    if matrix["demand_assignment_id"].isna().any():
        raise ValueError("demand_assignment_id에 결측치가 있습니다.")
    matrix_key = ["integration_option_id", "demand_assignment_id"]
    if matrix.duplicated(matrix_key).any():
        examples = matrix.loc[matrix.duplicated(matrix_key, keep=False), matrix_key].head(5)
        raise ValueError(f"학생-통합학교안 행렬 키 중복:\n{examples.to_string(index=False)}")
    if routes["route_request_id"].isna().any():
        raise ValueError("route_request_id에 결측치가 있습니다.")
    if routes["route_request_id"].duplicated().any():
        examples = routes.loc[
            routes["route_request_id"].duplicated(False), "route_request_id"
        ].head(5).tolist()
        raise ValueError(f"경로표 route_request_id 중복: {examples}")
    if routes["merge_check"].eq("FAIL").any():
        raise ValueError(
            f"경로표에 merge_check=FAIL이 {int(routes['merge_check'].eq('FAIL').sum())}건 있습니다."
        )

    to_numeric_strict(matrix, ["student_weight"], MATRIX_FILE)
    if matrix["student_weight"].isna().any() or (matrix["student_weight"] <= 0).any():
        raise ValueError("student_weight는 모든 행에서 0보다 큰 숫자여야 합니다.")

    route_numeric = [
        "analysis_road_distance_m", "analysis_road_duration_min",
        "analysis_road_to_straight_ratio",
    ]
    to_numeric_strict(routes, route_numeric, ROUTE_FILE)
    routes["analysis_eligible_flag"] = to_bool(routes["analysis_eligible_flag"])
    routes["ferry_flag"] = to_bool(routes["ferry_flag"])
    routes["zero_route_imputed_flag"] = to_bool(routes["zero_route_imputed_flag"])

    eligible_routes = routes["analysis_eligible_flag"]
    missing_eligible_metrics = eligible_routes & (
        routes["analysis_road_distance_m"].isna()
        | routes["analysis_road_duration_min"].isna()
    )
    if missing_eligible_metrics.any():
        raise ValueError(
            "analysis_eligible_flag=TRUE인데 분석용 거리 또는 시간이 없는 경로가 "
            f"{int(missing_eligible_metrics.sum())}건 있습니다."
        )
    if (eligible_routes & routes["ferry_flag"]).any():
        raise ValueError("페리 경로가 analysis_eligible_flag=TRUE로 남아 있습니다.")
    invalid_zero = routes["zero_route_imputed_flag"] & (
        routes["analysis_road_distance_m"].abs() > FLOAT_TOLERANCE
    )
    invalid_zero |= routes["zero_route_imputed_flag"] & (
        routes["analysis_road_duration_min"].abs() > FLOAT_TOLERANCE
    )
    if invalid_zero.any():
        raise ValueError("0분 보정 경로의 분석용 거리·시간이 0이 아닙니다.")

    route_ids = set(routes["route_request_id"])
    missing_current_ids = set(matrix["current_route_request_id"]) - route_ids
    missing_integrated_ids = set(matrix["integrated_route_request_id"]) - route_ids
    if missing_current_ids or missing_integrated_ids:
        raise ValueError(
            "학생-통합학교안 행렬의 요청키가 경로표에 없습니다: "
            f"current={len(missing_current_ids)}, integrated={len(missing_integrated_ids)}"
        )

    route_keep = [
        "route_request_id", "analysis_status", "analysis_eligible_flag",
        "analysis_exclusion_reason", "analysis_road_distance_m",
        "analysis_road_duration_min", "analysis_road_to_straight_ratio",
        "ferry_flag", "zero_route_imputed_flag", "merge_check",
    ]

    def prefixed_route_table(prefix: str) -> pd.DataFrame:
        renamed = {column: f"{prefix}_{column}" for column in route_keep if column != "route_request_id"}
        return routes[route_keep].rename(columns=renamed)

    detail = matrix.merge(
        prefixed_route_table("current"),
        left_on="current_route_request_id",
        right_on="route_request_id",
        how="left",
        validate="many_to_one",
    ).drop(columns="route_request_id")
    detail = detail.merge(
        prefixed_route_table("integrated"),
        left_on="integrated_route_request_id",
        right_on="route_request_id",
        how="left",
        validate="many_to_one",
    ).drop(columns="route_request_id")

    if len(detail) != len(matrix):
        raise ValueError(f"경로 연결 전후 행 수 변화: {len(matrix)} -> {len(detail)}")
    if detail.duplicated(matrix_key).any():
        raise ValueError("경로 연결 후 학생-통합학교안 키가 중복되었습니다.")

    detail["current_analysis_eligible_flag"] = to_bool(
        detail["current_analysis_eligible_flag"]
    )
    detail["integrated_analysis_eligible_flag"] = to_bool(
        detail["integrated_analysis_eligible_flag"]
    )
    detail["current_ferry_flag"] = to_bool(detail["current_ferry_flag"])
    detail["integrated_ferry_flag"] = to_bool(detail["integrated_ferry_flag"])
    detail["current_zero_route_imputed_flag"] = to_bool(
        detail["current_zero_route_imputed_flag"]
    )
    detail["integrated_zero_route_imputed_flag"] = to_bool(
        detail["integrated_zero_route_imputed_flag"]
    )

    current_ok = detail["current_analysis_eligible_flag"]
    integrated_ok = detail["integrated_analysis_eligible_flag"]
    detail["commute_analysis_eligible_flag"] = current_ok & integrated_ok
    detail["commute_analysis_status"] = np.select(
        [
            current_ok & integrated_ok,
            ~current_ok & integrated_ok,
            current_ok & ~integrated_ok,
        ],
        [
            "READY",
            "EXCLUDED_CURRENT_ROUTE",
            "EXCLUDED_INTEGRATED_ROUTE",
        ],
        default="EXCLUDED_BOTH_ROUTES",
    )

    analysis_ok = detail["commute_analysis_eligible_flag"]
    current_distance = detail["current_analysis_road_distance_m"]
    integrated_distance = detail["integrated_analysis_road_distance_m"]
    current_time = detail["current_analysis_road_duration_min"]
    integrated_time = detail["integrated_analysis_road_duration_min"]

    detail["current_road_distance_km"] = np.where(
        analysis_ok, current_distance / 1000.0, np.nan
    )
    detail["integrated_road_distance_km"] = np.where(
        analysis_ok, integrated_distance / 1000.0, np.nan
    )
    detail["road_distance_change_km"] = np.where(
        analysis_ok, (integrated_distance - current_distance) / 1000.0, np.nan
    )
    detail["additional_distance_burden_km"] = np.where(
        analysis_ok,
        np.maximum(detail["road_distance_change_km"], 0.0),
        np.nan,
    )
    detail["road_duration_change_min"] = np.where(
        analysis_ok, integrated_time - current_time, np.nan
    )
    detail["additional_time_burden_min"] = np.where(
        analysis_ok,
        np.maximum(detail["road_duration_change_min"], 0.0),
        np.nan,
    )
    detail["current_over_30_min_flag"] = np.where(
        analysis_ok, current_time > THIRTY_MINUTE_THRESHOLD, False
    )
    detail["integrated_over_30_min_flag"] = np.where(
        analysis_ok, integrated_time > THIRTY_MINUTE_THRESHOLD, False
    )
    detail["newly_over_30_min_flag"] = (
        detail["integrated_over_30_min_flag"] & ~detail["current_over_30_min_flag"]
    )
    detail["time_increase_flag"] = np.where(
        analysis_ok, detail["road_duration_change_min"] > FLOAT_TOLERANCE, False
    )
    detail["time_decrease_flag"] = np.where(
        analysis_ok, detail["road_duration_change_min"] < -FLOAT_TOLERANCE, False
    )

    moved = to_bool(detail["moved_student_flag"])
    same_route = to_bool(detail["same_route_reuse_flag"])
    if (moved == same_route).any():
        examples = detail.loc[
            moved == same_route,
            [
                "demand_assignment_id", "integration_option_id",
                "current_school_key", "integrated_school_key",
                "moved_student_flag", "same_route_reuse_flag",
            ],
        ].head(5)
        raise ValueError(
            "moved_student_flag와 same_route_reuse_flag의 관계가 모순입니다:\n"
            f"{examples.to_string(index=False)}"
        )
    unequal_same_route_time = analysis_ok & same_route & (
        detail["road_duration_change_min"].abs() > FLOAT_TOLERANCE
    )
    unequal_same_route_distance = analysis_ok & same_route & (
        detail["road_distance_change_km"].abs() > FLOAT_TOLERANCE
    )
    if unequal_same_route_time.any() or unequal_same_route_distance.any():
        raise ValueError("같은 경로를 재사용한 행에서 통합 전후 시간·거리가 달라졌습니다.")

    # ------------------------------------------------------------------
    # 통합학교안별 학생 가중 결과 요약
    # ------------------------------------------------------------------
    option_identity = [
        "final_cluster_id", "candidate_id", "capacity_scenario_id",
        "scenario_role", "integration_option_id",
    ]
    option_metadata_candidates = [
        "integrated_school_key", "integrated_school_name",
        "integrated_school_edu_office", "integrated_school_area_type",
        "integrated_school_population_decline_type",
        "merged_school_1_key", "merged_school_1_name",
        "merged_school_2_key", "merged_school_2_name",
        "students_per_class", "total_demand", "retained_capacity",
        "capacity_margin", "capacity_utilization_rate",
        "cluster_decline_school_count", "cluster_concern_school_count",
        "cluster_population_decline_group", "cluster_urban_school_count",
        "cluster_eup_school_count", "cluster_rural_school_count",
    ]
    option_metadata = [column for column in option_metadata_candidates if column in detail.columns]

    option_rows: list[dict[str, object]] = []
    for key, group in detail.groupby(option_identity, sort=True, dropna=False):
        row = dict(zip(option_identity, key))
        for column in option_metadata:
            if unique_non_null_count(group, column) > 1:
                raise ValueError(
                    f"통합학교안 {row['integration_option_id']}에서 {column} 값이 여러 개입니다."
                )
            row[column] = first_value(group, column)

        eligible = group["commute_analysis_eligible_flag"]
        eligible_group = group.loc[eligible].copy()
        excluded_group = group.loc[~eligible].copy()
        moved_group = eligible_group.loc[to_bool(eligible_group["moved_student_flag"])].copy()
        weights = eligible_group["student_weight"]
        moved_weights = moved_group["student_weight"]

        total_weight = float(group["student_weight"].sum())
        eligible_weight = float(weights.sum())
        excluded_weight = float(excluded_group["student_weight"].sum())
        coverage_rate = eligible_weight / total_weight if total_weight > 0 else np.nan

        integrated_over_30_weight = float(
            eligible_group.loc[
                eligible_group["integrated_over_30_min_flag"], "student_weight"
            ].sum()
        )
        newly_over_30_weight = float(
            eligible_group.loc[
                eligible_group["newly_over_30_min_flag"], "student_weight"
            ].sum()
        )
        current_over_30_weight = float(
            eligible_group.loc[
                eligible_group["current_over_30_min_flag"], "student_weight"
            ].sum()
        )
        time_increase_weight = float(
            eligible_group.loc[eligible_group["time_increase_flag"], "student_weight"].sum()
        )
        time_decrease_weight = float(
            eligible_group.loc[eligible_group["time_decrease_flag"], "student_weight"].sum()
        )

        row.update(
            {
                "option_analysis_status": (
                    "READY" if len(excluded_group) == 0
                    else "PARTIAL_ROUTE_EXCLUSION" if len(eligible_group) > 0
                    else "EXCLUDED_NO_USABLE_DEMAND"
                ),
                "demand_assignment_row_count": len(group),
                "unique_census_point_count": group["census_id"].nunique(),
                "student_weight_total": total_weight,
                "analysis_eligible_row_count": len(eligible_group),
                "analysis_excluded_row_count": len(excluded_group),
                "analysis_eligible_student_weight": eligible_weight,
                "analysis_excluded_student_weight": excluded_weight,
                "analysis_student_coverage_rate": coverage_rate,
                "ferry_affected_row_count": int(
                    (group["current_ferry_flag"] | group["integrated_ferry_flag"]).sum()
                ),
                "moved_analysis_eligible_student_weight": float(moved_weights.sum()),
                "current_weighted_mean_duration_min": weighted_mean(
                    eligible_group["current_analysis_road_duration_min"], weights
                ),
                "integrated_weighted_mean_duration_min": weighted_mean(
                    eligible_group["integrated_analysis_road_duration_min"], weights
                ),
                "weighted_mean_duration_change_min": weighted_mean(
                    eligible_group["road_duration_change_min"], weights
                ),
                "weighted_mean_additional_time_burden_min": weighted_mean(
                    eligible_group["additional_time_burden_min"], weights
                ),
                "weighted_p50_duration_change_min": weighted_quantile(
                    eligible_group["road_duration_change_min"], weights, 0.50
                ),
                "weighted_p90_duration_change_min": weighted_quantile(
                    eligible_group["road_duration_change_min"], weights, 0.90
                ),
                "max_duration_change_min": (
                    float(eligible_group["road_duration_change_min"].max())
                    if len(eligible_group) else np.nan
                ),
                "max_integrated_duration_min": (
                    float(eligible_group["integrated_analysis_road_duration_min"].max())
                    if len(eligible_group) else np.nan
                ),
                "moved_weighted_mean_duration_change_min": weighted_mean(
                    moved_group["road_duration_change_min"], moved_weights
                ),
                "moved_weighted_p90_duration_change_min": weighted_quantile(
                    moved_group["road_duration_change_min"], moved_weights, 0.90
                ),
                "moved_max_duration_change_min": (
                    float(moved_group["road_duration_change_min"].max())
                    if len(moved_group) else np.nan
                ),
                "current_weighted_mean_distance_km": weighted_mean(
                    eligible_group["current_road_distance_km"], weights
                ),
                "integrated_weighted_mean_distance_km": weighted_mean(
                    eligible_group["integrated_road_distance_km"], weights
                ),
                "weighted_mean_distance_change_km": weighted_mean(
                    eligible_group["road_distance_change_km"], weights
                ),
                "weighted_mean_additional_distance_burden_km": weighted_mean(
                    eligible_group["additional_distance_burden_km"], weights
                ),
                "current_over_30_student_weight": current_over_30_weight,
                "integrated_over_30_student_weight": integrated_over_30_weight,
                "integrated_over_30_student_rate": (
                    integrated_over_30_weight / eligible_weight if eligible_weight > 0 else np.nan
                ),
                "newly_over_30_student_weight": newly_over_30_weight,
                "newly_over_30_student_rate": (
                    newly_over_30_weight / eligible_weight if eligible_weight > 0 else np.nan
                ),
                "time_increase_student_weight": time_increase_weight,
                "time_decrease_student_weight": time_decrease_weight,
                "any_integrated_over_30_flag": integrated_over_30_weight > WEIGHT_TOLERANCE,
            }
        )

        if len(eligible_group):
            worst_index = eligible_group["road_duration_change_min"].idxmax()
            worst = eligible_group.loc[worst_index]
            row.update(
                {
                    "worst_census_id": worst["census_id"],
                    "worst_current_school_key": worst["current_school_key"],
                    "worst_current_school_name": worst["current_school_name"],
                    "worst_student_weight": worst["student_weight"],
                    "worst_current_duration_min": worst["current_analysis_road_duration_min"],
                    "worst_integrated_duration_min": worst["integrated_analysis_road_duration_min"],
                    "worst_duration_change_min": worst["road_duration_change_min"],
                }
            )
        else:
            for column in [
                "worst_census_id", "worst_current_school_key", "worst_current_school_name",
                "worst_student_weight", "worst_current_duration_min",
                "worst_integrated_duration_min", "worst_duration_change_min",
            ]:
                row[column] = np.nan
        option_rows.append(row)

    option_summary = pd.DataFrame(option_rows)

    # ------------------------------------------------------------------
    # 전역 품질 검증
    # ------------------------------------------------------------------
    expected_option_count = matrix["integration_option_id"].nunique()
    if len(option_summary) != expected_option_count:
        raise ValueError(
            f"통합학교안 요약 행 수 불일치: 기대 {expected_option_count}, 실제 {len(option_summary)}"
        )
    if option_summary["integration_option_id"].duplicated().any():
        raise ValueError("통합학교안 요약에 integration_option_id 중복이 있습니다.")

    expected_weight_by_option = matrix.groupby("integration_option_id")["student_weight"].sum()
    actual_weight_by_option = option_summary.set_index("integration_option_id")[
        "student_weight_total"
    ]
    weight_error = (
        expected_weight_by_option.sort_index() - actual_weight_by_option.sort_index()
    ).abs()
    max_option_weight_error = float(weight_error.max()) if len(weight_error) else 0.0
    if max_option_weight_error > WEIGHT_TOLERANCE:
        raise ValueError(
            f"통합학교안별 학생 가중치 합계 불일치: 최대오차={max_option_weight_error}"
        )

    eligible_detail = detail["commute_analysis_eligible_flag"]
    excluded_detail = ~eligible_detail
    if detail.loc[eligible_detail, [
        "road_duration_change_min", "road_distance_change_km"
    ]].isna().any().any():
        raise ValueError("분석 가능 행의 통합 전후 시간·거리 차이에 결측치가 있습니다.")
    if detail.loc[excluded_detail, [
        "road_duration_change_min", "road_distance_change_km"
    ]].notna().any().any():
        raise ValueError("분석 제외 행에 통합 전후 시간·거리 차이가 계산되어 있습니다.")

    # ------------------------------------------------------------------
    # 검증 요약표
    # ------------------------------------------------------------------
    summary_rows: list[dict[str, object]] = []

    def add_summary(
        section: str,
        metric: str,
        value: object,
        unit: str,
        check_status: str,
        note: str,
    ) -> None:
        summary_rows.append(
            {
                "section": section,
                "metric": metric,
                "value": value,
                "unit": unit,
                "check_status": check_status,
                "note": note,
            }
        )

    excluded_current = detail["commute_analysis_status"].eq("EXCLUDED_CURRENT_ROUTE")
    excluded_integrated = detail["commute_analysis_status"].eq("EXCLUDED_INTEGRATED_ROUTE")
    excluded_both = detail["commute_analysis_status"].eq("EXCLUDED_BOTH_ROUTES")
    over5_current = detail["current_analysis_road_to_straight_ratio"] > 5
    over5_integrated = detail["integrated_analysis_road_to_straight_ratio"] > 5

    add_summary("OVERALL", "overall_status", "READY", "status", "PASS", "04-4 계산 완료")
    add_summary("INPUT", "matrix_row_count", len(matrix), "rows", "PASS", MATRIX_FILE)
    add_summary("INPUT", "matrix_column_count", len(matrix.columns), "columns", "PASS", MATRIX_FILE)
    add_summary("INPUT", "route_row_count", len(routes), "routes", "PASS", ROUTE_FILE)
    add_summary("INPUT", "unique_route_request_count", routes["route_request_id"].nunique(), "routes", "PASS", "중복 없음")
    add_summary("JOIN", "missing_current_route_key_count", len(missing_current_ids), "keys", "PASS", "0이어야 함")
    add_summary("JOIN", "missing_integrated_route_key_count", len(missing_integrated_ids), "keys", "PASS", "0이어야 함")
    add_summary("JOIN", "detail_row_count_after_join", len(detail), "rows", "PASS", "입력 행 수 보존")
    add_summary("ANALYSIS", "analysis_eligible_row_count", int(eligible_detail.sum()), "rows", "PASS", "통합 전후 모두 분석 가능")
    add_summary("ANALYSIS", "analysis_excluded_row_count", int(excluded_detail.sum()), "rows", "INFO", "페리 경로 영향")
    add_summary("ANALYSIS", "excluded_current_route_row_count", int(excluded_current.sum()), "rows", "INFO", "현재 경로 제외")
    add_summary("ANALYSIS", "excluded_integrated_route_row_count", int(excluded_integrated.sum()), "rows", "INFO", "통합 후 경로 제외")
    add_summary("ANALYSIS", "excluded_both_routes_row_count", int(excluded_both.sum()), "rows", "INFO", "두 경로 모두 제외")
    add_summary("ANALYSIS", "analysis_eligible_student_weight", float(detail.loc[eligible_detail, "student_weight"].sum()), "students", "PASS", "행렬 전체에서 옵션 중복 포함")
    add_summary("ANALYSIS", "analysis_excluded_student_weight", float(detail.loc[excluded_detail, "student_weight"].sum()), "students", "INFO", "행렬 전체에서 옵션 중복 포함")
    add_summary("RULE", "zero_route_detail_row_count", int((detail["current_zero_route_imputed_flag"] | detail["integrated_zero_route_imputed_flag"]).sum()), "rows", "PASS", "0분으로 분석 포함")
    add_summary("RULE", "ferry_affected_detail_row_count", int((detail["current_ferry_flag"] | detail["integrated_ferry_flag"]).sum()), "rows", "PASS", "분석 제외")
    add_summary("RULE", "road_to_straight_over_5_detail_row_count", int((over5_current | over5_integrated).sum()), "rows", "PASS", "제외하지 않음")
    add_summary("QA", "duplicated_detail_key_count", int(detail.duplicated(matrix_key).sum()), "rows", "PASS", "통합학교안-수요배정키")
    add_summary("QA", "same_route_nonzero_change_count", int((unequal_same_route_time | unequal_same_route_distance).sum()), "rows", "PASS", "0이어야 함")
    add_summary("QA", "max_option_student_weight_error", max_option_weight_error, "students", "PASS", f"허용오차 {WEIGHT_TOLERANCE}")
    add_summary("SCOPE", "final_cluster_count", detail["final_cluster_id"].nunique(), "clusters", "PASS", "비중복 최종 군집")
    add_summary("SCOPE", "integration_option_count", expected_option_count, "options", "PASS", "통합학교안")
    for scenario, group in detail.groupby("capacity_scenario_id", sort=True):
        add_summary("SCENARIO", f"{scenario}_cluster_count", group["final_cluster_id"].nunique(), "clusters", "PASS", "수용력 시나리오")
        add_summary("SCENARIO", f"{scenario}_option_count", group["integration_option_id"].nunique(), "options", "PASS", "통합학교안")
        add_summary("SCENARIO", f"{scenario}_detail_row_count", len(group), "rows", "PASS", "수요배정행 x 통합학교안")

    add_summary("OUTPUT", "detail_output_rows", len(detail), "rows", "PASS", DETAIL_OUTPUT_FILE)
    add_summary("OUTPUT", "detail_output_columns", len(detail.columns), "columns", "PASS", DETAIL_OUTPUT_FILE)
    add_summary("OUTPUT", "option_output_rows", len(option_summary), "rows", "PASS", OPTION_OUTPUT_FILE)
    add_summary("OUTPUT", "option_output_columns", len(option_summary.columns), "columns", "PASS", OPTION_OUTPUT_FILE)

    calculation_summary = pd.DataFrame(summary_rows)
    add_summary_count = len(calculation_summary)
    # 요약표 자신의 크기를 마지막 두 행으로 기록한다.
    calculation_summary = pd.concat(
        [
            calculation_summary,
            pd.DataFrame(
                [
                    {
                        "section": "OUTPUT",
                        "metric": "calculation_summary_rows",
                        "value": add_summary_count + 2,
                        "unit": "rows",
                        "check_status": "PASS",
                        "note": SUMMARY_OUTPUT_FILE,
                    },
                    {
                        "section": "OUTPUT",
                        "metric": "calculation_summary_columns",
                        "value": 6,
                        "unit": "columns",
                        "check_status": "PASS",
                        "note": SUMMARY_OUTPUT_FILE,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    # 계산값은 반올림하지 않는다. 정렬만 고정하여 재현성을 확보한다.
    detail = detail.sort_values(
        [
            "final_cluster_id", "capacity_scenario_id", "integration_option_id",
            "census_id", "current_school_key",
        ],
        kind="stable",
    ).reset_index(drop=True)
    option_summary = option_summary.sort_values(
        ["final_cluster_id", "capacity_scenario_id", "integration_option_id"],
        kind="stable",
    ).reset_index(drop=True)

    detail_path = output_dir / DETAIL_OUTPUT_FILE
    option_path = output_dir / OPTION_OUTPUT_FILE
    summary_path = output_dir / SUMMARY_OUTPUT_FILE
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    option_summary.to_csv(option_path, index=False, encoding="utf-8-sig")
    calculation_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n[산출물]")
    print(f"- {DETAIL_OUTPUT_FILE}: {len(detail):,}행 x {len(detail.columns):,}열")
    print(f"- {OPTION_OUTPUT_FILE}: {len(option_summary):,}행 x {len(option_summary.columns):,}열")
    print(
        f"- {SUMMARY_OUTPUT_FILE}: "
        f"{len(calculation_summary):,}행 x {len(calculation_summary.columns):,}열"
    )
    print("\n[핵심 상태]")
    print(f"- 분석 가능 상세행: {int(eligible_detail.sum()):,}")
    print(f"- 페리 영향 분석 제외행: {int(excluded_detail.sum()):,}")
    print(f"- 통합학교안: {len(option_summary):,}개")
    print("- 계산 중 반올림 없음")
    print("- 04-4 완료")


if __name__ == "__main__":
    main()

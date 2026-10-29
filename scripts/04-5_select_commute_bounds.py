"""
04-5. 군집별 통합학교안 하한·상한 선택 및 전국 통학부담 집계

이 스크립트가 하는 일
1) 04-4의 통합학교안별 결과와 수요점별 상세 결과를 불러온다.
2) 최종 군집마다 통합학교안을 두 가지 학생 범위에서 비교한다.
   - ALL_STUDENTS: 통합학교에 계속 다니는 학생까지 포함한 군집 전체 학생
   - MOVED_STUDENTS: 폐교되어 다른 학교로 옮기는 학생만 포함
3) 각 학생 범위에서 가중 평균 통학시간 변화가 가장 작은 안을 하한,
   가장 큰 안을 상한으로 선택한다.
4) 선택된 통합학교안의 수요점 상세행을 연결하여 전국 학생 가중 결과를 집계한다.
5) 평균뿐 아니라 상위 10%, 최댓값, 30분 초과 학생 규모, 거리 변화도 함께 산출한다.
6) 통합학교안이 1개뿐인 군집은 같은 안을 하한과 상한에 모두 기록한다.

중요한 분석 원칙
- 시간 변화 = 통합 후 - 통합 전이며 음수를 0으로 바꾸지 않는다.
- 순증감과 별도로 양수 증가분만 계산한 additional_time_burden도 보존한다.
- ALL_STUDENTS의 선택 기준은 weighted_mean_duration_change_min이다.
- MOVED_STUDENTS의 선택 기준은 moved_weighted_mean_duration_change_min이다.
- 하한·상한 선택에는 실제 집계구 학생 가중 통학시간을 사용한다.
  앞 단계의 학교 간 시간은 선택 기준으로 다시 사용하지 않는다.
- option_analysis_status=READY인 안이 하나라도 있으면 READY 안끼리 비교한다.
  READY 안이 전혀 없는 군집은 PARTIAL_ROUTE_EXCLUSION 안을 사용하되 상태를 기록한다.
- 페리로 제외된 상세행은 전국 집계에서도 제외하고 포착률을 별도로 제공한다.
- ROAD_TO_STRAIGHT_OVER_5는 계속 포함한다.
- 계산 과정에서는 반올림하지 않는다.
- 앞 단계에서 확정된 720개 군집과 학교 표본을 그대로 신뢰하며 재필터링하지 않는다.

입력 파일
- data/03_outputs/commute_time_option_summary.csv
- data/03_outputs/commute_time_demand_option.csv

산출물
- commute_bound_cluster_selections.csv
  군집 x 선택학생범위 x 하한/상한별로 선택된 통합학교안
- commute_bound_demand_results.csv
  선택된 통합학교안의 수요점 상세 결과. 후속 지역·사례 분석 기준 파일
- commute_bound_national_summary.csv
  선택학생범위 x 하한/상한 x 집계학생범위별 전국 학생 가중 통학부담
- commute_bound_validation_summary.csv
  입력·선택·조인·가중치·출력 검증표

실행 예시
    python scripts/04-5_select_commute_bounds.py

다른 프로젝트 경로를 직접 지정하는 경우
    python scripts/04-5_select_commute_bounds.py --project-root "C:/.../final_project"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OPTION_FILE = "commute_time_option_summary.csv"
DETAIL_FILE = "commute_time_demand_option.csv"

SELECTION_OUTPUT_FILE = "commute_bound_cluster_selections.csv"
DEMAND_OUTPUT_FILE = "commute_bound_demand_results.csv"
NATIONAL_OUTPUT_FILE = "commute_bound_national_summary.csv"
VALIDATION_OUTPUT_FILE = "commute_bound_validation_summary.csv"

FLOAT_TOLERANCE = 1e-9
WEIGHT_TOLERANCE = 1e-6

SELECTION_SPECS = {
    "ALL_STUDENTS": {
        "metric": "weighted_mean_duration_change_min",
        "p90": "weighted_p90_duration_change_min",
        "maximum": "max_duration_change_min",
    },
    "MOVED_STUDENTS": {
        "metric": "moved_weighted_mean_duration_change_min",
        "p90": "moved_weighted_p90_duration_change_min",
        "maximum": "moved_max_duration_change_min",
    },
}

BOUND_TYPES = ("LOWER_BOUND", "UPPER_BOUND")
EVALUATION_POPULATIONS = ("ALL_STUDENTS", "MOVED_STUDENTS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="04-5 통학시간 하한·상한 분석")
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
        print(f"[주의] {filename}이 여러 폴더에 있습니다. 우선 파일 사용: {matches[0]}")
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
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
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


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = project_root / "data" / "03_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    option_path = find_input(project_root, OPTION_FILE)
    detail_path = find_input(project_root, DETAIL_FILE)
    print("[입력 파일]")
    print(f"- 통합학교안 요약: {option_path}")
    print(f"- 수요점 상세 결과: {detail_path}")

    options = read_csv_as_text(option_path)
    detail = read_csv_as_text(detail_path)

    option_required = [
        "final_cluster_id", "candidate_id", "capacity_scenario_id",
        "integration_option_id", "integrated_school_key", "integrated_school_name",
        "option_analysis_status", "student_weight_total",
        "analysis_eligible_student_weight", "analysis_excluded_student_weight",
        "analysis_student_coverage_rate", "weighted_mean_duration_change_min",
        "weighted_p90_duration_change_min", "max_duration_change_min",
        "moved_weighted_mean_duration_change_min",
        "moved_weighted_p90_duration_change_min", "moved_max_duration_change_min",
    ]
    detail_required = [
        "demand_assignment_id", "census_id", "student_weight", "final_cluster_id",
        "integration_option_id", "current_school_key", "current_school_name",
        "integrated_school_key", "integrated_school_name", "moved_student_flag",
        "commute_analysis_eligible_flag", "commute_analysis_status",
        "current_analysis_road_duration_min", "integrated_analysis_road_duration_min",
        "road_duration_change_min", "additional_time_burden_min",
        "current_road_distance_km", "integrated_road_distance_km",
        "road_distance_change_km", "additional_distance_burden_km",
        "current_over_30_min_flag", "integrated_over_30_min_flag",
        "newly_over_30_min_flag", "time_increase_flag", "time_decrease_flag",
    ]
    require_columns(options, option_required, OPTION_FILE)
    require_columns(detail, detail_required, DETAIL_FILE)

    if options.empty or detail.empty:
        raise ValueError("입력 파일이 비어 있습니다.")
    if options["integration_option_id"].duplicated().any():
        raise ValueError("통합학교안 요약에 integration_option_id 중복이 있습니다.")
    detail_key = ["integration_option_id", "demand_assignment_id"]
    if detail.duplicated(detail_key).any():
        raise ValueError("수요점 상세 결과에 integration_option_id-demand_assignment_id 중복이 있습니다.")
    if set(detail["integration_option_id"]) != set(options["integration_option_id"]):
        raise ValueError("통합학교안 요약과 상세 결과의 integration_option_id 집합이 다릅니다.")

    option_numeric = [
        "student_weight_total", "analysis_eligible_student_weight",
        "analysis_excluded_student_weight", "analysis_student_coverage_rate",
        "weighted_mean_duration_change_min", "weighted_p90_duration_change_min",
        "max_duration_change_min", "moved_weighted_mean_duration_change_min",
        "moved_weighted_p90_duration_change_min", "moved_max_duration_change_min",
    ]
    detail_numeric = [
        "student_weight", "current_analysis_road_duration_min",
        "integrated_analysis_road_duration_min", "road_duration_change_min",
        "additional_time_burden_min", "current_road_distance_km",
        "integrated_road_distance_km", "road_distance_change_km",
        "additional_distance_burden_km",
    ]
    to_numeric_strict(options, option_numeric, OPTION_FILE)
    to_numeric_strict(detail, detail_numeric, DETAIL_FILE)
    if detail["student_weight"].isna().any() or (detail["student_weight"] <= 0).any():
        raise ValueError("상세 결과의 student_weight는 모두 0보다 커야 합니다.")

    detail["moved_student_flag"] = to_bool(detail["moved_student_flag"])
    detail["commute_analysis_eligible_flag"] = to_bool(
        detail["commute_analysis_eligible_flag"]
    )
    for column in [
        "current_over_30_min_flag", "integrated_over_30_min_flag",
        "newly_over_30_min_flag", "time_increase_flag", "time_decrease_flag",
    ]:
        detail[column] = to_bool(detail[column])

    # 군집 안에서는 모든 통합학교안이 동일한 전체 학생 모집단을 가져야 한다.
    cluster_weight_range = options.groupby("final_cluster_id")["student_weight_total"].agg(
        lambda values: float(values.max() - values.min())
    )
    max_cluster_weight_range = float(cluster_weight_range.max())
    if max_cluster_weight_range > WEIGHT_TOLERANCE:
        raise ValueError(
            "동일 군집의 통합학교안별 전체 학생 가중치가 다릅니다: "
            f"최대 범위={max_cluster_weight_range}"
        )

    # ------------------------------------------------------------------
    # 1. 군집별 하한·상한 통합학교안 선택
    # ------------------------------------------------------------------
    selection_rows: list[pd.Series] = []
    cluster_option_counts = options.groupby("final_cluster_id").size().to_dict()

    for cluster_id, cluster_options in options.groupby("final_cluster_id", sort=True):
        ready_mask = cluster_options["option_analysis_status"].eq("READY")
        if ready_mask.any():
            pool = cluster_options.loc[ready_mask].copy()
            pool_status = "READY_ONLY"
        else:
            pool = cluster_options.copy()
            pool_status = "PARTIAL_ONLY"

        for selection_population, spec in SELECTION_SPECS.items():
            metric = spec["metric"]
            p90_metric = spec["p90"]
            max_metric = spec["maximum"]
            if pool[metric].isna().any():
                bad = pool.loc[pool[metric].isna(), "integration_option_id"].tolist()
                raise ValueError(
                    f"군집 {cluster_id}, {selection_population} 선택지표 결측: {bad}"
                )

            for bound_type in BOUND_TYPES:
                ascending = bound_type == "LOWER_BOUND"
                ranked = pool.sort_values(
                    [metric, p90_metric, max_metric, "integrated_school_key"],
                    ascending=[ascending, ascending, ascending, True],
                    kind="stable",
                ).reset_index(drop=True)
                selected = ranked.iloc[0].copy()
                selected_value = float(selected[metric])
                tie_count = int(
                    np.isclose(
                        pool[metric].to_numpy(dtype=float),
                        selected_value,
                        atol=FLOAT_TOLERANCE,
                        rtol=0,
                    ).sum()
                )
                selected["selection_population"] = selection_population
                selected["bound_type"] = bound_type
                selected["selection_metric"] = metric
                selected["selection_metric_value"] = selected_value
                selected["selection_pool_status"] = pool_status
                selected["available_option_count"] = int(cluster_option_counts[cluster_id])
                selected["compared_option_count"] = len(pool)
                selected["primary_metric_tie_count"] = tie_count
                selected["selection_id"] = (
                    f"{cluster_id}__{selection_population}__{bound_type}"
                )
                selection_rows.append(selected)

    selections = pd.DataFrame(selection_rows)
    selection_key = ["final_cluster_id", "selection_population", "bound_type"]
    expected_selection_rows = options["final_cluster_id"].nunique() * len(SELECTION_SPECS) * 2
    if len(selections) != expected_selection_rows:
        raise ValueError(
            f"하한·상한 선택 행 수 불일치: 기대 {expected_selection_rows}, 실제 {len(selections)}"
        )
    if selections.duplicated(selection_key).any() or selections["selection_id"].duplicated().any():
        raise ValueError("군집-선택학생범위-하한/상한 선택키가 중복되었습니다.")

    # 동일 관점에서 하한과 상한이 같은 통합학교안인지 표시한다.
    bound_pair = selections.pivot(
        index=["final_cluster_id", "selection_population"],
        columns="bound_type",
        values="integration_option_id",
    )
    bound_pair["same_option_for_bounds_flag"] = (
        bound_pair["LOWER_BOUND"] == bound_pair["UPPER_BOUND"]
    )
    selections = selections.merge(
        bound_pair[["same_option_for_bounds_flag"]].reset_index(),
        on=["final_cluster_id", "selection_population"],
        how="left",
        validate="many_to_one",
    )

    # 하한은 상한보다 클 수 없다.
    metric_check = selections.pivot(
        index=["final_cluster_id", "selection_population"],
        columns="bound_type",
        values="selection_metric_value",
    )
    invalid_bounds = metric_check["LOWER_BOUND"] > metric_check["UPPER_BOUND"] + FLOAT_TOLERANCE
    if invalid_bounds.any():
        raise ValueError(f"하한이 상한보다 큰 군집-관점이 {int(invalid_bounds.sum())}개 있습니다.")

    # ------------------------------------------------------------------
    # 2. 선택된 통합학교안과 수요점 상세 결과 연결
    # ------------------------------------------------------------------
    selection_tags = selections[
        [
            "selection_id", "final_cluster_id", "selection_population", "bound_type",
            "selection_metric", "selection_metric_value", "selection_pool_status",
            "available_option_count", "compared_option_count",
            "primary_metric_tie_count", "same_option_for_bounds_flag",
            "integration_option_id",
        ]
    ]
    selected_demand = selection_tags.merge(
        detail,
        on=["final_cluster_id", "integration_option_id"],
        how="left",
        # 동일 통합학교안이 전체학생/이동학생 관점 또는 하한/상한에서
        # 반복 선택될 수 있으므로 조인키 자체는 many-to-many가 정상이다.
        # 실제 중복 오류는 아래 selection_id-demand_assignment_id로 검증한다.
        validate="many_to_many",
    )
    if selected_demand["demand_assignment_id"].isna().any():
        raise ValueError("선택된 통합학교안의 상세행 연결에 실패했습니다.")
    if selected_demand.duplicated(["selection_id", "demand_assignment_id"]).any():
        raise ValueError("선택 결과 상세표에 selection_id-demand_assignment_id 중복이 있습니다.")

    expected_demand_rows = int(
        selection_tags.merge(
            detail.groupby("integration_option_id").size().rename("row_count").reset_index(),
            on="integration_option_id",
            how="left",
            validate="many_to_one",
        )["row_count"].sum()
    )
    if len(selected_demand) != expected_demand_rows:
        raise ValueError(
            f"선택 상세행 수 불일치: 기대 {expected_demand_rows}, 실제 {len(selected_demand)}"
        )

    # ------------------------------------------------------------------
    # 3. 선택 관점·하한/상한·집계 학생 범위별 전국 결과
    # ------------------------------------------------------------------
    national_rows: list[dict[str, object]] = []
    for selection_population in SELECTION_SPECS:
        for bound_type in BOUND_TYPES:
            selected_scope = selected_demand[
                selected_demand["selection_population"].eq(selection_population)
                & selected_demand["bound_type"].eq(bound_type)
            ].copy()
            if selected_scope["final_cluster_id"].nunique() != options["final_cluster_id"].nunique():
                raise ValueError(
                    f"{selection_population}-{bound_type}에서 일부 군집이 누락되었습니다."
                )

            for evaluation_population in EVALUATION_POPULATIONS:
                if evaluation_population == "MOVED_STUDENTS":
                    population = selected_scope[selected_scope["moved_student_flag"]].copy()
                else:
                    population = selected_scope.copy()

                eligible = population["commute_analysis_eligible_flag"]
                usable = population.loc[eligible].copy()
                weights = usable["student_weight"]
                total_weight = float(population["student_weight"].sum())
                eligible_weight = float(weights.sum())
                excluded_weight = float(population.loc[~eligible, "student_weight"].sum())

                integrated_over_30_weight = float(
                    usable.loc[usable["integrated_over_30_min_flag"], "student_weight"].sum()
                )
                newly_over_30_weight = float(
                    usable.loc[usable["newly_over_30_min_flag"], "student_weight"].sum()
                )
                current_over_30_weight = float(
                    usable.loc[usable["current_over_30_min_flag"], "student_weight"].sum()
                )
                time_increase_weight = float(
                    usable.loc[usable["time_increase_flag"], "student_weight"].sum()
                )
                time_decrease_weight = float(
                    usable.loc[usable["time_decrease_flag"], "student_weight"].sum()
                )

                cluster_over_30 = usable.groupby("final_cluster_id")[
                    "integrated_over_30_min_flag"
                ].any()
                cluster_newly_over_30 = usable.groupby("final_cluster_id")[
                    "newly_over_30_min_flag"
                ].any()

                national_rows.append(
                    {
                        "selection_population": selection_population,
                        "bound_type": bound_type,
                        "evaluation_population": evaluation_population,
                        "selection_metric": SELECTION_SPECS[selection_population]["metric"],
                        "selected_cluster_count": selected_scope["final_cluster_id"].nunique(),
                        "selected_option_count": selected_scope["integration_option_id"].nunique(),
                        "partial_selection_cluster_count": selected_scope.loc[
                            selected_scope["selection_pool_status"].eq("PARTIAL_ONLY"),
                            "final_cluster_id",
                        ].nunique(),
                        "demand_assignment_row_count": len(population),
                        "unique_census_point_count": population["census_id"].nunique(),
                        "student_weight_total": total_weight,
                        "analysis_eligible_student_weight": eligible_weight,
                        "analysis_excluded_student_weight": excluded_weight,
                        "analysis_student_coverage_rate": (
                            eligible_weight / total_weight if total_weight > 0 else np.nan
                        ),
                        "current_weighted_mean_duration_min": weighted_mean(
                            usable["current_analysis_road_duration_min"], weights
                        ),
                        "integrated_weighted_mean_duration_min": weighted_mean(
                            usable["integrated_analysis_road_duration_min"], weights
                        ),
                        "weighted_mean_duration_change_min": weighted_mean(
                            usable["road_duration_change_min"], weights
                        ),
                        "total_duration_change_student_minutes": float(
                            (usable["road_duration_change_min"] * weights).sum()
                        ),
                        "weighted_mean_additional_time_burden_min": weighted_mean(
                            usable["additional_time_burden_min"], weights
                        ),
                        "total_additional_time_burden_student_minutes": float(
                            (usable["additional_time_burden_min"] * weights).sum()
                        ),
                        "weighted_p50_duration_change_min": weighted_quantile(
                            usable["road_duration_change_min"], weights, 0.50
                        ),
                        "weighted_p90_duration_change_min": weighted_quantile(
                            usable["road_duration_change_min"], weights, 0.90
                        ),
                        "max_duration_change_min": (
                            float(usable["road_duration_change_min"].max())
                            if len(usable) else np.nan
                        ),
                        "max_integrated_duration_min": (
                            float(usable["integrated_analysis_road_duration_min"].max())
                            if len(usable) else np.nan
                        ),
                        "current_weighted_mean_distance_km": weighted_mean(
                            usable["current_road_distance_km"], weights
                        ),
                        "integrated_weighted_mean_distance_km": weighted_mean(
                            usable["integrated_road_distance_km"], weights
                        ),
                        "weighted_mean_distance_change_km": weighted_mean(
                            usable["road_distance_change_km"], weights
                        ),
                        "weighted_mean_additional_distance_burden_km": weighted_mean(
                            usable["additional_distance_burden_km"], weights
                        ),
                        "current_over_30_student_weight": current_over_30_weight,
                        "current_over_30_student_rate": (
                            current_over_30_weight / eligible_weight if eligible_weight > 0 else np.nan
                        ),
                        "integrated_over_30_student_weight": integrated_over_30_weight,
                        "integrated_over_30_student_rate": (
                            integrated_over_30_weight / eligible_weight if eligible_weight > 0 else np.nan
                        ),
                        "newly_over_30_student_weight": newly_over_30_weight,
                        "newly_over_30_student_rate": (
                            newly_over_30_weight / eligible_weight if eligible_weight > 0 else np.nan
                        ),
                        "time_increase_student_weight": time_increase_weight,
                        "time_increase_student_rate": (
                            time_increase_weight / eligible_weight if eligible_weight > 0 else np.nan
                        ),
                        "time_decrease_student_weight": time_decrease_weight,
                        "time_decrease_student_rate": (
                            time_decrease_weight / eligible_weight if eligible_weight > 0 else np.nan
                        ),
                        "integrated_over_30_cluster_count": int(cluster_over_30.sum()),
                        "newly_over_30_cluster_count": int(cluster_newly_over_30.sum()),
                    }
                )

    national_summary = pd.DataFrame(national_rows)
    if len(national_summary) != len(SELECTION_SPECS) * len(BOUND_TYPES) * len(
        EVALUATION_POPULATIONS
    ):
        raise ValueError("전국 결과 요약 행 수가 예상과 다릅니다.")

    # ALL_STUDENTS 선택의 하한·상한은 전국 합계에서도 순서가 유지되어야 한다.
    primary_national = national_summary[
        national_summary["selection_population"].eq("ALL_STUDENTS")
        & national_summary["evaluation_population"].eq("ALL_STUDENTS")
    ].set_index("bound_type")
    if (
        primary_national.loc["LOWER_BOUND", "weighted_mean_duration_change_min"]
        > primary_national.loc["UPPER_BOUND", "weighted_mean_duration_change_min"]
        + FLOAT_TOLERANCE
    ):
        raise ValueError("전국 전체학생 하한 평균이 상한 평균보다 큽니다.")

    # ------------------------------------------------------------------
    # 4. 검증 요약
    # ------------------------------------------------------------------
    validation_rows: list[dict[str, object]] = []

    def add_validation(
        section: str,
        metric: str,
        value: object,
        unit: str,
        status: str,
        note: str,
    ) -> None:
        validation_rows.append(
            {
                "section": section,
                "metric": metric,
                "value": value,
                "unit": unit,
                "check_status": status,
                "note": note,
            }
        )

    cluster_count = options["final_cluster_id"].nunique()
    single_option_clusters = int((options.groupby("final_cluster_id").size() == 1).sum())
    add_validation("OVERALL", "overall_status", "READY", "status", "PASS", "04-5 완료")
    add_validation("INPUT", "option_rows", len(options), "rows", "PASS", OPTION_FILE)
    add_validation("INPUT", "detail_rows", len(detail), "rows", "PASS", DETAIL_FILE)
    add_validation("SCOPE", "final_cluster_count", cluster_count, "clusters", "PASS", "비중복 군집")
    add_validation("SCOPE", "integration_option_count", len(options), "options", "PASS", "CAP28")
    add_validation("SCOPE", "single_option_cluster_count", single_option_clusters, "clusters", "PASS", "하한=상한")
    add_validation("QA", "max_cluster_student_weight_range", max_cluster_weight_range, "students", "PASS", f"허용오차 {WEIGHT_TOLERANCE}")
    add_validation("QA", "duplicated_selection_key_count", int(selections.duplicated(selection_key).sum()), "rows", "PASS", "0이어야 함")
    add_validation("QA", "invalid_bound_order_count", int(invalid_bounds.sum()), "cluster_lenses", "PASS", "하한<=상한")
    add_validation("SELECTION", "selection_lens_count", len(SELECTION_SPECS), "lenses", "PASS", "전체학생|이동학생")
    add_validation("SELECTION", "bound_type_count", len(BOUND_TYPES), "bounds", "PASS", "하한|상한")
    add_validation("SELECTION", "selection_row_count", len(selections), "rows", "PASS", "군집x관점x상하한")
    add_validation("SELECTION", "partial_only_selection_rows", int(selections["selection_pool_status"].eq("PARTIAL_ONLY").sum()), "rows", "INFO", "페리 영향 군집")
    add_validation("SELECTION", "negative_selected_metric_rows", int((selections["selection_metric_value"] < -FLOAT_TOLERANCE).sum()), "rows", "INFO", "음수 보존")
    for lens in SELECTION_SPECS:
        lens_pairs = selections[selections["selection_population"].eq(lens)].drop_duplicates(
            ["final_cluster_id", "selection_population"]
        )
        add_validation(
            "SELECTION",
            f"{lens}_same_option_for_bounds_cluster_count",
            int(lens_pairs["same_option_for_bounds_flag"].sum()),
            "clusters",
            "INFO",
            "통합학교안 1개 또는 동률",
        )
    add_validation("JOIN", "selected_demand_rows", len(selected_demand), "rows", "PASS", "선택안 상세행")
    add_validation("JOIN", "duplicated_selection_demand_key_count", int(selected_demand.duplicated(["selection_id", "demand_assignment_id"]).sum()), "rows", "PASS", "0이어야 함")
    add_validation("OUTPUT", "cluster_selection_rows", len(selections), "rows", "PASS", SELECTION_OUTPUT_FILE)
    add_validation("OUTPUT", "cluster_selection_columns", len(selections.columns), "columns", "PASS", SELECTION_OUTPUT_FILE)
    add_validation("OUTPUT", "bound_demand_rows", len(selected_demand), "rows", "PASS", DEMAND_OUTPUT_FILE)
    add_validation("OUTPUT", "bound_demand_columns", len(selected_demand.columns), "columns", "PASS", DEMAND_OUTPUT_FILE)
    add_validation("OUTPUT", "national_summary_rows", len(national_summary), "rows", "PASS", NATIONAL_OUTPUT_FILE)
    add_validation("OUTPUT", "national_summary_columns", len(national_summary.columns), "columns", "PASS", NATIONAL_OUTPUT_FILE)

    validation_summary = pd.DataFrame(validation_rows)
    final_validation_rows = len(validation_summary) + 2
    validation_summary = pd.concat(
        [
            validation_summary,
            pd.DataFrame(
                [
                    {
                        "section": "OUTPUT",
                        "metric": "validation_summary_rows",
                        "value": final_validation_rows,
                        "unit": "rows",
                        "check_status": "PASS",
                        "note": VALIDATION_OUTPUT_FILE,
                    },
                    {
                        "section": "OUTPUT",
                        "metric": "validation_summary_columns",
                        "value": 6,
                        "unit": "columns",
                        "check_status": "PASS",
                        "note": VALIDATION_OUTPUT_FILE,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # 5. 저장
    # ------------------------------------------------------------------
    selections = selections.sort_values(
        ["selection_population", "bound_type", "final_cluster_id"],
        kind="stable",
    ).reset_index(drop=True)
    selected_demand = selected_demand.sort_values(
        [
            "selection_population", "bound_type", "final_cluster_id",
            "census_id", "current_school_key",
        ],
        kind="stable",
    ).reset_index(drop=True)
    national_summary = national_summary.sort_values(
        ["selection_population", "bound_type", "evaluation_population"],
        kind="stable",
    ).reset_index(drop=True)

    selection_path = output_dir / SELECTION_OUTPUT_FILE
    demand_path = output_dir / DEMAND_OUTPUT_FILE
    national_path = output_dir / NATIONAL_OUTPUT_FILE
    validation_path = output_dir / VALIDATION_OUTPUT_FILE
    selections.to_csv(selection_path, index=False, encoding="utf-8-sig")
    selected_demand.to_csv(demand_path, index=False, encoding="utf-8-sig")
    national_summary.to_csv(national_path, index=False, encoding="utf-8-sig")
    validation_summary.to_csv(validation_path, index=False, encoding="utf-8-sig")

    print("\n[산출물]")
    print(f"- {SELECTION_OUTPUT_FILE}: {len(selections):,}행 x {len(selections.columns):,}열")
    print(f"- {DEMAND_OUTPUT_FILE}: {len(selected_demand):,}행 x {len(selected_demand.columns):,}열")
    print(f"- {NATIONAL_OUTPUT_FILE}: {len(national_summary):,}행 x {len(national_summary.columns):,}열")
    print(f"- {VALIDATION_OUTPUT_FILE}: {len(validation_summary):,}행 x {len(validation_summary.columns):,}열")
    print("\n[핵심 기준]")
    print("- 전체학생 관점: weighted_mean_duration_change_min 최소/최대")
    print("- 이동학생 관점: moved_weighted_mean_duration_change_min 최소/최대")
    print("- 음수 시간 변화 유지")
    print(f"- 통합학교안 1개 군집: {single_option_clusters:,}개, 하한=상한")
    print("- 04-5 완료")


if __name__ == "__main__":
    main()

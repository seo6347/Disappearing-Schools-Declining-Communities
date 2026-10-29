"""
통합 가능 시나리오 확정 3단계: 도로 접근성 분류·민감도 분석 (03-3)
=================================================================

목적
----
03-1의 수용 가능 후보–존치학교안에 03-2의 두 학교 간 도로경로를 연결하고,
도로시간·도로거리·우회비율·페리 여부를 이용해 접근성 상태를 분류한다.

이 단계는 30분 초과안이나 우회 경로를 원자료에서 삭제하지 않는다. 이들은
오류가 아니라 통폐합 시나리오의 현실성과 장거리 통학 가능성을 보여주는
분석 결과이므로 상세표에 전부 보존한다.

확정 처리
---------
- 페리 경로가 하나라도 포함된 후보–존치학교안은 `EXCLUDED_FERRY`로 표시한다.
- 페리 행도 상세표에는 보존하지만 모든 민감도 시나리오의 분석대상에서는
  제외한다.
- 30분 초과는 장거리 경로 플래그이며 자동 제외 기준이 아니다.
- 절대시간 30·35·40·50·60·80분 기준을 비교한다.
- 우회비율 3배 초과와 절대시간을 함께 적용한 구조적 우회 기준도 비교한다.
- 민감도 결과만 생성하며 최종 극단경로 제외 기준은 이 코드에서 확정하지
  않는다.

분석 단위
---------
후보–수용력 시나리오–존치학교안 1개당 1행이다. 한 안에는 다음 두 경로가
연결된다.

    폐교후보학교 1 -> 존치학교
    폐교후보학교 2 -> 존치학교

안의 대표 도로시간은 두 경로 중 더 긴 시간으로 계산한다. 한 학생이 두
경로를 모두 이동하는 것이 아니므로 두 시간의 합계를 통합 가능성 기준으로
사용하지 않는다. 합계는 참고 열로만 보존한다.

입력 파일
---------
data/02_processed/candidate_capacity_options.csv
    - 03-1의 CAP28·CAP19 후보–존치학교별 수용력 평가 결과

data/02_processed/school_road_routes.csv
    - 03-2의 학교 간 방향별 NAVER Directions 5 경로 결과

출력 파일
---------
data/02_processed/candidate_road_access_options.csv
    - 수용 가능한 모든 후보–존치학교안과 두 도로경로, 접근성 플래그
    - 페리·30분 초과·구조적 우회·극단시간 사례도 삭제하지 않고 보존

data/02_processed/road_access_sensitivity_summary.csv
    - CAP28·CAP19별 페리 제외 후 여러 시간·구조적 우회 기준을 적용했을 때
      남는 존치학교안 수와 한 곳 이상 가능한 3개교 후보 수

실행 방법
---------
프로젝트 루트에서:

    py scripts/03-3_analyze_road_access_sensitivity.py

주의
----
- `LONG_ROUTE_30`은 법적 위반 또는 최종 탈락을 뜻하지 않는다.
- `NETWORK_DETOUR`도 직선거리 대비 도로거리 비율이 큰 진단 플래그일 뿐
  절대시간이 짧으면 실제 부담은 제한적일 수 있다.
- `FERRY_REQUIRED`는 경로 오류가 아니라 일반 육상 통학 이외의 교통수단이
  필요한 특수 사례다. 분석범위에서는 제외하지만 행과 개수는 보존한다.
- 일요일 새벽에 조회한 학교 간 시간은 실제 등교시간이 아니라 도로망상
  과도한 원거리 후보를 진단하는 보조 지표다.
- 최종 학생 통학부담은 이후 집계구 대표점->현재학교·존치학교 경로로
  별도 계산한다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 분석 설정
# -----------------------------------------------------------------------------

CAPACITY_OPTIONS_FILE = "candidate_capacity_options.csv"
ROUTES_FILE = "school_road_routes.csv"
OUTPUT_OPTIONS = "candidate_road_access_options.csv"
OUTPUT_SENSITIVITY = "road_access_sensitivity_summary.csv"

EXPECTED_ROUTE_PROVIDER = "NAVER_DIRECTIONS5"
EXPECTED_ROUTE_OPTION = "traoptimal"
DETOUR_RATIO_THRESHOLD = 3.0
NUMERIC_TOLERANCE = 1e-9

CAPACITY_REQUIRED = {
    "capacity_option_id",
    "candidate_id",
    "capacity_scenario_id",
    "students_per_class",
    "school_edu_office",
    "candidate_type",
    "retained_school_key",
    "retained_school_name",
    "closed_school_1_key",
    "closed_school_1_name",
    "closed_school_2_key",
    "closed_school_2_name",
    "capacity_feasible",
    "capacity_status",
}

ROUTES_REQUIRED = {
    "route_request_id",
    "capacity_scenario_id",
    "school_edu_office",
    "origin_school_key",
    "origin_school_name",
    "destination_school_key",
    "destination_school_name",
    "straight_distance_km",
    "api_provider",
    "route_option",
    "route_status",
    "route_check",
    "review_flag",
    "road_distance_m",
    "road_duration_min",
    "road_to_straight_ratio",
    "requested_at_kst",
    "ferry_flag",
}


@dataclass(frozen=True)
class SensitivityRule:
    rule_id: str
    rule_name: str
    rule_type: str
    description: str
    absolute_time_limit_min: float | None = None
    structural_time_threshold_min: float | None = None
    structural_ratio_threshold: float | None = None
    extreme_absolute_time_min: float | None = None


SENSITIVITY_RULES = [
    SensitivityRule(
        "S00_FERRY_ONLY",
        "페리만 제외",
        "FERRY_ONLY",
        "페리 포함안만 제외하고 육상 경로는 모두 보존",
    ),
    SensitivityRule(
        "S01_ABS_30",
        "최대시간 30분",
        "ABSOLUTE_TIME",
        "페리 제외 후 두 학교 경로의 최대시간이 30분 이하",
        absolute_time_limit_min=30,
    ),
    SensitivityRule(
        "S02_ABS_35",
        "최대시간 35분",
        "ABSOLUTE_TIME",
        "페리 제외 후 두 학교 경로의 최대시간이 35분 이하",
        absolute_time_limit_min=35,
    ),
    SensitivityRule(
        "S03_ABS_40",
        "최대시간 40분",
        "ABSOLUTE_TIME",
        "페리 제외 후 두 학교 경로의 최대시간이 40분 이하",
        absolute_time_limit_min=40,
    ),
    SensitivityRule(
        "S04_ABS_50",
        "최대시간 50분",
        "ABSOLUTE_TIME",
        "페리 제외 후 두 학교 경로의 최대시간이 50분 이하",
        absolute_time_limit_min=50,
    ),
    SensitivityRule(
        "S05_ABS_60",
        "최대시간 60분",
        "ABSOLUTE_TIME",
        "페리 제외 후 두 학교 경로의 최대시간이 60분 이하",
        absolute_time_limit_min=60,
    ),
    SensitivityRule(
        "S06_ABS_80",
        "최대시간 80분",
        "ABSOLUTE_TIME",
        "페리 제외 후 두 학교 경로의 최대시간이 80분 이하",
        absolute_time_limit_min=80,
    ),
    SensitivityRule(
        "S11_STRUCT_30_R3",
        "30분·3배 동시 초과 제외",
        "STRUCTURAL_DETOUR",
        "페리 제외 후 최대시간 30분 초과이면서 최대 우회비율 3배 초과인 안 제외",
        structural_time_threshold_min=30,
        structural_ratio_threshold=DETOUR_RATIO_THRESHOLD,
    ),
    SensitivityRule(
        "S12_STRUCT_40_R3",
        "40분·3배 동시 초과 제외",
        "STRUCTURAL_DETOUR",
        "페리 제외 후 최대시간 40분 초과이면서 최대 우회비율 3배 초과인 안 제외",
        structural_time_threshold_min=40,
        structural_ratio_threshold=DETOUR_RATIO_THRESHOLD,
    ),
    SensitivityRule(
        "S13_STRUCT_60_R3",
        "60분·3배 동시 초과 제외",
        "STRUCTURAL_DETOUR",
        "페리 제외 후 최대시간 60분 초과이면서 최대 우회비율 3배 초과인 안 제외",
        structural_time_threshold_min=60,
        structural_ratio_threshold=DETOUR_RATIO_THRESHOLD,
    ),
    SensitivityRule(
        "S21_HYBRID_80_40_R3",
        "80분 또는 40분·3배 초과 제외",
        "EXTREME_HYBRID",
        "페리 제외 후 80분 초과 또는 40분·3배 동시 초과인 극단안 제외",
        extreme_absolute_time_min=80,
        structural_time_threshold_min=40,
        structural_ratio_threshold=DETOUR_RATIO_THRESHOLD,
    ),
    SensitivityRule(
        "S22_HYBRID_60_40_R3",
        "60분 또는 40분·3배 초과 제외",
        "EXTREME_HYBRID",
        "페리 제외 후 60분 초과 또는 40분·3배 동시 초과인 극단안 제외",
        extreme_absolute_time_min=60,
        structural_time_threshold_min=40,
        structural_ratio_threshold=DETOUR_RATIO_THRESHOLD,
    ),
]


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="수용 가능 3개교 후보–존치학교안의 도로 접근성 분류 및 민감도 분석"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="final_project 폴더. 기본값은 scripts 폴더의 상위 폴더입니다.",
    )
    parser.add_argument("--capacity-options-file", type=Path, default=None)
    parser.add_argument("--routes-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    project_root = args.project_root.resolve()
    processed_dir = project_root / "data" / "02_processed"
    capacity_path = (
        args.capacity_options_file or processed_dir / CAPACITY_OPTIONS_FILE
    ).resolve()
    routes_path = (args.routes_file or processed_dir / ROUTES_FILE).resolve()
    output_dir = (args.output_dir or processed_dir).resolve()
    return capacity_path, routes_path, output_dir


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"[{label}] 필수 컬럼이 없습니다: {missing}")


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
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
    capacity_path: Path,
    routes_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [capacity_path, routes_path]:
        if not path.exists():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    capacity = pd.read_csv(
        capacity_path,
        encoding="utf-8-sig",
        dtype={
            "capacity_option_id": "string",
            "candidate_id": "string",
            "retained_school_key": "string",
            "closed_school_1_key": "string",
            "closed_school_2_key": "string",
        },
        low_memory=False,
    )
    routes = pd.read_csv(
        routes_path,
        encoding="utf-8-sig",
        dtype={
            "route_request_id": "string",
            "origin_school_key": "string",
            "destination_school_key": "string",
        },
        low_memory=False,
    )
    require_columns(capacity, CAPACITY_REQUIRED, CAPACITY_OPTIONS_FILE)
    require_columns(routes, ROUTES_REQUIRED, ROUTES_FILE)

    for column in [
        "capacity_option_id",
        "candidate_id",
        "retained_school_key",
        "closed_school_1_key",
        "closed_school_2_key",
    ]:
        capacity[column] = normalize_id(capacity[column])
    for column in [
        "route_request_id",
        "origin_school_key",
        "destination_school_key",
    ]:
        routes[column] = normalize_id(routes[column])

    if capacity["capacity_option_id"].isna().any() or capacity[
        "capacity_option_id"
    ].duplicated().any():
        raise ValueError("capacity_option_id에 결측 또는 중복이 있습니다.")
    if routes["route_request_id"].isna().any() or routes[
        "route_request_id"
    ].duplicated().any():
        raise ValueError("route_request_id에 결측 또는 중복이 있습니다.")
    if routes.duplicated(["origin_school_key", "destination_school_key"]).any():
        raise ValueError("학교 간 방향 경로가 중복되어 있습니다.")

    capacity["capacity_feasible"] = parse_bool(
        capacity["capacity_feasible"], "capacity_feasible"
    )
    routes["ferry_flag"] = parse_bool(routes["ferry_flag"], "ferry_flag")

    if not routes["route_status"].eq("SUCCESS").all():
        counts = routes["route_status"].value_counts(dropna=False).to_dict()
        raise ValueError(f"SUCCESS가 아닌 학교 간 경로가 있습니다: {counts}")
    if not routes["api_provider"].eq(EXPECTED_ROUTE_PROVIDER).all():
        raise ValueError("예상하지 않은 길찾기 제공자가 포함되어 있습니다.")
    if not routes["route_option"].eq(EXPECTED_ROUTE_OPTION).all():
        raise ValueError("예상하지 않은 경로 옵션이 포함되어 있습니다.")

    for column in [
        "straight_distance_km",
        "road_distance_m",
        "road_duration_min",
        "road_to_straight_ratio",
    ]:
        routes[column] = pd.to_numeric(routes[column], errors="coerce")
        if routes[column].isna().any() or routes[column].le(0).any():
            raise ValueError(f"경로의 {column}에 결측 또는 0 이하 값이 있습니다.")

    return capacity, routes


def route_join_frame(routes: pd.DataFrame, position: int) -> pd.DataFrame:
    selected = routes[
        [
            "route_request_id",
            "school_edu_office",
            "origin_school_key",
            "origin_school_name",
            "destination_school_key",
            "destination_school_name",
            "straight_distance_km",
            "route_status",
            "route_check",
            "review_flag",
            "road_distance_m",
            "road_duration_min",
            "road_to_straight_ratio",
            "requested_at_kst",
            "ferry_flag",
        ]
    ].copy()
    return selected.rename(
        columns={
            "route_request_id": f"route_{position}_request_id",
            "school_edu_office": f"route_{position}_school_edu_office",
            "origin_school_key": f"closed_school_{position}_key",
            "origin_school_name": f"route_{position}_origin_school_name",
            "destination_school_key": "retained_school_key",
            "destination_school_name": f"route_{position}_destination_school_name",
            "straight_distance_km": f"route_{position}_straight_distance_km",
            "route_status": f"route_{position}_status",
            "route_check": f"route_{position}_check",
            "review_flag": f"route_{position}_review_flag",
            "road_distance_m": f"route_{position}_road_distance_m",
            "road_duration_min": f"route_{position}_road_duration_min",
            "road_to_straight_ratio": f"route_{position}_detour_ratio",
            "requested_at_kst": f"route_{position}_requested_at_kst",
            "ferry_flag": f"route_{position}_ferry_flag",
        }
    )


def classify_access_category(frame: pd.DataFrame) -> pd.Series:
    conditions = [
        frame["any_ferry_route"],
        frame["over_80min_flag"],
        frame["over_60min_flag"],
        frame["structural_detour_over_40min_flag"],
        frame["over_40min_flag"],
        frame["structural_detour_over_30min_flag"],
        frame["over_30min_flag"],
        frame["detour_ratio_over_3_flag"],
    ]
    choices = [
        "FERRY_REQUIRED",
        "EXTREME_OVER_80MIN",
        "EXTREME_OVER_60MIN",
        "STRUCTURAL_DETOUR_OVER_40MIN",
        "LONG_ROUTE_OVER_40MIN",
        "STRUCTURAL_DETOUR_OVER_30MIN",
        "LONG_ROUTE_OVER_30MIN",
        "NETWORK_DETOUR_ONLY",
    ]
    return pd.Series(
        np.select(conditions, choices, default="STANDARD_ROAD"),
        index=frame.index,
        dtype="string",
    )


def build_road_access_options(
    capacity: pd.DataFrame,
    routes: pd.DataFrame,
) -> pd.DataFrame:
    feasible = capacity.loc[capacity["capacity_feasible"]].copy()
    if feasible.empty:
        raise ValueError("수용 가능한 후보–존치학교안이 없습니다.")

    output = feasible.merge(
        route_join_frame(routes, 1),
        on=["closed_school_1_key", "retained_school_key"],
        how="left",
        validate="many_to_one",
    ).merge(
        route_join_frame(routes, 2),
        on=["closed_school_2_key", "retained_school_key"],
        how="left",
        validate="many_to_one",
    )

    required_joined = [
        "route_1_request_id",
        "route_2_request_id",
        "route_1_road_distance_m",
        "route_2_road_distance_m",
        "route_1_road_duration_min",
        "route_2_road_duration_min",
    ]
    if output[required_joined].isna().any().any():
        problems = output.loc[
            output[required_joined].isna().any(axis=1),
            [
                "capacity_option_id",
                "closed_school_1_key",
                "closed_school_2_key",
                "retained_school_key",
            ],
        ].head(10)
        raise ValueError(
            "수용 가능안에 연결되지 않은 학교 간 경로가 있습니다: "
            f"{problems.to_dict('records')}"
        )

    for position in [1, 2]:
        if not output[f"route_{position}_status"].eq("SUCCESS").all():
            raise ValueError(f"{position}번 경로에 SUCCESS가 아닌 행이 있습니다.")
        if not output[f"route_{position}_school_edu_office"].astype(
            "string"
        ).eq(output["school_edu_office"].astype("string")).all():
            raise ValueError(f"{position}번 경로와 후보의 교육지원청이 다릅니다.")
        if not output[f"route_{position}_origin_school_name"].astype(
            "string"
        ).eq(output[f"closed_school_{position}_name"].astype("string")).all():
            raise ValueError(f"{position}번 경로와 후보의 폐교후보 학교명이 다릅니다.")
        if not output[f"route_{position}_destination_school_name"].astype(
            "string"
        ).eq(output["retained_school_name"].astype("string")).all():
            raise ValueError(f"{position}번 경로와 후보의 존치학교명이 다릅니다.")

    time_columns = [
        "route_1_road_duration_min",
        "route_2_road_duration_min",
    ]
    distance_columns = [
        "route_1_road_distance_m",
        "route_2_road_distance_m",
    ]
    ratio_columns = [
        "route_1_detour_ratio",
        "route_2_detour_ratio",
    ]
    output["max_school_road_duration_min"] = output[time_columns].max(axis=1)
    output["mean_school_road_duration_min"] = output[time_columns].mean(axis=1)
    output["sum_school_road_duration_min"] = output[time_columns].sum(axis=1)
    output["max_school_road_distance_km"] = (
        output[distance_columns].max(axis=1) / 1_000
    )
    output["max_road_straight_ratio"] = output[ratio_columns].max(axis=1)
    output["any_ferry_route"] = output[
        ["route_1_ferry_flag", "route_2_ferry_flag"]
    ].any(axis=1)
    output["any_route_review"] = (
        output["route_1_check"].eq("REVIEW")
        | output["route_2_check"].eq("REVIEW")
    )

    for threshold in [30, 35, 40, 50, 60, 80]:
        output[f"over_{threshold}min_flag"] = output[
            "max_school_road_duration_min"
        ].gt(threshold)
    output["detour_ratio_over_3_flag"] = output[
        "max_road_straight_ratio"
    ].gt(DETOUR_RATIO_THRESHOLD)
    for threshold in [30, 40, 60]:
        output[f"structural_detour_over_{threshold}min_flag"] = (
            output[f"over_{threshold}min_flag"]
            & output["detour_ratio_over_3_flag"]
        )

    output["confirmed_scope_status"] = np.where(
        output["any_ferry_route"],
        "EXCLUDED_FERRY",
        "INCLUDED_NON_FERRY",
    )
    output["confirmed_scope_eligible"] = ~output["any_ferry_route"]
    output["road_access_category"] = classify_access_category(output)
    output["road_data_check"] = "PASS"

    # 조인 과정에서만 사용한 중복 교육지원청 열은 제거한다.
    output = output.drop(
        columns=[
            "route_1_school_edu_office",
            "route_2_school_edu_office",
        ]
    )
    return output.sort_values(
        [
            "students_per_class",
            "candidate_id",
            "retained_school_position",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def apply_rule(frame: pd.DataFrame, rule: SensitivityRule) -> pd.Series:
    """모든 규칙은 확정 범위인 비페리 안에서 시작한다."""
    eligible = frame["confirmed_scope_eligible"].copy()
    max_time = frame["max_school_road_duration_min"]
    max_ratio = frame["max_road_straight_ratio"]

    if rule.rule_type == "FERRY_ONLY":
        return eligible
    if rule.rule_type == "ABSOLUTE_TIME":
        if rule.absolute_time_limit_min is None:
            raise ValueError(f"{rule.rule_id}에 절대시간 기준이 없습니다.")
        return eligible & max_time.le(rule.absolute_time_limit_min)
    if rule.rule_type == "STRUCTURAL_DETOUR":
        if (
            rule.structural_time_threshold_min is None
            or rule.structural_ratio_threshold is None
        ):
            raise ValueError(f"{rule.rule_id}에 구조적 우회 기준이 없습니다.")
        structural_extreme = max_time.gt(
            rule.structural_time_threshold_min
        ) & max_ratio.gt(rule.structural_ratio_threshold)
        return eligible & ~structural_extreme
    if rule.rule_type == "EXTREME_HYBRID":
        if (
            rule.extreme_absolute_time_min is None
            or rule.structural_time_threshold_min is None
            or rule.structural_ratio_threshold is None
        ):
            raise ValueError(f"{rule.rule_id}에 혼합 극단 기준이 없습니다.")
        absolute_extreme = max_time.gt(rule.extreme_absolute_time_min)
        structural_extreme = max_time.gt(
            rule.structural_time_threshold_min
        ) & max_ratio.gt(rule.structural_ratio_threshold)
        return eligible & ~(absolute_extreme | structural_extreme)
    raise ValueError(f"알 수 없는 민감도 규칙입니다: {rule.rule_type}")


def build_sensitivity_summary(options: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_id, scenario in options.groupby(
        "capacity_scenario_id", sort=False
    ):
        base_option_count = len(scenario)
        base_candidate_count = scenario["candidate_id"].nunique()
        ferry_option_count = int(scenario["any_ferry_route"].sum())
        ferry_affected_candidate_count = scenario.loc[
            scenario["any_ferry_route"], "candidate_id"
        ].nunique()
        non_ferry_candidate_count = scenario.loc[
            scenario["confirmed_scope_eligible"], "candidate_id"
        ].nunique()

        for rule in SENSITIVITY_RULES:
            eligible = apply_rule(scenario, rule)
            eligible_option_count = int(eligible.sum())
            eligible_candidate_count = scenario.loc[
                eligible, "candidate_id"
            ].nunique()
            rows.append(
                {
                    "capacity_scenario_id": scenario_id,
                    "students_per_class": int(
                        scenario["students_per_class"].iloc[0]
                    ),
                    "rule_id": rule.rule_id,
                    "rule_name": rule.rule_name,
                    "rule_type": rule.rule_type,
                    "rule_description": rule.description,
                    "ferry_always_excluded": True,
                    "absolute_time_limit_min": rule.absolute_time_limit_min,
                    "structural_time_threshold_min": (
                        rule.structural_time_threshold_min
                    ),
                    "structural_ratio_threshold": (
                        rule.structural_ratio_threshold
                    ),
                    "extreme_absolute_time_min": (
                        rule.extreme_absolute_time_min
                    ),
                    "base_capacity_feasible_option_count": base_option_count,
                    "base_capacity_feasible_candidate_count": (
                        base_candidate_count
                    ),
                    "ferry_excluded_option_count": ferry_option_count,
                    "ferry_affected_candidate_count": (
                        ferry_affected_candidate_count
                    ),
                    "non_ferry_candidate_count": non_ferry_candidate_count,
                    "eligible_option_count": eligible_option_count,
                    "eligible_option_rate_of_base": (
                        eligible_option_count / base_option_count
                    ),
                    "excluded_option_count_total": (
                        base_option_count - eligible_option_count
                    ),
                    "eligible_candidate_count": eligible_candidate_count,
                    "eligible_candidate_rate_of_base": (
                        eligible_candidate_count / base_candidate_count
                    ),
                    "candidate_without_eligible_option_count": (
                        base_candidate_count - eligible_candidate_count
                    ),
                    "candidate_lost_beyond_ferry_count": (
                        non_ferry_candidate_count - eligible_candidate_count
                    ),
                    "long_route_over_30_option_count": int(
                        scenario["over_30min_flag"].sum()
                    ),
                    "extreme_over_60_option_count": int(
                        scenario["over_60min_flag"].sum()
                    ),
                    "extreme_over_80_option_count": int(
                        scenario["over_80min_flag"].sum()
                    ),
                    "structural_detour_over_30_option_count": int(
                        scenario[
                            "structural_detour_over_30min_flag"
                        ].sum()
                    ),
                    "structural_detour_over_40_option_count": int(
                        scenario[
                            "structural_detour_over_40min_flag"
                        ].sum()
                    ),
                }
            )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["students_per_class", "rule_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def validate_outputs(
    capacity: pd.DataFrame,
    routes: pd.DataFrame,
    options: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    feasible = capacity.loc[capacity["capacity_feasible"]]
    if len(options) != len(feasible):
        raise ValueError(
            f"도로접근성 상세안 행 수가 수용 가능안과 다릅니다: "
            f"예상 {len(feasible):,}, 실제 {len(options):,}"
        )
    if options["capacity_option_id"].duplicated().any():
        raise ValueError("도로접근성 상세안의 capacity_option_id가 중복되었습니다.")
    if not options["road_data_check"].eq("PASS").all():
        raise ValueError("road_data_check가 PASS가 아닌 행이 있습니다.")

    formula_max_time = options[
        ["route_1_road_duration_min", "route_2_road_duration_min"]
    ].max(axis=1)
    formula_max_distance = (
        options[
            ["route_1_road_distance_m", "route_2_road_distance_m"]
        ].max(axis=1)
        / 1_000
    )
    formula_max_ratio = options[
        ["route_1_detour_ratio", "route_2_detour_ratio"]
    ].max(axis=1)
    for actual_column, expected in [
        ("max_school_road_duration_min", formula_max_time),
        ("max_school_road_distance_km", formula_max_distance),
        ("max_road_straight_ratio", formula_max_ratio),
    ]:
        if not np.isclose(
            options[actual_column],
            expected,
            atol=NUMERIC_TOLERANCE,
            rtol=0,
        ).all():
            raise ValueError(f"{actual_column} 계산식 검증에 실패했습니다.")

    route_ferry_pairs = set(
        routes.loc[
            routes["ferry_flag"],
            ["origin_school_key", "destination_school_key"],
        ].itertuples(index=False, name=None)
    )
    option_ferry_formula = options.apply(
        lambda row: (
            (row["closed_school_1_key"], row["retained_school_key"])
            in route_ferry_pairs
            or (
                row["closed_school_2_key"],
                row["retained_school_key"],
            )
            in route_ferry_pairs
        ),
        axis=1,
    )
    if options["any_ferry_route"].ne(option_ferry_formula).any():
        raise ValueError("페리 포함 여부 계산식 검증에 실패했습니다.")
    if options.loc[
        options["any_ferry_route"], "confirmed_scope_eligible"
    ].any():
        raise ValueError("페리 포함안이 확정 분석범위에 포함되었습니다.")

    expected_summary_rows = (
        options["capacity_scenario_id"].nunique() * len(SENSITIVITY_RULES)
    )
    if len(summary) != expected_summary_rows:
        raise ValueError(
            f"민감도 요약 행 수가 예상과 다릅니다: "
            f"예상 {expected_summary_rows}, 실제 {len(summary)}"
        )
    if summary.duplicated(["capacity_scenario_id", "rule_id"]).any():
        raise ValueError("수용력 시나리오–민감도 규칙 요약이 중복되었습니다.")

    # 절대시간 기준이 완화될수록 적격안·후보 수가 감소해서는 안 된다.
    absolute = summary.loc[summary["rule_type"].eq("ABSOLUTE_TIME")].copy()
    for _, scenario in absolute.groupby("capacity_scenario_id"):
        scenario = scenario.sort_values("absolute_time_limit_min")
        if scenario["eligible_option_count"].diff().dropna().lt(0).any():
            raise ValueError("절대시간 기준 완화에 따라 적격안 수가 감소했습니다.")
        if scenario["eligible_candidate_count"].diff().dropna().lt(0).any():
            raise ValueError("절대시간 기준 완화에 따라 적격후보 수가 감소했습니다.")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def log_results(options: pd.DataFrame, summary: pd.DataFrame) -> None:
    logging.info("도로 접근성 상세안: %s", f"{len(options):,}")
    logging.info("접근성 분류")
    for category, count in options["road_access_category"].value_counts().items():
        logging.info("  %-35s %s", category, f"{count:,}")

    for scenario_id in sorted(
        summary["capacity_scenario_id"].unique(), reverse=True
    ):
        rows = summary.loc[summary["capacity_scenario_id"].eq(scenario_id)]
        logging.info("%s 민감도 결과", scenario_id)
        logging.info(
            "%-28s %12s %12s %12s",
            "기준",
            "적격안",
            "적격후보",
            "후보탈락",
        )
        for row in rows.itertuples(index=False):
            logging.info(
                "%-28s %12s %12s %12s",
                row.rule_id,
                f"{row.eligible_option_count:,}",
                f"{row.eligible_candidate_count:,}",
                f"{row.candidate_without_eligible_option_count:,}",
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    capacity_path, routes_path, output_dir = resolve_paths(args)
    logging.info("수용력 상세안: %s", capacity_path)
    logging.info("학교 간 경로: %s", routes_path)

    capacity, routes = load_inputs(capacity_path, routes_path)
    options = build_road_access_options(capacity, routes)
    summary = build_sensitivity_summary(options)
    validate_outputs(capacity, routes, options, summary)

    output_options_path = output_dir / OUTPUT_OPTIONS
    output_summary_path = output_dir / OUTPUT_SENSITIVITY
    write_csv(options, output_options_path)
    write_csv(summary, output_summary_path)
    log_results(options, summary)

    logging.info(
        "상세안 저장: %s (%s행 × %s열)",
        output_options_path,
        f"{len(options):,}",
        len(options.columns),
    )
    logging.info(
        "민감도 요약 저장: %s (%s행 × %s열)",
        output_summary_path,
        f"{len(summary):,}",
        len(summary.columns),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("실행 실패: %s", exc)
        sys.exit(1)

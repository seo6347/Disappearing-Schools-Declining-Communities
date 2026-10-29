# 공개용 기본 실행은 API를 호출하지 않고 data/01_raw/api_routes/school_road_routes.csv를 사용합니다.
# 이 수집 코드는 출처와 수집 방법을 보존하기 위한 선택 도구이며 --execute를 지정할 때만 API를 호출합니다.
"""
통합 가능 시나리오 확정 2단계: 학교 간 도로 이동시간 수집 (03-2)
================================================================

목적
----
03-1의 CAP28 수용 가능 후보–존치학교안에서 필요한 모든
`폐교 후보 학교 -> 존치학교` 방향쌍을 만들고, 동일 방향쌍을 중복 제거한 뒤
NAVER Cloud Maps Directions 5로 실시간 최적 자동차 경로의 거리와 시간을
수집한다.

이 단계에서 계산하는 학교–학교 시간은 지나치게 먼 통합안을 거르는
1차 검증용이다. 최종 학생 통학부담은 이후 집계구 대표점–존치학교 경로로
별도 계산한다.

확정 기준
---------
- 수용력 기준: CAP28에서 `capacity_feasible=True`인 안만 API 요청표에 포함
- CAP19는 CAP28 가능 방향쌍의 부분집합인지 검증하며 별도 호출하지 않음
- 방향 보존: A→B와 B→A는 일방통행·진입로·교통상황이 다를 수 있어 별도 요청
- 경로 옵션: `traoptimal`(실시간 최적)
- 요청 순서: 고정 난수 시드로 섞어 지역별 호출 시각 편향을 완화
- 반올림 없음: API 원자료(m, ms)를 보존하고 분 단위는 ms/60,000으로 계산

호출 수 절감
-----------
1. 모든 후보안이 아니라 CAP28 수용 가능안만 사용한다.
2. 후보안별 원시 구간을 `출발학교–도착학교` 기준으로 중복 제거한다.
3. CAP19는 CAP28 부분집합이므로 재호출하지 않는다.
4. 기존 결과 파일의 완료 상태는 재실행 시 건너뛴다.
5. 429·5xx·네트워크 오류만 제한적으로 재시도한다.

NAVER API의 `goal`에 목적지를 여러 개 입력하면 각 목적지별 결과가 아니라
비용이 가장 적은 목적지 하나의 경로만 반환된다. 따라서 본 분석에서는
여러 목적지를 묶는 방식으로 호출 수를 줄이면 안 된다.

입력 파일
---------
data/02_processed/candidate_capacity_options.csv
    - 03-1의 CAP28·CAP19 후보–존치학교별 수용력 평가 결과

data/02_processed/distance_graph_nodes.csv
    - 학교키, 학교명, 교육지원청, WGS84 위도·경도

출력 파일
---------
data/02_processed/school_road_routes.csv
    - 중복 제거된 `폐교 후보 학교 -> 존치학교` 방향쌍 1개당 1행
    - 직선거리, 도로거리, 도로시간, 조회시각, API 상태, 좌표 스냅 거리,
      페리 포함 여부와 검토 플래그

data/02_processed/school_road_routes_summary.csv
    - 실행·검증 결과를 1행으로 요약한 품질관리표

사전 준비
---------
프로젝트 루트의 `.env`에 인증키를 저장한다. 이 파일은 공유하거나 Git에
올리지 않는다.

    NAVER_MAPS_CLIENT_ID=발급받은_Client_ID
    NAVER_MAPS_CLIENT_SECRET=발급받은_Client_Secret

실행 방법
---------
1) API를 부르지 않고 요청 수·입력 검증 및 PENDING 표 생성:

    py scripts/03-2_fetch_school_road_routes.py

2) 실제 호출(권장: 화~목 07:10~07:35 KST에 시작):

    py scripts/03-2_fetch_school_road_routes.py --execute

중단 후 같은 명령을 다시 실행하면 기존 완료 행을 건너뛰고 이어서 수행한다.
권장 시간 밖에서 실행해야 할 때만 다음 옵션을 명시한다.

    --allow-outside-school-window

주의
----
- 콘솔에서 Directions 5 일·월 한도를 25,000 이하로 설정하고 '한도 초과
  사용'을 허용하지 않은 상태에서 실행하는 것을 권장한다.
- 스크립트의 25,000회 제한은 이 파일이 수행한 HTTP 요청만 센다. 다른
  애플리케이션·과거 호출은 알 수 없으므로 콘솔 당월 사용량도 확인한다.
- 실시간 교통을 반영하므로 같은 좌표도 호출 시점에 따라 결과가 달라진다.
- 공휴일, 방학, 폭설·태풍, 대형 사고 여부는 코드가 자동 판별하지 않는다.
- 도서지역의 페리 경로와 도로 주변이 아닌 좌표는 삭제하지 않고 REVIEW로
  보존하여 후속 단계에서 별도 판단한다.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, time as clock_time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 분석·API 설정
# -----------------------------------------------------------------------------

OPTIONS_FILE = "candidate_capacity_options.csv"
NODES_FILE = "distance_graph_nodes.csv"
OUTPUT_ROUTES = "school_road_routes.csv"
OUTPUT_SUMMARY = "school_road_routes_summary.csv"

API_URL = "https://maps.apigw.ntruss.com/map-direction/v1/driving"
API_PROVIDER = "NAVER_DIRECTIONS5"
DEFAULT_SCENARIO = "CAP28"
DEFAULT_ROUTE_OPTION = "traoptimal"
DEFAULT_EXPECTED_REQUESTS = 22_982
DEFAULT_MAX_API_CALLS = 25_000
DEFAULT_RANDOM_SEED = 20260825
DEFAULT_WORKERS = 8
DEFAULT_REQUESTS_PER_SECOND = 4.0
DEFAULT_BATCH_SIZE = 100
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_RETRIES = 2

KST = ZoneInfo("Asia/Seoul")
RECOMMENDED_WEEKDAYS = {1, 2, 3}  # 월=0, 화=1, 수=2, 목=3
RECOMMENDED_START = clock_time(7, 10)
RECOMMENDED_END = clock_time(7, 35)
FERRY_GUIDE_TYPES = {48, 49}

TERMINAL_STATUSES = {
    "SUCCESS",
    "SAME_POINT",
    "POINT_NOT_NEAR_ROAD",
    "NO_ROUTE",
    "WAYPOINT_NOT_NEAR_ROAD",
    "DISTANCE_TOO_LONG",
    "HTTP_CLIENT_ERROR",
}

OPTIONS_REQUIRED = {
    "capacity_option_id",
    "candidate_id",
    "capacity_scenario_id",
    "school_edu_office",
    "retained_school_key",
    "closed_school_1_key",
    "closed_school_2_key",
    "capacity_feasible",
}

NODES_REQUIRED = {
    "school_key",
    "school_name",
    "school_edu_office",
    "school_lat",
    "school_lon",
}

RESULT_COLUMNS = [
    "route_request_id",
    "request_order",
    "capacity_scenario_id",
    "school_edu_office",
    "origin_school_key",
    "origin_school_name",
    "origin_lat",
    "origin_lon",
    "destination_school_key",
    "destination_school_name",
    "destination_lat",
    "destination_lon",
    "straight_distance_km",
    "source_capacity_option_count",
    "source_candidate_count",
    "api_provider",
    "route_option",
    "route_status",
    "route_check",
    "review_flag",
    "http_status",
    "api_code",
    "api_message",
    "road_distance_m",
    "road_duration_ms",
    "road_duration_sec",
    "road_duration_min",
    "road_to_straight_ratio",
    "api_current_datetime",
    "api_departure_time",
    "requested_at_kst",
    "snapped_start_lon",
    "snapped_start_lat",
    "snapped_goal_lon",
    "snapped_goal_lat",
    "start_snap_distance_m",
    "goal_snap_distance_m",
    "ferry_flag",
    "attempt_count",
    "last_error",
]


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="CAP28 가능안의 학교 간 NAVER Directions 5 도로시간 수집"
    )
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--options-file", type=Path, default=None)
    parser.add_argument("--nodes-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--route-option",
        default=DEFAULT_ROUTE_OPTION,
        choices=[
            "trafast",
            "tracomfort",
            "traoptimal",
            "traavoidtoll",
            "traavoidcaronly",
        ],
    )
    parser.add_argument("--expected-requests", type=int, default=DEFAULT_EXPECTED_REQUESTS)
    parser.add_argument("--skip-expected-count-check", action="store_true")
    parser.add_argument("--max-api-calls", type=int, default=DEFAULT_MAX_API_CALLS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--requests-per-second", type=float, default=DEFAULT_REQUESTS_PER_SECOND
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="지정할 때만 실제 API를 호출합니다. 생략하면 검증·요청표만 생성합니다.",
    )
    parser.add_argument("--allow-outside-school-window", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    project_root = args.project_root.resolve()
    processed_dir = project_root / "data" / "02_processed"
    options_path = (args.options_file or processed_dir / OPTIONS_FILE).resolve()
    nodes_path = (args.nodes_file or processed_dir / NODES_FILE).resolve()
    output_dir = (args.output_dir or processed_dir).resolve()
    routes_path = output_dir / OUTPUT_ROUTES
    summary_path = output_dir / OUTPUT_SUMMARY
    return project_root, options_path, nodes_path, routes_path, summary_path


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


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_m = 6_371_008.8
    lon1_r, lat1_r, lon2_r, lat2_r = map(
        math.radians, [lon1, lat1, lon2, lat2]
    )
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(a))


def load_env_file(path: Path) -> None:
    """외부 패키지 없이 단순 KEY=VALUE 형식의 .env를 읽는다."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_inputs(
    options_path: Path,
    nodes_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [options_path, nodes_path]:
        if not path.exists():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    options = pd.read_csv(
        options_path,
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
    nodes = pd.read_csv(
        nodes_path,
        encoding="utf-8-sig",
        dtype={"school_key": "string"},
        low_memory=False,
    )
    require_columns(options, OPTIONS_REQUIRED, OPTIONS_FILE)
    require_columns(nodes, NODES_REQUIRED, NODES_FILE)

    for column in [
        "capacity_option_id",
        "candidate_id",
        "retained_school_key",
        "closed_school_1_key",
        "closed_school_2_key",
    ]:
        options[column] = normalize_id(options[column])
    nodes["school_key"] = normalize_id(nodes["school_key"])

    if options["capacity_option_id"].isna().any() or options[
        "capacity_option_id"
    ].duplicated().any():
        raise ValueError("capacity_option_id에 결측 또는 중복이 있습니다.")
    if nodes["school_key"].isna().any() or nodes["school_key"].duplicated().any():
        raise ValueError("노드 school_key에 결측 또는 중복이 있습니다.")

    options["capacity_feasible"] = parse_bool(
        options["capacity_feasible"], "capacity_feasible"
    )
    for column in ["school_lat", "school_lon"]:
        nodes[column] = pd.to_numeric(nodes[column], errors="coerce")
    coordinate_problem = (
        nodes["school_lat"].isna()
        | nodes["school_lon"].isna()
        | ~nodes["school_lat"].between(30, 40)
        | ~nodes["school_lon"].between(120, 135)
    )
    if coordinate_problem.any():
        problems = nodes.loc[
            coordinate_problem,
            ["school_key", "school_name", "school_lat", "school_lon"],
        ].head(10)
        raise ValueError(
            "학교 좌표가 없거나 한국 WGS84 범위를 벗어났습니다: "
            f"{problems.to_dict('records')}"
        )
    return options, nodes


def explode_route_legs(feasible: pd.DataFrame) -> pd.DataFrame:
    common = [
        "capacity_option_id",
        "candidate_id",
        "capacity_scenario_id",
        "school_edu_office",
        "retained_school_key",
    ]
    frames: list[pd.DataFrame] = []
    for closed_column in ["closed_school_1_key", "closed_school_2_key"]:
        frame = feasible[common + [closed_column]].copy()
        frame = frame.rename(
            columns={
                closed_column: "origin_school_key",
                "retained_school_key": "destination_school_key",
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_request_table(
    options: pd.DataFrame,
    nodes: pd.DataFrame,
    scenario: str,
    route_option: str,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario_rows = options.loc[options["capacity_scenario_id"].eq(scenario)].copy()
    if scenario_rows.empty:
        raise ValueError(f"수용력 시나리오 {scenario} 행이 없습니다.")
    feasible = scenario_rows.loc[scenario_rows["capacity_feasible"]].copy()
    if feasible.empty:
        raise ValueError(f"{scenario}에 수용 가능한 후보–존치학교안이 없습니다.")

    raw_legs = explode_route_legs(feasible)
    if raw_legs[["origin_school_key", "destination_school_key"]].isna().any().any():
        raise ValueError("API 요청 학교키에 결측이 있습니다.")
    if raw_legs["origin_school_key"].eq(raw_legs["destination_school_key"]).any():
        raise ValueError("출발학교와 도착학교가 같은 요청이 있습니다.")

    grouped = raw_legs.groupby(
        ["origin_school_key", "destination_school_key"],
        as_index=False,
        sort=True,
    ).agg(
        school_edu_office=("school_edu_office", "first"),
        source_capacity_option_count=("capacity_option_id", "nunique"),
        source_candidate_count=("candidate_id", "nunique"),
    )

    node_columns = [
        "school_key",
        "school_name",
        "school_edu_office",
        "school_lat",
        "school_lon",
    ]
    origin = nodes[node_columns].rename(
        columns={
            "school_key": "origin_school_key",
            "school_name": "origin_school_name",
            "school_edu_office": "origin_school_edu_office",
            "school_lat": "origin_lat",
            "school_lon": "origin_lon",
        }
    )
    destination = nodes[node_columns].rename(
        columns={
            "school_key": "destination_school_key",
            "school_name": "destination_school_name",
            "school_edu_office": "destination_school_edu_office",
            "school_lat": "destination_lat",
            "school_lon": "destination_lon",
        }
    )
    requests_frame = grouped.merge(
        origin, on="origin_school_key", how="left", validate="many_to_one"
    ).merge(
        destination,
        on="destination_school_key",
        how="left",
        validate="many_to_one",
    )
    if requests_frame[["origin_school_name", "destination_school_name"]].isna().any().any():
        raise ValueError("노드표에 없는 출발학교 또는 도착학교가 있습니다.")
    if not (
        requests_frame["school_edu_office"]
        .astype("string")
        .eq(requests_frame["origin_school_edu_office"].astype("string"))
        & requests_frame["school_edu_office"]
        .astype("string")
        .eq(requests_frame["destination_school_edu_office"].astype("string"))
    ).all():
        raise ValueError("같은 교육지원청이 아닌 학교 간 요청이 있습니다.")

    requests_frame["route_request_id"] = (
        requests_frame["origin_school_key"].astype(str)
        + "__"
        + requests_frame["destination_school_key"].astype(str)
    )
    requests_frame["capacity_scenario_id"] = scenario
    requests_frame["straight_distance_km"] = requests_frame.apply(
        lambda row: haversine_m(
            row["origin_lon"],
            row["origin_lat"],
            row["destination_lon"],
            row["destination_lat"],
        )
        / 1_000,
        axis=1,
    )
    requests_frame["api_provider"] = API_PROVIDER
    requests_frame["route_option"] = route_option

    # 지역 순으로 호출해 특정 지역만 항상 같은 시각대에 측정되는 문제를 줄인다.
    shuffled_ids = requests_frame["route_request_id"].tolist()
    random.Random(random_seed).shuffle(shuffled_ids)
    order_map = {route_id: position for position, route_id in enumerate(shuffled_ids, 1)}
    requests_frame["request_order"] = requests_frame["route_request_id"].map(order_map)

    requests_frame["route_status"] = "PENDING"
    requests_frame["route_check"] = "PENDING"
    requests_frame["review_flag"] = ""
    for column in [
        "http_status",
        "api_code",
        "api_message",
        "road_distance_m",
        "road_duration_ms",
        "road_duration_sec",
        "road_duration_min",
        "road_to_straight_ratio",
        "api_current_datetime",
        "api_departure_time",
        "requested_at_kst",
        "snapped_start_lon",
        "snapped_start_lat",
        "snapped_goal_lon",
        "snapped_goal_lat",
        "start_snap_distance_m",
        "goal_snap_distance_m",
        "ferry_flag",
        "last_error",
    ]:
        requests_frame[column] = pd.NA
    requests_frame["attempt_count"] = 0

    requests_frame = requests_frame.drop(
        columns=["origin_school_edu_office", "destination_school_edu_office"]
    )
    requests_frame = requests_frame[RESULT_COLUMNS].sort_values(
        "request_order"
    ).reset_index(drop=True)

    metrics: dict[str, Any] = {
        "input_option_rows": len(options),
        "scenario_option_rows": len(scenario_rows),
        "feasible_option_rows": len(feasible),
        "raw_route_legs": len(raw_legs),
        "unique_directed_routes": len(requests_frame),
        "duplicate_route_legs_removed": len(raw_legs) - len(requests_frame),
        "deduplication_rate": 1 - len(requests_frame) / len(raw_legs),
    }

    cap19 = options.loc[
        options["capacity_scenario_id"].eq("CAP19") & options["capacity_feasible"]
    ]
    if not cap19.empty:
        cap19_pairs = set(
            explode_route_legs(cap19)[
                ["origin_school_key", "destination_school_key"]
            ].itertuples(index=False, name=None)
        )
        scenario_pairs = set(
            requests_frame[
                ["origin_school_key", "destination_school_key"]
            ].itertuples(index=False, name=None)
        )
        extra_cap19 = cap19_pairs - scenario_pairs
        if extra_cap19:
            raise ValueError(
                "CAP19 가능 방향쌍이 CAP28 요청표의 부분집합이 아닙니다: "
                f"{list(extra_cap19)[:10]}"
            )
        metrics["cap19_unique_directed_routes"] = len(cap19_pairs)
        metrics["cap19_subset_of_cap28"] = True
    else:
        metrics["cap19_unique_directed_routes"] = 0
        metrics["cap19_subset_of_cap28"] = "NOT_AVAILABLE"
    return requests_frame, metrics


def merge_previous_results(
    planned: pd.DataFrame,
    routes_path: Path,
) -> pd.DataFrame:
    if not routes_path.exists():
        return planned
    previous = pd.read_csv(
        routes_path,
        encoding="utf-8-sig",
        dtype={"route_request_id": "string"},
        low_memory=False,
    )
    require_columns(previous, set(RESULT_COLUMNS), OUTPUT_ROUTES)
    if previous["route_request_id"].isna().any() or previous[
        "route_request_id"
    ].duplicated().any():
        raise ValueError("기존 경로 결과의 route_request_id에 결측 또는 중복이 있습니다.")
    planned_ids = set(planned["route_request_id"].astype(str))
    previous_ids = set(previous["route_request_id"].astype(str))
    if planned_ids != previous_ids:
        raise ValueError(
            "기존 경로 결과와 현재 요청표의 ID 집합이 다릅니다. 입력이 바뀌었다면 "
            "기존 파일을 별도 보관한 뒤 새로 실행하십시오."
        )
    if set(previous["route_option"].dropna().astype(str)) != set(
        planned["route_option"].dropna().astype(str)
    ):
        raise ValueError("기존 결과와 현재 실행의 route_option이 다릅니다.")
    if set(previous["capacity_scenario_id"].dropna().astype(str)) != set(
        planned["capacity_scenario_id"].dropna().astype(str)
    ):
        raise ValueError("기존 결과와 현재 실행의 수용력 시나리오가 다릅니다.")

    preserved = previous.set_index("route_request_id")
    merged = planned.set_index("route_request_id")
    result_columns = [
        column
        for column in RESULT_COLUMNS
        if column not in {
            "route_request_id",
            "request_order",
            "capacity_scenario_id",
            "school_edu_office",
            "origin_school_key",
            "origin_school_name",
            "origin_lat",
            "origin_lon",
            "destination_school_key",
            "destination_school_name",
            "destination_lat",
            "destination_lon",
            "straight_distance_km",
            "source_capacity_option_count",
            "source_candidate_count",
            "api_provider",
            "route_option",
        }
    ]
    for column in result_columns:
        merged[column] = preserved[column]
    merged = merged.reset_index()[RESULT_COLUMNS]
    return merged.sort_values("request_order").reset_index(drop=True)


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second는 0보다 커야 합니다.")
        self.interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.next_allowed = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self.next_allowed - now)
            self.next_allowed = max(now, self.next_allowed) + self.interval
        if wait_seconds:
            time.sleep(wait_seconds)


class CallBudgetExceeded(RuntimeError):
    pass


class CallBudget:
    def __init__(self, maximum: int) -> None:
        if maximum <= 0:
            raise ValueError("max_api_calls는 0보다 커야 합니다.")
        self.maximum = maximum
        self.used = 0
        self.lock = threading.Lock()

    def reserve(self) -> int:
        with self.lock:
            if self.used >= self.maximum:
                raise CallBudgetExceeded(
                    f"이 실행의 API 호출 상한 {self.maximum:,}회에 도달했습니다."
                )
            self.used += 1
            return self.used


@dataclass
class ApiConfig:
    client_id: str
    client_secret: str
    route_option: str
    timeout: float
    max_retries: int
    rate_limiter: RateLimiter
    call_budget: CallBudget


def empty_result(status: str, message: str, attempt_count: int) -> dict[str, Any]:
    return {
        "route_status": status,
        "route_check": "REVIEW",
        "review_flag": status,
        "http_status": pd.NA,
        "api_code": pd.NA,
        "api_message": pd.NA,
        "road_distance_m": pd.NA,
        "road_duration_ms": pd.NA,
        "road_duration_sec": pd.NA,
        "road_duration_min": pd.NA,
        "road_to_straight_ratio": pd.NA,
        "api_current_datetime": pd.NA,
        "api_departure_time": pd.NA,
        "requested_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "snapped_start_lon": pd.NA,
        "snapped_start_lat": pd.NA,
        "snapped_goal_lon": pd.NA,
        "snapped_goal_lat": pd.NA,
        "start_snap_distance_m": pd.NA,
        "goal_snap_distance_m": pd.NA,
        "ferry_flag": pd.NA,
        "attempt_count": attempt_count,
        "last_error": message,
    }


def classify_business_code(code: int) -> str:
    return {
        1: "SAME_POINT",
        2: "POINT_NOT_NEAR_ROAD",
        3: "NO_ROUTE",
        4: "WAYPOINT_NOT_NEAR_ROAD",
        5: "DISTANCE_TOO_LONG",
    }.get(code, "API_BUSINESS_ERROR")


def parse_success_response(
    payload: dict[str, Any],
    row: pd.Series,
    route_option: str,
    http_status: int,
    attempt_count: int,
) -> dict[str, Any]:
    routes = payload.get("route", {}).get(route_option, [])
    if not routes:
        result = empty_result("MISSING_ROUTE", "성공 코드지만 경로 배열이 없습니다.", attempt_count)
        result.update(
            http_status=http_status,
            api_code=payload.get("code"),
            api_message=payload.get("message"),
            api_current_datetime=payload.get("currentDateTime"),
        )
        return result

    route = routes[0]
    summary = route.get("summary", {})
    start_location = summary.get("start", {}).get("location", [None, None])
    goal_location = summary.get("goal", {}).get("location", [None, None])
    distance_m = summary.get("distance")
    duration_ms = summary.get("duration")
    if (
        distance_m is None
        or duration_ms is None
        or len(start_location) < 2
        or len(goal_location) < 2
        or start_location[0] is None
        or start_location[1] is None
        or goal_location[0] is None
        or goal_location[1] is None
    ):
        return empty_result("MISSING_ROUTE", "경로 요약에 거리 또는 시간이 없습니다.", attempt_count)

    start_snap_m = haversine_m(
        float(row["origin_lon"]),
        float(row["origin_lat"]),
        float(start_location[0]),
        float(start_location[1]),
    )
    goal_snap_m = haversine_m(
        float(row["destination_lon"]),
        float(row["destination_lat"]),
        float(goal_location[0]),
        float(goal_location[1]),
    )
    ferry_flag = any(
        guide.get("type") in FERRY_GUIDE_TYPES for guide in route.get("guide", [])
    )
    straight_m = float(row["straight_distance_km"]) * 1_000
    ratio = float(distance_m) / straight_m if straight_m > 0 else np.nan

    flags: list[str] = []
    if ferry_flag:
        flags.append("FERRY_ROUTE")
    if start_snap_m > 100:
        flags.append("START_SNAP_OVER_100M")
    if goal_snap_m > 100:
        flags.append("GOAL_SNAP_OVER_100M")
    if ratio > 3:
        flags.append("ROAD_STRAIGHT_RATIO_OVER_3")

    return {
        "route_status": "SUCCESS",
        "route_check": "REVIEW" if flags else "PASS",
        "review_flag": "|".join(flags),
        "http_status": http_status,
        "api_code": payload.get("code"),
        "api_message": payload.get("message"),
        "road_distance_m": distance_m,
        "road_duration_ms": duration_ms,
        "road_duration_sec": float(duration_ms) / 1_000,
        "road_duration_min": float(duration_ms) / 60_000,
        "road_to_straight_ratio": ratio,
        "api_current_datetime": payload.get("currentDateTime"),
        "api_departure_time": summary.get("departureTime"),
        "requested_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "snapped_start_lon": start_location[0],
        "snapped_start_lat": start_location[1],
        "snapped_goal_lon": goal_location[0],
        "snapped_goal_lat": goal_location[1],
        "start_snap_distance_m": start_snap_m,
        "goal_snap_distance_m": goal_snap_m,
        "ferry_flag": ferry_flag,
        "attempt_count": attempt_count,
        "last_error": "",
    }


def fetch_one_route(row: pd.Series, config: ApiConfig) -> dict[str, Any]:
    headers = {
        "x-ncp-apigw-api-key-id": config.client_id,
        "x-ncp-apigw-api-key": config.client_secret,
        "User-Agent": "small-school-analysis/03-2",
    }
    params = {
        "start": f"{float(row['origin_lon']):.8f},{float(row['origin_lat']):.8f}",
        "goal": (
            f"{float(row['destination_lon']):.8f},"
            f"{float(row['destination_lat']):.8f}"
        ),
        "option": config.route_option,
    }
    prior_attempts = int(pd.to_numeric(row.get("attempt_count", 0), errors="coerce") or 0)
    last_result: dict[str, Any] | None = None

    for retry_index in range(config.max_retries + 1):
        attempt_count = prior_attempts + retry_index + 1
        if retry_index:
            time.sleep(min(8.0, 2.0 ** retry_index))
        http_status: int | None = None
        body_text = ""
        try:
            config.call_budget.reserve()
            config.rate_limiter.wait()
            request = Request(
                API_URL + "?" + urlencode(params),
                headers=headers,
                method="GET",
            )
            with urlopen(request, timeout=config.timeout) as response:
                http_status = int(response.status)
                body_text = response.read().decode("utf-8", errors="replace")
        except CallBudgetExceeded as exc:
            return empty_result("BUDGET_STOPPED", str(exc), prior_attempts + retry_index)
        except HTTPError as exc:
            http_status = int(exc.code)
            body_text = exc.read().decode("utf-8", errors="replace")
        except (URLError, TimeoutError, OSError) as exc:
            last_result = empty_result("NETWORK_ERROR", repr(exc), attempt_count)
            continue

        requested_at = datetime.now(KST).isoformat(timespec="seconds")
        if http_status in {401, 403}:
            result = empty_result(
                "AUTH_ERROR",
                "인증 실패: Client ID/Secret 및 Directions 5 선택 여부를 확인하십시오.",
                attempt_count,
            )
            result.update(http_status=http_status, requested_at_kst=requested_at)
            return result
        if http_status == 429:
            last_result = empty_result("RATE_LIMITED", "HTTP 429", attempt_count)
            last_result.update(http_status=http_status, requested_at_kst=requested_at)
            continue
        if http_status is not None and 500 <= http_status < 600:
            last_result = empty_result(
                "HTTP_SERVER_ERROR", f"HTTP {http_status}", attempt_count
            )
            last_result.update(http_status=http_status, requested_at_kst=requested_at)
            continue
        if http_status is not None and http_status >= 400:
            result = empty_result(
                "HTTP_CLIENT_ERROR",
                f"HTTP {http_status}: {body_text[:300]}",
                attempt_count,
            )
            result.update(http_status=http_status, requested_at_kst=requested_at)
            return result

        try:
            payload = json.loads(body_text)
        except (TypeError, json.JSONDecodeError) as exc:
            last_result = empty_result("JSON_ERROR", repr(exc), attempt_count)
            last_result.update(http_status=http_status, requested_at_kst=requested_at)
            continue

        code = payload.get("code")
        if code == 0:
            return parse_success_response(
                payload,
                row,
                config.route_option,
                int(http_status or 200),
                attempt_count,
            )
        status = classify_business_code(int(code)) if code is not None else "API_BUSINESS_ERROR"
        result = empty_result(status, str(payload.get("message", "")), attempt_count)
        result.update(
            http_status=http_status,
            api_code=code,
            api_message=payload.get("message"),
            api_current_datetime=payload.get("currentDateTime"),
            requested_at_kst=requested_at,
        )
        return result

    return last_result or empty_result(
        "UNKNOWN_ERROR", "알 수 없는 오류", prior_attempts
    )


def apply_result(frame: pd.DataFrame, index: int, result: dict[str, Any]) -> None:
    for column, value in result.items():
        frame.at[index, column] = value


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def validate_schedule(now_kst: datetime, allow_outside: bool) -> None:
    in_window = (
        now_kst.weekday() in RECOMMENDED_WEEKDAYS
        and RECOMMENDED_START <= now_kst.time().replace(tzinfo=None) <= RECOMMENDED_END
    )
    if not in_window and not allow_outside:
        raise ValueError(
            "현재는 권장 시작 시간(화~목 07:10~07:35 KST)이 아닙니다. "
            "공휴일·방학·특이 기상 여부를 확인한 뒤 권장 시간에 실행하거나, "
            "의도적으로 다른 시간에 실행할 경우 --allow-outside-school-window를 "
            "추가하십시오."
        )


def validate_parameters(args: argparse.Namespace) -> None:
    if args.workers <= 0 or args.batch_size <= 0:
        raise ValueError("workers와 batch_size는 0보다 커야 합니다.")
    if args.requests_per_second <= 0 or args.timeout <= 0:
        raise ValueError("requests_per_second와 timeout은 0보다 커야 합니다.")
    if args.max_retries < 0:
        raise ValueError("max_retries는 0 이상이어야 합니다.")
    if args.max_api_calls > 60_000:
        raise ValueError("무료 이용량 보호를 위해 max_api_calls는 60,000 이하만 허용합니다.")


def run_api_requests(
    routes: pd.DataFrame,
    routes_path: Path,
    config: ApiConfig,
    workers: int,
    batch_size: int,
) -> pd.DataFrame:
    pending_indices = routes.index[
        ~routes["route_status"].astype("string").isin(TERMINAL_STATUSES)
    ].tolist()
    if not pending_indices:
        logging.info("모든 방향쌍이 완료 상태이므로 API를 호출하지 않습니다.")
        return routes

    # 첫 번째 실제 요청을 단독 처리한다. 인증 오류가 있으면 대량 요청 전에 중단한다.
    first_index = pending_indices.pop(0)
    first_result = fetch_one_route(routes.loc[first_index], config)
    apply_result(routes, first_index, first_result)
    write_csv_atomic(routes, routes_path)
    if first_result["route_status"] == "AUTH_ERROR":
        raise RuntimeError(
            "첫 실제 요청에서 인증에 실패하여 대량 호출을 중단했습니다. "
            "저장된 AUTH_ERROR 행은 인증 수정 후 자동 재시도됩니다."
        )

    completed_before = int(routes["route_status"].isin(TERMINAL_STATUSES).sum())
    total = len(routes)
    for batch_start in range(0, len(pending_indices), batch_size):
        batch = pending_indices[batch_start : batch_start + batch_size]
        if not batch:
            break
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch_one_route, routes.loc[index].copy(), config): index
                for index in batch
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # 예기치 않은 작업 단위 오류도 행으로 보존
                    prior = int(pd.to_numeric(routes.at[index, "attempt_count"], errors="coerce") or 0)
                    result = empty_result("WORKER_ERROR", repr(exc), prior)
                apply_result(routes, index, result)

        write_csv_atomic(routes, routes_path)
        terminal = int(routes["route_status"].isin(TERMINAL_STATUSES).sum())
        success = int(routes["route_status"].eq("SUCCESS").sum())
        logging.info(
            "진행: 완료상태 %s/%s, 성공 %s, 이번 실행 API 호출 %s/%s",
            f"{terminal:,}",
            f"{total:,}",
            f"{success:,}",
            f"{config.call_budget.used:,}",
            f"{config.call_budget.maximum:,}",
        )
        if config.call_budget.used >= config.call_budget.maximum:
            logging.warning("API 호출 상한에 도달하여 현재 체크포인트에서 중단합니다.")
            break

    logging.info("실행 전 완료 행: %s", f"{completed_before:,}")
    return routes


def build_summary(
    routes: pd.DataFrame,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    actual_api_calls: int,
) -> pd.DataFrame:
    status_counts = routes["route_status"].value_counts(dropna=False)
    success = routes.loc[routes["route_status"].eq("SUCCESS")]
    summary: dict[str, Any] = {
        "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
        "capacity_scenario_id": args.scenario,
        "route_option": args.route_option,
        **metrics,
        "actual_api_calls_this_run": actual_api_calls,
        "success_count": int(status_counts.get("SUCCESS", 0)),
        "pass_count": int(routes["route_check"].eq("PASS").sum()),
        "review_count": int(routes["route_check"].eq("REVIEW").sum()),
        "pending_count": int(routes["route_status"].eq("PENDING").sum()),
        "non_success_terminal_count": int(
            routes["route_status"].isin(TERMINAL_STATUSES - {"SUCCESS"}).sum()
        ),
        "retryable_or_stopped_count": int(
            (~routes["route_status"].isin(TERMINAL_STATUSES | {"PENDING"})).sum()
        ),
        "ferry_route_count": int(success["ferry_flag"].eq(True).sum()),
        "snap_over_100m_count": int(
            (
                pd.to_numeric(success["start_snap_distance_m"], errors="coerce").gt(100)
                | pd.to_numeric(success["goal_snap_distance_m"], errors="coerce").gt(100)
            ).sum()
        ),
        "road_duration_min_mean": pd.NA,
        "road_duration_min_p50": pd.NA,
        "road_duration_min_p90": pd.NA,
        "road_duration_min_max": pd.NA,
        "road_distance_km_mean": pd.NA,
        "road_distance_km_p50": pd.NA,
        "road_distance_km_p90": pd.NA,
        "road_distance_km_max": pd.NA,
    }
    if not success.empty:
        duration = pd.to_numeric(success["road_duration_min"], errors="coerce")
        distance = pd.to_numeric(success["road_distance_m"], errors="coerce") / 1_000
        summary.update(
            road_duration_min_mean=duration.mean(),
            road_duration_min_p50=duration.quantile(0.50),
            road_duration_min_p90=duration.quantile(0.90),
            road_duration_min_max=duration.max(),
            road_distance_km_mean=distance.mean(),
            road_distance_km_p50=distance.quantile(0.50),
            road_distance_km_p90=distance.quantile(0.90),
            road_distance_km_max=distance.max(),
        )
    return pd.DataFrame([summary])


def validate_outputs(routes: pd.DataFrame, expected_requests: int | None) -> None:
    if routes["route_request_id"].isna().any() or routes[
        "route_request_id"
    ].duplicated().any():
        raise ValueError("route_request_id에 결측 또는 중복이 있습니다.")
    if routes.duplicated(["origin_school_key", "destination_school_key"]).any():
        raise ValueError("출발학교–도착학교 방향쌍이 중복되었습니다.")
    if routes["origin_school_key"].eq(routes["destination_school_key"]).any():
        raise ValueError("출발학교와 도착학교가 같은 행이 있습니다.")
    if expected_requests is not None and len(routes) != expected_requests:
        raise ValueError(
            f"고유 방향쌍 수가 예상과 다릅니다: 예상 {expected_requests:,}, "
            f"실제 {len(routes):,}. 입력 버전과 CAP28 기준을 확인하십시오."
        )
    success = routes["route_status"].eq("SUCCESS")
    if routes.loc[
        success,
        ["road_distance_m", "road_duration_ms", "requested_at_kst"],
    ].isna().any().any():
        raise ValueError("SUCCESS 행에 도로거리·도로시간·조회시각 결측이 있습니다.")
    if (
        pd.to_numeric(routes.loc[success, "road_distance_m"], errors="coerce").le(0).any()
        or pd.to_numeric(routes.loc[success, "road_duration_ms"], errors="coerce").le(0).any()
    ):
        raise ValueError("SUCCESS 행에 0 이하 도로거리 또는 도로시간이 있습니다.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    validate_parameters(args)
    project_root, options_path, nodes_path, routes_path, summary_path = resolve_paths(args)

    logging.info("수용력안: %s", options_path)
    logging.info("학교 노드: %s", nodes_path)
    options, nodes = load_inputs(options_path, nodes_path)
    routes, metrics = build_request_table(
        options,
        nodes,
        args.scenario,
        args.route_option,
        args.random_seed,
    )
    routes = merge_previous_results(routes, routes_path)
    expected = None if args.skip_expected_count_check else args.expected_requests
    validate_outputs(routes, expected)

    logging.info("CAP28 수용 가능안: %s", f"{metrics['feasible_option_rows']:,}")
    logging.info("원시 방향 구간: %s", f"{metrics['raw_route_legs']:,}")
    logging.info("중복 제거 후 API 요청: %s", f"{len(routes):,}")
    logging.info(
        "중복 제거율: %.2f%%", float(metrics["deduplication_rate"]) * 100
    )

    call_budget = CallBudget(args.max_api_calls)
    if args.execute:
        now_kst = datetime.now(KST)
        validate_schedule(now_kst, args.allow_outside_school_window)
        load_env_file(project_root / ".env")
        client_id = os.environ.get("NAVER_MAPS_CLIENT_ID", "").strip()
        client_secret = os.environ.get("NAVER_MAPS_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            raise ValueError(
                "NAVER_MAPS_CLIENT_ID 또는 NAVER_MAPS_CLIENT_SECRET이 없습니다. "
                "프로젝트 루트 .env를 확인하십시오."
            )
        config = ApiConfig(
            client_id=client_id,
            client_secret=client_secret,
            route_option=args.route_option,
            timeout=args.timeout,
            max_retries=args.max_retries,
            rate_limiter=RateLimiter(args.requests_per_second),
            call_budget=call_budget,
        )
        routes = run_api_requests(
            routes,
            routes_path,
            config,
            args.workers,
            args.batch_size,
        )
    else:
        logging.info("검증 모드: --execute가 없어 API를 호출하지 않았습니다.")

    validate_outputs(routes, expected)
    summary = build_summary(routes, metrics, args, call_budget.used)
    write_csv_atomic(routes, routes_path)
    write_csv_atomic(summary, summary_path)

    logging.info("경로 결과 저장: %s (%s행 × %s열)", routes_path, f"{len(routes):,}", len(routes.columns))
    logging.info("요약 저장: %s (%s행 × %s열)", summary_path, f"{len(summary):,}", len(summary.columns))
    logging.info("이번 실행의 실제 API 호출: %s", f"{call_budget.used:,}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("실행 실패: %s", exc)
        sys.exit(1)

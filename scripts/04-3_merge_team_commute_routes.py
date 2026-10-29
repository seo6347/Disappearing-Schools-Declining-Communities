"""
04-3. 팀원별 네이버 통학경로 결과 병합 및 무결성 검증

이 스크립트가 하는 일
1) 04-1 산출물 commute_route_requests.csv를 기준 모집단으로 불러온다.
2) data/01_raw/team_route_results의 팀원별 04-2 결과를 모두 읽는다.
3) route_request_id, 배치, 수집 상태, 성공 거리·시간과 중복 결과를 검증한다.
4) 동일 요청이 여러 번 수집됐다면 SUCCESS > NO_ROUTE > API_ERROR >
   RETRYABLE_ERROR 순으로 우선하고, 같은 상태에서는 가장 최근 행을 선택한다.
5) 출발지와 목적지가 같은 Directions code 1 사례는 도보 이동 가능한 동일 지점으로
   보고 분석용 도로거리 0m, 도로시간 0분으로 전환한다.
6) 페리 경로는 원시 결과를 보존하지만 후속 통학부담 분석에서는 제외한다.
7) ROAD_TO_STRAIGHT_OVER_5는 검토 플래그만 유지하고 이 단계에서 제외하지 않는다.
8) 원본 요청 1행당 정확히 1행인 통합 경로표와 QA 요약·검토표를 생성한다.

중요한 처리 원칙
- 원시 API 결과 칼럼은 수정하지 않는다.
- 후속 분석에서는 analysis_road_distance_m, analysis_road_duration_min을 사용한다.
- analysis_eligible_flag=True인 경로만 통학부담 계산에 사용한다.
- FERRY_ROUTE는 analysis_status=EXCLUDED_FERRY로 명시한다.
- START_GOAL_SAME_ZERO는 유효한 0분 경로로 분석에 포함한다.
- ROAD_TO_STRAIGHT_OVER_5는 analysis_eligible_flag=True를 유지한다.
- 팀원 결과가 참조한 요청 파일의 바이트 해시가 현재 파일과 달라도, 모든 요청 ID와
  배치가 완전하게 일치하고 팀원 결과끼리 동일한 원본 해시를 공유하면 전면 실패시키지
  않고 BYTE_HASH_MISMATCH_LOGICAL_KEYS_MATCH로 기록한다.

입력 파일
- data/03_outputs/commute_route_requests.csv
- data/01_raw/team_route_results/naver_commute_routes_B01_of_02.csv
- data/01_raw/team_route_results/naver_commute_routes_B02_of_02.csv

실행
    python scripts/04-3_merge_team_commute_routes.py

결과 파일을 직접 지정하는 경우
    python scripts/04-3_merge_team_commute_routes.py \
        --result-files path/to/batch1.csv path/to/batch2.csv

산출물
- commute_road_routes.csv
  원본 요청과 선택된 API 결과, 분석 사용 여부 및 분석용 도로거리·시간을 결합한 1:1 표.
- commute_road_routes_summary.csv
  전체·배치별 완결성, 상태값, 페리 제외, 0분 전환 및 해시 검증 요약.
- commute_road_routes_review.csv
  FAIL/REVIEW 경로와 기준 요청표에 없는 결과를 모은 검토 목록.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


REQUEST_FILE = "commute_route_requests.csv"
OUTPUT_ROUTES_FILE = "commute_road_routes.csv"
OUTPUT_SUMMARY_FILE = "commute_road_routes_summary.csv"
OUTPUT_REVIEW_FILE = "commute_road_routes_review.csv"
RESULT_PATTERN = "naver_commute_routes_B??_of_??.csv"

STATUS_PRIORITY = {
    "SUCCESS": 4,
    "NO_ROUTE": 3,
    "API_ERROR": 2,
    "RETRYABLE_ERROR": 1,
}
SUCCESS_DISTANCE_TOLERANCE_M = 1.0
SUCCESS_DURATION_TOLERANCE_MS = 1_000.0
START_GOAL_SAME_MAX_STRAIGHT_DISTANCE_KM = 0.1

REQUIRED_REQUEST_COLUMNS = [
    "route_request_id", "census_id", "origin_lon", "origin_lat",
    "destination_school_key", "destination_school_lon", "destination_school_lat",
    "straight_distance_km", "request_batch_number", "request_batch_id",
]

REQUIRED_RESULT_COLUMNS = [
    "route_request_id", "request_batch_number", "request_batch_id", "collector_label",
    "source_request_file_sha256", "route_option", "route_status", "route_check",
    "review_flag", "http_status", "api_code", "api_message", "road_distance_m",
    "road_duration_ms", "road_duration_sec", "road_duration_min",
    "road_to_straight_ratio", "api_current_datetime", "api_departure_time",
    "requested_at_kst", "collected_at_utc", "snapped_start_lon", "snapped_start_lat",
    "snapped_goal_lon", "snapped_goal_lat", "start_snap_distance_m",
    "goal_snap_distance_m", "ferry_flag", "attempt_count", "last_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="04-3 팀원 경로 결과 병합·검증")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1],
        help="final_project 폴더. 기본값은 scripts의 상위 폴더",
    )
    parser.add_argument(
        "--result-dir", type=Path, default=None,
        help="팀원 결과 폴더. 기본값: data/01_raw/team_route_results",
    )
    parser.add_argument(
        "--result-files", type=Path, nargs="*", default=None,
        help="팀원 결과 CSV 경로를 직접 지정",
    )
    return parser.parse_args()


def read_csv_text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_request_file(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "03_outputs" / REQUEST_FILE,
        project_root / "data" / "02_processed" / REQUEST_FILE,
        project_root / "data" / "01_raw" / REQUEST_FILE,
        project_root / "upload" / REQUEST_FILE,
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(
            f"{REQUEST_FILE}을 찾지 못했습니다. 확인 경로: {[str(p) for p in candidates]}"
        )
    if len(existing) > 1:
        print(f"[주의] 요청 파일이 여러 곳에 있어 우선순위가 높은 파일 사용: {existing[0]}")
    return existing[0]


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} 필수 칼럼 누락: {missing}")


def parse_bool(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper().isin(
        {"TRUE", "1", "Y", "YES"}
    )


def unique_join(series: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in series.dropna() if str(value).strip()}))


def add_flag(existing: pd.Series, condition: pd.Series, flag: str) -> pd.Series:
    updated = existing.fillna("").astype(str)
    updated.loc[condition] = updated.loc[condition].map(
        lambda value: flag if not value else f"{value}|{flag}"
    )
    return updated


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = project_root / "data" / "03_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    request_path = find_request_file(project_root)
    if args.result_files:
        result_paths = [path.resolve() for path in args.result_files]
    else:
        result_dir = (
            args.result_dir.resolve()
            if args.result_dir is not None
            else project_root / "data" / "01_raw" / "team_route_results"
        )
        result_paths = sorted(result_dir.glob(RESULT_PATTERN))

    if not result_paths:
        raise FileNotFoundError(
            "팀원 결과 CSV가 없습니다. data/01_raw/team_route_results에 "
            f"{RESULT_PATTERN} 형식으로 넣거나 --result-files를 사용하세요."
        )
    missing_files = [str(path) for path in result_paths if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"지정한 결과 파일이 없습니다: {missing_files}")

    print(f"[기준 요청 파일] {request_path}")
    print("[팀원 결과 파일]")
    for path in result_paths:
        print(f"- {path}")

    # 1. 기준 요청 모집단 검증
    requests = read_csv_text(request_path)
    require_columns(requests, REQUIRED_REQUEST_COLUMNS, REQUEST_FILE)
    if requests["route_request_id"].isna().any():
        raise ValueError("기준 요청 파일의 route_request_id에 결측이 있습니다.")
    if requests["route_request_id"].duplicated().any():
        raise ValueError("기준 요청 파일의 route_request_id가 중복되어 있습니다.")

    request_hash = sha256_file(request_path)
    request_result_placeholders = [
        "routing_provider", "routing_status", "road_distance_m",
        "road_duration_min", "routing_error",
    ]
    requests = requests.drop(
        columns=[column for column in request_result_placeholders if column in requests.columns]
    )
    requests["request_batch_number"] = pd.to_numeric(
        requests["request_batch_number"], errors="raise"
    ).astype(int)
    requests["straight_distance_km"] = pd.to_numeric(
        requests["straight_distance_km"], errors="raise"
    )
    request_id_set = set(requests["route_request_id"])
    expected_batch_map = requests.set_index("route_request_id")[[
        "request_batch_number", "request_batch_id",
    ]]

    # 2. 팀원 결과 적재 및 ID·배치·해시 검증
    raw_parts = []
    for file_order, path in enumerate(result_paths):
        part = read_csv_text(path)
        require_columns(part, REQUIRED_RESULT_COLUMNS, path.name)
        part["source_result_file"] = path.name
        part["source_file_order"] = file_order
        part["source_row_order"] = np.arange(len(part))
        raw_parts.append(part)
    raw = pd.concat(raw_parts, ignore_index=True)

    raw["result_route_id_in_request_flag"] = raw["route_request_id"].isin(request_id_set)
    raw["result_source_hash_match_flag"] = raw[
        "source_request_file_sha256"
    ].eq(request_hash)
    raw = raw.merge(
        expected_batch_map.rename(
            columns={
                "request_batch_number": "expected_request_batch_number",
                "request_batch_id": "expected_request_batch_id",
            }
        ),
        left_on="route_request_id", right_index=True, how="left", validate="many_to_one",
    )
    raw["request_batch_number_num"] = pd.to_numeric(
        raw["request_batch_number"], errors="coerce"
    )
    raw["result_batch_match_flag"] = (
        raw["request_batch_number_num"].eq(raw["expected_request_batch_number"])
        & raw["request_batch_id"].eq(raw["expected_request_batch_id"])
    )

    source_hashes = sorted({
        str(value)
        for value in raw["source_request_file_sha256"].dropna()
        if str(value).strip()
    })
    source_hash_consistent = len(source_hashes) == 1
    source_hash_exact_match = source_hashes == [request_hash]
    unexpected = raw[~raw["result_route_id_in_request_flag"]].copy()
    valid_raw = raw[raw["result_route_id_in_request_flag"]].copy()

    for column in [
        "road_distance_m", "road_duration_ms", "road_duration_sec",
        "road_duration_min", "road_to_straight_ratio",
        "start_snap_distance_m", "goal_snap_distance_m",
    ]:
        valid_raw[f"{column}_num"] = pd.to_numeric(valid_raw[column], errors="coerce")

    valid_raw["_status_priority"] = valid_raw["route_status"].map(
        STATUS_PRIORITY
    ).fillna(0)
    valid_raw["_collected_at"] = pd.to_datetime(
        valid_raw["collected_at_utc"], errors="coerce", utc=True
    )
    valid_raw["_global_row_order"] = np.arange(len(valid_raw))
    duplicate_counts = valid_raw.groupby("route_request_id").size()
    duplicate_route_ids = set(duplicate_counts[duplicate_counts > 1].index)

    success_raw = valid_raw[valid_raw["route_status"].eq("SUCCESS")].copy()
    success_conflict_ids: set[str] = set()
    if not success_raw.empty:
        success_spread = success_raw.groupby("route_request_id").agg(
            success_count=("route_request_id", "size"),
            distance_min=("road_distance_m_num", "min"),
            distance_max=("road_distance_m_num", "max"),
            duration_min=("road_duration_ms_num", "min"),
            duration_max=("road_duration_ms_num", "max"),
        )
        conflict_mask = (
            (success_spread["success_count"] > 1)
            & (
                (success_spread["distance_max"] - success_spread["distance_min"]
                 > SUCCESS_DISTANCE_TOLERANCE_M)
                | (success_spread["duration_max"] - success_spread["duration_min"]
                   > SUCCESS_DURATION_TOLERANCE_MS)
            )
        )
        success_conflict_ids = set(success_spread[conflict_mask].index)

    selected = valid_raw.sort_values(
        [
            "route_request_id", "_status_priority", "_collected_at",
            "source_file_order", "source_row_order", "_global_row_order",
        ],
        kind="stable",
    ).drop_duplicates("route_request_id", keep="last")

    result_columns_for_merge = [
        "route_request_id", "collector_label", "source_request_file_sha256",
        "route_option", "route_status", "route_check", "review_flag", "http_status",
        "api_code", "api_message", "road_distance_m", "road_duration_ms",
        "road_duration_sec", "road_duration_min", "road_to_straight_ratio",
        "api_current_datetime", "api_departure_time", "requested_at_kst",
        "collected_at_utc", "snapped_start_lon", "snapped_start_lat",
        "snapped_goal_lon", "snapped_goal_lat", "start_snap_distance_m",
        "goal_snap_distance_m", "ferry_flag", "attempt_count", "last_error",
        "source_result_file", "result_source_hash_match_flag", "result_batch_match_flag",
    ]
    selected = selected[result_columns_for_merge]

    # 3. 기준 요청 1행당 수집 결과 1행 병합
    merged = requests.merge(selected, on="route_request_id", how="left", validate="one_to_one")
    merged["raw_result_duplicate_flag"] = merged["route_request_id"].isin(
        duplicate_route_ids
    )
    merged["success_result_conflict_flag"] = merged["route_request_id"].isin(
        success_conflict_ids
    )
    merged["ferry_flag"] = parse_bool(merged["ferry_flag"])
    for column in [
        "road_distance_m", "road_duration_ms", "road_duration_sec",
        "road_duration_min", "road_to_straight_ratio",
        "start_snap_distance_m", "goal_snap_distance_m",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    missing_result = merged["route_status"].isna()
    success = merged["route_status"].eq("SUCCESS")
    no_route = merged["route_status"].eq("NO_ROUTE")
    api_error = merged["route_status"].eq("API_ERROR")
    retryable = merged["route_status"].eq("RETRYABLE_ERROR")
    invalid_success_metrics = success & (
        merged["road_distance_m"].isna()
        | merged["road_duration_ms"].isna()
        | (merged["road_distance_m"] <= 0)
        | (merged["road_duration_ms"] <= 0)
    )

    error_text = (
        merged["api_message"].fillna("").astype(str)
        + " "
        + merged["last_error"].fillna("").astype(str)
    )
    start_goal_same_text = error_text.str.contains(
        r"출발지와 도착지가 동일|출발.*도착.*동일|\"code\"\s*:\s*1|Directions code 1",
        regex=True, case=False, na=False,
    )
    zero_route_imputed = (
        api_error
        & merged["http_status"].astype(str).eq("400")
        & start_goal_same_text
        & merged["straight_distance_km"].le(START_GOAL_SAME_MAX_STRAIGHT_DISTANCE_KM)
    )
    unresolved_api_error = api_error & ~zero_route_imputed
    ferry_route = success & merged["ferry_flag"]
    road_ratio_review = merged["review_flag"].fillna("").str.contains(
        "ROAD_TO_STRAIGHT_OVER_5", regex=False
    )

    # 4. 원시 결과와 분리된 분석용 상태·거리·시간 생성
    merged["zero_route_imputed_flag"] = zero_route_imputed
    merged["analysis_status"] = np.select(
        [
            missing_result, retryable, unresolved_api_error, no_route,
            invalid_success_metrics, ferry_route, zero_route_imputed, success,
        ],
        [
            "MISSING_RESULT", "RETRYABLE_ERROR", "API_ERROR_UNRESOLVED", "NO_ROUTE",
            "INVALID_SUCCESS_METRICS", "EXCLUDED_FERRY", "USABLE_ZERO_ROUTE", "USABLE",
        ],
        default="UNKNOWN_STATUS",
    )
    merged["analysis_eligible_flag"] = merged["analysis_status"].isin(
        {"USABLE", "USABLE_ZERO_ROUTE"}
    )
    merged["analysis_exclusion_reason"] = np.select(
        [
            ferry_route, missing_result, retryable, unresolved_api_error,
            no_route, invalid_success_metrics,
        ],
        [
            "FERRY_ROUTE", "MISSING_RESULT", "RETRYABLE_ERROR",
            "API_ERROR_UNRESOLVED", "NO_ROUTE", "INVALID_SUCCESS_METRICS",
        ],
        default="",
    )

    merged["analysis_road_distance_m"] = merged["road_distance_m"]
    merged["analysis_road_duration_ms"] = merged["road_duration_ms"]
    merged["analysis_road_duration_sec"] = merged["road_duration_sec"]
    merged["analysis_road_duration_min"] = merged["road_duration_min"]
    merged["analysis_road_to_straight_ratio"] = merged["road_to_straight_ratio"]
    merged.loc[zero_route_imputed, "analysis_road_distance_m"] = 0.0
    merged.loc[zero_route_imputed, "analysis_road_duration_ms"] = 0.0
    merged.loc[zero_route_imputed, "analysis_road_duration_sec"] = 0.0
    merged.loc[zero_route_imputed, "analysis_road_duration_min"] = 0.0
    merged.loc[zero_route_imputed, "analysis_road_to_straight_ratio"] = 0.0

    analysis_ineligible = ~merged["analysis_eligible_flag"]
    for column in [
        "analysis_road_distance_m", "analysis_road_duration_ms",
        "analysis_road_duration_sec", "analysis_road_duration_min",
        "analysis_road_to_straight_ratio",
    ]:
        merged.loc[analysis_ineligible, column] = np.nan

    merged["merge_status"] = np.select(
        [missing_result, success, zero_route_imputed, no_route, unresolved_api_error, retryable],
        ["MISSING_RESULT", "SUCCESS", "START_GOAL_SAME_ZERO", "NO_ROUTE", "API_ERROR", "RETRYABLE_ERROR"],
        default="UNKNOWN_STATUS",
    )

    merged["merge_review_flag"] = merged["review_flag"].fillna("")
    merged["merge_review_flag"] = add_flag(
        merged["merge_review_flag"], missing_result, "MISSING_RESULT"
    )
    merged["merge_review_flag"] = add_flag(
        merged["merge_review_flag"], no_route, "NO_ROUTE"
    )
    merged["merge_review_flag"] = add_flag(
        merged["merge_review_flag"], retryable, "RETRYABLE_ERROR"
    )
    merged["merge_review_flag"] = add_flag(
        merged["merge_review_flag"], zero_route_imputed, "START_GOAL_SAME_IMPUTED_ZERO"
    )
    merged["merge_review_flag"] = add_flag(
        merged["merge_review_flag"], unresolved_api_error, "API_ERROR_UNRESOLVED"
    )
    merged["merge_review_flag"] = add_flag(
        merged["merge_review_flag"], merged["result_batch_match_flag"].eq(False),
        "REQUEST_BATCH_MISMATCH",
    )
    merged["merge_review_flag"] = add_flag(
        merged["merge_review_flag"], merged["success_result_conflict_flag"],
        "SUCCESS_RESULT_CONFLICT",
    )
    merged["merge_review_flag"] = add_flag(
        merged["merge_review_flag"], invalid_success_metrics, "INVALID_SUCCESS_METRICS",
    )

    logical_key_complete = (
        missing_result.sum() == 0
        and len(unexpected) == 0
        and merged["result_batch_match_flag"].fillna(False).all()
    )
    if source_hash_exact_match:
        source_hash_validation = "EXACT_MATCH"
    elif source_hash_consistent and logical_key_complete:
        source_hash_validation = "BYTE_HASH_MISMATCH_LOGICAL_KEYS_MATCH"
    else:
        source_hash_validation = "SOURCE_HASH_VALIDATION_FAILED"
    merged["source_hash_validation"] = source_hash_validation

    source_hash_fail = source_hash_validation == "SOURCE_HASH_VALIDATION_FAILED"
    structural_fail = (
        missing_result
        | retryable
        | unresolved_api_error
        | no_route
        | invalid_success_metrics
        | merged["result_batch_match_flag"].eq(False)
        | merged["success_result_conflict_flag"]
    )
    if source_hash_fail:
        structural_fail = structural_fail | True

    # ROAD_TO_STRAIGHT_OVER_5는 REVIEW일 뿐 분석 제외 조건이 아니다.
    review_needed = (
        ferry_route
        | zero_route_imputed
        | road_ratio_review
        | merged["route_check"].eq("REVIEW")
    )
    merged["merge_check"] = np.select(
        [structural_fail, review_needed], ["FAIL", "REVIEW"], default="PASS"
    )

    # 5. 검토 목록
    review_columns = [
        "route_request_id", "request_batch_id", "census_id",
        "destination_school_key", "destination_school_name", "collector_label",
        "source_result_file", "route_status", "merge_status", "analysis_status",
        "analysis_eligible_flag", "analysis_exclusion_reason", "merge_check",
        "merge_review_flag", "straight_distance_km", "road_distance_m",
        "road_duration_min", "analysis_road_distance_m", "analysis_road_duration_min",
        "road_to_straight_ratio", "start_snap_distance_m", "goal_snap_distance_m",
        "ferry_flag", "zero_route_imputed_flag", "api_code", "api_message", "last_error",
    ]
    review_columns = [column for column in review_columns if column in merged.columns]
    review = merged[merged["merge_check"].ne("PASS")][review_columns].copy()

    if not unexpected.empty:
        unexpected_review = pd.DataFrame({
            "route_request_id": unexpected["route_request_id"],
            "request_batch_id": unexpected["request_batch_id"],
            "collector_label": unexpected["collector_label"],
            "source_result_file": unexpected["source_result_file"],
            "route_status": unexpected["route_status"],
            "merge_status": "UNEXPECTED_ROUTE_ID",
            "analysis_status": "UNEXPECTED_ROUTE_ID",
            "analysis_eligible_flag": False,
            "analysis_exclusion_reason": "UNEXPECTED_ROUTE_ID",
            "merge_check": "FAIL",
            "merge_review_flag": "UNEXPECTED_ROUTE_ID_NOT_IN_REQUEST_FILE",
            "api_code": unexpected["api_code"],
            "api_message": unexpected["api_message"],
            "last_error": unexpected["last_error"],
        })
        review = pd.concat([review, unexpected_review], ignore_index=True, sort=False)

    # 6. 전체·배치별 요약
    summary_rows: list[dict[str, object]] = []

    def add_summary(
        section: str, group: str, metric: str, value: object,
        unit: str, status: str, note: str,
    ) -> None:
        summary_rows.append({
            "section": section, "group": group, "metric": metric,
            "value": value, "unit": unit, "check_status": status, "note": note,
        })

    missing_count = int(missing_result.sum())
    retryable_count = int(retryable.sum())
    no_route_count = int(no_route.sum())
    success_count = int(success.sum())
    api_error_count = int(api_error.sum())
    zero_route_count = int(zero_route_imputed.sum())
    unresolved_api_error_count = int(unresolved_api_error.sum())
    ferry_count = int(ferry_route.sum())
    road_ratio_review_count = int(road_ratio_review.sum())
    structural_fail_count = int((merged["merge_check"] == "FAIL").sum())
    analysis_eligible_count = int(merged["analysis_eligible_flag"].sum())

    blocking_count = (
        missing_count + retryable_count + no_route_count + unresolved_api_error_count
        + int(invalid_success_metrics.sum()) + len(success_conflict_ids) + len(unexpected)
    )
    if source_hash_fail or blocking_count > 0 or structural_fail_count > 0:
        overall_status = "INCOMPLETE_OR_FAILED"
    elif ferry_count > 0 or zero_route_count > 0 or road_ratio_review_count > 0:
        overall_status = "READY_WITH_REVIEW_FLAGS"
    else:
        overall_status = "READY_FOR_ANALYSIS"

    add_summary("OVERALL", "ALL", "overall_status", overall_status, "status", overall_status, "")
    add_summary("INPUT", "ALL", "request_file_sha256", request_hash, "hash", "INFO", request_path.name)
    add_summary("INPUT", "ALL", "result_source_request_sha256", "|".join(source_hashes), "hash", "INFO", "팀원 결과가 기록한 원본 요청 파일 해시")
    add_summary(
        "INPUT", "ALL", "source_hash_validation", source_hash_validation, "status",
        "PASS" if not source_hash_fail else "FAIL",
        "바이트 해시 불일치 시 전체 요청 ID·배치 완전 일치 여부로 논리 검증",
    )
    add_summary("INPUT", "ALL", "required_route_count", len(requests), "routes", "PASS", "기준 모집단")
    add_summary("INPUT", "ALL", "result_file_count", len(result_paths), "files", "PASS", unique_join(pd.Series([p.name for p in result_paths])))
    add_summary("RAW", "ALL", "raw_result_row_count", len(raw), "rows", "INFO", "재시도 중복 포함")
    add_summary("RAW", "ALL", "unexpected_route_id_rows", len(unexpected), "rows", "PASS" if len(unexpected) == 0 else "FAIL", "기준 요청에 없는 ID")
    add_summary("RAW", "ALL", "duplicated_route_id_count", len(duplicate_route_ids), "routes", "INFO", "상태 우선순위로 자동 해소")
    add_summary("RAW", "ALL", "success_result_conflict_count", len(success_conflict_ids), "routes", "PASS" if len(success_conflict_ids) == 0 else "FAIL", "실시간 재호출 결과 차이")

    add_summary("MERGED", "ALL", "success_count", success_count, "routes", "PASS", "원시 API 성공")
    add_summary("MERGED", "ALL", "api_error_count", api_error_count, "routes", "INFO", "0분 전환 포함 원시 API_ERROR")
    add_summary("MERGED", "ALL", "start_goal_same_zero_count", zero_route_count, "routes", "PASS", "도보 이동 가능한 동일 지점으로 0m·0분 전환")
    add_summary("MERGED", "ALL", "unresolved_api_error_count", unresolved_api_error_count, "routes", "PASS" if unresolved_api_error_count == 0 else "FAIL", "0분 전환으로 해결되지 않은 API 오류")
    add_summary("MERGED", "ALL", "no_route_count", no_route_count, "routes", "PASS" if no_route_count == 0 else "FAIL", "")
    add_summary("MERGED", "ALL", "retryable_error_count", retryable_count, "routes", "PASS" if retryable_count == 0 else "FAIL", "Colab 재실행 대상")
    add_summary("MERGED", "ALL", "missing_result_count", missing_count, "routes", "PASS" if missing_count == 0 else "FAIL", "팀원 결과 누락")
    add_summary("MERGED", "ALL", "pass_count", int((merged["merge_check"] == "PASS").sum()), "routes", "INFO", "")
    add_summary("MERGED", "ALL", "review_count", int((merged["merge_check"] == "REVIEW").sum()), "routes", "INFO", "")
    add_summary("MERGED", "ALL", "fail_count", structural_fail_count, "routes", "PASS" if structural_fail_count == 0 else "FAIL", "")

    add_summary("ANALYSIS", "ALL", "analysis_eligible_count", analysis_eligible_count, "routes", "PASS", "ROAD_TO_STRAIGHT_OVER_5 포함")
    add_summary("ANALYSIS", "ALL", "ferry_excluded_count", ferry_count, "routes", "PASS", "원시 결과 보존, 후속 분석 제외")
    add_summary("ANALYSIS", "ALL", "road_to_straight_over_5_count", road_ratio_review_count, "routes", "INFO", "검토 플래그만 유지하고 분석 포함")
    add_summary("ANALYSIS", "ALL", "zero_minute_imputed_count", zero_route_count, "routes", "PASS", "START_GOAL_SAME")

    add_summary("OUTPUT", "ALL", "merged_route_rows", len(merged), "rows", "PASS", OUTPUT_ROUTES_FILE)
    add_summary("OUTPUT", "ALL", "merged_route_columns", len(merged.columns), "columns", "PASS", OUTPUT_ROUTES_FILE)
    add_summary("OUTPUT", "ALL", "review_rows", len(review), "rows", "INFO", OUTPUT_REVIEW_FILE)

    for batch_id, batch_group in merged.groupby("request_batch_id", sort=True):
        batch_missing = int(batch_group["route_status"].isna().sum())
        batch_retry = int(batch_group["route_status"].eq("RETRYABLE_ERROR").sum())
        batch_success = int(batch_group["route_status"].eq("SUCCESS").sum())
        batch_api_error = int(batch_group["route_status"].eq("API_ERROR").sum())
        batch_zero = int(batch_group["zero_route_imputed_flag"].sum())
        batch_ferry = int(batch_group["analysis_status"].eq("EXCLUDED_FERRY").sum())
        batch_eligible = int(batch_group["analysis_eligible_flag"].sum())
        batch_fail = int(batch_group["merge_check"].eq("FAIL").sum())
        batch_status = "PASS" if batch_missing == 0 and batch_retry == 0 and batch_fail == 0 else "FAIL"
        add_summary("BATCH", batch_id, "assigned_route_count", len(batch_group), "routes", "PASS", "")
        add_summary("BATCH", batch_id, "success_count", batch_success, "routes", "INFO", "")
        add_summary("BATCH", batch_id, "api_error_count", batch_api_error, "routes", "INFO", "")
        add_summary("BATCH", batch_id, "zero_route_imputed_count", batch_zero, "routes", batch_status, "")
        add_summary("BATCH", batch_id, "ferry_excluded_count", batch_ferry, "routes", "INFO", "")
        add_summary("BATCH", batch_id, "analysis_eligible_count", batch_eligible, "routes", batch_status, "")
        add_summary("BATCH", batch_id, "retryable_error_count", batch_retry, "routes", batch_status, "")
        add_summary("BATCH", batch_id, "missing_result_count", batch_missing, "routes", batch_status, "")

    summary = pd.DataFrame(summary_rows)
    merged_path = output_dir / OUTPUT_ROUTES_FILE
    summary_path = output_dir / OUTPUT_SUMMARY_FILE
    review_path = output_dir / OUTPUT_REVIEW_FILE
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    review.to_csv(review_path, index=False, encoding="utf-8-sig", float_format="%.10f")

    print("\n[병합 완료]")
    print(f"- {merged_path}: {merged.shape[0]:,}행 x {merged.shape[1]:,}열")
    print(f"- {summary_path}: {summary.shape[0]:,}행 x {summary.shape[1]:,}열")
    print(f"- {review_path}: {review.shape[0]:,}행 x {review.shape[1]:,}열")
    print(f"- 전체 상태: {overall_status}")
    print(f"- 원시 SUCCESS: {success_count:,}건")
    print(f"- 0분 전환: {zero_route_count:,}건")
    print(f"- 분석 사용 가능: {analysis_eligible_count:,}건")
    print(f"- 페리 제외: {ferry_count:,}건")
    print(f"- ROAD_TO_STRAIGHT_OVER_5 유지: {road_ratio_review_count:,}건")
    print(f"- FAIL: {structural_fail_count:,}건")
    print(f"- 해시 검증: {source_hash_validation}")
    if overall_status == "INCOMPLETE_OR_FAILED":
        print("[다음 조치] commute_road_routes_review.csv의 FAIL 사유를 먼저 해결하세요.")
    else:
        print("[다음 조치] analysis_eligible_flag=TRUE 경로로 통합 전후 통학부담을 계산할 수 있습니다.")


if __name__ == "__main__":
    main()

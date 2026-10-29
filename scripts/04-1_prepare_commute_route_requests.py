"""
04-1. 통합 전·후 통학부담 분석용 학생-통합학교안 행렬 및 경로 요청 목록 생성

이 스크립트가 하는 일
1) 03단계에서 확정된 비중복 최종 군집(3개교)을 불러온다.
2) 각 최종 군집의 보정 완료 학생 수요점과 가능한 통합학교안을 결합한다.
3) 통합 전(집계구 대표점 -> 현재 학교)과 통합 후
   (집계구 대표점 -> 통합학교)에 필요한 경로를 만든다.
4) 여러 통합학교안에서 반복되는 동일한
   '집계구 대표점-목적 학교' 쌍을 하나의 요청으로 중복 제거한다.
5) 학교별 학생 가중치 합계, 좌표, 연결키, 중복 여부를 검증한다.
6) 필요하면 하나의 요청표 안에서 팀 분담용 배치 번호를 부여한다.
   같은 집계구의 통합 전·후 경로가 서로 다른 팀에 갈라지지 않도록 census_id 단위로 묶는다.

중요한 분석 원칙
- 후보 군집: A-B-C처럼 함께 검토하는 3개교 묶음
- 통합학교안: 후보 군집에서 한 학교를 통합학교로 두고 나머지 두 학교를 합치는 안
- student_weight가 0보다 큰 행만 실제 통학부담 분석 및 경로 계산에 사용한다.
  0명 행을 제외해도 학교별/전체 학생 가중치 합계는 변하지 않아야 한다.
- 같은 경로는 시나리오와 통합학교안이 달라도 한 번만 계산하고 결과를 재사용한다.
- 기본 분석은 CAP28이다. CAP19는 CAP28의 목적학교 부분집합이므로 CAP19를 추가해도
  고유 경로 수는 늘지 않지만, 분석 행렬 행 수는 증가한다.
- 이 스크립트는 길찾기 API를 호출하지 않는다. 예상 호출량을 먼저 확정하는 준비 단계다.
- 통합버스 30분 약속 검증의 1차 결과는 '대표점에서 통합학교까지 직행 차량 시간'이다.
  여러 마을을 순회하는 실제 통학버스 탑승시간보다 유리한 하한값임을 후속 기사에서 명시한다.

입력 파일
- current_student_distribution.csv
- nonoverlap_cluster_selection.csv
- final_selected_cluster_options.csv
- school_road_routes.csv (선택: 같은 달 기존 API 사용량의 최소치 참고용)

핵심 산출물
- commute_demand_option_matrix.csv
  보정 학생 배정행 x 통합학교안 분석 행렬. 이후 통합 전후 시간/거리 차이를 붙이는 기준표.
- commute_route_requests.csv
  중복 제거된 집계구 대표점 -> 학교 경로 목록. 로컬 라우팅 또는 API 입력표.
- commute_route_request_summary.csv
  표본 수, 예상 경로 수, 검증 결과, 무료 한도 관련 경고를 모은 요약표.

실행 예시
    python scripts/04-1_prepare_commute_route_requests.py

CAP28 요청을 3개 작업 묶음으로 나눠 팀에 분담하는 경우
    python scripts/04-1_prepare_commute_route_requests.py --batch-count 3

CAP19 재평가 행렬까지 함께 만드는 경우
    python scripts/04-1_prepare_commute_route_requests.py --scenarios CAP28,CAP19

다른 프로젝트 경로를 직접 지정하는 경우
    python scripts/04-1_prepare_commute_route_requests.py --project-root "C:/.../final_project"
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


# 2025년 출시 Maps 상품 공지 기준 Directions 5 대표계정 월 무료 이용량.
# 실제 계정의 남은 무료량은 콘솔에서 반드시 별도로 확인해야 한다.
NAVER_DIRECTIONS5_MONTHLY_FREE_QUOTA = 60_000

CURRENT_DISTRIBUTION_FILE = "current_student_distribution.csv"
SELECTED_CLUSTER_FILE = "nonoverlap_cluster_selection.csv"
SELECTED_OPTION_FILE = "final_selected_cluster_options.csv"
PREVIOUS_ROUTE_FILE = "school_road_routes.csv"

OUTPUT_MATRIX_FILE = "commute_demand_option_matrix.csv"
OUTPUT_REQUEST_FILE = "commute_route_requests.csv"
OUTPUT_SUMMARY_FILE = "commute_route_request_summary.csv"

WEIGHT_EPSILON = 1e-12
CALIBRATION_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="04-1 통학경로 요청 준비")
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="final_project 폴더. 기본값은 scripts 폴더의 상위 폴더",
    )
    parser.add_argument(
        "--scenarios",
        default="CAP28",
        help="쉼표로 구분한 수용력 시나리오. 기본값: CAP28",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=1,
        help="경로 요청을 안정적으로 나눌 팀 작업 묶음 수. 기본값: 1",
    )
    return parser.parse_args()


def find_input(project_root: Path, filename: str, required: bool = True) -> Path | None:
    """프로젝트의 표준 데이터 폴더에서 입력 파일을 우선순위대로 찾는다."""
    search_dirs = [
        project_root / "data" / "03_outputs",
        project_root / "data" / "02_processed",
        project_root / "data" / "01_raw",
        
    ]
    matches = [folder / filename for folder in search_dirs if (folder / filename).exists()]
    if not matches:
        if required:
            raise FileNotFoundError(
                f"입력 파일을 찾지 못했습니다: {filename}\n"
                f"확인한 폴더: {[str(p) for p in search_dirs]}"
            )
        return None
    if len(matches) > 1:
        print(f"[주의] {filename}이 여러 폴더에 있습니다. 우선순위가 가장 높은 파일 사용: {matches[0]}")
    return matches[0]


def read_csv_as_text(path: Path) -> pd.DataFrame:
    """ID의 앞자리 0과 문자열 키를 보존하기 위해 우선 문자열로 읽는다."""
    return pd.read_csv(path, dtype=str, low_memory=False)


def require_columns(df: pd.DataFrame, columns: list[str], file_label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{file_label} 필수 칼럼 누락: {missing}")


def to_bool(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper().isin({"TRUE", "1", "Y", "YES"})


def stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def haversine_km(
    lon1: pd.Series,
    lat1: pd.Series,
    lon2: pd.Series,
    lat2: pd.Series,
) -> np.ndarray:
    """WGS84 경위도 두 점 사이의 대권거리(km)를 벡터 연산으로 계산한다."""
    lon1_rad = np.radians(lon1.astype(float).to_numpy())
    lat1_rad = np.radians(lat1.astype(float).to_numpy())
    lon2_rad = np.radians(lon2.astype(float).to_numpy())
    lat2_rad = np.radians(lat2.astype(float).to_numpy())
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def unique_join(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna() if str(value).strip()})
    return "|".join(values)


def check_coordinate_range(df: pd.DataFrame, lon_col: str, lat_col: str, label: str) -> None:
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    invalid = lon.isna() | lat.isna() | ~lon.between(124, 132) | ~lat.between(32, 39.5)
    if invalid.any():
        examples = df.loc[invalid, [lon_col, lat_col]].head(5).to_dict("records")
        raise ValueError(f"{label} 좌표 누락 또는 대한민국 범위 밖: {invalid.sum()}행, 예시={examples}")


def decline_group(decline_count: int, concern_count: int) -> str:
    if decline_count == 3:
        return "ALL_3_DECLINE"
    if decline_count > 0:
        return "MIXED_1_OR_2_DECLINE"
    if concern_count > 0:
        return "CONCERN_ONLY"
    return "NO_DECLINE_OR_CONCERN"


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    if args.batch_count < 1:
        raise ValueError("--batch-count는 1 이상의 정수여야 합니다.")
    requested_scenarios = [
        value.strip().upper() for value in args.scenarios.split(",") if value.strip()
    ]
    if not requested_scenarios:
        raise ValueError("--scenarios에 하나 이상의 시나리오를 입력해야 합니다.")
    output_dir = project_root / "data" / "03_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution_path = find_input(project_root, CURRENT_DISTRIBUTION_FILE)
    cluster_path = find_input(project_root, SELECTED_CLUSTER_FILE)
    option_path = find_input(project_root, SELECTED_OPTION_FILE)
    previous_route_path = find_input(project_root, PREVIOUS_ROUTE_FILE, required=False)

    print("[입력 파일]")
    print(f"- 학생분포: {distribution_path}")
    print(f"- 최종 군집: {cluster_path}")
    print(f"- 통합학교안: {option_path}")
    print(f"- 기존 학교 간 경로(선택): {previous_route_path}")

    distribution = read_csv_as_text(distribution_path)
    clusters_all = read_csv_as_text(cluster_path)
    options = read_csv_as_text(option_path)

    require_columns(
        distribution,
        [
            "census_id", "longitude", "latitude", "school_key", "school_name",
            "school_lon", "school_lat", "school_edu_office", "school_area_type",
            "population_decline_type", "actual_enrollment", "student_weight",
            "assignment_status", "calibration_status",
        ],
        CURRENT_DISTRIBUTION_FILE,
    )
    require_columns(
        clusters_all,
        [
            "final_cluster_id", "selected_cluster_flag", "candidate_id",
            "school_1_key", "school_1_name", "school_2_key", "school_2_name",
            "school_3_key", "school_3_name",
        ],
        SELECTED_CLUSTER_FILE,
    )
    require_columns(
        options,
        [
            "final_cluster_id", "candidate_id", "capacity_scenario_id",
            "scenario_role", "capacity_option_id", "retained_school_key",
            "retained_school_name", "closed_school_1_key", "closed_school_1_name",
            "closed_school_2_key", "closed_school_2_name", "students_per_class",
            "total_demand", "retained_capacity", "capacity_margin",
            "capacity_utilization_rate", "final_road_eligible",
        ],
        SELECTED_OPTION_FILE,
    )
    available_scenarios = set(options["capacity_scenario_id"].dropna().str.upper())
    missing_scenarios = sorted(set(requested_scenarios) - available_scenarios)
    if missing_scenarios:
        raise ValueError(
            f"요청한 수용력 시나리오가 파일에 없습니다: {missing_scenarios}; "
            f"사용 가능={sorted(available_scenarios)}"
        )
    options = options[
        options["capacity_scenario_id"].str.upper().isin(requested_scenarios)
    ].copy()
    print(f"- 분석 수용력 시나리오: {requested_scenarios}")
    print(f"- 팀 작업 묶음 수: {args.batch_count}")

    # ------------------------------------------------------------------
    # 1. 비중복 최종 군집과 3개교 구성표 생성
    # ------------------------------------------------------------------
    selected_clusters = clusters_all[to_bool(clusters_all["selected_cluster_flag"])].copy()
    if selected_clusters.empty:
        raise ValueError("selected_cluster_flag=TRUE인 최종 군집이 없습니다.")
    if selected_clusters["final_cluster_id"].duplicated().any():
        raise ValueError("선택된 final_cluster_id가 중복되어 있습니다.")

    membership_parts = []
    for position in (1, 2, 3):
        part = selected_clusters[
            ["final_cluster_id", "candidate_id", f"school_{position}_key", f"school_{position}_name"]
        ].rename(
            columns={
                f"school_{position}_key": "school_key",
                f"school_{position}_name": "cluster_school_name",
            }
        )
        part["cluster_school_position"] = position
        membership_parts.append(part)
    membership = pd.concat(membership_parts, ignore_index=True)

    if membership["school_key"].isna().any():
        raise ValueError("최종 군집의 학교키에 결측치가 있습니다.")
    if membership["school_key"].duplicated().any():
        duplicated = membership.loc[membership["school_key"].duplicated(False), "school_key"].tolist()[:10]
        raise ValueError(f"비중복 군집인데 학교가 여러 군집에 중복됨: {duplicated}")

    # ------------------------------------------------------------------
    # 2. 최종 군집에 속한 보정 학생 배정행 추출 및 검증
    # ------------------------------------------------------------------
    distribution["student_weight"] = pd.to_numeric(distribution["student_weight"], errors="coerce")
    distribution["actual_enrollment"] = pd.to_numeric(distribution["actual_enrollment"], errors="coerce")
    for column in ["longitude", "latitude", "school_lon", "school_lat"]:
        distribution[column] = pd.to_numeric(distribution[column], errors="coerce")

    selected_distribution = distribution.merge(
        membership,
        on="school_key",
        how="inner",
        validate="many_to_one",
    )
    expected_school_count = len(membership)
    actual_school_count = selected_distribution["school_key"].nunique()
    if actual_school_count != expected_school_count:
        missing = sorted(set(membership["school_key"]) - set(selected_distribution["school_key"]))
        raise ValueError(
            f"최종 군집 학교의 학생분포 연결 실패: 기대 {expected_school_count}개교, "
            f"연결 {actual_school_count}개교, 미연결 예시={missing[:10]}"
        )
    if selected_distribution.duplicated(["census_id", "school_key"]).any():
        raise ValueError("current_student_distribution에 census_id-school_key 중복행이 있습니다.")
    if not selected_distribution["assignment_status"].eq("ASSIGNED").all():
        raise ValueError("최종 군집 학생분포에 ASSIGNED가 아닌 행이 포함되어 있습니다.")
    if not selected_distribution["calibration_status"].eq("CALIBRATED").all():
        raise ValueError("최종 군집 학생분포에 CALIBRATED가 아닌 행이 포함되어 있습니다.")
    if selected_distribution["student_weight"].isna().any():
        raise ValueError("최종 군집 학생분포의 student_weight에 숫자 변환 불가 값이 있습니다.")
    if (selected_distribution["student_weight"] < -WEIGHT_EPSILON).any():
        raise ValueError("student_weight에 음수가 있습니다.")

    check_coordinate_range(selected_distribution, "longitude", "latitude", "집계구 대표점")
    check_coordinate_range(selected_distribution, "school_lon", "school_lat", "학교")

    school_reconciliation = (
        selected_distribution.groupby("school_key", as_index=False)
        .agg(
            calibrated_student_sum=("student_weight", "sum"),
            actual_enrollment=("actual_enrollment", "first"),
            actual_enrollment_nunique=("actual_enrollment", "nunique"),
        )
    )
    if (school_reconciliation["actual_enrollment_nunique"] > 1).any():
        raise ValueError("동일 학교키에 actual_enrollment 값이 여러 개 존재합니다.")
    school_reconciliation["calibration_abs_error"] = (
        school_reconciliation["calibrated_student_sum"]
        - school_reconciliation["actual_enrollment"]
    ).abs()
    max_calibration_error = float(school_reconciliation["calibration_abs_error"].max())
    if max_calibration_error > CALIBRATION_TOLERANCE:
        raise ValueError(
            f"학교별 학생 가중치 합계가 실제 재학생 수와 불일치합니다. 최대 절대오차={max_calibration_error}"
        )

    zero_weight_rows = int((selected_distribution["student_weight"] <= WEIGHT_EPSILON).sum())
    full_weight_sum = float(selected_distribution["student_weight"].sum())
    demand = selected_distribution[selected_distribution["student_weight"] > WEIGHT_EPSILON].copy()
    positive_weight_sum = float(demand["student_weight"].sum())
    if not math.isclose(full_weight_sum, positive_weight_sum, abs_tol=CALIBRATION_TOLERANCE):
        raise ValueError("0명 행 제외 전후 전체 학생 가중치 합계가 달라졌습니다.")

    demand["demand_assignment_id"] = [
        stable_id("DA", census_id, school_key)
        for census_id, school_key in zip(demand["census_id"], demand["school_key"])
    ]
    if demand["demand_assignment_id"].duplicated().any():
        raise ValueError("demand_assignment_id 해시 충돌 또는 원자료 중복이 발생했습니다.")

    # 학교별 좌표/분류 메타데이터는 한 학교에서 하나의 값이어야 한다.
    school_meta_columns = [
        "school_key", "school_name", "school_edu_office", "school_lon", "school_lat",
        "school_area_type", "population_decline_type", "actual_enrollment",
    ]
    for column in school_meta_columns[1:]:
        max_nunique = selected_distribution.groupby("school_key")[column].nunique(dropna=True).max()
        if max_nunique > 1:
            raise ValueError(f"동일 school_key에 {column} 값이 여러 개 존재합니다.")
    school_meta = selected_distribution[school_meta_columns].drop_duplicates("school_key")

    # 기사에서 사용할 군집별 인구감소지역/지역유형 비교 변수를 미리 만든다.
    cluster_school_meta = membership.merge(school_meta, on="school_key", how="left", validate="one_to_one")
    cluster_school_meta["is_decline"] = cluster_school_meta["population_decline_type"].eq("인구감소지역")
    cluster_school_meta["is_concern"] = cluster_school_meta["population_decline_type"].eq("인구감소관심지역")
    cluster_profile = (
        cluster_school_meta.groupby("final_cluster_id", as_index=False)
        .agg(
            cluster_decline_school_count=("is_decline", "sum"),
            cluster_concern_school_count=("is_concern", "sum"),
            cluster_urban_school_count=("school_area_type", lambda s: int(s.eq("도시").sum())),
            cluster_eup_school_count=("school_area_type", lambda s: int(s.eq("읍지역").sum())),
            cluster_rural_school_count=("school_area_type", lambda s: int(s.eq("면도서벽지").sum())),
        )
    )
    cluster_profile["cluster_population_decline_group"] = [
        decline_group(int(decline), int(concern))
        for decline, concern in zip(
            cluster_profile["cluster_decline_school_count"],
            cluster_profile["cluster_concern_school_count"],
        )
    ]

    # ------------------------------------------------------------------
    # 3. 통합학교안 정리 및 학생 배정행과 결합
    # ------------------------------------------------------------------
    if not to_bool(options["final_road_eligible"]).all():
        raise ValueError("final_selected_cluster_options에 final_road_eligible=FALSE 행이 포함되어 있습니다.")
    if options.duplicated(["final_cluster_id", "capacity_scenario_id", "retained_school_key"]).any():
        raise ValueError("동일 군집-수용력 시나리오-통합학교 조합이 중복되어 있습니다.")
    if set(options["final_cluster_id"]) - set(selected_clusters["final_cluster_id"]):
        raise ValueError("통합학교안에 비선정 군집이 포함되어 있습니다.")

    option_keep = [
        "final_cluster_id", "candidate_id", "capacity_scenario_id", "scenario_role",
        "cap19_cluster_status", "capacity_option_id", "students_per_class", "total_demand",
        "retained_school_key", "retained_school_name", "closed_school_1_key",
        "closed_school_1_name", "closed_school_2_key", "closed_school_2_name",
        "retained_capacity", "capacity_margin", "capacity_utilization_rate",
        "max_school_road_duration_min", "mean_school_road_duration_min",
        "max_school_road_distance_km", "min_eligible_max_road_duration_min",
    ]
    option_keep = [column for column in option_keep if column in options.columns]
    option_master = options[option_keep].copy().rename(
        columns={
            "capacity_option_id": "integration_option_id",
            "retained_school_key": "integrated_school_key",
            "retained_school_name": "integrated_school_name",
            "closed_school_1_key": "merged_school_1_key",
            "closed_school_1_name": "merged_school_1_name",
            "closed_school_2_key": "merged_school_2_key",
            "closed_school_2_name": "merged_school_2_name",
        }
    )
    valid_cluster_school_pairs = set(zip(membership["final_cluster_id"], membership["school_key"]))
    invalid_integrated = [
        pair for pair in zip(option_master["final_cluster_id"], option_master["integrated_school_key"])
        if pair not in valid_cluster_school_pairs
    ]
    if invalid_integrated:
        raise ValueError(f"통합학교가 해당 3개교 군집에 속하지 않는 사례: {invalid_integrated[:10]}")

    integrated_meta = school_meta.rename(
        columns={
            "school_key": "integrated_school_key",
            "school_name": "integrated_school_name_master",
            "school_edu_office": "integrated_school_edu_office",
            "school_lon": "integrated_school_lon",
            "school_lat": "integrated_school_lat",
            "school_area_type": "integrated_school_area_type",
            "population_decline_type": "integrated_school_population_decline_type",
            "actual_enrollment": "integrated_school_current_enrollment",
        }
    )
    option_master = option_master.merge(
        integrated_meta,
        on="integrated_school_key",
        how="left",
        validate="many_to_one",
    ).merge(cluster_profile, on="final_cluster_id", how="left", validate="many_to_one")
    if option_master[["integrated_school_lon", "integrated_school_lat"]].isna().any().any():
        raise ValueError("통합학교 좌표 연결에 실패했습니다.")

    demand_base = demand.rename(
        columns={
            "longitude": "origin_lon",
            "latitude": "origin_lat",
            "school_key": "current_school_key",
            "school_name": "current_school_name",
            "school_lon": "current_school_lon",
            "school_lat": "current_school_lat",
            "school_edu_office": "current_school_edu_office",
            "school_area_type": "current_school_area_type",
            "population_decline_type": "current_school_population_decline_type",
            "actual_enrollment": "current_school_actual_enrollment",
        }
    )
    demand_keep = [
        "demand_assignment_id", "census_id", "origin_lon", "origin_lat",
        "zone_key", "zone_id", "zone_name", "final_cluster_id", "candidate_id",
        "current_school_key", "current_school_name", "current_school_lon", "current_school_lat",
        "current_school_edu_office", "current_school_area_type",
        "current_school_population_decline_type", "current_school_actual_enrollment",
        "student_weight",
    ]
    demand_keep = [column for column in demand_keep if column in demand_base.columns]
    demand_base = demand_base[demand_keep]

    matrix = demand_base.merge(
        option_master,
        on=["final_cluster_id", "candidate_id"],
        how="inner",
        validate="many_to_many",
    )
    if matrix.empty:
        raise ValueError("학생 배정행과 통합학교안 결합 결과가 0행입니다.")
    matrix["moved_student_flag"] = matrix["current_school_key"].ne(matrix["integrated_school_key"])
    matrix["current_route_request_id"] = [
        stable_id("RT", census_id, school_key)
        for census_id, school_key in zip(matrix["census_id"], matrix["current_school_key"])
    ]
    matrix["integrated_route_request_id"] = [
        stable_id("RT", census_id, school_key)
        for census_id, school_key in zip(matrix["census_id"], matrix["integrated_school_key"])
    ]
    matrix["same_route_reuse_flag"] = matrix["current_route_request_id"].eq(
        matrix["integrated_route_request_id"]
    )
    matrix["commute_result_status"] = "ROUTE_PENDING"

    # ------------------------------------------------------------------
    # 4. 동일 집계구 대표점-목적학교 경로를 전 시나리오에서 중복 제거
    # ------------------------------------------------------------------
    current_usage = demand_base[
        [
            "demand_assignment_id", "census_id", "origin_lon", "origin_lat",
            "final_cluster_id", "current_school_key", "student_weight",
        ]
    ].rename(columns={"current_school_key": "destination_school_key"})
    current_usage["route_role"] = "CURRENT"
    current_usage["capacity_scenario_id"] = ""
    current_usage["integration_option_id"] = ""
    current_usage["source_current_school_key"] = current_usage["destination_school_key"]

    integrated_usage = matrix[
        [
            "demand_assignment_id", "census_id", "origin_lon", "origin_lat",
            "final_cluster_id", "current_school_key", "student_weight",
            "integrated_school_key", "capacity_scenario_id", "integration_option_id",
        ]
    ].rename(
        columns={
            "current_school_key": "source_current_school_key",
            "integrated_school_key": "destination_school_key",
        }
    )
    integrated_usage["route_role"] = "INTEGRATED"

    usage = pd.concat([current_usage, integrated_usage], ignore_index=True)
    usage["current_role_flag"] = usage["route_role"].eq("CURRENT")
    usage["integrated_role_flag"] = usage["route_role"].eq("INTEGRATED")
    usage["cap28_role_flag"] = usage["capacity_scenario_id"].eq("CAP28")
    usage["cap19_role_flag"] = usage["capacity_scenario_id"].eq("CAP19")

    request_agg = (
        usage.groupby(["census_id", "destination_school_key"], as_index=False)
        .agg(
            route_roles=("route_role", unique_join),
            required_for_current_flag=("current_role_flag", "max"),
            required_for_integrated_flag=("integrated_role_flag", "max"),
            required_for_cap28_flag=("cap28_role_flag", "max"),
            required_for_cap19_flag=("cap19_role_flag", "max"),
            final_cluster_count=("final_cluster_id", "nunique"),
            final_cluster_ids=("final_cluster_id", unique_join),
            integration_option_count=("integration_option_id", lambda s: s[s.ne("")].nunique()),
        )
    )

    # 노출 학생수는 CAP28/CAP19 중복을 제거한 현재학교-목적학교 단위로 합산한다.
    current_exposure = (
        current_usage.groupby(["census_id", "destination_school_key"], as_index=False)["student_weight"]
        .sum()
        .rename(columns={"student_weight": "current_exposed_student_weight"})
    )
    integrated_exposure = (
        integrated_usage.drop_duplicates(
            ["census_id", "source_current_school_key", "destination_school_key"]
        )
        .groupby(["census_id", "destination_school_key"], as_index=False)["student_weight"]
        .sum()
        .rename(columns={"student_weight": "integrated_exposed_student_weight"})
    )
    request_agg = request_agg.merge(
        current_exposure,
        on=["census_id", "destination_school_key"],
        how="left",
        validate="one_to_one",
    ).merge(
        integrated_exposure,
        on=["census_id", "destination_school_key"],
        how="left",
        validate="one_to_one",
    )
    request_agg[["current_exposed_student_weight", "integrated_exposed_student_weight"]] = (
        request_agg[["current_exposed_student_weight", "integrated_exposed_student_weight"]].fillna(0.0)
    )

    origin_meta = demand_base[["census_id", "origin_lon", "origin_lat"]].drop_duplicates()
    if origin_meta["census_id"].duplicated().any():
        raise ValueError("동일 census_id에 서로 다른 대표점 좌표가 있습니다.")
    destination_meta = school_meta.rename(
        columns={
            "school_key": "destination_school_key",
            "school_name": "destination_school_name",
            "school_edu_office": "destination_school_edu_office",
            "school_lon": "destination_school_lon",
            "school_lat": "destination_school_lat",
            "school_area_type": "destination_school_area_type",
            "population_decline_type": "destination_school_population_decline_type",
            "actual_enrollment": "destination_school_current_enrollment",
        }
    )
    requests = request_agg.merge(
        origin_meta,
        on="census_id",
        how="left",
        validate="many_to_one",
    ).merge(
        destination_meta,
        on="destination_school_key",
        how="left",
        validate="many_to_one",
    )
    check_coordinate_range(requests, "origin_lon", "origin_lat", "경로 출발점")
    check_coordinate_range(requests, "destination_school_lon", "destination_school_lat", "경로 도착학교")

    requests["route_request_id"] = [
        stable_id("RT", census_id, school_key)
        for census_id, school_key in zip(requests["census_id"], requests["destination_school_key"])
    ]
    if requests["route_request_id"].duplicated().any():
        raise ValueError("route_request_id 해시 충돌이 발생했습니다.")
    # 실시간 교통 차이가 추가 통학시간에 섞이지 않도록 동일 census_id의
    # 현재학교·통합학교 경로를 반드시 같은 작업 배치에 넣는다.
    requests["request_batch_number"] = [
        int(hashlib.sha256(str(census_id).encode("utf-8")).hexdigest()[:12], 16)
        % args.batch_count
        + 1
        for census_id in requests["census_id"]
    ]
    batch_width = max(2, len(str(args.batch_count)))
    requests["request_batch_id"] = requests["request_batch_number"].map(
        lambda number: f"B{number:0{batch_width}d}_of_{args.batch_count:0{batch_width}d}"
    )
    requests["straight_distance_km"] = haversine_km(
        requests["origin_lon"],
        requests["origin_lat"],
        requests["destination_school_lon"],
        requests["destination_school_lat"],
    )
    requests["routing_status"] = "PENDING"
    requests["routing_provider"] = "UNDECIDED"
    requests["road_distance_m"] = np.nan
    requests["road_duration_min"] = np.nan
    requests["routing_error"] = ""

    request_order = [
        "route_request_id", "census_id", "origin_lon", "origin_lat",
        "destination_school_key", "destination_school_name", "destination_school_lon",
        "destination_school_lat", "destination_school_edu_office",
        "destination_school_area_type", "destination_school_population_decline_type",
        "destination_school_current_enrollment", "straight_distance_km", "route_roles",
        "required_for_current_flag", "required_for_integrated_flag", "required_for_cap28_flag",
        "required_for_cap19_flag", "final_cluster_count", "final_cluster_ids",
        "integration_option_count", "current_exposed_student_weight",
        "integrated_exposed_student_weight", "request_batch_number", "request_batch_id",
        "routing_provider", "routing_status",
        "road_distance_m", "road_duration_min", "routing_error",
    ]
    requests = requests[request_order].sort_values(
        ["destination_school_edu_office", "destination_school_key", "census_id"],
        kind="stable",
    ).reset_index(drop=True)

    # 행렬의 두 요청키가 실제 요청표에 모두 존재하는지 검증한다.
    request_ids = set(requests["route_request_id"])
    missing_current = set(matrix["current_route_request_id"]) - request_ids
    missing_integrated = set(matrix["integrated_route_request_id"]) - request_ids
    if missing_current or missing_integrated:
        raise ValueError(
            f"분석 행렬의 요청키가 요청표에 없음: current={len(missing_current)}, "
            f"integrated={len(missing_integrated)}"
        )

    matrix_order = [
        "demand_assignment_id", "census_id", "origin_lon", "origin_lat",
        "student_weight", "final_cluster_id", "candidate_id", "capacity_scenario_id",
        "scenario_role", "integration_option_id", "current_school_key", "current_school_name",
        "current_school_lon", "current_school_lat", "current_school_edu_office",
        "current_school_area_type", "current_school_population_decline_type",
        "current_school_actual_enrollment", "integrated_school_key", "integrated_school_name",
        "integrated_school_lon", "integrated_school_lat", "integrated_school_edu_office",
        "integrated_school_area_type", "integrated_school_population_decline_type",
        "integrated_school_current_enrollment", "merged_school_1_key", "merged_school_1_name",
        "merged_school_2_key", "merged_school_2_name", "students_per_class", "total_demand",
        "retained_capacity", "capacity_margin", "capacity_utilization_rate",
        "cluster_decline_school_count", "cluster_concern_school_count",
        "cluster_population_decline_group", "cluster_urban_school_count",
        "cluster_eup_school_count", "cluster_rural_school_count", "moved_student_flag",
        "current_route_request_id", "integrated_route_request_id", "same_route_reuse_flag",
        "commute_result_status",
    ]
    matrix_order = [column for column in matrix_order if column in matrix.columns]
    matrix = matrix[matrix_order].sort_values(
        ["final_cluster_id", "capacity_scenario_id", "integration_option_id", "census_id", "current_school_key"],
        kind="stable",
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. 검증·호출량 요약
    # ------------------------------------------------------------------
    summary_rows: list[dict[str, object]] = []

    def add_summary(section: str, metric: str, value: object, unit: str, status: str, note: str) -> None:
        summary_rows.append(
            {
                "section": section,
                "metric": metric,
                "value": value,
                "unit": unit,
                "check_status": status,
                "note": note,
            }
        )

    add_summary("SCOPE", "selected_cluster_count", len(selected_clusters), "clusters", "PASS", "비중복 최종 군집")
    add_summary("SCOPE", "requested_capacity_scenarios", "|".join(requested_scenarios), "scenarios", "PASS", "기본값 CAP28")
    add_summary("SCOPE", "selected_school_count", expected_school_count, "schools", "PASS", "군집당 3개교")
    add_summary("SCOPE", "positive_demand_assignment_rows", len(demand_base), "rows", "PASS", "student_weight>0")
    add_summary("SCOPE", "positive_unique_census_points", demand_base["census_id"].nunique(), "points", "PASS", "선정학교 학생이 배정된 대표점")
    add_summary("SCOPE", "selected_student_weight_sum", positive_weight_sum, "students", "PASS", "보정 학생 가중치 합계")
    add_summary("EXCLUSION", "zero_weight_rows_excluded", zero_weight_rows, "rows", "PASS", "학생 합계에 영향을 주지 않는 0명 행")
    add_summary("QA", "max_school_calibration_abs_error", max_calibration_error, "students", "PASS", f"허용오차 {CALIBRATION_TOLERANCE}")
    add_summary("QA", "duplicated_school_across_selected_clusters", 0, "schools", "PASS", "비중복 군집 조건")
    add_summary("QA", "missing_route_request_keys", 0, "keys", "PASS", "현재/통합 후 요청키 모두 존재")

    for scenario_id, group in option_master.groupby("capacity_scenario_id", dropna=False):
        add_summary(
            "OPTIONS", f"{scenario_id}_cluster_count", group["final_cluster_id"].nunique(),
            "clusters", "PASS", "해당 수용력 시나리오에서 유지되는 최종 군집",
        )
        add_summary(
            "OPTIONS", f"{scenario_id}_integration_option_count", len(group),
            "options", "PASS", "가능한 통합학교안 수",
        )
        add_summary(
            "MATRIX", f"{scenario_id}_analysis_rows",
            int(matrix["capacity_scenario_id"].eq(scenario_id).sum()), "rows", "PASS",
            "보정 학생 배정행 x 통합학교안",
        )

    current_request_count = int(requests["required_for_current_flag"].sum())
    integrated_request_count = int(requests["required_for_integrated_flag"].sum())
    both_role_count = int(
        (requests["required_for_current_flag"] & requests["required_for_integrated_flag"]).sum()
    )
    unique_request_count = len(requests)
    add_summary("ROUTES", "current_unique_route_count", current_request_count, "routes", "PASS", "통합 전 대표점->현재학교")
    add_summary("ROUTES", "integrated_unique_route_count", integrated_request_count, "routes", "PASS", "통합 후 대표점->통합학교")
    add_summary("ROUTES", "current_and_integrated_reused_count", both_role_count, "routes", "PASS", "두 역할에서 공통으로 재사용")
    add_summary("ROUTES", "all_unique_route_count", unique_request_count, "routes", "PASS", "전 시나리오 중복 제거 후")
    add_summary("ROUTES", "request_batch_count", args.batch_count, "batches", "PASS", "동일 census_id의 모든 목적학교 경로는 같은 배치")
    for batch_id, batch_group in requests.groupby("request_batch_id", sort=True):
        add_summary(
            "ROUTE_BATCH", f"{batch_id}_route_count", len(batch_group), "routes", "PASS",
            "팀 분담 시 해당 request_batch_id만 필터링",
        )
    add_summary("OUTPUT", "commute_demand_option_matrix_rows", len(matrix), "rows", "PASS", OUTPUT_MATRIX_FILE)
    add_summary("OUTPUT", "commute_demand_option_matrix_columns", len(matrix.columns), "columns", "PASS", OUTPUT_MATRIX_FILE)
    add_summary("OUTPUT", "commute_route_requests_rows", len(requests), "rows", "PASS", OUTPUT_REQUEST_FILE)
    add_summary("OUTPUT", "commute_route_requests_columns", len(requests.columns), "columns", "PASS", OUTPUT_REQUEST_FILE)

    previous_success_calls = 0
    if previous_route_path is not None:
        previous_routes = read_csv_as_text(previous_route_path)
        if "route_status" in previous_routes.columns:
            previous_success_calls = int(previous_routes["route_status"].eq("SUCCESS").sum())
        else:
            previous_success_calls = len(previous_routes)
    remaining_free_upper_bound = max(
        0,
        NAVER_DIRECTIONS5_MONTHLY_FREE_QUOTA - previous_success_calls,
    )
    calls_over_upper_bound = max(0, unique_request_count - remaining_free_upper_bound)
    add_summary(
        "NAVER_QUOTA", "monthly_free_quota_reference",
        NAVER_DIRECTIONS5_MONTHLY_FREE_QUOTA, "calls", "REFERENCE",
        "대표계정 Directions 5 공지 기준; 콘솔에서 최신 정책 재확인",
    )
    add_summary(
        "NAVER_QUOTA", "previous_project_success_calls_observed",
        previous_success_calls, "calls", "INFO",
        "같은 달·같은 대표계정이라는 가정의 최소 사용량이며 다른 호출은 알 수 없음",
    )
    add_summary(
        "NAVER_QUOTA", "remaining_free_calls_upper_bound",
        remaining_free_upper_bound, "calls", "INFO",
        "실제 남은 무료량은 이 값보다 작을 수 있음",
    )
    add_summary(
        "NAVER_QUOTA", "new_calls_over_remaining_upper_bound",
        calls_over_upper_bound, "calls", "WARN" if calls_over_upper_bound > 0 else "PASS",
        "0보다 크면 NAVER 전체 호출 금지; 로컬 라우팅 권장",
    )
    add_summary(
        "DECISION", "recommended_full_routing_method",
        "LOCAL_OSRM_WITH_OSM", "method", "RECOMMENDED",
        "전국 OSM PBF를 이용해 비용 없이 정적 자동차 도로시간·거리 행렬 계산",
    )

    summary = pd.DataFrame(summary_rows)

    matrix_path = output_dir / OUTPUT_MATRIX_FILE
    request_path = output_dir / OUTPUT_REQUEST_FILE
    summary_path = output_dir / OUTPUT_SUMMARY_FILE
    matrix.to_csv(matrix_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    requests.to_csv(request_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n[완료]")
    print(f"- {matrix_path} : {matrix.shape[0]:,}행 x {matrix.shape[1]:,}열")
    print(f"- {request_path} : {requests.shape[0]:,}행 x {requests.shape[1]:,}열")
    print(f"- {summary_path} : {summary.shape[0]:,}행 x {summary.shape[1]:,}열")
    print(f"- 선택 군집/학교: {len(selected_clusters):,}개 / {expected_school_count:,}개교")
    print(f"- 분석 학생 가중치: {positive_weight_sum:,.3f}명")
    print(f"- 중복 제거 경로: {unique_request_count:,}건")
    if calls_over_upper_bound > 0:
        print(
            "[중요] 관찰된 기존 사용량과 공식 무료량을 고려해도 전체 NAVER 호출은 무료 범위를 "
            f"최소 {calls_over_upper_bound:,}건 초과합니다. 이 스크립트는 API를 호출하지 않았습니다."
        )


if __name__ == "__main__":
    main()

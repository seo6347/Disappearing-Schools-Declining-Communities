"""
현재 학생분포 추정 단계
======================

목적
----
집계구 대표점을 초등학교 통학구역과 공간 결합하고, 통학구역에서 갈 수 있는
공립초등학교에 집계구별 초등학령인구를 균등 배분한 뒤 학교별 실제 재학생 수에
맞게 학생가중치를 보정한다.

입력 파일 (data/01_raw)
-----------------------
1. elementary_master.csv
2. elementary_polygon.gpkg
3. school_area_master.csv
4. census_demand_points.gpkg
5. census_demand_points.csv

출력 파일 (data/02_processed)
-----------------------------
1. current_student_distribution.csv
   - 배정 성공: 집계구-학교 관계 1개당 1행
   - 배정 실패/제외: 집계구당 1행, assignment_status에 사유 기록
2. school_calibration_summary.csv
   - 공립초등학교별 초기 추정치, 실제 재학생 수, 보정계수, 검증 결과

핵심 계산
---------
P_i = 0.6 * pop_5_9_i + 0.6 * pop_10_14_i
R_is = P_i / k_z
c_s = N_s / sum_i(R_is)
F_is = R_is * c_s

소수점 처리 원칙
----------------
- 중간 계산에서는 절대 올림·내림·반올림하지 않는다.
- 학생 수는 실제 인원 정수가 아니라 분석용 가중치이므로 float64로 유지한다.
- CSV에는 계산 정밀도를 유지해 저장하고, 기사·표의 최종 집계값에서만 반올림한다.

실행 환경
---------
Windows 터미널에서 최초 1회:
    py -m pip install pandas geopandas pyogrio

프로젝트 루트에서 실행:
    py scripts/01_estimate_current_student_distribution.py

스크립트는 자신의 위치를 기준으로 프로젝트 루트를 자동 인식한다.
테스트 등 다른 위치를 사용할 때는 --project-root 옵션을 지정할 수 있다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover - 사용자 환경 안내용
    raise SystemExit(
        "geopandas가 설치되어 있지 않습니다. "
        "터미널에서 'py -m pip install pandas geopandas pyogrio'를 실행하세요."
    ) from exc


# -----------------------------------------------------------------------------
# 분석 설정
# -----------------------------------------------------------------------------

MASTER_FILE = "elementary_master.csv"
ZONE_FILE = "elementary_polygon.gpkg"
BRIDGE_FILE = "school_area_master.csv"
POINT_GPKG_FILE = "census_demand_points.gpkg"
POINT_CSV_FILE = "census_demand_points.csv"

# 제외 통학구역까지 구분하기 위해 '전체' 레이어를 읽고,
# 실제 배정은 in_master=True, match_status='매칭'인 통학구역만 사용한다.
ZONE_LAYER = "통학구역_전체"
POINT_LAYER = "census_demand_points"

OUTPUT_DISTRIBUTION = "current_student_distribution.csv"
OUTPUT_SCHOOL_QC = "school_calibration_summary.csv"

FORMULA_TOLERANCE = 1e-9
CALIBRATION_TOLERANCE = 1e-6

# 계산 오류가 아니라 수동 검토를 위한 진단 기준이다.
EXTREME_FACTOR_LOW = 0.2
EXTREME_FACTOR_HIGH = 5.0


MASTER_REQUIRED = {
    "학교고유ID",
    "학교명",
    "교육지원청",
    "설립형태",
    "총계",
    "위도",
    "경도",
    "본교분교구분",
    "작은학교",
}

BRIDGE_REQUIRED = {
    "학구ID",
    "학교ID",
    "학교고유ID",
    "학교명_학구도",
    "학교급구분",
    "in_zone_shp",
    "match_status",
}

DEMAND_REQUIRED = {
    "census_id",
    "pop_5_9",
    "pop_10_14",
    "elem_pop_est",
    "longitude",
    "latitude",
}

ZONE_REQUIRED = {
    "zone_key",
    "HAKGUDO_ID",
    "HAKGUDO_NM",
    "EDU_NM",
    "in_master",
    "match_status",
    "exclude_reason",
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="집계구별 현재 학생분포 추정")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="final_project 폴더 경로. 기본값은 scripts 폴더의 상위 폴더입니다.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"[{label}] 필수 컬럼이 없습니다: {missing}")


def require_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    null_count = int(frame[column].isna().sum())
    duplicate_count = int(frame.duplicated([column]).sum())
    if null_count or duplicate_count:
        raise ValueError(
            f"[{label}] {column} 키 검증 실패: "
            f"결측 {null_count:,}개, 중복 {duplicate_count:,}개"
        )


def normalize_id(series: pd.Series) -> pd.Series:
    """식별자를 문자열로 통일하고 양쪽 공백을 제거한다."""
    return series.astype("string").str.strip()


def to_boolean(series: pd.Series) -> pd.Series:
    """CSV/GPKG에서 읽은 다양한 불리언 표현을 True/False로 통일한다."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "y", "yes", "t"})


def join_unique(values: Iterable[object]) -> str:
    """여러 공식 학교ID를 중복 제거 후 하나의 문자열로 보존한다."""
    unique_values: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        unique_values.update(
            item.strip() for item in str(value).split("|") if item.strip()
        )
    return "|".join(sorted(unique_values))


def log_count(label: str, value: int | float) -> None:
    if isinstance(value, int):
        logging.info("%-42s %12s", label, f"{value:,}")
    else:
        logging.info("%-42s %12.6f", label, value)


def load_inputs(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """세 CSV를 원본 식별자가 손상되지 않도록 명시적 dtype으로 읽는다."""
    paths = {
        "학교 마스터": raw_dir / MASTER_FILE,
        "학교-학구도 연결": raw_dir / BRIDGE_FILE,
        "집계구 수요점 CSV": raw_dir / POINT_CSV_FILE,
        "통학구역 GPKG": raw_dir / ZONE_FILE,
        "집계구 수요점 GPKG": raw_dir / POINT_GPKG_FILE,
    }
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"입력 파일이 없습니다: {missing_files}")

    master = pd.read_csv(
        paths["학교 마스터"],
        encoding="utf-8-sig",
        dtype={"학교고유ID": "string"},
        low_memory=False,
    )
    bridge = pd.read_csv(
        paths["학교-학구도 연결"],
        encoding="utf-8-sig",
        dtype={"학구ID": "string", "학교ID": "string", "학교고유ID": "string"},
        low_memory=False,
    )
    demand = pd.read_csv(
        paths["집계구 수요점 CSV"],
        encoding="utf-8-sig",
        dtype={"census_id": "string"},
        low_memory=False,
    )

    require_columns(master, MASTER_REQUIRED, "elementary_master.csv")
    require_columns(bridge, BRIDGE_REQUIRED, "school_area_master.csv")
    require_columns(demand, DEMAND_REQUIRED, "census_demand_points.csv")

    master["학교고유ID"] = normalize_id(master["학교고유ID"])
    bridge["학구ID"] = normalize_id(bridge["학구ID"])
    bridge["학교ID"] = normalize_id(bridge["학교ID"])
    bridge["학교고유ID"] = normalize_id(bridge["학교고유ID"])
    demand["census_id"] = normalize_id(demand["census_id"])

    require_unique(master, "학교고유ID", "학교 마스터")
    require_unique(demand, "census_id", "집계구 수요점 CSV")

    return master, bridge, demand


def validate_demand_formula(demand: pd.DataFrame) -> float:
    """elem_pop_est가 7~12세 추정식(각 연령구간의 3/5)과 일치하는지 확인한다."""
    numeric_columns = ["pop_5_9", "pop_10_14", "elem_pop_est", "longitude", "latitude"]
    for column in numeric_columns:
        demand[column] = pd.to_numeric(demand[column], errors="raise")

    if (demand[["pop_5_9", "pop_10_14", "elem_pop_est"]].lt(0)).any().any():
        raise ValueError("집계구 인구 컬럼에 음수가 있습니다.")

    calculated = 0.6 * demand["pop_5_9"] + 0.6 * demand["pop_10_14"]
    max_error = float((demand["elem_pop_est"] - calculated).abs().max())
    if max_error > FORMULA_TOLERANCE:
        raise ValueError(
            "elem_pop_est 검증 실패: "
            f"3/5 추정식과의 최대 차이={max_error:.12g}"
        )
    return max_error


def build_public_zone_school_bridge(
    master: pd.DataFrame,
    bridge_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    공립초등학교만 남기고 통학구역-현재 학교 관계를 유일하게 만든다.

    분교 흡수 등으로 서로 다른 공식 학교ID가 하나의 학교고유ID에 연결된 경우,
    통학 선택지는 학교고유ID 기준으로 한 번만 센다. 공식 학교ID들은 추적을 위해
    official_school_ids 컬럼에 모두 보존한다.
    """
    public_master = master.loc[master["설립형태"].eq("공립")].copy()
    public_master["총계"] = pd.to_numeric(public_master["총계"], errors="raise")
    if public_master["총계"].isna().any() or (public_master["총계"] < 0).any():
        raise ValueError("공립초등학교 실제 재학생 수에 결측 또는 음수가 있습니다.")

    bridge_raw = bridge_raw.copy()
    bridge_raw["in_zone_shp"] = to_boolean(bridge_raw["in_zone_shp"])

    valid = bridge_raw.loc[
        bridge_raw["학교급구분"].eq("초등학교")
        & bridge_raw["in_zone_shp"]
        & bridge_raw["match_status"].eq("매칭")
        & bridge_raw["학교고유ID"].notna()
    ].copy()

    missing_master_key_count = int(
        bridge_raw.loc[
            bridge_raw["학교급구분"].eq("초등학교")
            & bridge_raw["in_zone_shp"]
            & bridge_raw["match_status"].eq("매칭"),
            "학교고유ID",
        ].isna().sum()
    )

    valid = valid.loc[valid["학교고유ID"].isin(public_master["학교고유ID"])].copy()
    before_dedup = len(valid)

    relation = (
        valid.groupby(["학구ID", "학교고유ID"], as_index=False, dropna=False)
        .agg(
            official_school_ids=("학교ID", join_unique),
            relation_source_rows=("학교ID", "size"),
        )
    )
    collapsed_duplicate_count = before_dedup - len(relation)

    school_columns = [
        "학교고유ID",
        "학교명",
        "교육지원청",
        "총계",
        "위도",
        "경도",
        "본교분교구분",
        "작은학교",
        "지역구분",
        "인구감소지역",
    ]
    relation = relation.merge(
        public_master[school_columns],
        on="학교고유ID",
        how="left",
        validate="many_to_one",
    )
    relation["eligible_school_count"] = relation.groupby("학구ID")[
        "학교고유ID"
    ].transform("nunique")

    if relation.duplicated(["학구ID", "학교고유ID"]).any():
        raise ValueError("통학구역-학교고유ID 관계가 유일하지 않습니다.")
    if (relation["eligible_school_count"] <= 0).any():
        raise ValueError("연결 학교 수가 0 이하인 통학구역이 있습니다.")

    relation = relation.rename(
        columns={
            "학구ID": "zone_id",
            "학교고유ID": "school_key",
            "학교명": "school_name",
            "교육지원청": "school_edu_office",
            "총계": "actual_enrollment",
            "위도": "school_lat",
            "경도": "school_lon",
            "본교분교구분": "main_branch_type",
            "작은학교": "small_school_flag",
            "지역구분": "school_area_type",
            "인구감소지역": "population_decline_type",
        }
    )
    public_master = public_master.rename(columns={"학교고유ID": "school_key"})

    return (
        public_master,
        relation,
        missing_master_key_count,
        collapsed_duplicate_count,
    )


def spatial_join_points_to_zones(
    raw_dir: Path,
    demand: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """대표점을 통학구역과 공간 결합하고 공간 데이터 품질을 검증한다."""
    point_path = raw_dir / POINT_GPKG_FILE
    zone_path = raw_dir / ZONE_FILE

    points = gpd.read_file(point_path, layer=POINT_LAYER)
    zones = gpd.read_file(zone_path, layer=ZONE_LAYER)

    require_columns(points, {"census_id"}, f"{POINT_GPKG_FILE}:{POINT_LAYER}")
    require_columns(zones, ZONE_REQUIRED, f"{ZONE_FILE}:{ZONE_LAYER}")

    points["census_id"] = normalize_id(points["census_id"])
    zones["zone_key"] = normalize_id(zones["zone_key"])
    zones["HAKGUDO_ID"] = normalize_id(zones["HAKGUDO_ID"])
    zones["in_master"] = to_boolean(zones["in_master"])

    require_unique(points, "census_id", "집계구 수요점 GPKG")
    require_unique(zones, "zone_key", "통학구역 GPKG")

    csv_ids = set(demand["census_id"])
    gpkg_ids = set(points["census_id"])
    if csv_ids != gpkg_ids:
        raise ValueError(
            "집계구 CSV와 GPKG의 census_id 집합이 다릅니다: "
            f"CSV에만 {len(csv_ids - gpkg_ids):,}개, "
            f"GPKG에만 {len(gpkg_ids - csv_ids):,}개"
        )

    if points.crs is None or zones.crs is None:
        raise ValueError("집계구 또는 통학구역 공간 파일에 좌표계가 없습니다.")

    invalid_zone_count = int((~zones.geometry.is_valid).sum())
    if invalid_zone_count:
        zones.loc[~zones.geometry.is_valid, "geometry"] = zones.loc[
            ~zones.geometry.is_valid, "geometry"
        ].make_valid()
    invalid_after_repair = int((~zones.geometry.is_valid).sum())
    if invalid_after_repair:
        raise ValueError(f"유효성 복구에 실패한 통학구역이 {invalid_after_repair:,}개 있습니다.")

    matched_zone_subset = zones.loc[
        zones["in_master"] & zones["match_status"].eq("매칭")
    ]
    if matched_zone_subset["HAKGUDO_ID"].duplicated().any():
        duplicated = matched_zone_subset.loc[
            matched_zone_subset["HAKGUDO_ID"].duplicated(False), "HAKGUDO_ID"
        ].unique()[:10]
        raise ValueError(
            "분석 대상 통학구역의 HAKGUDO_ID가 중복됩니다: "
            f"{duplicated.tolist()}"
        )

    source_crs = str(points.crs)
    target_crs = str(zones.crs)
    points_projected = points[["census_id", "geometry"]].to_crs(zones.crs)

    zone_columns = [
        "zone_key",
        "HAKGUDO_ID",
        "HAKGUDO_NM",
        "EDU_NM",
        "in_master",
        "match_status",
        "exclude_reason",
        "geometry",
    ]
    point_zone = gpd.sjoin(
        points_projected,
        zones[zone_columns],
        how="left",
        predicate="within",
    )

    duplicated_points = point_zone.loc[
        point_zone.duplicated(["census_id"], keep=False), "census_id"
    ].unique()
    if len(duplicated_points):
        raise ValueError(
            "한 집계구 대표점이 여러 통학구역에 포함되었습니다. "
            f"중복 집계구 {len(duplicated_points):,}개, "
            f"예시 {duplicated_points[:10].tolist()}"
        )

    point_zone = pd.DataFrame(
        point_zone.drop(columns=["geometry", "index_right"], errors="ignore")
    ).rename(
        columns={
            "HAKGUDO_ID": "zone_id",
            "HAKGUDO_NM": "zone_name",
            "EDU_NM": "zone_edu_office",
            "match_status": "zone_match_status",
            "exclude_reason": "zone_exclude_reason",
        }
    )

    base = demand.merge(point_zone, on="census_id", how="left", validate="one_to_one")

    diagnostics: dict[str, int | str] = {
        "point_crs_before": source_crs,
        "zone_crs": target_crs,
        "invalid_zones_repaired": invalid_zone_count,
        "points_in_any_zone": int(base["zone_key"].notna().sum()),
        "points_without_zone": int(base["zone_key"].isna().sum()),
        "points_in_matched_zone": int(base["zone_match_status"].eq("매칭").sum()),
        "points_in_unmatched_zone": int(base["zone_match_status"].eq("통학구역만").sum()),
        "points_in_excluded_zone": int(base["zone_match_status"].eq("분석제외").sum()),
    }
    return base, diagnostics


def assign_and_calibrate(
    point_zone: pd.DataFrame,
    relation: pd.DataFrame,
    public_master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    """균등배정, 학교별 보정계수 계산, 최종 학생가중치 검증을 수행한다."""
    relation_zone_ids = set(relation["zone_id"])
    point_zone_ids = set(point_zone["zone_id"].dropna())
    missing_zone_ids = relation_zone_ids - point_zone_ids

    expanded = point_zone.merge(relation, on="zone_id", how="left")
    in_master_flag = expanded["in_master"].fillna(False).astype(bool)

    expanded["assignment_status"] = np.select(
        [
            expanded["zone_key"].isna(),
            expanded["zone_match_status"].eq("분석제외"),
            expanded["zone_match_status"].eq("통학구역만")
            | ~in_master_flag,
            expanded["school_key"].isna(),
        ],
        [
            "NO_ZONE",
            "EXCLUDED_ZONE",
            "ZONE_NOT_MATCHED",
            "NO_PUBLIC_SCHOOL_LINK",
        ],
        default="ASSIGNED",
    )

    expanded["initial_share"] = 0.0
    expanded["initial_student_est"] = 0.0
    assigned_mask = expanded["assignment_status"].eq("ASSIGNED")
    if not assigned_mask.any():
        raise ValueError(
            "정상 배정된 집계구-학교 관계가 0개입니다. "
            "통학구역 상태값과 학교 연결 키를 확인하세요."
        )

    expanded.loc[assigned_mask, "initial_share"] = 1.0 / expanded.loc[
        assigned_mask, "eligible_school_count"
    ].astype(float)
    expanded.loc[assigned_mask, "initial_student_est"] = (
        expanded.loc[assigned_mask, "elem_pop_est"].astype(float)
        * expanded.loc[assigned_mask, "initial_share"]
    )

    # 검증 1: 정상 배정된 집계구는 학교별 초기 추정값의 합이 elem_pop_est와 같아야 한다.
    assigned_point_totals = (
        expanded.loc[assigned_mask]
        .groupby("census_id", as_index=False)
        .agg(
            allocated=("initial_student_est", "sum"),
            expected=("elem_pop_est", "first"),
        )
    )
    point_allocation_max_error = float(
        (assigned_point_totals["allocated"] - assigned_point_totals["expected"])
        .abs()
        .max()
    )
    if point_allocation_max_error > FORMULA_TOLERANCE:
        raise ValueError(
            "집계구 초기 균등배정 검증 실패: "
            f"최대 오차={point_allocation_max_error:.12g}"
        )

    initial_by_school = (
        expanded.loc[assigned_mask]
        .groupby("school_key", as_index=False)
        .agg(
            school_initial_est_total=("initial_student_est", "sum"),
            assigned_point_count=("census_id", "nunique"),
        )
    )

    relation_stats = (
        relation.groupby("school_key", as_index=False)
        .agg(
            linked_zone_count=("zone_id", "nunique"),
            official_school_ids=("official_school_ids", join_unique),
        )
    )

    school_qc = public_master[
        [
            "school_key",
            "학교명",
            "교육지원청",
            "총계",
            "본교분교구분",
            "작은학교",
            "지역구분",
            "인구감소지역",
        ]
    ].rename(
        columns={
            "학교명": "school_name",
            "교육지원청": "school_edu_office",
            "총계": "actual_enrollment",
            "본교분교구분": "main_branch_type",
            "작은학교": "small_school_flag",
            "지역구분": "school_area_type",
            "인구감소지역": "population_decline_type",
        }
    )
    school_qc = school_qc.merge(relation_stats, on="school_key", how="left")
    school_qc = school_qc.merge(initial_by_school, on="school_key", how="left")
    school_qc["linked_zone_count"] = school_qc["linked_zone_count"].fillna(0).astype(int)
    school_qc["assigned_point_count"] = school_qc["assigned_point_count"].fillna(0).astype(int)

    has_no_point = school_qc["school_initial_est_total"].isna()
    has_zero_initial = school_qc["school_initial_est_total"].fillna(0).le(0) & ~has_no_point
    calibratable = ~(has_no_point | has_zero_initial)
    if not calibratable.any():
        raise ValueError("실제 재학생 수에 맞춰 보정할 수 있는 학교가 없습니다.")

    school_qc["calibration_factor"] = np.nan
    school_qc.loc[calibratable, "calibration_factor"] = (
        school_qc.loc[calibratable, "actual_enrollment"]
        / school_qc.loc[calibratable, "school_initial_est_total"]
    )
    school_qc["calibration_status"] = np.select(
        [has_no_point, has_zero_initial],
        ["NO_DEMAND_POINT", "ZERO_INITIAL_ESTIMATE"],
        default="CALIBRATED",
    )
    school_qc["review_flag"] = ""
    extreme_factor = calibratable & (
        school_qc["calibration_factor"].lt(EXTREME_FACTOR_LOW)
        | school_qc["calibration_factor"].gt(EXTREME_FACTOR_HIGH)
    )
    school_qc.loc[extreme_factor, "review_flag"] = "EXTREME_FACTOR"

    expanded = expanded.merge(
        school_qc[
            [
                "school_key",
                "school_initial_est_total",
                "calibration_factor",
                "calibration_status",
            ]
        ],
        on="school_key",
        how="left",
        validate="many_to_one",
    )
    expanded["calibration_status"] = expanded["calibration_status"].fillna(
        "NOT_APPLICABLE"
    )

    # 소수점을 유지한다. ZERO_INITIAL_ESTIMATE는 임의 배분하지 않고 0으로 둔다.
    expanded["student_weight"] = 0.0
    calibrated_row_mask = assigned_mask & expanded["calibration_status"].eq("CALIBRATED")
    expanded.loc[calibrated_row_mask, "student_weight"] = (
        expanded.loc[calibrated_row_mask, "initial_student_est"]
        * expanded.loc[calibrated_row_mask, "calibration_factor"]
    )

    final_by_school = (
        expanded.loc[expanded["school_key"].notna()]
        .groupby("school_key", as_index=False)["student_weight"]
        .sum()
        .rename(columns={"student_weight": "final_student_weight_total"})
    )
    school_qc = school_qc.merge(final_by_school, on="school_key", how="left")
    school_qc["final_student_weight_total"] = school_qc[
        "final_student_weight_total"
    ].fillna(0.0)
    school_qc["calibration_error"] = (
        school_qc["final_student_weight_total"] - school_qc["actual_enrollment"]
    )

    calibrated_school_mask = school_qc["calibration_status"].eq("CALIBRATED")
    max_calibration_error = float(
        school_qc.loc[calibrated_school_mask, "calibration_error"].abs().max()
    )
    if max_calibration_error > CALIBRATION_TOLERANCE:
        raise ValueError(
            "학교별 실제 재학생 수 보정 검증 실패: "
            f"최대 오차={max_calibration_error:.12g}"
        )
    if (expanded["student_weight"] < 0).any():
        raise ValueError("최종 학생가중치에 음수가 있습니다.")

    school_qc["calibration_check"] = np.where(
        school_qc["calibration_status"].eq("CALIBRATED")
        & school_qc["calibration_error"].abs().le(CALIBRATION_TOLERANCE),
        "PASS",
        "REVIEW",
    )

    diagnostics: dict[str, int | float] = {
        "relation_zone_ids_without_demand_point": len(missing_zone_ids),
        "expanded_rows": len(expanded),
        "assigned_rows": int(assigned_mask.sum()),
        "assigned_points": int(expanded.loc[assigned_mask, "census_id"].nunique()),
        "no_zone_points": int(
            expanded.loc[expanded["assignment_status"].eq("NO_ZONE"), "census_id"].nunique()
        ),
        "zone_not_matched_points": int(
            expanded.loc[
                expanded["assignment_status"].eq("ZONE_NOT_MATCHED"), "census_id"
            ].nunique()
        ),
        "excluded_zone_points": int(
            expanded.loc[
                expanded["assignment_status"].eq("EXCLUDED_ZONE"), "census_id"
            ].nunique()
        ),
        "no_public_school_link_points": int(
            expanded.loc[
                expanded["assignment_status"].eq("NO_PUBLIC_SCHOOL_LINK"),
                "census_id",
            ].nunique()
        ),
        "point_initial_allocation_max_error": point_allocation_max_error,
        "calibrated_school_count": int(calibrated_school_mask.sum()),
        "no_demand_point_school_count": int(has_no_point.sum()),
        "zero_initial_school_count": int(has_zero_initial.sum()),
        "extreme_factor_school_count": int(extreme_factor.sum()),
        "school_calibration_max_error": max_calibration_error,
    }
    return expanded, school_qc, diagnostics


def prepare_output_columns(distribution: pd.DataFrame) -> pd.DataFrame:
    """다음 분석에서 필요한 핵심 컬럼만 정리한다."""
    output_columns = [
        "census_id",
        "pop_5_9",
        "pop_10_14",
        "elem_pop_est",
        "longitude",
        "latitude",
        "zone_key",
        "zone_id",
        "zone_name",
        "zone_edu_office",
        "zone_match_status",
        "zone_exclude_reason",
        "assignment_status",
        "eligible_school_count",
        "school_key",
        "official_school_ids",
        "school_name",
        "school_edu_office",
        "school_lon",
        "school_lat",
        "main_branch_type",
        "small_school_flag",
        "school_area_type",
        "population_decline_type",
        "actual_enrollment",
        "initial_share",
        "initial_student_est",
        "school_initial_est_total",
        "calibration_factor",
        "student_weight",
        "calibration_status",
    ]
    return distribution.reindex(columns=output_columns)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    raw_dir = project_root / "data" / "01_raw"
    processed_dir = project_root / "data" / "02_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info("현재 학생분포 추정을 시작합니다.")
    logging.info("프로젝트 루트: %s", project_root)

    master, bridge_raw, demand = load_inputs(raw_dir)
    formula_error = validate_demand_formula(demand)
    public_master, relation, missing_key_count, collapsed_count = (
        build_public_zone_school_bridge(master, bridge_raw)
    )

    log_count("학교 마스터 전체 행", len(master))
    log_count("공립초등학교 수", len(public_master))
    log_count("유일 통학구역-공립학교 관계", len(relation))
    log_count("학교고유ID 결측으로 제외한 연결행", missing_key_count)
    log_count("같은 현재 학교로 통합한 중복 관계행", collapsed_count)
    log_count("집계구 수", len(demand))
    log_count("3/5 학령인구 공식 최대 오차", formula_error)

    point_zone, spatial_diag = spatial_join_points_to_zones(raw_dir, demand)
    for key, value in spatial_diag.items():
        logging.info("%-42s %s", key, value)

    distribution, school_qc, allocation_diag = assign_and_calibrate(
        point_zone,
        relation,
        public_master,
    )
    for key, value in allocation_diag.items():
        logging.info("%-42s %s", key, value)

    distribution_out = prepare_output_columns(distribution)
    distribution_path = processed_dir / OUTPUT_DISTRIBUTION
    school_qc_path = processed_dir / OUTPUT_SCHOOL_QC

    # float_format을 지정하지 않아 pandas가 계산 정밀도를 최대한 유지하도록 한다.
    # 표시용 반올림은 이후 기사 표·시각화 단계에서만 수행한다.
    # Write and validate a complete temporary file before exposing it to later stages.
    # This changes output handling only; allocation/calibration formulas are unchanged.
    for frame, destination in ((distribution_out, distribution_path), (school_qc, school_qc_path)):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        written = pd.read_csv(temporary, dtype=str, low_memory=False)
        if len(written) != len(frame) or list(written.columns) != list(frame.columns):
            raise IOError(f"Incomplete CSV write: {destination.name}")
        if "school_key" in frame and written["school_key"].nunique() != frame["school_key"].nunique():
            raise IOError(f"School IDs were lost while writing {destination.name}")
        temporary.replace(destination)

    logging.info("검증 완료")
    logging.info("학생분포 파일: %s", distribution_path)
    logging.info("학교 보정 검증 파일: %s", school_qc_path)

    review_count = int(school_qc["calibration_check"].eq("REVIEW").sum())
    if review_count:
        logging.warning(
            "학생분포를 계산할 수 없는 학교가 %s개 있습니다. "
            "school_calibration_summary.csv에서 calibration_status를 확인하세요.",
            f"{review_count:,}",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("분석이 중단되었습니다: %s", exc)
        sys.exit(1)

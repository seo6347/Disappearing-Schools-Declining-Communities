"""
05. 초등학교 공백지대(교육 인프라 공동화) 기본 분석

목적
----
1) 하한(LOWER_BOUND) 통합학교안을 적용했을 때 읍·면별 공립 초등학교 본교 수가
   통합 전후에 어떻게 달라지는지 계산한다.
2) 마지막 공립 초등학교 본교가 사라지는 읍·면과 그 지역의 집계구·학생 노출량을 구한다.
3) 이후 전국/지역/인구감소지역 등 원하는 기사·시각화 단위로 즉시 재집계할 수 있도록
   학교 단위, 집계구 단위, 읍·면 단위 기본 테이블을 만든다.

중요한 해석 기준
----------------
- 현재 학교 모집단: elementary_master.csv의 '공립'이면서 '본교'인 초등학교 전체.
- 통합 후 학교: 현재 모집단에서 하한 통합학교안의 폐교 대상 2개교만 제거.
- 읍·면: 학교 주소에서 읍/면 명칭을 추출하고, 집계구 경계의 ADM_CD와 공간 연결하여
  집계구를 같은 읍·면에 귀속시킨다. 동 지역은 이번 읍·면 공백 분석에서 제외한다.
- 학생 노출량은 두 값을 함께 보존한다.
  * elem_pop_est: 5~9세와 10~14세에 3/5를 적용한 집계구 초등학령인구 추정값
  * calibrated_student_weight: 실제 학교 재학생 수에 맞춰 보정한 집계구 학생 가중치
- '통합학교까지 30분 초과'는 현재 선택된 통합학교안의 지정 통합학교까지 걸리는
  직행 자동차 시간이다.
- '30분 안에 남아 있는 공립학교가 하나도 없음'은 모든 잔존학교까지의 경로가 필요하다.
  현재 경로자료는 선택된 통합학교까지의 경로만 있으므로 이 지표는 0으로 간주하지 않고
  NOT_EVALUATED로 명시한다. 추후 별도 라우팅 단계에서 계산해야 한다.

입력(기본 위치: data/01_raw, 없으면 data/02_processed 또는 data/03_outputs 탐색)
-------------------------------------------------------------------------------
- elementary_master.csv
- census_demand_points.gpkg  (census_boundary 레이어 필요)
- census_demand_points.csv
- current_student_distribution.csv
- commute_bound_demand_results.csv

산출물(기본 위치: data/03_outputs)
---------------------------------
- school_void_school_status.csv
  공립 본교별 통합 전후 존치/폐교 상태와 읍·면·ADM_CD 연결 상태
- school_void_demand_exposure.csv
  모든 집계구별 인구·학생가중치·하한 통합학교안 노출 및 마지막 학교 소멸 여부
- school_void_eupmyeon_summary.csv
  읍·면별 통합 전후 학교 수, 마지막 학교 소멸, 학생·통학시간 노출 요약
- school_void_validation_summary.csv
  입력·선택·공간연결·산출물 검증값과 아직 계산하지 않은 지표의 명시적 기록

실행 예시
---------
python scripts/05_analyze_school_voids.py

필요 패키지
-----------
pip install pandas geopandas pyogrio
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover - 실행환경 안내용
    raise ImportError("geopandas가 필요합니다: pip install geopandas pyogrio") from exc


SELECTION_POPULATION = "ALL_STUDENTS"
BOUND_TYPE = "LOWER_BOUND"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="하한 통합학교안 기준 초등학교 공백지대 분석")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--elementary-master", type=Path, default=None)
    parser.add_argument("--census-gpkg", type=Path, default=None)
    parser.add_argument("--census-csv", type=Path, default=None)
    parser.add_argument("--student-distribution", type=Path, default=None)
    parser.add_argument("--commute-bound-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def infer_project_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "scripts":
        return script_dir.parent
    return Path.cwd().resolve()


def find_input(root: Path, filename: str, explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = [
        root / "data" / "03_outputs" / filename,
        root / "data" / "02_processed" / filename,
        root / "data" / "01_raw" / filename,
        root / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{filename}을 찾지 못했습니다. 탐색 위치: {candidates}")


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label}에 필수 열이 없습니다: {missing}")


def clean_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_key_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).strip())


def parse_bool(series: pd.Series) -> pd.Series:
    true_values = {"TRUE", "T", "1", "Y", "YES"}
    false_values = {"FALSE", "F", "0", "N", "NO"}
    normalized = clean_string(series).str.upper()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out.loc[normalized.isin(true_values)] = True
    out.loc[normalized.isin(false_values)] = False
    return out


def first_nonempty(*values: object) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


EUPMYEON_PATTERN = re.compile(r"(?:^|\s)([^\s]+(?:읍|면))(?=\s|$)")


def extract_eupmyeon(address: object) -> str | pd.NA:
    if pd.isna(address):
        return pd.NA
    match = EUPMYEON_PATTERN.search(str(address).strip())
    return match.group(1) if match else pd.NA


def split_region(region: object) -> tuple[str | pd.NA, str | pd.NA]:
    if pd.isna(region):
        return pd.NA, pd.NA
    tokens = str(region).strip().split()
    sido = tokens[0] if tokens else pd.NA
    sigungu = tokens[1] if len(tokens) >= 2 else pd.NA
    return sido, sigungu


def unique_join(values: pd.Series) -> str:
    items = sorted({str(v).strip() for v in values if pd.notna(v) and str(v).strip()})
    return " | ".join(items)


def weighted_mean_from_sums(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator > 0))


def load_lower_bound_detail(path: Path) -> pd.DataFrame:
    detail = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = [
        "selection_id", "final_cluster_id", "selection_population", "bound_type",
        "integration_option_id", "demand_assignment_id", "census_id", "student_weight",
        "current_school_key", "current_school_name", "integrated_school_key",
        "integrated_school_name", "merged_school_1_key", "merged_school_1_name",
        "merged_school_2_key", "merged_school_2_name", "moved_student_flag",
        "commute_analysis_eligible_flag", "current_analysis_road_duration_min",
        "integrated_analysis_road_duration_min", "road_duration_change_min",
        "integrated_over_30_min_flag",
    ]
    require_columns(detail, required, path.name)
    detail = detail.loc[
        clean_string(detail["selection_population"]).eq(SELECTION_POPULATION)
        & clean_string(detail["bound_type"]).eq(BOUND_TYPE)
    ].copy()
    if detail.empty:
        raise ValueError(f"{SELECTION_POPULATION}/{BOUND_TYPE} 행이 없습니다.")

    for col in ["student_weight", "current_analysis_road_duration_min",
                "integrated_analysis_road_duration_min", "road_duration_change_min"]:
        detail[col] = pd.to_numeric(detail[col], errors="coerce")
    for col in ["moved_student_flag", "commute_analysis_eligible_flag",
                "integrated_over_30_min_flag"]:
        detail[col] = parse_bool(detail[col])
    for col in ["census_id", "final_cluster_id", "selection_id", "integration_option_id",
                "demand_assignment_id", "current_school_key", "integrated_school_key",
                "merged_school_1_key", "merged_school_2_key"]:
        detail[col] = clean_string(detail[col])

    cluster_option_n = detail.groupby("final_cluster_id")["integration_option_id"].nunique()
    if cluster_option_n.gt(1).any():
        bad = cluster_option_n[cluster_option_n.gt(1)].head().to_dict()
        raise ValueError(f"하한에서 한 군집에 복수 통합학교안이 선택되었습니다: {bad}")
    if detail["demand_assignment_id"].duplicated().any():
        examples = detail.loc[detail["demand_assignment_id"].duplicated(False),
                              "demand_assignment_id"].head().tolist()
        raise ValueError(f"하한 상세행의 demand_assignment_id가 중복됩니다: {examples}")
    return detail


def build_cluster_decisions(detail: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "selection_id", "final_cluster_id", "integration_option_id",
        "integrated_school_key", "integrated_school_name",
        "merged_school_1_key", "merged_school_1_name",
        "merged_school_2_key", "merged_school_2_name",
    ]
    decisions = detail[cols].drop_duplicates().copy()
    if decisions["final_cluster_id"].duplicated().any():
        raise ValueError("최종 군집별 하한 통합학교안이 1개로 확정되지 않았습니다.")
    closed_long = pd.concat([
        decisions[["final_cluster_id", "merged_school_1_key"]]
        .rename(columns={"merged_school_1_key": "school_key"}),
        decisions[["final_cluster_id", "merged_school_2_key"]]
        .rename(columns={"merged_school_2_key": "school_key"}),
    ], ignore_index=True)
    if closed_long["school_key"].duplicated().any():
        raise ValueError("폐교 대상 학교가 여러 최종 군집에 중복됩니다.")
    retained = set(decisions["integrated_school_key"])
    closed = set(closed_long["school_key"])
    overlap = sorted(retained & closed)
    if overlap:
        raise ValueError(f"같은 학교가 존치와 폐교에 동시에 포함됩니다: {overlap[:5]}")
    return decisions


def build_school_universe(master_path: Path, decisions: pd.DataFrame) -> pd.DataFrame:
    master = pd.read_csv(master_path, encoding="utf-8-sig", low_memory=False)
    required = [
        "학교고유ID", "지역", "학교명", "설립형태", "본교분교구분",
        "소재지지번주소", "소재지도로명주소", "위도", "경도",
        "교육지원청", "지역구분", "인구감소지역",
    ]
    require_columns(master, required, master_path.name)
    for col in ["학교고유ID", "설립형태", "본교분교구분"]:
        master[col] = clean_string(master[col])
    schools = master.loc[
        master["설립형태"].eq("공립") & master["본교분교구분"].eq("본교")
    ].copy()
    schools = schools.rename(columns={"학교고유ID": "school_key"})
    if schools["school_key"].duplicated().any():
        raise ValueError("공립 본교 모집단의 school_key가 중복됩니다.")

    retained = set(decisions["integrated_school_key"])
    closed = set(decisions["merged_school_1_key"]) | set(decisions["merged_school_2_key"])
    selected = retained | closed
    missing = sorted(selected - set(schools["school_key"]))
    if missing:
        raise ValueError(f"하한 군집 학교가 공립 본교 모집단에 없습니다: {missing[:10]}")
    schools["scenario_school_status"] = np.select(
        [schools["school_key"].isin(closed), schools["school_key"].isin(retained)],
        ["CLOSED_BY_LOWER_BOUND", "RETAINED_IN_LOWER_BOUND"],
        default="UNCHANGED_REMAINING",
    )
    schools["remaining_after_lower_bound_flag"] = ~schools["school_key"].isin(closed)

    schools["eupmyeon_name"] = schools["소재지지번주소"].map(extract_eupmyeon)
    fallback = schools["eupmyeon_name"].isna()
    schools.loc[fallback, "eupmyeon_name"] = schools.loc[
        fallback, "소재지도로명주소"
    ].map(extract_eupmyeon)
    region_parts = schools["지역"].map(split_region)
    schools["sido_name"] = region_parts.map(lambda x: x[0])
    schools["sigungu_name"] = region_parts.map(lambda x: x[1])
    schools["eupmyeon_key"] = pd.NA
    has_unit = schools["eupmyeon_name"].notna()
    schools.loc[has_unit, "eupmyeon_key"] = (
        schools.loc[has_unit, "지역"].map(normalize_key_text)
        + "|" + schools.loc[has_unit, "eupmyeon_name"].map(normalize_key_text)
    )
    schools["longitude"] = pd.to_numeric(schools["경도"], errors="coerce")
    schools["latitude"] = pd.to_numeric(schools["위도"], errors="coerce")
    return schools


def assign_school_admin_codes(schools: pd.DataFrame, gpkg_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    boundaries = gpd.read_file(gpkg_path, layer="census_boundary")
    require_columns(boundaries, ["ADM_CD", "census_id", "geometry"], "census_boundary")
    boundaries["ADM_CD"] = clean_string(boundaries["ADM_CD"])
    boundaries = boundaries.loc[boundaries.geometry.notna(), ["ADM_CD", "geometry"]].copy()
    boundaries = boundaries.loc[boundaries["ADM_CD"].notna() & boundaries["ADM_CD"].ne("")]
    boundaries = boundaries.dissolve(by="ADM_CD", as_index=False)
    if boundaries.crs is None:
        raise ValueError("census_boundary 레이어의 CRS가 없습니다.")

    valid = schools["longitude"].notna() & schools["latitude"].notna()
    points = gpd.GeoDataFrame(
        schools.loc[valid, ["school_key"]].copy(),
        geometry=gpd.points_from_xy(
            schools.loc[valid, "longitude"], schools.loc[valid, "latitude"]
        ),
        crs="EPSG:4326",
    ).to_crs(boundaries.crs)
    hits = gpd.sjoin(points, boundaries, how="left", predicate="intersects")
    hit_lists = (
        hits.groupby("school_key")["ADM_CD"]
        .agg(lambda s: sorted({str(v) for v in s.dropna() if str(v).strip()}))
    )
    out = schools.copy()
    out["adm_cd_candidates"] = out["school_key"].map(hit_lists).map(
        lambda x: x if isinstance(x, list) else []
    )
    out["school_admin_match_status"] = out["adm_cd_candidates"].map(
        lambda x: "SPATIAL_UNIQUE" if len(x) == 1
        else ("SPATIAL_MULTIPLE" if len(x) > 1 else "SPATIAL_NONE")
    )
    out["adm_cd"] = out["adm_cd_candidates"].map(
        lambda x: x[0] if len(x) == 1 else pd.NA
    )

    # 경계선 오차로 미매칭된 학교는 동일한 주소 읍·면의 다른 학교가 하나의 ADM_CD로
    # 일관되게 연결된 경우에만 보완한다. 복수 후보는 임의 배정하지 않는다.
    unique_unit_adm = (
        out.loc[out["eupmyeon_key"].notna() & out["adm_cd"].notna(),
                ["eupmyeon_key", "adm_cd"]]
        .drop_duplicates()
        .groupby("eupmyeon_key")["adm_cd"]
        .agg(list)
    )
    unique_unit_adm = unique_unit_adm[unique_unit_adm.map(len).eq(1)].map(lambda x: x[0])
    fill_mask = out["adm_cd"].isna() & out["eupmyeon_key"].isin(unique_unit_adm.index)
    out.loc[fill_mask, "adm_cd"] = out.loc[fill_mask, "eupmyeon_key"].map(unique_unit_adm)
    out.loc[fill_mask, "school_admin_match_status"] = "FILLED_BY_UNIQUE_EUPMYEON"
    out["adm_cd_candidates"] = out["adm_cd_candidates"].map(lambda x: " | ".join(x))

    pairs = out.loc[out["eupmyeon_key"].notna() & out["adm_cd"].notna(),
                    ["adm_cd", "eupmyeon_key", "sido_name", "sigungu_name",
                     "eupmyeon_name", "지역"]].drop_duplicates()
    adm_n = pairs.groupby("adm_cd")["eupmyeon_key"].nunique()
    unit_n = pairs.groupby("eupmyeon_key")["adm_cd"].nunique()
    pairs["adm_mapping_status"] = np.select(
        [
            pairs["adm_cd"].map(adm_n).eq(1) & pairs["eupmyeon_key"].map(unit_n).eq(1),
            pairs["adm_cd"].map(adm_n).gt(1),
            pairs["eupmyeon_key"].map(unit_n).gt(1),
        ],
        ["UNIQUE", "ADM_TO_MULTIPLE_EUPMYEON", "EUPMYEON_TO_MULTIPLE_ADM"],
        default="REVIEW",
    )
    adm_map = pairs.loc[pairs["adm_mapping_status"].eq("UNIQUE")].copy()
    adm_map = adm_map.drop_duplicates("adm_cd")
    return out, adm_map


def aggregate_current_student_weights(path: Path) -> pd.DataFrame:
    dist = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    require_columns(
        dist, ["census_id", "assignment_status", "calibration_status", "student_weight"],
        path.name,
    )
    dist["census_id"] = clean_string(dist["census_id"])
    dist["student_weight"] = pd.to_numeric(dist["student_weight"], errors="coerce")
    usable = dist.loc[
        clean_string(dist["assignment_status"]).eq("ASSIGNED")
        & clean_string(dist["calibration_status"]).eq("CALIBRATED")
        & dist["student_weight"].notna()
    ].copy()
    return (
        usable.groupby("census_id", as_index=False)["student_weight"].sum()
        .rename(columns={"student_weight": "calibrated_student_weight"})
    )


def aggregate_lower_detail_by_census(detail: pd.DataFrame) -> pd.DataFrame:
    d = detail.copy()
    weight = d["student_weight"].fillna(0.0)
    eligible = d["commute_analysis_eligible_flag"].fillna(False)
    moved = d["moved_student_flag"].fillna(False)
    over30 = d["integrated_over_30_min_flag"].fillna(False)

    d["selected_cluster_student_weight"] = weight
    d["moved_student_weight"] = weight.where(moved, 0.0)
    d["moved_route_eligible_student_weight"] = weight.where(moved & eligible, 0.0)
    d["designated_integrated_over30_student_weight"] = weight.where(moved & eligible & over30, 0.0)
    d["weighted_current_duration_moved"] = (
        weight * d["current_analysis_road_duration_min"]
    ).where(moved & eligible, 0.0)
    d["weighted_integrated_duration_moved"] = (
        weight * d["integrated_analysis_road_duration_min"]
    ).where(moved & eligible, 0.0)
    d["weighted_duration_change_moved"] = (
        weight * d["road_duration_change_min"]
    ).where(moved & eligible, 0.0)
    d["moved_integrated_duration_for_max"] = d[
        "integrated_analysis_road_duration_min"
    ].where(moved & eligible)
    d["moved_integrated_duration_for_min"] = d[
        "integrated_analysis_road_duration_min"
    ].where(moved & eligible)
    d["designated_integrated_over30_point"] = (moved & eligible & over30).astype(int)

    grouped = d.groupby("census_id", as_index=False).agg(
        selected_cluster_student_weight=("selected_cluster_student_weight", "sum"),
        moved_student_weight=("moved_student_weight", "sum"),
        moved_route_eligible_student_weight=("moved_route_eligible_student_weight", "sum"),
        moved_weighted_current_duration_sum=("weighted_current_duration_moved", "sum"),
        moved_weighted_integrated_duration_sum=("weighted_integrated_duration_moved", "sum"),
        moved_weighted_duration_change_sum=("weighted_duration_change_moved", "sum"),
        designated_integrated_over30_student_weight=("designated_integrated_over30_student_weight", "sum"),
        designated_integrated_over30_point_flag=("designated_integrated_over30_point", "max"),
        designated_integrated_duration_min=("moved_integrated_duration_for_min", "min"),
        designated_integrated_duration_max=("moved_integrated_duration_for_max", "max"),
        designated_integrated_school_keys=("integrated_school_key", unique_join),
        designated_integrated_school_names=("integrated_school_name", unique_join),
        selected_final_cluster_ids=("final_cluster_id", unique_join),
    )
    grouped["moved_weighted_current_duration_mean"] = weighted_mean_from_sums(
        grouped["moved_weighted_current_duration_sum"],
        grouped["moved_route_eligible_student_weight"],
    )
    grouped["moved_weighted_integrated_duration_mean"] = weighted_mean_from_sums(
        grouped["moved_weighted_integrated_duration_sum"],
        grouped["moved_route_eligible_student_weight"],
    )
    grouped["moved_weighted_duration_change_mean"] = weighted_mean_from_sums(
        grouped["moved_weighted_duration_change_sum"],
        grouped["moved_route_eligible_student_weight"],
    )
    return grouped


def build_school_unit_summary(schools: pd.DataFrame) -> pd.DataFrame:
    subset = schools.loc[schools["eupmyeon_key"].notna()].copy()
    subset["closed_flag"] = subset["scenario_school_status"].eq("CLOSED_BY_LOWER_BOUND").astype(int)
    subset["retained_cluster_flag"] = subset["scenario_school_status"].eq("RETAINED_IN_LOWER_BOUND").astype(int)
    subset["remaining_flag"] = subset["remaining_after_lower_bound_flag"].astype(int)
    summary = subset.groupby("eupmyeon_key", as_index=False).agg(
        sido_name=("sido_name", "first"),
        sigungu_name=("sigungu_name", "first"),
        region_name=("지역", "first"),
        eupmyeon_name=("eupmyeon_name", "first"),
        adm_cd=("adm_cd", lambda s: unique_join(s) if s.nunique(dropna=True) == 1 else pd.NA),
        school_admin_match_statuses=("school_admin_match_status", unique_join),
        current_public_main_school_count=("school_key", "nunique"),
        closed_school_count=("closed_flag", "sum"),
        retained_cluster_school_count=("retained_cluster_flag", "sum"),
        remaining_public_main_school_count=("remaining_flag", "sum"),
        current_school_names=("학교명", unique_join),
        closed_school_names=("학교명", lambda s: ""),
        remaining_school_names=("학교명", lambda s: ""),
    )
    closed_names = (
        subset.loc[subset["closed_flag"].eq(1)]
        .groupby("eupmyeon_key")["학교명"].agg(unique_join)
    )
    remaining_names = (
        subset.loc[subset["remaining_flag"].eq(1)]
        .groupby("eupmyeon_key")["학교명"].agg(unique_join)
    )
    summary["closed_school_names"] = summary["eupmyeon_key"].map(closed_names).fillna("")
    summary["remaining_school_names"] = summary["eupmyeon_key"].map(remaining_names).fillna("")
    summary["last_public_elementary_school_lost_flag"] = (
        summary["current_public_main_school_count"].gt(0)
        & summary["remaining_public_main_school_count"].eq(0)
    )
    return summary


def build_demand_exposure(
    census_csv_path: Path,
    current_weights: pd.DataFrame,
    lower_census: pd.DataFrame,
    adm_map: pd.DataFrame,
    school_units: pd.DataFrame,
) -> pd.DataFrame:
    census = pd.read_csv(census_csv_path, encoding="utf-8-sig", low_memory=False)
    require_columns(census, ["census_id", "elem_pop_est", "longitude", "latitude"], census_csv_path.name)
    census["census_id"] = clean_string(census["census_id"])
    if census["census_id"].duplicated().any():
        raise ValueError("census_demand_points.csv의 census_id가 중복됩니다.")
    for col in ["elem_pop_est", "longitude", "latitude"]:
        census[col] = pd.to_numeric(census[col], errors="coerce")
    census["adm_cd"] = census["census_id"].str[:8]

    admin_cols = ["adm_cd", "eupmyeon_key", "sido_name", "sigungu_name", "eupmyeon_name", "지역"]
    demand = census.merge(adm_map[admin_cols], on="adm_cd", how="left", validate="many_to_one")
    demand = demand.rename(columns={"지역": "region_name"})
    demand["eupmyeon_mapping_status"] = np.where(
        demand["eupmyeon_key"].notna(), "MAPPED_BY_ADM_CD", "NOT_MAPPED_TO_ANALYZED_EUPMYEON"
    )
    demand = demand.merge(current_weights, on="census_id", how="left", validate="one_to_one")
    demand = demand.merge(lower_census, on="census_id", how="left", validate="one_to_one")

    zero_cols = [
        "calibrated_student_weight", "selected_cluster_student_weight", "moved_student_weight",
        "moved_route_eligible_student_weight", "moved_weighted_current_duration_sum",
        "moved_weighted_integrated_duration_sum", "moved_weighted_duration_change_sum",
        "designated_integrated_over30_student_weight", "designated_integrated_over30_point_flag",
    ]
    for col in zero_cols:
        demand[col] = pd.to_numeric(demand[col], errors="coerce").fillna(0.0)
    for col in ["designated_integrated_school_keys", "designated_integrated_school_names",
                "selected_final_cluster_ids"]:
        demand[col] = demand[col].fillna("")

    unit_cols = [
        "eupmyeon_key", "current_public_main_school_count", "closed_school_count",
        "remaining_public_main_school_count", "last_public_elementary_school_lost_flag",
    ]
    demand = demand.merge(school_units[unit_cols], on="eupmyeon_key", how="left", validate="many_to_one")
    demand["last_public_elementary_school_lost_flag"] = demand[
        "last_public_elementary_school_lost_flag"
    ].eq(True)
    demand["nearest_remaining_public_school_30min_status"] = "NOT_EVALUATED"
    demand["nearest_remaining_public_school_duration_min"] = np.nan
    demand["no_remaining_public_school_within_30min_flag"] = pd.NA
    return demand


def aggregate_demand_by_unit(demand: pd.DataFrame) -> pd.DataFrame:
    mapped = demand.loc[demand["eupmyeon_key"].notna()].copy()
    mapped["selected_census_point_flag"] = mapped["selected_cluster_student_weight"].gt(0).astype(int)
    mapped["moved_census_point_flag"] = mapped["moved_student_weight"].gt(0).astype(int)
    unit = mapped.groupby("eupmyeon_key", as_index=False).agg(
        census_point_count=("census_id", "nunique"),
        elem_pop_est_sum=("elem_pop_est", "sum"),
        calibrated_student_weight_sum=("calibrated_student_weight", "sum"),
        selected_census_point_count=("selected_census_point_flag", "sum"),
        selected_cluster_student_weight_sum=("selected_cluster_student_weight", "sum"),
        moved_census_point_count=("moved_census_point_flag", "sum"),
        moved_student_weight_sum=("moved_student_weight", "sum"),
        moved_route_eligible_student_weight_sum=("moved_route_eligible_student_weight", "sum"),
        moved_weighted_current_duration_sum=("moved_weighted_current_duration_sum", "sum"),
        moved_weighted_integrated_duration_sum=("moved_weighted_integrated_duration_sum", "sum"),
        moved_weighted_duration_change_sum=("moved_weighted_duration_change_sum", "sum"),
        designated_integrated_duration_max=("designated_integrated_duration_max", "max"),
        designated_integrated_over30_census_point_count=("designated_integrated_over30_point_flag", "sum"),
        designated_integrated_over30_student_weight_sum=("designated_integrated_over30_student_weight", "sum"),
    )
    denom = unit["moved_route_eligible_student_weight_sum"]
    unit["moved_weighted_current_duration_mean"] = weighted_mean_from_sums(
        unit["moved_weighted_current_duration_sum"], denom
    )
    unit["moved_weighted_integrated_duration_mean"] = weighted_mean_from_sums(
        unit["moved_weighted_integrated_duration_sum"], denom
    )
    unit["moved_weighted_duration_change_mean"] = weighted_mean_from_sums(
        unit["moved_weighted_duration_change_sum"], denom
    )
    unit["nearest_remaining_public_school_30min_status"] = "NOT_EVALUATED"
    unit["no_remaining_public_school_within_30min_census_count"] = np.nan
    unit["no_remaining_public_school_within_30min_student_weight"] = np.nan
    return unit


def validation_row(metric: str, value: object, status: str, note: str = "") -> dict[str, object]:
    return {"metric": metric, "value": value, "status": status, "note": note}


def main() -> None:
    args = parse_args()
    root = infer_project_root(args.project_root)
    paths = {
        "master": find_input(root, "elementary_master.csv", args.elementary_master),
        "gpkg": find_input(root, "census_demand_points.gpkg", args.census_gpkg),
        "census": find_input(root, "census_demand_points.csv", args.census_csv),
        "distribution": find_input(root, "current_student_distribution.csv", args.student_distribution),
        "bounds": find_input(root, "commute_bound_demand_results.csv", args.commute_bound_results),
    }
    output_dir = (args.output_dir or root / "data" / "03_outputs").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] 하한 통합학교안 확인")
    detail = load_lower_bound_detail(paths["bounds"])
    decisions = build_cluster_decisions(detail)
    print("[2/7] 전국 공립 초등학교 본교 모집단 구성")
    schools = build_school_universe(paths["master"], decisions)
    print("[3/7] 학교 좌표와 집계구 행정경계 연결")
    schools, adm_map = assign_school_admin_codes(schools, paths["gpkg"])
    print("[4/7] 집계구 학생가중치와 하한 통학노출 집계")
    current_weights = aggregate_current_student_weights(paths["distribution"])
    lower_census = aggregate_lower_detail_by_census(detail)
    print("[5/7] 학교·집계구·읍면 기본 테이블 생성")
    school_units = build_school_unit_summary(schools)
    demand = build_demand_exposure(
        paths["census"], current_weights, lower_census, adm_map, school_units
    )
    demand_units = aggregate_demand_by_unit(demand)
    eupmyeon = school_units.merge(demand_units, on="eupmyeon_key", how="left", validate="one_to_one")

    count_cols = [
        "census_point_count", "elem_pop_est_sum", "calibrated_student_weight_sum",
        "selected_census_point_count", "selected_cluster_student_weight_sum",
        "moved_census_point_count", "moved_student_weight_sum",
        "moved_route_eligible_student_weight_sum", "moved_weighted_current_duration_sum",
        "moved_weighted_integrated_duration_sum", "moved_weighted_duration_change_sum",
        "designated_integrated_over30_census_point_count",
        "designated_integrated_over30_student_weight_sum",
    ]
    for col in count_cols:
        eupmyeon[col] = pd.to_numeric(eupmyeon[col], errors="coerce").fillna(0.0)
    eupmyeon["nearest_remaining_public_school_30min_status"] = "NOT_EVALUATED"
    eupmyeon["no_remaining_public_school_within_30min_census_count"] = np.nan
    eupmyeon["no_remaining_public_school_within_30min_student_weight"] = np.nan

    print("[6/7] 검증")
    retained_count = decisions["integrated_school_key"].nunique()
    closed_keys = set(decisions["merged_school_1_key"]) | set(decisions["merged_school_2_key"])
    validation = [
        validation_row("selection_population", SELECTION_POPULATION, "PASS"),
        validation_row("bound_type", BOUND_TYPE, "PASS", "기사 기본값으로 사용할 하한 통합학교안"),
        validation_row("final_cluster_count", decisions["final_cluster_id"].nunique(), "PASS"),
        validation_row("retained_cluster_school_count", retained_count, "PASS"),
        validation_row("closed_school_count", len(closed_keys), "PASS"),
        validation_row("public_main_school_count_before", len(schools), "PASS"),
        validation_row(
            "public_main_school_count_after",
            int(schools["remaining_after_lower_bound_flag"].sum()),
            "PASS",
        ),
        validation_row(
            "school_count_identity_check",
            int(schools["remaining_after_lower_bound_flag"].sum()) + len(closed_keys),
            "PASS" if int(schools["remaining_after_lower_bound_flag"].sum()) + len(closed_keys) == len(schools) else "FAIL",
            "통합 후 잔존학교 + 폐교학교 = 통합 전 공립 본교",
        ),
        validation_row("school_eupmyeon_parsed_count", int(schools["eupmyeon_key"].notna().sum()), "INFO"),
        validation_row("school_admin_code_mapped_count", int(schools["adm_cd"].notna().sum()), "INFO"),
        validation_row("unique_adm_to_eupmyeon_count", len(adm_map), "INFO"),
        validation_row("census_point_count", len(demand), "PASS"),
        validation_row("census_mapped_to_analyzed_eupmyeon_count", int(demand["eupmyeon_key"].notna().sum()), "INFO"),
        validation_row("analyzed_eupmyeon_count", len(eupmyeon), "PASS"),
        validation_row(
            "last_public_elementary_school_lost_eupmyeon_count",
            int(eupmyeon["last_public_elementary_school_lost_flag"].sum()),
            "INFO",
        ),
        validation_row(
            "last_school_lost_elem_pop_est_sum",
            float(eupmyeon.loc[eupmyeon["last_public_elementary_school_lost_flag"], "elem_pop_est_sum"].sum()),
            "INFO",
        ),
        validation_row(
            "last_school_lost_calibrated_student_weight_sum",
            float(eupmyeon.loc[eupmyeon["last_public_elementary_school_lost_flag"], "calibrated_student_weight_sum"].sum()),
            "INFO",
        ),
        validation_row(
            "no_remaining_school_within_30min_metric",
            "NOT_EVALUATED",
            "PENDING",
            "모든 집계구-잔존 공립학교 경로가 없어 지정 통합학교 30분 초과와 구분하여 보류",
        ),
    ]

    school_output_cols = [
        "school_key", "학교명", "시도교육청", "교육지원청", "지역",
        "sido_name", "sigungu_name", "eupmyeon_name", "eupmyeon_key", "adm_cd",
        "school_admin_match_status", "adm_cd_candidates", "scenario_school_status",
        "remaining_after_lower_bound_flag", "지역구분", "인구감소지역",
        "소재지지번주소", "소재지도로명주소", "longitude", "latitude",
    ]
    require_columns(schools, school_output_cols, "school output")
    schools_out = schools[school_output_cols].sort_values(
        ["지역", "eupmyeon_name", "학교명"], na_position="last"
    )
    demand = demand.sort_values("census_id")
    eupmyeon = eupmyeon.sort_values(["sido_name", "sigungu_name", "eupmyeon_name"])
    validation_df = pd.DataFrame(validation)

    outputs = {
        "school_void_school_status.csv": schools_out,
        "school_void_demand_exposure.csv": demand,
        "school_void_eupmyeon_summary.csv": eupmyeon,
        "school_void_validation_summary.csv": validation_df,
    }
    print("[7/7] 산출물 저장")
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"- {filename}: {len(frame):,}행 x {len(frame.columns):,}열 -> {path}")

    fail_count = validation_df["status"].eq("FAIL").sum()
    print("\n완료 상태:", "FAIL" if fail_count else "READY_WITH_PENDING_NEAREST_SCHOOL_ROUTING")
    print("주의: 지정 통합학교 30분 초과는 계산되지만, '30분 내 잔존학교 없음'은 아직 미평가입니다.")


if __name__ == "__main__":
    main()

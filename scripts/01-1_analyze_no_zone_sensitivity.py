"""
NO_ZONE 민감도 분석 단계 (01-1)
================================

목적
----
`current_student_distribution.csv`에서 `assignment_status=NO_ZONE`인 집계구만
추출하여, 대표점이 통학구역에 포함되지 않은 원인이 엄격한 `within` 판정 때문인지
실제 공간적 공백 때문인지 판단할 수 있는 진단자료를 만든다.

이 코드는 통학구역을 새로 배정하거나 학교별 학생가중치를 다시 보정하지 않는다.
허용거리별 민감도 결과를 확인한 뒤, 다음 단계에서 보완 기준을 확정하고 학생분포
추정 전체를 다시 실행해야 한다.

입력 파일
---------
data/01_raw/census_demand_points.gpkg
    - `census_demand_points` 레이어의 집계구 대표점
data/01_raw/elementary_polygon.gpkg
    - `통학구역_전체` 레이어의 통학구역 폴리곤
data/02_processed/current_student_distribution.csv
    - 기존 학생분포 추정 결과와 `NO_ZONE` 상태

출력 파일
---------
data/02_processed/no_zone_sensitivity_detail.csv
    - NO_ZONE 집계구 1개당 1행
    - 최근접 통학구역, 거리, 통학구역 상태, 허용거리별 후보 수·판정 결과

data/02_processed/no_zone_sensitivity_summary.csv
    - 허용거리별 누적 복구 가능 집계구 수와 학령인구 합계
    - `UNIQUE_USABLE_ZONE`은 공간적으로 후보가 하나이고 전처리 매칭도 정상인 경우

기본 민감도 기준
----------------
0m, 0.1m, 0.5m, 1m, 3m, 5m, 10m, 25m, 50m, 100m

판정값
------
UNIQUE_USABLE_ZONE
    허용거리 안에 통학구역이 정확히 하나 있고 `in_master=True`,
    `match_status='매칭'`인 경우. 최종 보완 후보이다.
UNIQUE_UNUSABLE_ZONE
    공간 후보는 하나지만 전처리 매칭 대상이 아닌 통학구역인 경우이다.
    임의 배정하지 않는다.
AMBIGUOUS_MULTIPLE_ZONES
    허용거리 안에 통학구역이 둘 이상인 경우이다. 임의 선택하지 않는다.
OUTSIDE_TOLERANCE
    해당 허용거리 안에 통학구역이 없는 경우이다.

실행 환경
---------
Windows 터미널에서 최초 1회:
    py -m pip install pandas numpy geopandas pyogrio

프로젝트 루트에서 실행:
    py scripts/01-1_analyze_no_zone_sensitivity.py

다른 허용거리로 실행하는 예:
    py scripts/01-1_analyze_no_zone_sensitivity.py --tolerances 0 0.5 1 2 5 10

소수점 처리
-----------
거리와 인구는 계산 정밀도를 그대로 보존한다. CSV 저장 단계에서 반올림하지 않는다.
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
        "터미널에서 'py -m pip install pandas numpy geopandas pyogrio'를 실행하세요."
    ) from exc


# -----------------------------------------------------------------------------
# 분석 설정
# -----------------------------------------------------------------------------

POINT_FILE = "census_demand_points.gpkg"
ZONE_FILE = "elementary_polygon.gpkg"
DISTRIBUTION_FILE = "current_student_distribution.csv"

POINT_LAYER = "census_demand_points"
ZONE_LAYER = "통학구역_전체"

OUTPUT_DETAIL = "no_zone_sensitivity_detail.csv"
OUTPUT_SUMMARY = "no_zone_sensitivity_summary.csv"

DEFAULT_TOLERANCES_M = [0.0, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0]

# 거리 0 판정에서 부동소수점 오차만 흡수하기 위한 값이다.
# 실제 공간 허용거리로 사용하지 않는다.
DISTANCE_EPSILON_M = 1e-8

POINT_REQUIRED = {"census_id"}
ZONE_REQUIRED = {
    "zone_key",
    "HAKGUDO_ID",
    "HAKGUDO_NM",
    "EDU_NM",
    "in_master",
    "match_status",
    "exclude_reason",
}
DISTRIBUTION_REQUIRED = {
    "census_id",
    "elem_pop_est",
    "longitude",
    "latitude",
    "assignment_status",
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="NO_ZONE 집계구 민감도 분석")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="final_project 폴더 경로. 기본값은 scripts 폴더의 상위 폴더입니다.",
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=DEFAULT_TOLERANCES_M,
        help="민감도 분석에 사용할 허용거리(m) 목록",
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
    return series.astype("string").str.strip()


def to_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "y", "yes", "t"})


def join_unique(values: Iterable[object]) -> str:
    result = sorted(
        {
            str(value).strip()
            for value in values
            if pd.notna(value) and str(value).strip()
        }
    )
    return "|".join(result)


def tolerance_label(tolerance: float) -> str:
    """CSV 컬럼명에 안전한 허용거리 라벨을 만든다."""
    if float(tolerance).is_integer():
        return f"{int(tolerance)}m"
    return f"{tolerance:g}".replace(".", "p") + "m"


def validate_tolerances(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("허용거리를 하나 이상 입력해야 합니다.")
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("허용거리는 0 이상의 유한한 숫자여야 합니다.")
    return sorted(set(float(value) for value in values))


def load_no_zone_distribution(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / DISTRIBUTION_FILE
    if not path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    distribution = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"census_id": "string"},
        low_memory=False,
    )
    require_columns(distribution, DISTRIBUTION_REQUIRED, DISTRIBUTION_FILE)
    distribution["census_id"] = normalize_id(distribution["census_id"])

    no_zone = distribution.loc[
        distribution["assignment_status"].eq("NO_ZONE"),
        ["census_id", "elem_pop_est", "longitude", "latitude", "assignment_status"],
    ].copy()

    # NO_ZONE은 학교로 확장되지 않으므로 집계구당 정확히 한 행이어야 한다.
    require_unique(no_zone, "census_id", "NO_ZONE 집계구")
    if no_zone.empty:
        raise ValueError("current_student_distribution.csv에 NO_ZONE 집계구가 없습니다.")

    numeric_columns = ["elem_pop_est", "longitude", "latitude"]
    for column in numeric_columns:
        no_zone[column] = pd.to_numeric(no_zone[column], errors="raise")
    if no_zone["elem_pop_est"].isna().any() or (no_zone["elem_pop_est"] < 0).any():
        raise ValueError("NO_ZONE 집계구의 elem_pop_est에 결측 또는 음수가 있습니다.")
    return no_zone


def load_spatial_data(
    raw_dir: Path,
    no_zone: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, object]]:
    point_path = raw_dir / POINT_FILE
    zone_path = raw_dir / ZONE_FILE
    for path in [point_path, zone_path]:
        if not path.exists():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    points = gpd.read_file(point_path, layer=POINT_LAYER)
    zones = gpd.read_file(zone_path, layer=ZONE_LAYER)
    require_columns(points, POINT_REQUIRED, f"{POINT_FILE}:{POINT_LAYER}")
    require_columns(zones, ZONE_REQUIRED, f"{ZONE_FILE}:{ZONE_LAYER}")

    points["census_id"] = normalize_id(points["census_id"])
    zones["zone_key"] = normalize_id(zones["zone_key"])
    zones["HAKGUDO_ID"] = normalize_id(zones["HAKGUDO_ID"])
    zones["in_master"] = to_boolean(zones["in_master"])
    zones["zone_usable"] = zones["in_master"] & zones["match_status"].eq("매칭")

    require_unique(points, "census_id", "집계구 대표점 GPKG")
    require_unique(zones, "zone_key", "통학구역 GPKG")
    if points.crs is None or zones.crs is None:
        raise ValueError("집계구 또는 통학구역 공간 파일에 좌표계가 없습니다.")

    missing_point_ids = set(no_zone["census_id"]) - set(points["census_id"])
    if missing_point_ids:
        raise ValueError(
            "NO_ZONE 집계구가 대표점 GPKG에 없습니다: "
            f"{len(missing_point_ids):,}개, 예시 {sorted(missing_point_ids)[:10]}"
        )

    invalid_zone_count = int((~zones.geometry.is_valid).sum())
    if invalid_zone_count:
        zones.loc[~zones.geometry.is_valid, "geometry"] = zones.loc[
            ~zones.geometry.is_valid, "geometry"
        ].make_valid()
    invalid_after_repair = int((~zones.geometry.is_valid).sum())
    if invalid_after_repair:
        raise ValueError(
            f"유효성 복구에 실패한 통학구역이 {invalid_after_repair:,}개 있습니다."
        )

    no_zone_points = points.loc[
        points["census_id"].isin(no_zone["census_id"]),
        ["census_id", "geometry"],
    ].copy()
    no_zone_points = no_zone_points.to_crs(zones.crs)
    no_zone_points = no_zone_points.merge(
        no_zone,
        on="census_id",
        how="left",
        validate="one_to_one",
    )

    # 이후 index_right를 안정적으로 참조하기 위해 0부터 다시 부여한다.
    zones = zones.reset_index(drop=True)
    diagnostics = {
        "point_crs_before": str(points.crs),
        "zone_crs": str(zones.crs),
        "no_zone_point_count": len(no_zone_points),
        "no_zone_population": float(no_zone_points["elem_pop_est"].sum()),
        "zone_count": len(zones),
        "usable_zone_count": int(zones["zone_usable"].sum()),
        "invalid_zones_repaired": invalid_zone_count,
    }
    return no_zone_points, zones, diagnostics


def build_nearest_zone_table(
    no_zone_points: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """모든 NO_ZONE 대표점에 대해 거리 제한 없는 최근접 통학구역을 찾는다."""
    zone_columns = [
        "zone_key",
        "HAKGUDO_ID",
        "HAKGUDO_NM",
        "EDU_NM",
        "in_master",
        "match_status",
        "exclude_reason",
        "zone_usable",
        "geometry",
    ]
    nearest_raw = gpd.sjoin_nearest(
        no_zone_points[["census_id", "geometry"]],
        zones[zone_columns],
        how="left",
        distance_col="nearest_distance_m",
    )
    if nearest_raw["zone_key"].isna().any():
        raise ValueError("최근접 통학구역을 찾지 못한 NO_ZONE 집계구가 있습니다.")
    if (nearest_raw["nearest_distance_m"] < -DISTANCE_EPSILON_M).any():
        raise ValueError("최근접 거리에서 음수가 발견되었습니다.")

    # sjoin_nearest는 같은 최단거리의 폴리곤이 여러 개면 여러 행을 반환한다.
    nearest = (
        pd.DataFrame(nearest_raw.drop(columns="geometry", errors="ignore"))
        .groupby("census_id", as_index=False)
        .agg(
            nearest_distance_m=("nearest_distance_m", "min"),
            equidistant_nearest_zone_count=("zone_key", "nunique"),
            nearest_zone_keys=("zone_key", join_unique),
            nearest_zone_ids=("HAKGUDO_ID", join_unique),
            nearest_zone_names=("HAKGUDO_NM", join_unique),
            nearest_zone_edu_offices=("EDU_NM", join_unique),
            nearest_zone_match_statuses=("match_status", join_unique),
            nearest_zone_exclude_reasons=("exclude_reason", join_unique),
            nearest_usable_zone_count=("zone_usable", "sum"),
        )
    )
    nearest["nearest_zone_is_unique_usable"] = (
        nearest["equidistant_nearest_zone_count"].eq(1)
        & nearest["nearest_usable_zone_count"].eq(1)
    )
    return nearest


def build_candidate_pairs(
    no_zone_points: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    max_tolerance_m: float,
) -> pd.DataFrame:
    """최대 허용거리 안의 모든 대표점-통학구역 후보와 실제 거리를 만든다."""
    buffers = no_zone_points[["census_id", "geometry"]].copy()
    # Point.buffer(0)은 빈 도형이 될 수 있으므로, 0m만 요청했을 때는 원래 점을 쓴다.
    if max_tolerance_m > 0:
        buffers["geometry"] = buffers.geometry.buffer(max_tolerance_m)

    zone_columns = [
        "zone_key",
        "HAKGUDO_ID",
        "HAKGUDO_NM",
        "zone_usable",
        "geometry",
    ]
    pairs = gpd.sjoin(
        buffers,
        zones[zone_columns],
        how="inner",
        predicate="intersects",
    )
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "census_id",
                "zone_key",
                "HAKGUDO_ID",
                "HAKGUDO_NM",
                "zone_usable",
                "distance_m",
            ]
        )

    point_geometry = no_zone_points.set_index("census_id").geometry
    zone_geometry = zones.geometry
    pairs = pd.DataFrame(pairs.drop(columns="geometry", errors="ignore"))
    pairs["distance_m"] = [
        point_geometry.at[census_id].distance(zone_geometry.iloc[int(zone_index)])
        for census_id, zone_index in zip(pairs["census_id"], pairs["index_right"])
    ]
    pairs = pairs.loc[
        pairs["distance_m"].le(max_tolerance_m + DISTANCE_EPSILON_M)
    ].copy()
    pairs = pairs.drop(columns="index_right", errors="ignore")
    pairs = pairs.sort_values(["census_id", "distance_m", "zone_key"])

    duplicate_pair_count = int(pairs.duplicated(["census_id", "zone_key"]).sum())
    if duplicate_pair_count:
        raise ValueError(
            "동일 집계구-통학구역 후보가 중복되었습니다: "
            f"{duplicate_pair_count:,}행"
        )
    return pairs


def add_tolerance_results(
    detail: pd.DataFrame,
    pairs: pd.DataFrame,
    tolerances: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """허용거리별 후보 수·판정값과 요약표를 생성한다."""
    detail = detail.copy()
    summaries: list[dict[str, int | float]] = []
    total_points = len(detail)
    total_population = float(detail["elem_pop_est"].sum())

    if pairs.empty:
        candidate_text = pd.Series(dtype="string")
    else:
        pairs_for_text = pairs.copy()
        pairs_for_text["candidate_description"] = [
            f"{zone_id}:{distance:.12g}m"
            for zone_id, distance in zip(
                pairs_for_text["HAKGUDO_ID"], pairs_for_text["distance_m"]
            )
        ]
        candidate_text = pairs_for_text.groupby("census_id")[
            "candidate_description"
        ].agg("|".join)
    detail["candidate_zones_within_max_tolerance"] = detail["census_id"].map(
        candidate_text
    )

    for tolerance in tolerances:
        label = tolerance_label(tolerance)
        within = pairs.loc[
            pairs["distance_m"].le(tolerance + DISTANCE_EPSILON_M)
        ]
        if within.empty:
            candidate_count = pd.Series(dtype="int64")
            usable_count = pd.Series(dtype="int64")
        else:
            candidate_count = within.groupby("census_id")["zone_key"].nunique()
            usable_count = (
                within.loc[within["zone_usable"]]
                .groupby("census_id")["zone_key"]
                .nunique()
            )

        count_column = f"candidate_zone_count_{label}"
        usable_column = f"usable_zone_count_{label}"
        decision_column = f"decision_{label}"
        detail[count_column] = detail["census_id"].map(candidate_count).fillna(0).astype(int)
        detail[usable_column] = detail["census_id"].map(usable_count).fillna(0).astype(int)
        detail[decision_column] = np.select(
            [
                detail[count_column].eq(0),
                detail[count_column].gt(1),
                detail[count_column].eq(1) & detail[usable_column].eq(1),
            ],
            [
                "OUTSIDE_TOLERANCE",
                "AMBIGUOUS_MULTIPLE_ZONES",
                "UNIQUE_USABLE_ZONE",
            ],
            default="UNIQUE_UNUSABLE_ZONE",
        )

        status_count = detail[decision_column].value_counts()
        status_population = detail.groupby(decision_column)["elem_pop_est"].sum()
        partition_count = int(status_count.sum())
        partition_population = float(status_population.sum())
        if partition_count != total_points:
            raise ValueError(f"{tolerance:g}m 판정 건수 합계가 NO_ZONE 전체와 다릅니다.")
        if not np.isclose(partition_population, total_population, atol=1e-9):
            raise ValueError(f"{tolerance:g}m 판정 인구 합계가 NO_ZONE 전체와 다릅니다.")

        recovered_count = int(status_count.get("UNIQUE_USABLE_ZONE", 0))
        recovered_population = float(
            status_population.get("UNIQUE_USABLE_ZONE", 0.0)
        )
        summaries.append(
            {
                "tolerance_m": tolerance,
                "no_zone_point_count": total_points,
                "no_zone_elem_pop_est": total_population,
                "within_tolerance_point_count": int(
                    total_points - status_count.get("OUTSIDE_TOLERANCE", 0)
                ),
                "unique_usable_point_count": recovered_count,
                "unique_usable_elem_pop_est": recovered_population,
                "unique_unusable_point_count": int(
                    status_count.get("UNIQUE_UNUSABLE_ZONE", 0)
                ),
                "unique_unusable_elem_pop_est": float(
                    status_population.get("UNIQUE_UNUSABLE_ZONE", 0.0)
                ),
                "ambiguous_point_count": int(
                    status_count.get("AMBIGUOUS_MULTIPLE_ZONES", 0)
                ),
                "ambiguous_elem_pop_est": float(
                    status_population.get("AMBIGUOUS_MULTIPLE_ZONES", 0.0)
                ),
                "outside_tolerance_point_count": int(
                    status_count.get("OUTSIDE_TOLERANCE", 0)
                ),
                "outside_tolerance_elem_pop_est": float(
                    status_population.get("OUTSIDE_TOLERANCE", 0.0)
                ),
                "recoverable_point_rate": recovered_count / total_points,
                "recoverable_elem_pop_rate": (
                    recovered_population / total_population
                    if total_population > 0
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(summaries)
    if not summary["within_tolerance_point_count"].is_monotonic_increasing:
        raise ValueError("허용거리가 커졌는데 거리 내 후보 집계구 수가 감소했습니다.")
    return detail, summary


def prepare_detail_output(
    no_zone_points: gpd.GeoDataFrame,
    nearest: pd.DataFrame,
) -> pd.DataFrame:
    detail = pd.DataFrame(
        no_zone_points.drop(columns="geometry", errors="ignore")
    ).merge(nearest, on="census_id", how="left", validate="one_to_one")
    require_unique(detail, "census_id", "NO_ZONE 민감도 상세 결과")
    if detail["nearest_distance_m"].isna().any():
        raise ValueError("최근접 거리가 결측인 NO_ZONE 집계구가 있습니다.")
    return detail.sort_values(["nearest_distance_m", "census_id"]).reset_index(drop=True)


def log_summary(summary: pd.DataFrame) -> None:
    logging.info("허용거리별 누적 민감도 결과")
    logging.info(
        "%-10s %12s %12s %12s %12s",
        "허용거리",
        "복구가능",
        "복구인구",
        "복수후보",
        "거리밖",
    )
    for row in summary.itertuples(index=False):
        logging.info(
            "%-10s %12s %12.1f %12s %12s",
            f"{row.tolerance_m:g}m",
            f"{int(row.unique_usable_point_count):,}",
            row.unique_usable_elem_pop_est,
            f"{int(row.ambiguous_point_count):,}",
            f"{int(row.outside_tolerance_point_count):,}",
        )


def main() -> None:
    args = parse_args()
    tolerances = validate_tolerances(args.tolerances)
    project_root = args.project_root.resolve()
    raw_dir = project_root / "data" / "01_raw"
    processed_dir = project_root / "data" / "02_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("NO_ZONE 민감도 분석을 시작합니다.")
    logging.info("프로젝트 루트: %s", project_root)
    logging.info("허용거리(m): %s", tolerances)

    no_zone = load_no_zone_distribution(processed_dir)
    no_zone_points, zones, spatial_diagnostics = load_spatial_data(raw_dir, no_zone)
    for key, value in spatial_diagnostics.items():
        logging.info("%-38s %s", key, value)

    nearest = build_nearest_zone_table(no_zone_points, zones)
    pairs = build_candidate_pairs(no_zone_points, zones, max(tolerances))
    detail = prepare_detail_output(no_zone_points, nearest)
    detail, summary = add_tolerance_results(detail, pairs, tolerances)

    # 검증: 현재 NO_ZONE 집계구 수와 출력 상세 행 수가 정확히 일치해야 한다.
    if len(detail) != len(no_zone):
        raise ValueError(
            "상세 결과 행 수가 NO_ZONE 집계구 수와 다릅니다: "
            f"입력 {len(no_zone):,}, 출력 {len(detail):,}"
        )
    if (detail["nearest_distance_m"] < -DISTANCE_EPSILON_M).any():
        raise ValueError("상세 결과에 음수 거리가 있습니다.")

    detail_path = processed_dir / OUTPUT_DETAIL
    summary_path = processed_dir / OUTPUT_SUMMARY
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    log_summary(summary)
    logging.info("민감도 분석 완료")
    logging.info("상세 파일: %s", detail_path)
    logging.info("요약 파일: %s", summary_path)
    logging.info(
        "주의: 이 코드는 재배정·재보정을 하지 않았습니다. "
        "결과 확인 후 다음 단계에서 기준을 확정하세요."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("분석이 중단되었습니다: %s", exc)
        sys.exit(1)

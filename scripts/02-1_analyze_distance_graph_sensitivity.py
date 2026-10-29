"""
거리 기반 인접 그래프 민감도 분석 단계 (02-1)
================================================

목적
----
통폐합 후보 학교의 노드 모집단을 확정하고, 같은 교육지원청 내 모든 학교쌍의
직선거리를 계산한다. 여러 거리 임계값에서 간선·고립학교·연결요소·연결된
3개교 후보 수가 어떻게 달라지는지 비교하여 최종 간선 기준을 정할 근거를 만든다.

이 단계에서는 최종 인접 그래프와 3개교 후보 목록을 확정하지 않는다.
도로 거리·도로 이동시간·학교 수용력·존치학교 시나리오도 계산하지 않는다.

확정 노드 기준
-------------
1. `school_calibration_summary.csv`의 `calibration_check=PASS`
2. 설립형태가 공립
3. 작은학교가 Y
4. 본교분교구분이 본교
5. 교육지원청과 학교 위·경도가 존재

간선 민감도 기준
----------------
- 동일한 교육지원청 안의 학교쌍만 계산한다.
- 학교 좌표의 대권거리(Haversine)를 직선거리로 사용한다.
- 기본 임계값: 3km, 5km, 10km, 15km, 20km, 30km
- 임계값 D에서 직선거리 <= D인 학교쌍을 임시 간선으로 본다.
- 직선거리는 후보 선별용이며 실제 도로 통학거리로 해석하지 않는다.

연결된 3개교 후보 수 계산
------------------------
- 간선 2개로 연결되는 조합: CHAIN
- 간선 3개인 조합: TRIANGLE
- 간선 0~1개인 조합: 제외
- 이 단계에서는 후보 수만 계산하고 후보 목록은 02-2에서 생성한다.

입력 파일
---------
data/01_raw/elementary_master.csv
data/02_processed/school_calibration_summary.csv

출력 파일
---------
data/02_processed/distance_graph_nodes.csv
    - 확정 노드 학교 1개당 1행
    - 같은 교육지원청 내 1·2·3번째 최근접 학교와 직선거리

data/02_processed/same_office_school_distance_pairs.csv
    - 같은 교육지원청 내 학교쌍 1개당 1행
    - 02-2의 최종 간선 생성에 재사용하는 기본 거리표

data/02_processed/distance_graph_sensitivity_summary.csv
    - 전국 및 교육지원청별·임계값별 그래프 지표
    - 간선 수, 고립학교 수, 연결요소, CHAIN·TRIANGLE 후보 수 등

실행 방법
---------
프로젝트 루트에서:
    py scripts/02-1_analyze_distance_graph_sensitivity.py

다른 임계값을 사용할 때:
    py scripts/02-1_analyze_distance_graph_sensitivity.py --thresholds-km 3 5 8 10 15

소수점 처리
-----------
학교 간 거리는 계산 정밀도를 유지하며 CSV 저장 단계에서 반올림하지 않는다.
표·기사·시각화 단계에서만 표시용 반올림을 적용한다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 분석 설정
# -----------------------------------------------------------------------------

MASTER_FILE = "elementary_master.csv"
CALIBRATION_FILE = "school_calibration_summary.csv"

OUTPUT_NODES = "distance_graph_nodes.csv"
OUTPUT_PAIRS = "same_office_school_distance_pairs.csv"
OUTPUT_SENSITIVITY = "distance_graph_sensitivity_summary.csv"

DEFAULT_THRESHOLDS_KM = [3.0, 5.0, 10.0, 15.0, 20.0, 30.0]
EARTH_RADIUS_KM = 6371.0088
DISTANCE_EPSILON_KM = 1e-12

MASTER_REQUIRED = {
    "학교고유ID",
    "시도교육청",
    "교육지원청",
    "지역",
    "학교명",
    "총계",
    "설립형태",
    "본교분교구분",
    "위도",
    "경도",
    "지역구분",
    "작은학교",
    "일반교실수",
    "인구감소지역",
}

CALIBRATION_REQUIRED = {
    "school_key",
    "school_name",
    "school_edu_office",
    "actual_enrollment",
    "calibration_status",
    "calibration_check",
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="거리 기반 인접 그래프 민감도 분석")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="final_project 폴더 경로. 기본값은 scripts 폴더의 상위 폴더입니다.",
    )
    parser.add_argument(
        "--thresholds-km",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS_KM,
        help="민감도 분석에 사용할 직선거리 임계값(km)",
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


def validate_thresholds(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("거리 임계값을 하나 이상 입력해야 합니다.")
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("거리 임계값은 0보다 큰 유한한 숫자여야 합니다.")
    return sorted(set(float(value) for value in values))


def load_target_nodes(
    raw_dir: Path,
    processed_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """확정 기준을 적용해 거리 그래프의 학교 노드 모집단을 만든다."""
    master_path = raw_dir / MASTER_FILE
    calibration_path = processed_dir / CALIBRATION_FILE
    for path in [master_path, calibration_path]:
        if not path.exists():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    master = pd.read_csv(
        master_path,
        encoding="utf-8-sig",
        dtype={"학교고유ID": "string"},
        low_memory=False,
    )
    calibration = pd.read_csv(
        calibration_path,
        encoding="utf-8-sig",
        dtype={"school_key": "string"},
        low_memory=False,
    )
    require_columns(master, MASTER_REQUIRED, MASTER_FILE)
    require_columns(calibration, CALIBRATION_REQUIRED, CALIBRATION_FILE)

    master["학교고유ID"] = normalize_id(master["학교고유ID"])
    calibration["school_key"] = normalize_id(calibration["school_key"])
    require_unique(master, "학교고유ID", "초등학교 마스터")
    require_unique(calibration, "school_key", "학교 보정 요약")

    joined = master.merge(
        calibration[
            [
                "school_key",
                "school_name",
                "school_edu_office",
                "actual_enrollment",
                "calibration_status",
                "calibration_check",
            ]
        ],
        left_on="학교고유ID",
        right_on="school_key",
        how="left",
        validate="one_to_one",
    )

    public_small_main = (
        joined["설립형태"].eq("공립")
        & joined["작은학교"].eq("Y")
        & joined["본교분교구분"].eq("본교")
    )
    target_mask = public_small_main & joined["calibration_check"].eq("PASS")
    targets = joined.loc[target_mask].copy()

    # 분석대상에서 좌표나 교육지원청이 조용히 빠지지 않도록 명시적으로 중단한다.
    missing_office = targets["교육지원청"].isna() | targets["교육지원청"].astype(
        "string"
    ).str.strip().eq("")
    targets["위도"] = pd.to_numeric(targets["위도"], errors="coerce")
    targets["경도"] = pd.to_numeric(targets["경도"], errors="coerce")
    targets["총계"] = pd.to_numeric(targets["총계"], errors="coerce")
    targets["일반교실수"] = pd.to_numeric(targets["일반교실수"], errors="coerce")
    missing_coordinates = targets[["위도", "경도"]].isna().any(axis=1)
    invalid_coordinates = ~targets["위도"].between(-90, 90) | ~targets["경도"].between(
        -180, 180
    )
    if missing_office.any() or missing_coordinates.any() or invalid_coordinates.any():
        problems = targets.loc[
            missing_office | missing_coordinates | invalid_coordinates,
            ["학교고유ID", "학교명", "교육지원청", "위도", "경도"],
        ]
        raise ValueError(
            "그래프 대상 학교에 교육지원청·좌표 문제가 있습니다. "
            f"문제 학교 {len(problems):,}개, "
            f"예시 {problems.head(10).to_dict('records')}"
        )
    if targets["총계"].isna().any() or (targets["총계"] < 0).any():
        raise ValueError("그래프 대상 학교의 실제 재학생 수에 결측 또는 음수가 있습니다.")

    # 마스터와 보정표가 같은 학교 속성을 가리키는지 검증한다.
    name_mismatch = targets["학교명"].astype("string").ne(
        targets["school_name"].astype("string")
    )
    office_mismatch = targets["교육지원청"].astype("string").ne(
        targets["school_edu_office"].astype("string")
    )
    enrollment_mismatch = ~np.isclose(
        targets["총계"].astype(float),
        pd.to_numeric(targets["actual_enrollment"], errors="coerce"),
        atol=1e-9,
        rtol=0,
    )
    if name_mismatch.any() or office_mismatch.any() or enrollment_mismatch.any():
        raise ValueError(
            "elementary_master.csv와 school_calibration_summary.csv의 "
            "학교명·교육지원청·재학생 수가 일치하지 않는 대상 학교가 있습니다."
        )

    # 두 파일의 키가 일치함을 확인했으므로 오른쪽 보정표 키는 제거하고,
    # 마스터의 학교고유ID 하나만 최종 school_key로 사용한다.
    targets = targets.drop(columns="school_key")
    targets = targets.rename(
        columns={
            "학교고유ID": "school_key",
            "학교명": "school_name_master",
            "시도교육청": "province_edu_office",
            "교육지원청": "school_edu_office_master",
            "지역": "region_name",
            "총계": "actual_enrollment_master",
            "위도": "school_lat",
            "경도": "school_lon",
            "지역구분": "school_area_type",
            "작은학교": "small_school_flag",
            "본교분교구분": "main_branch_type",
            "일반교실수": "classroom_count",
            "인구감소지역": "population_decline_type",
        }
    )
    targets = targets[
        [
            "school_key",
            "school_name_master",
            "province_edu_office",
            "school_edu_office_master",
            "region_name",
            "actual_enrollment_master",
            "school_lat",
            "school_lon",
            "school_area_type",
            "small_school_flag",
            "main_branch_type",
            "classroom_count",
            "population_decline_type",
            "calibration_status",
            "calibration_check",
        ]
    ].rename(
        columns={
            "school_name_master": "school_name",
            "school_edu_office_master": "school_edu_office",
            "actual_enrollment_master": "actual_enrollment",
        }
    )
    targets = targets.sort_values(
        ["school_edu_office", "school_key"]
    ).reset_index(drop=True)
    require_unique(targets, "school_key", "거리 그래프 노드")

    diagnostics = {
        "master_school_count": len(master),
        "calibration_pass_school_count": int(
            calibration["calibration_check"].eq("PASS").sum()
        ),
        "public_small_main_before_pass_count": int(public_small_main.sum()),
        "public_small_main_review_excluded_count": int(
            (public_small_main & ~joined["calibration_check"].eq("PASS")).sum()
        ),
        "target_node_count": len(targets),
        "education_office_count": int(targets["school_edu_office"].nunique()),
    }
    return targets, diagnostics


def haversine_distance_km(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    """두 위·경도 배열의 대권거리(km)를 벡터 연산으로 계산한다."""
    lat1 = np.radians(lat1_deg.astype(float))
    lon1 = np.radians(lon1_deg.astype(float))
    lat2 = np.radians(lat2_deg.astype(float))
    lon2 = np.radians(lon2_deg.astype(float))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    central_angle = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * central_angle


def build_same_office_pairs(nodes: pd.DataFrame) -> pd.DataFrame:
    """같은 교육지원청 내 모든 비순서 학교쌍과 직선거리를 만든다."""
    pair_frames: list[pd.DataFrame] = []
    endpoint_columns = [
        "school_key",
        "school_name",
        "school_lat",
        "school_lon",
        "actual_enrollment",
        "school_area_type",
        "population_decline_type",
    ]

    for office, group in nodes.groupby("school_edu_office", sort=True):
        group = group.sort_values("school_key").reset_index(drop=True)
        node_count = len(group)
        if node_count < 2:
            continue
        index_a, index_b = np.triu_indices(node_count, k=1)
        distance = haversine_distance_km(
            group.loc[index_a, "school_lat"].to_numpy(),
            group.loc[index_a, "school_lon"].to_numpy(),
            group.loc[index_b, "school_lat"].to_numpy(),
            group.loc[index_b, "school_lon"].to_numpy(),
        )
        data: dict[str, object] = {"school_edu_office": office}
        for column in endpoint_columns:
            data[f"school_a_{column.removeprefix('school_')}"] = group.loc[
                index_a, column
            ].to_numpy()
            data[f"school_b_{column.removeprefix('school_')}"] = group.loc[
                index_b, column
            ].to_numpy()
        data["straight_distance_km"] = distance
        pair_frames.append(pd.DataFrame(data))

    if not pair_frames:
        raise ValueError("같은 교육지원청에 속한 학교쌍이 한 건도 없습니다.")
    pairs = pd.concat(pair_frames, ignore_index=True)
    pairs = pairs.sort_values(
        ["school_edu_office", "school_a_key", "school_b_key"]
    ).reset_index(drop=True)

    expected_pair_count = int(
        nodes.groupby("school_edu_office")
        .size()
        .map(lambda count: count * (count - 1) // 2)
        .sum()
    )
    if len(pairs) != expected_pair_count:
        raise ValueError(
            "학교쌍 수 검증 실패: "
            f"예상 {expected_pair_count:,}, 실제 {len(pairs):,}"
        )
    if pairs.duplicated(["school_a_key", "school_b_key"]).any():
        raise ValueError("같은 학교쌍이 중복 생성되었습니다.")
    if (pairs["school_a_key"] >= pairs["school_b_key"]).any():
        raise ValueError("학교쌍 ID 정렬 규칙 school_a_key < school_b_key를 위반했습니다.")
    if (pairs["straight_distance_km"] < -DISTANCE_EPSILON_KM).any():
        raise ValueError("학교 간 직선거리에서 음수가 발견되었습니다.")
    return pairs


def build_nearest_school_columns(
    nodes: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """각 학교의 같은 교육지원청 내 1·2·3번째 최근접 학교를 붙인다."""
    a_to_b = pairs[
        [
            "school_a_key",
            "school_b_key",
            "school_b_name",
            "straight_distance_km",
        ]
    ].rename(
        columns={
            "school_a_key": "school_key",
            "school_b_key": "neighbor_school_key",
            "school_b_name": "neighbor_school_name",
        }
    )
    b_to_a = pairs[
        [
            "school_b_key",
            "school_a_key",
            "school_a_name",
            "straight_distance_km",
        ]
    ].rename(
        columns={
            "school_b_key": "school_key",
            "school_a_key": "neighbor_school_key",
            "school_a_name": "neighbor_school_name",
        }
    )
    directed = pd.concat([a_to_b, b_to_a], ignore_index=True)
    directed = directed.sort_values(
        ["school_key", "straight_distance_km", "neighbor_school_key"]
    )
    directed["neighbor_rank"] = directed.groupby("school_key").cumcount() + 1

    result = nodes.copy()
    partner_count = directed.groupby("school_key")["neighbor_school_key"].nunique()
    result["same_office_partner_count"] = (
        result["school_key"].map(partner_count).fillna(0).astype(int)
    )
    for rank in [1, 2, 3]:
        nearest = directed.loc[
            directed["neighbor_rank"].eq(rank),
            [
                "school_key",
                "neighbor_school_key",
                "neighbor_school_name",
                "straight_distance_km",
            ],
        ].rename(
            columns={
                "neighbor_school_key": f"nearest_{rank}_school_key",
                "neighbor_school_name": f"nearest_{rank}_school_name",
                "straight_distance_km": f"nearest_{rank}_distance_km",
            }
        )
        result = result.merge(nearest, on="school_key", how="left", validate="one_to_one")
    require_unique(result, "school_key", "최근접 학교 결과")
    return result


def graph_statistics(
    node_ids: list[str],
    edges: pd.DataFrame,
) -> dict[str, int | float]:
    """외부 그래프 라이브러리 없이 무방향 그래프의 핵심 통계를 계산한다."""
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for row in edges[["school_a_key", "school_b_key"]].itertuples(index=False):
        adjacency[row.school_a_key].add(row.school_b_key)
        adjacency[row.school_b_key].add(row.school_a_key)

    degrees = {node_id: len(neighbors) for node_id, neighbors in adjacency.items()}
    isolated_count = sum(degree == 0 for degree in degrees.values())
    nodes_with_edge = len(node_ids) - isolated_count

    visited: set[str] = set()
    component_sizes: list[int] = []
    for start in node_ids:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)

    order = {node_id: index for index, node_id in enumerate(sorted(node_ids))}
    triangle_count = 0
    for node_a in node_ids:
        for node_b in adjacency[node_a]:
            if order[node_b] <= order[node_a]:
                continue
            common_neighbors = adjacency[node_a].intersection(adjacency[node_b])
            triangle_count += sum(
                order[node_c] > order[node_b] for node_c in common_neighbors
            )

    centered_neighbor_pairs = sum(
        degree * (degree - 1) // 2 for degree in degrees.values()
    )
    chain_count = centered_neighbor_pairs - 3 * triangle_count
    connected_triple_count = chain_count + triangle_count
    if chain_count < 0 or connected_triple_count < 0:
        raise ValueError("연결된 3개교 후보 수 계산에서 음수가 발생했습니다.")

    nodes_in_components_ge3 = sum(size for size in component_sizes if size >= 3)
    node_count = len(node_ids)
    edge_count = len(edges)
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "nodes_with_edge_count": nodes_with_edge,
        "isolated_node_count": isolated_count,
        "node_edge_coverage_rate": nodes_with_edge / node_count if node_count else np.nan,
        "connected_component_count": len(component_sizes),
        "largest_component_size": max(component_sizes, default=0),
        "nodes_in_components_ge3_count": nodes_in_components_ge3,
        "triple_eligible_node_rate": (
            nodes_in_components_ge3 / node_count if node_count else np.nan
        ),
        "connected_three_school_candidate_count": connected_triple_count,
        "chain_candidate_count": chain_count,
        "triangle_candidate_count": triangle_count,
    }


def summarize_graph_sensitivity(
    nodes: pd.DataFrame,
    pairs: pd.DataFrame,
    thresholds_km: list[float],
) -> pd.DataFrame:
    """전국 및 교육지원청별로 임계값별 그래프 통계를 만든다."""
    records: list[dict[str, object]] = []
    office_node_ids = {
        office: group["school_key"].tolist()
        for office, group in nodes.groupby("school_edu_office", sort=True)
    }

    for threshold in thresholds_km:
        edges = pairs.loc[
            pairs["straight_distance_km"].le(threshold + DISTANCE_EPSILON_KM)
        ].copy()
        scopes: list[tuple[str, str, list[str], pd.DataFrame]] = [
            ("NATIONAL", "ALL", nodes["school_key"].tolist(), edges)
        ]
        for office, node_ids in office_node_ids.items():
            office_edges = edges.loc[edges["school_edu_office"].eq(office)]
            scopes.append(("EDUCATION_OFFICE", office, node_ids, office_edges))

        for scope_type, office, node_ids, scope_edges in scopes:
            stats = graph_statistics(node_ids, scope_edges)
            if scope_edges.empty:
                distance_mean = np.nan
                distance_median = np.nan
                distance_max = np.nan
            else:
                distance_mean = float(scope_edges["straight_distance_km"].mean())
                distance_median = float(scope_edges["straight_distance_km"].median())
                distance_max = float(scope_edges["straight_distance_km"].max())
            records.append(
                {
                    "scope_type": scope_type,
                    "school_edu_office": office,
                    "threshold_km": threshold,
                    **stats,
                    "edge_distance_mean_km": distance_mean,
                    "edge_distance_median_km": distance_median,
                    "edge_distance_max_km": distance_max,
                }
            )

    summary = pd.DataFrame(records)
    national = summary.loc[summary["scope_type"].eq("NATIONAL")].sort_values(
        "threshold_km"
    )
    for column in [
        "edge_count",
        "nodes_with_edge_count",
        "connected_three_school_candidate_count",
        "triangle_candidate_count",
    ]:
        if not national[column].is_monotonic_increasing:
            raise ValueError(f"임계값 증가에 따라 {column}이 감소했습니다.")

    office_summary = summary.loc[summary["scope_type"].eq("EDUCATION_OFFICE")]
    for threshold in thresholds_km:
        national_row = national.loc[national["threshold_km"].eq(threshold)].iloc[0]
        office_rows = office_summary.loc[office_summary["threshold_km"].eq(threshold)]
        for column in ["node_count", "edge_count", "connected_three_school_candidate_count"]:
            office_total = int(office_rows[column].sum())
            national_total = int(national_row[column])
            if office_total != national_total:
                raise ValueError(
                    f"{threshold:g}km {column} 전국값과 교육지원청 합계가 다릅니다: "
                    f"전국 {national_total:,}, 지원청합 {office_total:,}"
                )
    return summary.sort_values(
        ["scope_type", "school_edu_office", "threshold_km"]
    ).reset_index(drop=True)


def log_nearest_distance_distribution(nodes_with_nearest: pd.DataFrame) -> None:
    logging.info("최근접 학교 거리 분포(km)")
    logging.info("%-8s %10s %10s %10s %10s %10s", "순위", "중앙값", "75%", "90%", "95%", "최대")
    for rank in [1, 2, 3]:
        values = nodes_with_nearest[f"nearest_{rank}_distance_km"].dropna()
        if values.empty:
            continue
        quantiles = values.quantile([0.5, 0.75, 0.9, 0.95, 1.0])
        logging.info(
            "%-8s %10.3f %10.3f %10.3f %10.3f %10.3f",
            f"{rank}번째",
            quantiles.loc[0.5],
            quantiles.loc[0.75],
            quantiles.loc[0.9],
            quantiles.loc[0.95],
            quantiles.loc[1.0],
        )


def log_national_sensitivity(summary: pd.DataFrame) -> None:
    national = summary.loc[summary["scope_type"].eq("NATIONAL")].sort_values(
        "threshold_km"
    )
    logging.info("전국 거리 그래프 민감도 결과")
    logging.info(
        "%-8s %10s %10s %10s %12s %12s %12s",
        "기준",
        "간선",
        "고립학교",
        "최대요소",
        "3개교후보",
        "CHAIN",
        "TRIANGLE",
    )
    for row in national.itertuples(index=False):
        logging.info(
            "%-8s %10s %10s %10s %12s %12s %12s",
            f"{row.threshold_km:g}km",
            f"{int(row.edge_count):,}",
            f"{int(row.isolated_node_count):,}",
            f"{int(row.largest_component_size):,}",
            f"{int(row.connected_three_school_candidate_count):,}",
            f"{int(row.chain_candidate_count):,}",
            f"{int(row.triangle_candidate_count):,}",
        )


def main() -> None:
    args = parse_args()
    thresholds_km = validate_thresholds(args.thresholds_km)
    project_root = args.project_root.resolve()
    raw_dir = project_root / "data" / "01_raw"
    processed_dir = project_root / "data" / "02_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("거리 기반 인접 그래프 민감도 분석을 시작합니다.")
    logging.info("프로젝트 루트: %s", project_root)
    logging.info("거리 임계값(km): %s", thresholds_km)

    nodes, node_diagnostics = load_target_nodes(raw_dir, processed_dir)
    for key, value in node_diagnostics.items():
        logging.info("%-44s %s", key, f"{value:,}")

    pairs = build_same_office_pairs(nodes)
    nodes_with_nearest = build_nearest_school_columns(nodes, pairs)
    sensitivity = summarize_graph_sensitivity(nodes, pairs, thresholds_km)

    zero_distance_pair_count = int(
        pairs["straight_distance_km"].le(DISTANCE_EPSILON_KM).sum()
    )
    logging.info("%-44s %s", "같은 교육지원청 학교쌍 수", f"{len(pairs):,}")
    logging.info("%-44s %s", "좌표가 사실상 같은 학교쌍", f"{zero_distance_pair_count:,}")
    log_nearest_distance_distribution(nodes_with_nearest)
    log_national_sensitivity(sensitivity)

    nodes_path = processed_dir / OUTPUT_NODES
    pairs_path = processed_dir / OUTPUT_PAIRS
    sensitivity_path = processed_dir / OUTPUT_SENSITIVITY
    nodes_with_nearest.to_csv(nodes_path, index=False, encoding="utf-8-sig")
    pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    sensitivity.to_csv(sensitivity_path, index=False, encoding="utf-8-sig")

    logging.info("거리 그래프 민감도 분석 완료")
    logging.info("노드·최근접 거리 파일: %s", nodes_path)
    logging.info("같은 교육지원청 학교쌍 거리표: %s", pairs_path)
    logging.info("임계값 민감도 요약: %s", sensitivity_path)
    logging.info(
        "주의: 아직 최종 거리 임계값·간선·3개교 후보 목록을 확정하지 않았습니다. "
        "민감도 결과를 검토한 뒤 02-2에서 생성하세요."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("분석이 중단되었습니다: %s", exc)
        sys.exit(1)

"""
10km 거리 인접 그래프 및 연결된 3개교 후보 생성 단계 (02-2)
=========================================================

목적
----
02-1 민감도 분석에서 확정한 단일 기준인 10km를 적용하여 다음 두 작업을 수행한다.

1. 같은 교육지원청에 속하면서 학교 간 직선거리가 10km 이하인 학교쌍을
   최종 그래프 간선으로 확정한다.
2. 최종 그래프에서 세 학교가 하나로 연결되는 모든 비순서 3개교 조합을
   생성한다. 간선 2개의 CHAIN과 간선 3개의 TRIANGLE을 모두 포함한다.

이 단계의 그래프는 머신러닝 모형이 아니라 다음 두 표로 표현되는 관계 구조다.
- 노드: 02-1의 distance_graph_nodes.csv
- 간선: 이 코드가 만드는 school_adjacency_edges.csv

확정 기준
---------
- 노드 모집단: 02-1에서 확정한 PASS + 공립 + 소규모 + 본교 2,503개교
- 간선: 같은 교육지원청 AND Haversine 직선거리 <= 10km
- 후보: 세 학교의 유도 부분그래프에 간선이 2개 이상
- CHAIN: A-B-C처럼 간선 2개로 연결
- TRIANGLE: 세 학교쌍 모두 간선으로 연결
- 한 학교가 여러 후보에 중복되는 것은 이 단계에서는 정상

주의
----
- 10km는 '각 그래프 간선'의 기준이다. CHAIN의 양 끝 학교는 직접 간선이
  없으므로 서로의 직선거리가 10km를 초과할 수 있다.
- 직선거리는 후보 선별용이다. 실제 도로거리·이동시간·수용력·존치학교
  선택·학교 중복 없는 최종 군집 선정은 뒤 단계에서 수행한다.
- 거리 계산값은 반올림하지 않고 저장한다. 반올림은 표시 단계에서만 한다.

입력 파일
---------
data/02_processed/distance_graph_nodes.csv
data/02_processed/same_office_school_distance_pairs.csv
data/02_processed/distance_graph_sensitivity_summary.csv  # 있으면 결과 교차검증

출력 파일
---------
data/02_processed/school_adjacency_edges.csv
    - 최종 간선 1개당 1행
    - 10km 이하 학교쌍, 연결요소 ID·크기, 양 끝 학교 속성

data/02_processed/connected_three_school_candidates.csv
    - 연결된 3개교 후보 1개당 1행
    - CHAIN/TRIANGLE, 세 학교 속성, 세 쌍의 거리와 간선 여부

실행 방법
---------
프로젝트 루트에서:
    py scripts/02-2_build_adjacency_graph_and_triples.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 분석 설정: 민감도 분석을 거쳐 10km 단일 기준으로 확정
# -----------------------------------------------------------------------------

GRAPH_THRESHOLD_KM = 10.0
DISTANCE_EPSILON_KM = 1e-12

NODES_FILE = "distance_graph_nodes.csv"
PAIRS_FILE = "same_office_school_distance_pairs.csv"
SENSITIVITY_FILE = "distance_graph_sensitivity_summary.csv"

OUTPUT_EDGES = "school_adjacency_edges.csv"
OUTPUT_CANDIDATES = "connected_three_school_candidates.csv"

NODES_REQUIRED = {
    "school_key",
    "school_name",
    "province_edu_office",
    "school_edu_office",
    "region_name",
    "actual_enrollment",
    "school_lat",
    "school_lon",
    "school_area_type",
    "small_school_flag",
    "main_branch_type",
    "classroom_count",
    "population_decline_type",
    "calibration_status",
    "calibration_check",
}

PAIRS_REQUIRED = {
    "school_edu_office",
    "school_a_key",
    "school_b_key",
    "school_a_name",
    "school_b_name",
    "school_a_lat",
    "school_b_lat",
    "school_a_lon",
    "school_b_lon",
    "school_a_actual_enrollment",
    "school_b_actual_enrollment",
    "school_a_area_type",
    "school_b_area_type",
    "school_a_population_decline_type",
    "school_b_population_decline_type",
    "straight_distance_km",
}

SENSITIVITY_REQUIRED = {
    "scope_type",
    "school_edu_office",
    "threshold_km",
    "node_count",
    "edge_count",
    "connected_component_count",
    "isolated_node_count",
    "connected_three_school_candidate_count",
    "chain_candidate_count",
    "triangle_candidate_count",
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="10km 거리 인접 그래프 및 연결된 3개교 후보 생성"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="final_project 폴더 경로. 기본값은 scripts 폴더의 상위 폴더입니다.",
    )
    parser.add_argument(
        "--nodes-file",
        type=Path,
        default=None,
        help="노드 CSV 경로. 생략하면 data/02_processed의 기본 파일을 사용합니다.",
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=None,
        help="학교쌍 거리 CSV 경로. 생략하면 data/02_processed의 기본 파일을 사용합니다.",
    )
    parser.add_argument(
        "--sensitivity-file",
        type=Path,
        default=None,
        help="민감도 요약 CSV 경로. 파일이 존재하면 10km 결과를 교차검증합니다.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="출력 폴더. 생략하면 data/02_processed를 사용합니다.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"[{label}] 필수 컬럼이 없습니다: {missing}")


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    project_root = args.project_root.resolve()
    processed_dir = project_root / "data" / "02_processed"
    nodes_path = (args.nodes_file or processed_dir / NODES_FILE).resolve()
    pairs_path = (args.pairs_file or processed_dir / PAIRS_FILE).resolve()
    sensitivity_path = (
        args.sensitivity_file or processed_dir / SENSITIVITY_FILE
    ).resolve()
    output_dir = (args.output_dir or processed_dir).resolve()
    return nodes_path, pairs_path, sensitivity_path, output_dir


def load_and_validate_inputs(
    nodes_path: Path,
    pairs_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [nodes_path, pairs_path]:
        if not path.exists():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    nodes = pd.read_csv(
        nodes_path,
        encoding="utf-8-sig",
        dtype={"school_key": "string"},
        low_memory=False,
    )
    pairs = pd.read_csv(
        pairs_path,
        encoding="utf-8-sig",
        dtype={"school_a_key": "string", "school_b_key": "string"},
        low_memory=False,
    )
    require_columns(nodes, NODES_REQUIRED, NODES_FILE)
    require_columns(pairs, PAIRS_REQUIRED, PAIRS_FILE)

    nodes["school_key"] = normalize_id(nodes["school_key"])
    pairs["school_a_key"] = normalize_id(pairs["school_a_key"])
    pairs["school_b_key"] = normalize_id(pairs["school_b_key"])
    for frame, columns in [
        (nodes, ["school_edu_office", "school_name"]),
        (pairs, ["school_edu_office"]),
    ]:
        for column in columns:
            frame[column] = frame[column].astype("string").str.strip()

    if nodes["school_key"].isna().any() or nodes.duplicated("school_key").any():
        raise ValueError("노드의 school_key에 결측 또는 중복이 있습니다.")
    if not nodes["calibration_check"].eq("PASS").all():
        raise ValueError("노드 파일에 calibration_check가 PASS가 아닌 학교가 있습니다.")
    if not nodes["small_school_flag"].eq("Y").all():
        raise ValueError("노드 파일에 작은학교가 아닌 학교가 있습니다.")
    if not nodes["main_branch_type"].eq("본교").all():
        raise ValueError("노드 파일에 분교가 포함되어 있습니다.")

    if pairs[["school_a_key", "school_b_key"]].isna().any().any():
        raise ValueError("학교쌍 키에 결측이 있습니다.")
    if pairs.duplicated(["school_a_key", "school_b_key"]).any():
        raise ValueError("학교쌍이 중복되어 있습니다.")
    if (pairs["school_a_key"] >= pairs["school_b_key"]).any():
        raise ValueError("학교쌍 정렬 규칙 school_a_key < school_b_key를 위반했습니다.")

    pairs["straight_distance_km"] = pd.to_numeric(
        pairs["straight_distance_km"], errors="coerce"
    )
    if (
        pairs["straight_distance_km"].isna().any()
        or (pairs["straight_distance_km"] < -DISTANCE_EPSILON_KM).any()
    ):
        raise ValueError("학교쌍 직선거리에 결측 또는 음수가 있습니다.")

    node_keys = set(nodes["school_key"])
    pair_keys = set(pairs["school_a_key"]).union(pairs["school_b_key"])
    if not pair_keys.issubset(node_keys):
        unknown = sorted(pair_keys - node_keys)[:10]
        raise ValueError(f"노드 파일에 없는 학교키가 거리표에 있습니다: {unknown}")

    office_lookup = nodes.set_index("school_key")["school_edu_office"].to_dict()
    office_a = pairs["school_a_key"].map(office_lookup)
    office_b = pairs["school_b_key"].map(office_lookup)
    office_mismatch = (
        office_a.ne(office_b)
        | office_a.ne(pairs["school_edu_office"])
        | office_b.ne(pairs["school_edu_office"])
    )
    if office_mismatch.any():
        raise ValueError("거리표에 서로 다른 교육지원청 학교쌍이 있습니다.")

    # 같은 교육지원청의 모든 비순서 학교쌍 nC2가 들어 있는지 지원청별로 검증한다.
    node_counts = nodes.groupby("school_edu_office").size()
    expected_pair_counts = node_counts.map(lambda n: n * (n - 1) // 2)
    actual_pair_counts = pairs.groupby("school_edu_office").size().reindex(
        node_counts.index, fill_value=0
    )
    mismatch = expected_pair_counts.ne(actual_pair_counts)
    if mismatch.any():
        examples = pd.DataFrame(
            {
                "expected": expected_pair_counts[mismatch],
                "actual": actual_pair_counts[mismatch],
            }
        ).head(10)
        raise ValueError(
            "교육지원청별 학교쌍 수가 nC2와 일치하지 않습니다: "
            f"{examples.to_dict('index')}"
        )

    return (
        nodes.sort_values(["school_edu_office", "school_key"]).reset_index(drop=True),
        pairs.sort_values(
            ["school_edu_office", "school_a_key", "school_b_key"]
        ).reset_index(drop=True),
    )


def canonical_pair(key_a: str, key_b: str) -> tuple[str, str]:
    return (key_a, key_b) if key_a < key_b else (key_b, key_a)


def build_adjacency(
    node_keys: list[str],
    edges: pd.DataFrame,
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {key: set() for key in node_keys}
    for row in edges[["school_a_key", "school_b_key"]].itertuples(index=False):
        adjacency[row.school_a_key].add(row.school_b_key)
        adjacency[row.school_b_key].add(row.school_a_key)
    return adjacency


def build_components(
    nodes: pd.DataFrame,
    adjacency: dict[str, set[str]],
) -> tuple[dict[str, str], dict[str, int], int]:
    """고립학교까지 포함한 연결요소를 안정적인 ID로 만든다."""
    office_lookup = nodes.set_index("school_key")["school_edu_office"].to_dict()
    ordered_nodes = nodes.sort_values(
        ["school_edu_office", "school_key"]
    )["school_key"].tolist()
    visited: set[str] = set()
    components: list[tuple[str, tuple[str, ...]]] = []

    for start in ordered_nodes:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        members: list[str] = []
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        members_tuple = tuple(sorted(members))
        offices = {office_lookup[key] for key in members_tuple}
        if len(offices) != 1:
            raise ValueError("한 연결요소 안에 여러 교육지원청이 포함되었습니다.")
        components.append((next(iter(offices)), members_tuple))

    components.sort(key=lambda item: (item[0], item[1][0]))
    component_id_by_node: dict[str, str] = {}
    component_size_by_node: dict[str, int] = {}
    for index, (_, members) in enumerate(components, start=1):
        component_id = f"CC{index:04d}"
        size = len(members)
        for key in members:
            component_id_by_node[key] = component_id
            component_size_by_node[key] = size
    return component_id_by_node, component_size_by_node, len(components)


def create_final_edges(
    nodes: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, set[str]], int]:
    edges = pairs.loc[
        pairs["straight_distance_km"].le(
            GRAPH_THRESHOLD_KM + DISTANCE_EPSILON_KM
        )
    ].copy()
    if edges.empty:
        raise ValueError("10km 기준 간선이 한 건도 없습니다.")
    if (edges["straight_distance_km"] > GRAPH_THRESHOLD_KM + DISTANCE_EPSILON_KM).any():
        raise ValueError("10km를 초과하는 간선이 생성되었습니다.")

    adjacency = build_adjacency(nodes["school_key"].tolist(), edges)
    component_id, component_size, component_count = build_components(nodes, adjacency)
    edges.insert(0, "graph_threshold_km", GRAPH_THRESHOLD_KM)
    edges.insert(1, "component_id", edges["school_a_key"].map(component_id))
    edges.insert(2, "component_size", edges["school_a_key"].map(component_size))

    component_b = edges["school_b_key"].map(component_id)
    size_b = edges["school_b_key"].map(component_size)
    if edges["component_id"].ne(component_b).any() or edges["component_size"].ne(size_b).any():
        raise ValueError("간선 양 끝 학교의 연결요소가 일치하지 않습니다.")

    return (
        edges.sort_values(
            ["school_edu_office", "component_id", "school_a_key", "school_b_key"]
        ).reset_index(drop=True),
        adjacency,
        component_count,
    )


def enumerate_connected_triples(
    nodes: pd.DataFrame,
    pairs: pd.DataFrame,
    adjacency: dict[str, set[str]],
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """이웃 두 개를 가진 중심학교를 기준으로 후보를 만들고 중복을 제거한다."""
    triple_keys: set[tuple[str, str, str]] = set()
    for center in sorted(adjacency):
        for neighbor_a, neighbor_b in combinations(sorted(adjacency[center]), 2):
            triple_keys.add(tuple(sorted((center, neighbor_a, neighbor_b))))

    if not triple_keys:
        raise ValueError("연결된 3개교 후보가 한 건도 생성되지 않았습니다.")

    node_lookup = nodes.set_index("school_key").to_dict("index")
    distance_lookup = {
        (row.school_a_key, row.school_b_key): float(row.straight_distance_km)
        for row in pairs[
            ["school_a_key", "school_b_key", "straight_distance_km"]
        ].itertuples(index=False)
    }
    component_lookup = edges.set_index("school_a_key")["component_id"].to_dict()
    component_size_lookup = edges.set_index("school_a_key")["component_size"].to_dict()
    # school_a로 한 번도 나오지 않는 노드도 있으므로 양 끝에서 완성한다.
    component_lookup.update(edges.set_index("school_b_key")["component_id"].to_dict())
    component_size_lookup.update(
        edges.set_index("school_b_key")["component_size"].to_dict()
    )

    records: list[dict[str, object]] = []
    ordered_triples = sorted(
        triple_keys,
        key=lambda triple: (
            node_lookup[triple[0]]["school_edu_office"],
            triple[0],
            triple[1],
            triple[2],
        ),
    )

    for index, (key_1, key_2, key_3) in enumerate(ordered_triples, start=1):
        keys = [key_1, key_2, key_3]
        attrs = [node_lookup[key] for key in keys]
        offices = {attr["school_edu_office"] for attr in attrs}
        if len(offices) != 1:
            raise ValueError("3개교 후보에 여러 교육지원청이 섞였습니다.")

        pair_12 = canonical_pair(key_1, key_2)
        pair_13 = canonical_pair(key_1, key_3)
        pair_23 = canonical_pair(key_2, key_3)
        distances = [
            distance_lookup[pair_12],
            distance_lookup[pair_13],
            distance_lookup[pair_23],
        ]
        edge_flags = [
            key_2 in adjacency[key_1],
            key_3 in adjacency[key_1],
            key_3 in adjacency[key_2],
        ]
        induced_edge_count = int(sum(edge_flags))
        if induced_edge_count not in {2, 3}:
            raise ValueError("연결되지 않은 조합이 3개교 후보에 포함되었습니다.")
        candidate_type = "TRIANGLE" if induced_edge_count == 3 else "CHAIN"

        connector_keys = [
            key
            for key in keys
            if sum(other in adjacency[key] for other in keys if other != key) == 2
        ]
        if candidate_type == "CHAIN":
            if len(connector_keys) != 1:
                raise ValueError("CHAIN 후보의 연결 중심학교를 하나로 정할 수 없습니다.")
            connector_key = connector_keys[0]
            connector_name = node_lookup[connector_key]["school_name"]
        else:
            connector_key = pd.NA
            connector_name = pd.NA

        graph_edge_distance_sum = sum(
            distance for distance, is_edge in zip(distances, edge_flags) if is_edge
        )
        record: dict[str, object] = {
            "candidate_id": f"C3_{index:08d}",
            "graph_threshold_km": GRAPH_THRESHOLD_KM,
            "school_edu_office": next(iter(offices)),
            "component_id": component_lookup[key_1],
            "component_size": component_size_lookup[key_1],
            "candidate_type": candidate_type,
            "induced_edge_count": induced_edge_count,
            "connector_school_key": connector_key,
            "connector_school_name": connector_name,
        }
        for position, (key, attr) in enumerate(zip(keys, attrs), start=1):
            record.update(
                {
                    f"school_{position}_key": key,
                    f"school_{position}_name": attr["school_name"],
                    f"school_{position}_lat": attr["school_lat"],
                    f"school_{position}_lon": attr["school_lon"],
                    f"school_{position}_actual_enrollment": attr[
                        "actual_enrollment"
                    ],
                    f"school_{position}_area_type": attr["school_area_type"],
                    f"school_{position}_population_decline_type": attr[
                        "population_decline_type"
                    ],
                }
            )
        record.update(
            {
                "distance_12_km": distances[0],
                "distance_13_km": distances[1],
                "distance_23_km": distances[2],
                "edge_12": edge_flags[0],
                "edge_13": edge_flags[1],
                "edge_23": edge_flags[2],
                "min_pair_distance_km": min(distances),
                "mean_pair_distance_km": float(np.mean(distances)),
                "max_pair_distance_km": max(distances),
                "graph_edge_distance_sum_km": graph_edge_distance_sum,
                "total_actual_enrollment": sum(
                    float(attr["actual_enrollment"]) for attr in attrs
                ),
            }
        )
        records.append(record)

    candidates = pd.DataFrame(records)
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("candidate_id가 중복되었습니다.")
    if candidates.duplicated(
        ["school_1_key", "school_2_key", "school_3_key"]
    ).any():
        raise ValueError("동일한 3개교 조합이 중복 생성되었습니다.")
    if not (
        (candidates["school_1_key"] < candidates["school_2_key"])
        & (candidates["school_2_key"] < candidates["school_3_key"])
    ).all():
        raise ValueError("후보 학교키 정렬 규칙을 위반했습니다.")
    chain_mask = candidates["candidate_type"].eq("CHAIN")
    triangle_mask = candidates["candidate_type"].eq("TRIANGLE")
    if candidates.loc[
        chain_mask, ["connector_school_key", "connector_school_name"]
    ].isna().any().any():
        raise ValueError("CHAIN 후보의 연결 중심학교가 누락되었습니다.")
    if candidates.loc[
        triangle_mask, ["connector_school_key", "connector_school_name"]
    ].notna().any().any():
        raise ValueError("TRIANGLE 후보에는 단일 연결 중심학교를 기록하지 않아야 합니다.")
    non_connector_columns = [
        column
        for column in candidates.columns
        if column not in {"connector_school_key", "connector_school_name"}
    ]
    if candidates[non_connector_columns].isna().any().any():
        missing = candidates[non_connector_columns].isna().sum()
        raise ValueError(
            "3개교 후보의 필수 열에 결측이 있습니다: "
            f"{missing[missing.gt(0)].to_dict()}"
        )

    # 그래프 공식으로 독립 계산한 후보 수와 전수 열거 결과를 교차검증한다.
    degrees = {key: len(neighbors) for key, neighbors in adjacency.items()}
    centered_neighbor_pairs = sum(
        degree * (degree - 1) // 2 for degree in degrees.values()
    )
    triangle_count = int(candidates["candidate_type"].eq("TRIANGLE").sum())
    chain_count = centered_neighbor_pairs - 3 * triangle_count
    expected_total = chain_count + triangle_count
    actual_chain = int(candidates["candidate_type"].eq("CHAIN").sum())
    if chain_count != actual_chain or expected_total != len(candidates):
        raise ValueError(
            "그래프 공식과 후보 전수 열거 결과가 다릅니다: "
            f"공식 CHAIN {chain_count:,}, 실제 {actual_chain:,}, "
            f"공식 전체 {expected_total:,}, 실제 {len(candidates):,}"
        )
    return candidates


def validate_against_sensitivity(
    sensitivity_path: Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    candidates: pd.DataFrame,
    component_count: int,
    adjacency: dict[str, set[str]],
) -> None:
    if not sensitivity_path.exists():
        logging.warning(
            "민감도 요약 파일이 없어 02-1의 10km 결과와 교차검증하지 않습니다: %s",
            sensitivity_path,
        )
        return
    sensitivity = pd.read_csv(sensitivity_path, encoding="utf-8-sig")
    require_columns(sensitivity, SENSITIVITY_REQUIRED, SENSITIVITY_FILE)
    national = sensitivity.loc[
        sensitivity["scope_type"].eq("NATIONAL")
        & np.isclose(
            pd.to_numeric(sensitivity["threshold_km"], errors="coerce"),
            GRAPH_THRESHOLD_KM,
            atol=1e-12,
            rtol=0,
        )
    ]
    if len(national) != 1:
        raise ValueError("민감도 요약에서 전국 10km 행을 정확히 하나 찾지 못했습니다.")
    row = national.iloc[0]
    actual = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "connected_component_count": component_count,
        "isolated_node_count": sum(not neighbors for neighbors in adjacency.values()),
        "connected_three_school_candidate_count": len(candidates),
        "chain_candidate_count": int(candidates["candidate_type"].eq("CHAIN").sum()),
        "triangle_candidate_count": int(
            candidates["candidate_type"].eq("TRIANGLE").sum()
        ),
    }
    mismatches = {
        key: {"sensitivity": int(row[key]), "generated": value}
        for key, value in actual.items()
        if int(row[key]) != value
    }
    if mismatches:
        raise ValueError(f"02-1 민감도 결과와 02-2 생성 결과가 다릅니다: {mismatches}")


def log_results(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    candidates: pd.DataFrame,
    adjacency: dict[str, set[str]],
    component_count: int,
) -> None:
    isolated_count = sum(not neighbors for neighbors in adjacency.values())
    chain_count = int(candidates["candidate_type"].eq("CHAIN").sum())
    triangle_count = int(candidates["candidate_type"].eq("TRIANGLE").sum())
    chain_max_over_10 = int(
        (
            candidates["candidate_type"].eq("CHAIN")
            & candidates["max_pair_distance_km"].gt(
                GRAPH_THRESHOLD_KM + DISTANCE_EPSILON_KM
            )
        ).sum()
    )
    participating_schools = pd.unique(
        candidates[["school_1_key", "school_2_key", "school_3_key"]].to_numpy().ravel()
    )

    logging.info("%-42s %s", "확정 노드 수", f"{len(nodes):,}")
    logging.info("%-42s %s", "10km 최종 간선 수", f"{len(edges):,}")
    logging.info("%-42s %s", "연결요소 수(고립학교 포함)", f"{component_count:,}")
    logging.info("%-42s %s", "고립학교 수", f"{isolated_count:,}")
    logging.info("%-42s %s", "연결된 3개교 후보 수", f"{len(candidates):,}")
    logging.info("%-42s %s", "CHAIN 후보 수", f"{chain_count:,}")
    logging.info("%-42s %s", "TRIANGLE 후보 수", f"{triangle_count:,}")
    logging.info("%-42s %s", "후보에 1회 이상 참여한 학교 수", f"{len(participating_schools):,}")
    logging.info(
        "%-42s %s",
        "양 끝 거리가 10km 초과인 CHAIN 수",
        f"{chain_max_over_10:,}",
    )
    logging.info(
        "%-42s %.3f / %.3f / %.3f",
        "후보 최대 학교쌍 거리 중앙값/90%/최대(km)",
        candidates["max_pair_distance_km"].median(),
        candidates["max_pair_distance_km"].quantile(0.9),
        candidates["max_pair_distance_km"].max(),
    )


def main() -> None:
    args = parse_args()
    nodes_path, pairs_path, sensitivity_path, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("10km 인접 그래프와 연결된 3개교 후보 생성을 시작합니다.")
    logging.info("노드 파일: %s", nodes_path)
    logging.info("학교쌍 거리표: %s", pairs_path)
    logging.info("확정 거리 임계값: %.1fkm", GRAPH_THRESHOLD_KM)

    nodes, pairs = load_and_validate_inputs(nodes_path, pairs_path)
    edges, adjacency, component_count = create_final_edges(nodes, pairs)
    candidates = enumerate_connected_triples(nodes, pairs, adjacency, edges)
    validate_against_sensitivity(
        sensitivity_path,
        nodes,
        edges,
        candidates,
        component_count,
        adjacency,
    )
    log_results(nodes, edges, candidates, adjacency, component_count)

    edges_path = output_dir / OUTPUT_EDGES
    candidates_path = output_dir / OUTPUT_CANDIDATES
    edges.to_csv(edges_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")

    logging.info("02-2 작업 완료")
    logging.info("최종 인접 그래프 간선: %s", edges_path)
    logging.info("연결된 3개교 후보: %s", candidates_path)
    logging.info(
        "주의: 이 후보는 아직 도로 이동시간·수용력·학교 중복 제거를 적용하지 않은 "
        "전수 후보입니다."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("분석이 중단되었습니다: %s", exc)
        sys.exit(1)

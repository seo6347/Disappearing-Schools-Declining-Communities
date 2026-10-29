"""
통합 가능 시나리오 확정 5단계: 비중복 기준 군집 선정 (03-5)
================================================================

목적
----
03-4에서 수용력과 최종 도로 접근성 기준을 통과한 3개교 후보 중에서,
같은 학교가 둘 이상의 군집에 중복되지 않도록 CAP28 기준 군집을 선정한다.
선정된 동일 군집이 CAP19에서도 유지 가능한지 재평가하여 단계 03의
'통합 가능한 최종 기준 군집'을 확정한다.

왜 최적화가 필요한가
---------------------
02-2에서 만든 3개교 후보는 전수 조합이므로 한 학교가 여러 후보에 반복된다.
후보를 단순히 앞에서부터 고르면 먼저 등장한 후보에 따라 결과가 달라지고,
더 많은 학교를 포함할 수 있는 조합을 놓칠 수 있다. 따라서 교육지원청별로
다음 3원소 집합 패킹 정수최적화를 정확히 푼다.

최종 군집 선정 기준
-------------------
1순위: 선택 군집 수 최대화
    - 학교는 최대 한 군집에만 포함될 수 있다.
    - 모든 군집이 3개교이므로 군집 수 최대화 = 포함 학교 수 최대화다.

2순위: 같은 최대 군집 수라면 학교 간 접근시간 최소화
    - 후보별 점수는 가능한 CAP28 존치학교안 중
      `max_school_road_duration_min`이 가장 작은 값이다.
    - 즉, 어느 한 학교를 존치했을 때 두 폐교예정학교 중 더 먼 학교의
      이동시간을 가능한 한 작게 만드는 군집을 선호한다.
    - 이 값은 후보 요약표의 `min_eligible_max_road_duration_min`이다.
    - 최적화에서는 초 단위 정수로 반올림해 합계를 최소화하지만 원래 분 단위
      실수값은 그대로 보존한다. 이는 최종 학생 통학시간이 아니라 동률 해소용
      학교 간 접근성 보조지표다.

3순위: 1·2순위도 같으면 후보 ID 순위합 최소화
    - 실행할 때마다 같은 결과가 나오도록 결정론적 동률 처리를 한다.

CAP28·CAP19 비교 원칙
---------------------
- 기준 비중복 군집은 CAP28에서 한 번만 선정한다.
- CAP19에서는 같은 candidate_id가 수용력·도로 기준을 통과하는지만 재평가한다.
- CAP19에서 불가능해진 군집을 다른 군집으로 교체하여 재최적화하지 않는다.
- 따라서 두 시나리오 차이는 학교 조합 변경이 아니라 수용력 가정의 차이로
  해석할 수 있다.

중요한 해석 원칙
----------------
- 03-5에서는 최종 3개교 묶음만 확정한다.
- 존치학교는 아직 하나로 선택하지 않는다.
- 선정 군집에 남아 있는 모든 적격 존치학교안을 단계 04로 넘긴다.
- 단계 04에서 집계구별 통합 전후 통학시간을 계산한 뒤, 군집별 부담이
  최소인 존치안과 최대인 존치안을 각각 하한·상한으로 확정한다.

입력 파일
---------
data/02_processed/final_road_candidate_summary.csv
    - 03-4 후보 단위 최종 도로 접근성 요약

data/02_processed/final_road_access_options.csv
    - 03-4 후보–존치학교안 단위 상세표

출력 파일
---------
data/02_processed/nonoverlap_cluster_selection.csv
    - CAP28 최종 가능 후보 전체를 보존한 선택 결과
    - 선택 여부, 최종 군집 ID, 겹치는 선택 군집, CAP19 유지 여부 포함

data/02_processed/final_selected_cluster_options.csv
    - 최종 선택 군집에 속한 CAP28·CAP19 적격 존치학교안만 추출
    - 단계 04의 집계구→현재학교·존치학교 경로 요청 구성에 사용

data/02_processed/nonoverlap_selection_summary.csv
    - 교육지원청별 후보·학교·선택 군집·포함 학교·CAP19 유지 군집 수
    - 각 교육지원청의 최적화 성공 여부와 전국 합계 포함

실행 방법
---------
프로젝트 루트에서:

    py scripts/03-5_select_nonoverlapping_clusters.py

필요 패키지
-----------
    pandas, numpy, scipy

주의
----
- scipy.optimize.milp를 사용하므로 비교적 최신 SciPy가 필요하다.
- `ROAD_FEASIBLE_CANDIDATE`는 선택 가능한 후보이고,
  `SELECTED_FINAL_CLUSTER`만 실제 최종 기준 군집이다.
- 선택되지 않은 후보는 오류가 아니라 같은 학교를 공유하는 다른 후보가
  최적해에서 선택된 결과다.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, csr_matrix, vstack
except ImportError as error:  # pragma: no cover - 환경 안내용
    raise ImportError(
        "03-5는 scipy.optimize.milp가 필요합니다. "
        "프로젝트 가상환경에서 'pip install scipy'를 실행하세요."
    ) from error


# -----------------------------------------------------------------------------
# 분석 설정
# -----------------------------------------------------------------------------

CANDIDATE_FILE = "final_road_candidate_summary.csv"
OPTION_FILE = "final_road_access_options.csv"

OUTPUT_SELECTION = "nonoverlap_cluster_selection.csv"
OUTPUT_OPTIONS = "final_selected_cluster_options.csv"
OUTPUT_SUMMARY = "nonoverlap_selection_summary.csv"

BASE_SCENARIO = "CAP28"
STRICT_SCENARIO = "CAP19"
NUMERIC_TOLERANCE = 1e-7

CANDIDATE_REQUIRED = {
    "capacity_scenario_id",
    "candidate_id",
    "students_per_class",
    "school_edu_office",
    "component_id",
    "candidate_type",
    "total_demand",
    "school_1_key",
    "school_1_name",
    "school_2_key",
    "school_2_name",
    "school_3_key",
    "school_3_name",
    "final_eligible_option_count",
    "eligible_retained_school_keys",
    "eligible_retained_school_names",
    "min_eligible_max_road_duration_min",
    "candidate_road_eligible",
    "candidate_road_status",
    "candidate_road_check",
}

OPTION_REQUIRED = {
    "capacity_option_id",
    "candidate_id",
    "capacity_scenario_id",
    "school_edu_office",
    "retained_school_key",
    "retained_school_name",
    "final_road_eligible",
    "final_road_status",
    "final_rule_check",
}


@dataclass
class OfficeOptimizationResult:
    office: str
    candidates: pd.DataFrame
    selected_indices: np.ndarray
    candidate_school_count: int
    theoretical_upper_bound: int
    selected_cluster_count: int
    compactness_total_seconds: int
    stage_1_status: int
    stage_2_status: int
    stage_3_status: int
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="CAP28 기준 학교 중복 없는 최종 3개교 군집 선정"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="final_project 폴더. 기본값은 scripts 폴더의 상위 폴더입니다.",
    )
    parser.add_argument("--candidate-file", type=Path, default=None)
    parser.add_argument("--option-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--solver-time-limit-seconds",
        type=float,
        default=120.0,
        help="교육지원청별·최적화 단계별 최대 풀이 시간. 기본값 120초",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    project_root = args.project_root.resolve()
    processed_dir = project_root / "data" / "02_processed"
    candidate_path = (
        args.candidate_file or processed_dir / CANDIDATE_FILE
    ).resolve()
    option_path = (args.option_file or processed_dir / OPTION_FILE).resolve()
    output_dir = (args.output_dir or processed_dir).resolve()
    return candidate_path, option_path, output_dir


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"[{label}] 필수 컬럼이 없습니다: {missing}")


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{label}에 결측값이 있습니다.")
        return series.astype(bool)

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
    candidate_path: Path,
    option_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [candidate_path, option_path]:
        if not path.exists():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    candidate_dtypes = {
        "capacity_scenario_id": "string",
        "candidate_id": "string",
        "school_1_key": "string",
        "school_2_key": "string",
        "school_3_key": "string",
    }
    option_dtypes = {
        "capacity_option_id": "string",
        "candidate_id": "string",
        "capacity_scenario_id": "string",
        "retained_school_key": "string",
    }
    candidates = pd.read_csv(
        candidate_path,
        encoding="utf-8-sig",
        dtype=candidate_dtypes,
        low_memory=False,
    )
    options = pd.read_csv(
        option_path,
        encoding="utf-8-sig",
        dtype=option_dtypes,
        low_memory=False,
    )
    require_columns(candidates, CANDIDATE_REQUIRED, CANDIDATE_FILE)
    require_columns(options, OPTION_REQUIRED, OPTION_FILE)

    for column in candidate_dtypes:
        candidates[column] = normalize_id(candidates[column])
    for column in option_dtypes:
        options[column] = normalize_id(options[column])

    candidates["candidate_road_eligible"] = parse_bool(
        candidates["candidate_road_eligible"], "candidate_road_eligible"
    )
    options["final_road_eligible"] = parse_bool(
        options["final_road_eligible"], "final_road_eligible"
    )
    return candidates, options


def validate_inputs(
    candidates: pd.DataFrame,
    options: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_result = candidates.copy()
    option_result = options.copy()

    if candidate_result.empty or option_result.empty:
        raise ValueError("입력 후보 또는 존치안 파일이 비어 있습니다.")
    if candidate_result.duplicated(["capacity_scenario_id", "candidate_id"]).any():
        raise ValueError("후보–수용력 시나리오 키가 중복됩니다.")
    if option_result["capacity_option_id"].isna().any():
        raise ValueError("capacity_option_id에 결측값이 있습니다.")
    if option_result["capacity_option_id"].duplicated().any():
        raise ValueError("capacity_option_id가 중복됩니다.")
    if not candidate_result["candidate_road_check"].eq("PASS").all():
        raise ValueError("candidate_road_check가 PASS가 아닌 후보가 있습니다.")
    if not option_result["final_rule_check"].eq("PASS").all():
        raise ValueError("final_rule_check가 PASS가 아닌 존치안이 있습니다.")

    scenario_set = set(candidate_result["capacity_scenario_id"].dropna())
    if not {BASE_SCENARIO, STRICT_SCENARIO}.issubset(scenario_set):
        raise ValueError(
            f"{BASE_SCENARIO}과 {STRICT_SCENARIO} 후보가 모두 필요합니다: {scenario_set}"
        )

    for column in ["final_eligible_option_count", "min_eligible_max_road_duration_min"]:
        candidate_result[column] = pd.to_numeric(
            candidate_result[column], errors="coerce"
        )

    base = candidate_result.loc[
        candidate_result["capacity_scenario_id"].eq(BASE_SCENARIO)
        & candidate_result["candidate_road_eligible"]
    ]
    if base.empty:
        raise ValueError("CAP28 최종 가능 후보가 없습니다.")
    if base["min_eligible_max_road_duration_min"].isna().any():
        raise ValueError("CAP28 최종 가능 후보의 접근시간 점수에 결측값이 있습니다.")
    if base["min_eligible_max_road_duration_min"].lt(0).any():
        raise ValueError("CAP28 후보 접근시간 점수에 음수가 있습니다.")
    if base["final_eligible_option_count"].lt(1).any():
        raise ValueError("최종 가능 CAP28 후보에 적격 존치안이 없는 행이 있습니다.")
    if not base["candidate_road_status"].eq("ROAD_FEASIBLE_CANDIDATE").all():
        raise ValueError("최종 가능 CAP28 후보의 상태값이 일치하지 않습니다.")

    school_columns = ["school_1_key", "school_2_key", "school_3_key"]
    if base[school_columns].isna().any().any():
        raise ValueError("CAP28 후보의 학교 키에 결측값이 있습니다.")
    duplicate_school = base.apply(
        lambda row: len(set(row[school_columns])) != 3,
        axis=1,
    )
    if duplicate_school.any():
        raise ValueError("동일 후보 안에 같은 학교가 중복된 행이 있습니다.")
    if not (
        (base["school_1_key"] < base["school_2_key"])
        & (base["school_2_key"] < base["school_3_key"])
    ).all():
        raise ValueError("후보 학교 키의 오름차순 정렬 규칙이 깨졌습니다.")

    strict_ids = set(
        candidate_result.loc[
            candidate_result["capacity_scenario_id"].eq(STRICT_SCENARIO),
            "candidate_id",
        ]
    )
    base_ids = set(
        candidate_result.loc[
            candidate_result["capacity_scenario_id"].eq(BASE_SCENARIO),
            "candidate_id",
        ]
    )
    if not strict_ids.issubset(base_ids):
        raise ValueError("CAP19 후보 집합이 CAP28 후보 집합의 부분집합이 아닙니다.")

    eligible_options = option_result.loc[option_result["final_road_eligible"]]
    detail_counts = eligible_options.groupby(
        ["capacity_scenario_id", "candidate_id"]
    ).size()
    summary_counts = candidate_result.set_index(
        ["capacity_scenario_id", "candidate_id"]
    )["final_eligible_option_count"]
    common_index = detail_counts.index.intersection(summary_counts.index)
    if not detail_counts.loc[common_index].eq(summary_counts.loc[common_index]).all():
        raise ValueError("후보 요약표와 상세표의 최종 적격 존치안 수가 다릅니다.")

    return candidate_result, option_result


def solve_office_set_packing(
    office: str,
    office_candidates: pd.DataFrame,
    time_limit_seconds: float,
) -> OfficeOptimizationResult:
    start = time.perf_counter()
    frame = office_candidates.sort_values("candidate_id").reset_index(drop=True).copy()
    school_columns = ["school_1_key", "school_2_key", "school_3_key"]
    schools = sorted(set(frame[school_columns].to_numpy().ravel()))
    school_to_row = {school_key: index for index, school_key in enumerate(schools)}

    row_indices: list[int] = []
    column_indices: list[int] = []
    for candidate_index, row in frame.iterrows():
        for column in school_columns:
            row_indices.append(school_to_row[row[column]])
            column_indices.append(candidate_index)

    school_matrix = coo_matrix(
        (
            np.ones(len(row_indices), dtype=float),
            (row_indices, column_indices),
        ),
        shape=(len(schools), len(frame)),
    ).tocsr()

    variable_count = len(frame)
    integrality = np.ones(variable_count, dtype=int)
    bounds = Bounds(np.zeros(variable_count), np.ones(variable_count))
    solver_options = {
        "presolve": True,
        "mip_rel_gap": 0.0,
        "time_limit": float(time_limit_seconds),
    }

    # 1단계: 선택 군집 수 최대화
    school_constraint = LinearConstraint(
        school_matrix,
        lb=np.full(len(schools), -np.inf),
        ub=np.ones(len(schools)),
    )
    stage_1 = milp(
        c=-np.ones(variable_count),
        integrality=integrality,
        bounds=bounds,
        constraints=school_constraint,
        options=solver_options,
    )
    if not stage_1.success or stage_1.x is None:
        raise RuntimeError(
            f"[{office}] 1단계 군집 수 최대화 실패: {stage_1.message}"
        )
    stage_1_selected = stage_1.x > 0.5
    maximum_cluster_count = int(stage_1_selected.sum())

    # 후보 접근시간은 최적화의 정확한 정수 동률 기준으로 쓰기 위해 초 단위로
    # 반올림한다. 원래 분 단위 실수값은 출력에 그대로 보존한다.
    compactness_seconds = np.rint(
        frame["min_eligible_max_road_duration_min"].to_numpy(dtype=float) * 60
    ).astype(np.int64)

    count_row = csr_matrix(np.ones((1, variable_count), dtype=float))
    stage_2_matrix = vstack([school_matrix, count_row], format="csr")
    stage_2_constraint = LinearConstraint(
        stage_2_matrix,
        lb=np.r_[np.full(len(schools), -np.inf), maximum_cluster_count],
        ub=np.r_[np.ones(len(schools)), maximum_cluster_count],
    )

    # 2단계: 최대 군집 수를 고정한 뒤 접근시간 합 최소화
    stage_2 = milp(
        c=compactness_seconds.astype(float),
        integrality=integrality,
        bounds=bounds,
        constraints=stage_2_constraint,
        options=solver_options,
    )
    if not stage_2.success or stage_2.x is None:
        raise RuntimeError(
            f"[{office}] 2단계 접근시간 최소화 실패: {stage_2.message}"
        )
    stage_2_selected = stage_2.x > 0.5
    minimum_compactness_seconds = int(
        compactness_seconds[stage_2_selected].sum()
    )

    # 3단계: 앞의 두 최적값을 고정하고 candidate_id 순위합을 최소화한다.
    compactness_row = csr_matrix(compactness_seconds.reshape(1, -1))
    stage_3_matrix = vstack(
        [school_matrix, count_row, compactness_row],
        format="csr",
    )
    stage_3_constraint = LinearConstraint(
        stage_3_matrix,
        lb=np.r_[
            np.full(len(schools), -np.inf),
            maximum_cluster_count,
            minimum_compactness_seconds,
        ],
        ub=np.r_[
            np.ones(len(schools)),
            maximum_cluster_count,
            minimum_compactness_seconds,
        ],
    )
    candidate_rank = np.arange(1, variable_count + 1, dtype=float)
    stage_3 = milp(
        c=candidate_rank,
        integrality=integrality,
        bounds=bounds,
        constraints=stage_3_constraint,
        options=solver_options,
    )
    if not stage_3.success or stage_3.x is None:
        raise RuntimeError(
            f"[{office}] 3단계 결정론적 동률 처리 실패: {stage_3.message}"
        )

    selected = stage_3.x > 0.5
    selected_indices = np.flatnonzero(selected)
    if len(selected_indices) != maximum_cluster_count:
        raise AssertionError(f"[{office}] 최종 선택 군집 수가 1단계 최적값과 다릅니다.")
    if int(compactness_seconds[selected].sum()) != minimum_compactness_seconds:
        raise AssertionError(f"[{office}] 최종 접근시간 합이 2단계 최적값과 다릅니다.")

    selected_school_values = frame.loc[selected, school_columns].to_numpy().ravel()
    if len(selected_school_values) != len(set(selected_school_values)):
        raise AssertionError(f"[{office}] 최종 선택 군집 사이에 학교 중복이 있습니다.")

    return OfficeOptimizationResult(
        office=office,
        candidates=frame,
        selected_indices=selected_indices,
        candidate_school_count=len(schools),
        theoretical_upper_bound=len(schools) // 3,
        selected_cluster_count=maximum_cluster_count,
        compactness_total_seconds=minimum_compactness_seconds,
        stage_1_status=int(stage_1.status),
        stage_2_status=int(stage_2.status),
        stage_3_status=int(stage_3.status),
        elapsed_seconds=time.perf_counter() - start,
    )


def optimize_all_offices(
    candidates: pd.DataFrame,
    time_limit_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = candidates.loc[
        candidates["capacity_scenario_id"].eq(BASE_SCENARIO)
        & candidates["candidate_road_eligible"]
    ].copy()

    selection_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    office_groups = list(base.groupby("school_edu_office", sort=True))
    for sequence, (office, office_candidates) in enumerate(office_groups, start=1):
        result = solve_office_set_packing(
            office,
            office_candidates,
            time_limit_seconds,
        )
        frame = result.candidates.copy()
        frame["candidate_compactness_seconds"] = np.rint(
            frame["min_eligible_max_road_duration_min"].to_numpy(dtype=float) * 60
        ).astype(np.int64)
        frame["selected_cluster_flag"] = False
        frame.loc[result.selected_indices, "selected_cluster_flag"] = True
        frame["selection_status"] = np.where(
            frame["selected_cluster_flag"],
            "SELECTED_FINAL_CLUSTER",
            "NOT_SELECTED_OVERLAP_OPTIMIZATION",
        )
        selection_parts.append(frame)

        summary_rows.append(
            {
                "school_edu_office": office,
                "cap28_eligible_candidate_count": len(frame),
                "candidate_school_count": result.candidate_school_count,
                "theoretical_upper_bound_cluster_count": result.theoretical_upper_bound,
                "selected_cluster_count": result.selected_cluster_count,
                "selected_school_count": result.selected_cluster_count * 3,
                "unselected_candidate_school_count": (
                    result.candidate_school_count - result.selected_cluster_count * 3
                ),
                "school_coverage_rate": (
                    result.selected_cluster_count * 3 / result.candidate_school_count
                ),
                "compactness_total_seconds": result.compactness_total_seconds,
                "compactness_total_minutes": result.compactness_total_seconds / 60,
                "stage_1_solver_status": result.stage_1_status,
                "stage_2_solver_status": result.stage_2_status,
                "stage_3_solver_status": result.stage_3_status,
                "optimization_elapsed_seconds": result.elapsed_seconds,
                "optimization_check": "PASS",
            }
        )
        if sequence % 25 == 0 or sequence == len(office_groups):
            logging.info(
                "교육지원청 최적화 %s/%s 완료",
                sequence,
                len(office_groups),
            )

    selection = pd.concat(selection_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    return selection, summary


def attach_cluster_ids_and_conflicts(selection: pd.DataFrame) -> pd.DataFrame:
    result = selection.copy()
    selected_mask = result["selected_cluster_flag"]
    selected_rows = result.loc[selected_mask].sort_values(
        ["school_edu_office", "candidate_id"]
    )
    cluster_id_map = {
        candidate_id: f"FC_{sequence:05d}"
        for sequence, candidate_id in enumerate(
            selected_rows["candidate_id"],
            start=1,
        )
    }
    result["final_cluster_id"] = result["candidate_id"].map(cluster_id_map)

    school_to_cluster: dict[str, str] = {}
    for row in selected_rows.itertuples(index=False):
        cluster_id = cluster_id_map[row.candidate_id]
        for school_key in [row.school_1_key, row.school_2_key, row.school_3_key]:
            if school_key in school_to_cluster:
                raise AssertionError(f"선택 학교가 여러 군집에 중복됩니다: {school_key}")
            school_to_cluster[school_key] = cluster_id

    conflict_cluster_values: list[str] = []
    conflict_school_values: list[str] = []
    for row in result.itertuples(index=False):
        if row.selected_cluster_flag:
            conflict_cluster_values.append("NONE")
            conflict_school_values.append("NONE")
            continue
        schools = [row.school_1_key, row.school_2_key, row.school_3_key]
        conflicting_schools = sorted(
            school_key for school_key in schools if school_key in school_to_cluster
        )
        conflicting_clusters = sorted(
            {school_to_cluster[school_key] for school_key in conflicting_schools}
        )
        if not conflicting_clusters:
            raise AssertionError(
                f"선택 군집과 겹치지 않는데 미선택된 후보가 있습니다: {row.candidate_id}"
            )
        conflict_cluster_values.append("|".join(conflicting_clusters))
        conflict_school_values.append("|".join(conflicting_schools))

    result["conflicting_selected_cluster_ids"] = conflict_cluster_values
    result["conflicting_selected_school_keys"] = conflict_school_values
    result["selection_check"] = np.where(
        result["selected_cluster_flag"]
        == result["final_cluster_id"].notna(),
        "PASS",
        "FAIL",
    )
    if not result["selection_check"].eq("PASS").all():
        raise AssertionError("최종 군집 ID와 선택 플래그가 일치하지 않습니다.")
    return result


def attach_cap19_reevaluation(
    selection: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    result = selection.copy()
    strict = candidates.loc[
        candidates["capacity_scenario_id"].eq(STRICT_SCENARIO),
        [
            "candidate_id",
            "candidate_road_eligible",
            "candidate_road_status",
            "final_eligible_option_count",
            "eligible_retained_school_keys",
            "eligible_retained_school_names",
            "candidate_exclusion_reason",
        ],
    ].copy()
    strict = strict.rename(
        columns={
            "candidate_road_eligible": "cap19_candidate_road_eligible",
            "candidate_road_status": "cap19_candidate_road_status",
            "final_eligible_option_count": "cap19_final_eligible_option_count",
            "eligible_retained_school_keys": "cap19_eligible_retained_school_keys",
            "eligible_retained_school_names": "cap19_eligible_retained_school_names",
            "candidate_exclusion_reason": "cap19_candidate_exclusion_reason",
        }
    )
    result = result.merge(
        strict,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    result["cap19_candidate_exists"] = result[
        "cap19_candidate_road_status"
    ].notna()
    # NaN은 CAP19 후보 자체가 없다는 뜻이다. eq(True)는 NaN을 False로
    # 처리하면서 object dtype 자동 다운캐스팅 경고도 만들지 않는다.
    result["cap19_candidate_road_eligible"] = result[
        "cap19_candidate_road_eligible"
    ].eq(True)
    result["cap19_final_eligible_option_count"] = result[
        "cap19_final_eligible_option_count"
    ].fillna(0).astype(int)
    for column in [
        "cap19_candidate_road_status",
        "cap19_eligible_retained_school_keys",
        "cap19_eligible_retained_school_names",
        "cap19_candidate_exclusion_reason",
    ]:
        result[column] = result[column].fillna("NONE")

    result["cap19_cluster_status"] = np.select(
        [
            result["cap19_candidate_road_eligible"],
            ~result["cap19_candidate_exists"],
            result["cap19_candidate_exists"]
            & ~result["cap19_candidate_road_eligible"],
        ],
        [
            "MAINTAINED_CAP19",
            "NOT_CAPACITY_FEASIBLE_CAP19",
            "NO_ROAD_FEASIBLE_OPTION_CAP19",
        ],
        default="CHECK_REQUIRED",
    )
    if result["cap19_cluster_status"].eq("CHECK_REQUIRED").any():
        raise AssertionError("CAP19 재평가 상태를 결정하지 못한 후보가 있습니다.")
    return result


def build_selected_options(
    selection: pd.DataFrame,
    options: pd.DataFrame,
) -> pd.DataFrame:
    selected = selection.loc[
        selection["selected_cluster_flag"],
        [
            "candidate_id",
            "final_cluster_id",
            "min_eligible_max_road_duration_min",
            "candidate_compactness_seconds",
            "cap19_cluster_status",
        ],
    ].copy()
    selected_ids = set(selected["candidate_id"])
    result = options.loc[
        options["candidate_id"].isin(selected_ids)
        & options["final_road_eligible"]
        & options["capacity_scenario_id"].isin([BASE_SCENARIO, STRICT_SCENARIO])
    ].copy()
    result = result.merge(
        selected,
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )
    result["scenario_role"] = np.where(
        result["capacity_scenario_id"].eq(BASE_SCENARIO),
        "BASELINE_CAP28",
        "STRICT_CAP19_REEVALUATION",
    )

    front_columns = [
        "final_cluster_id",
        "candidate_id",
        "capacity_scenario_id",
        "scenario_role",
        "cap19_cluster_status",
        "min_eligible_max_road_duration_min",
        "candidate_compactness_seconds",
    ]
    remaining_columns = [
        column for column in result.columns if column not in front_columns
    ]
    result = result[front_columns + remaining_columns].sort_values(
        [
            "final_cluster_id",
            "capacity_scenario_id",
            "retained_school_position",
        ],
        ascending=[True, False, True],
    )
    if result["capacity_option_id"].duplicated().any():
        raise AssertionError("선택 군집 존치안 파일에 옵션 ID 중복이 있습니다.")
    if not result["final_road_eligible"].all():
        raise AssertionError("선택 군집 존치안 파일에 도로 기준 탈락안이 있습니다.")
    return result


def finalize_summary(
    summary: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    result = summary.copy()
    selected = selection.loc[selection["selected_cluster_flag"]]
    strict_counts = selected.groupby("school_edu_office").agg(
        cap19_maintained_cluster_count=(
            "cap19_cluster_status",
            lambda values: int((values == "MAINTAINED_CAP19").sum()),
        ),
        cap19_not_capacity_feasible_cluster_count=(
            "cap19_cluster_status",
            lambda values: int((values == "NOT_CAPACITY_FEASIBLE_CAP19").sum()),
        ),
        cap19_no_road_feasible_cluster_count=(
            "cap19_cluster_status",
            lambda values: int((values == "NO_ROAD_FEASIBLE_OPTION_CAP19").sum()),
        ),
    ).reset_index()
    result = result.merge(
        strict_counts,
        on="school_edu_office",
        how="left",
        validate="one_to_one",
    )
    for column in [
        "cap19_maintained_cluster_count",
        "cap19_not_capacity_feasible_cluster_count",
        "cap19_no_road_feasible_cluster_count",
    ]:
        result[column] = result[column].fillna(0).astype(int)
    result["cap19_maintenance_rate"] = (
        result["cap19_maintained_cluster_count"] / result["selected_cluster_count"]
    )
    result["optimization_check"] = np.where(
        result["selected_cluster_count"]
        == result["cap19_maintained_cluster_count"]
        + result["cap19_not_capacity_feasible_cluster_count"]
        + result["cap19_no_road_feasible_cluster_count"],
        result["optimization_check"],
        "FAIL",
    )

    total = {
        "school_edu_office": "ALL",
        "cap28_eligible_candidate_count": int(
            result["cap28_eligible_candidate_count"].sum()
        ),
        "candidate_school_count": int(result["candidate_school_count"].sum()),
        "theoretical_upper_bound_cluster_count": int(
            result["theoretical_upper_bound_cluster_count"].sum()
        ),
        "selected_cluster_count": int(result["selected_cluster_count"].sum()),
        "selected_school_count": int(result["selected_school_count"].sum()),
        "unselected_candidate_school_count": int(
            result["unselected_candidate_school_count"].sum()
        ),
        "school_coverage_rate": (
            result["selected_school_count"].sum()
            / result["candidate_school_count"].sum()
        ),
        "compactness_total_seconds": int(
            result["compactness_total_seconds"].sum()
        ),
        "compactness_total_minutes": result["compactness_total_minutes"].sum(),
        "stage_1_solver_status": 0,
        "stage_2_solver_status": 0,
        "stage_3_solver_status": 0,
        "optimization_elapsed_seconds": result[
            "optimization_elapsed_seconds"
        ].sum(),
        "optimization_check": (
            "PASS" if result["optimization_check"].eq("PASS").all() else "FAIL"
        ),
        "cap19_maintained_cluster_count": int(
            result["cap19_maintained_cluster_count"].sum()
        ),
        "cap19_not_capacity_feasible_cluster_count": int(
            result["cap19_not_capacity_feasible_cluster_count"].sum()
        ),
        "cap19_no_road_feasible_cluster_count": int(
            result["cap19_no_road_feasible_cluster_count"].sum()
        ),
        "cap19_maintenance_rate": (
            result["cap19_maintained_cluster_count"].sum()
            / result["selected_cluster_count"].sum()
        ),
    }
    result = pd.concat([result, pd.DataFrame([total])], ignore_index=True)
    if not result["optimization_check"].eq("PASS").all():
        raise AssertionError("교육지원청별 최적화 요약 검증에 실패했습니다.")
    return result


def validate_final_outputs(
    selection: pd.DataFrame,
    selected_options: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    selected = selection.loc[selection["selected_cluster_flag"]].copy()
    school_columns = ["school_1_key", "school_2_key", "school_3_key"]
    selected_schools = selected[school_columns].to_numpy().ravel()
    if len(selected_schools) != len(set(selected_schools)):
        raise AssertionError("전국 최종 선택 군집에 중복 학교가 있습니다.")
    if selected["final_cluster_id"].isna().any():
        raise AssertionError("선택 군집에 final_cluster_id 결측값이 있습니다.")
    if selected["final_cluster_id"].duplicated().any():
        raise AssertionError("final_cluster_id가 중복됩니다.")
    if selection.loc[~selection["selected_cluster_flag"], "final_cluster_id"].notna().any():
        raise AssertionError("미선택 후보에 final_cluster_id가 부여되었습니다.")

    selected_id_set = set(selected["candidate_id"])
    if set(selected_options["candidate_id"]) != selected_id_set:
        raise AssertionError("선택 군집과 단계 04용 존치안의 후보 집합이 다릅니다.")
    cap28_option_candidates = set(
        selected_options.loc[
            selected_options["capacity_scenario_id"].eq(BASE_SCENARIO),
            "candidate_id",
        ]
    )
    if cap28_option_candidates != selected_id_set:
        raise AssertionError("모든 선택 군집에 CAP28 적격 존치안이 존재하지 않습니다.")

    maintained_ids = set(
        selected.loc[
            selected["cap19_cluster_status"].eq("MAINTAINED_CAP19"),
            "candidate_id",
        ]
    )
    cap19_option_candidates = set(
        selected_options.loc[
            selected_options["capacity_scenario_id"].eq(STRICT_SCENARIO),
            "candidate_id",
        ]
    )
    if cap19_option_candidates != maintained_ids:
        raise AssertionError("CAP19 유지 군집과 CAP19 적격 존치안 집합이 다릅니다.")

    total_summary = summary.loc[summary["school_edu_office"].eq("ALL")].iloc[0]
    if int(total_summary["selected_cluster_count"]) != len(selected):
        raise AssertionError("전국 요약의 선택 군집 수가 상세 선택 결과와 다릅니다.")
    if int(total_summary["selected_school_count"]) != len(set(selected_schools)):
        raise AssertionError("전국 요약의 선택 학교 수가 상세 선택 결과와 다릅니다.")


def order_selection_columns(selection: pd.DataFrame) -> pd.DataFrame:
    front_columns = [
        "final_cluster_id",
        "selected_cluster_flag",
        "selection_status",
        "selection_check",
        "candidate_id",
        "capacity_scenario_id",
        "school_edu_office",
        "component_id",
        "candidate_type",
        "school_1_key",
        "school_1_name",
        "school_2_key",
        "school_2_name",
        "school_3_key",
        "school_3_name",
        "min_eligible_max_road_duration_min",
        "candidate_compactness_seconds",
        "conflicting_selected_cluster_ids",
        "conflicting_selected_school_keys",
        "cap19_cluster_status",
        "cap19_candidate_exists",
        "cap19_candidate_road_eligible",
        "cap19_final_eligible_option_count",
        "cap19_eligible_retained_school_keys",
        "cap19_eligible_retained_school_names",
        "cap19_candidate_exclusion_reason",
    ]
    remaining = [column for column in selection.columns if column not in front_columns]
    return selection[front_columns + remaining].sort_values(
        ["school_edu_office", "selected_cluster_flag", "candidate_id"],
        ascending=[True, False, True],
    )


def log_results(
    selection: pd.DataFrame,
    selected_options: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    selected = selection.loc[selection["selected_cluster_flag"]]
    total = summary.loc[summary["school_edu_office"].eq("ALL")].iloc[0]
    logging.info("CAP28 최종 가능 후보: %s", f"{len(selection):,}")
    logging.info("최종 비중복 군집: %s", f"{len(selected):,}")
    logging.info("최종 포함 학교: %s", f"{len(selected) * 3:,}")
    logging.info(
        "후보 참여 학교 포괄률: %.2f%%",
        float(total["school_coverage_rate"]) * 100,
    )
    logging.info(
        "CAP19 유지 군집: %s / CAP19 미유지 군집: %s",
        f"{int(total['cap19_maintained_cluster_count']):,}",
        f"{int(total['selected_cluster_count'] - total['cap19_maintained_cluster_count']):,}",
    )
    logging.info(
        "단계 04용 적격 존치안: 전체 %s / CAP28 %s / CAP19 %s",
        f"{len(selected_options):,}",
        f"{int(selected_options['capacity_scenario_id'].eq(BASE_SCENARIO).sum()):,}",
        f"{int(selected_options['capacity_scenario_id'].eq(STRICT_SCENARIO).sum()):,}",
    )


def main() -> int:
    args = parse_args()
    if args.solver_time_limit_seconds <= 0:
        raise ValueError("solver-time-limit-seconds는 0보다 커야 합니다.")

    candidate_path, option_path, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("입력 후보표: %s", candidate_path)
    logging.info("입력 존치안표: %s", option_path)

    candidates, options = load_inputs(candidate_path, option_path)
    candidates, options = validate_inputs(candidates, options)
    selection, optimization_summary = optimize_all_offices(
        candidates,
        args.solver_time_limit_seconds,
    )
    selection = attach_cluster_ids_and_conflicts(selection)
    selection = attach_cap19_reevaluation(selection, candidates)
    selected_options = build_selected_options(selection, options)
    optimization_summary = finalize_summary(optimization_summary, selection)
    validate_final_outputs(selection, selected_options, optimization_summary)
    selection = order_selection_columns(selection)

    selection_path = output_dir / OUTPUT_SELECTION
    selected_options_path = output_dir / OUTPUT_OPTIONS
    summary_path = output_dir / OUTPUT_SUMMARY
    selection.to_csv(selection_path, index=False, encoding="utf-8-sig")
    selected_options.to_csv(
        selected_options_path,
        index=False,
        encoding="utf-8-sig",
    )
    optimization_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    log_results(selection, selected_options, optimization_summary)
    logging.info("저장 완료: %s", selection_path)
    logging.info("저장 완료: %s", selected_options_path)
    logging.info("저장 완료: %s", summary_path)
    logging.info("03-5 검증 PASS: 단계 03 통합 가능 최종 기준 군집 확정 완료")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        raise SystemExit(main())
    except Exception as error:
        logging.exception("03-5 실행 실패: %s", error)
        sys.exit(1)

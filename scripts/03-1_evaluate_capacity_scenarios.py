"""
통합 가능 시나리오 확정 1단계: 학교 수용력 평가 (03-1)
=====================================================

목적
----
02-2에서 생성한 연결된 3개교 전수 후보마다 세 학교를 각각 존치학교로
가정하고, 통합 후 학생 전체를 수용할 수 있는지 평가한다.

이 코드는 7개 핵심 분석 단계 중 '통합 가능 시나리오 확정' 단계의 첫 번째
하위 작업이다. 수용 가능한 후보–존치학교안을 확정하지만, 학교 간 도로
이동시간과 후보 간 학교 중복은 아직 평가하지 않는다. 따라서 이 단계의
`CAPACITY_FEASIBLE`은 최종 통합 가능을 뜻하지 않는다.

확정 기준
---------
- 통합 수요: 후보에 포함된 세 학교의 현재 실제 재학생 수 합계
- 학교 수용력: 존치학교의 일반교실수 × 교실당 학생 수
- 기본 교실당 학생 수: 28명
- 수용 가능: 학교 수용력 >= 통합 수요
- 소수점 반올림·올림 없이 원자료의 학생 수와 교실 수를 그대로 계산

엄격한 수용력 시나리오
----------------------
28명보다 낮은 구체적 값은 아직 확정하지 않았으므로 기본 실행은 28명만
계산한다. 팀에서 값을 확정하면 동일 코드에 다음처럼 추가한다.

    py scripts/03-1_evaluate_capacity_scenarios.py \
        --students-per-class 28 24 20

입력 파일
---------
data/02_processed/distance_graph_nodes.csv
    - 학교별 일반교실수와 실제 재학생 수

data/02_processed/connected_three_school_candidates.csv
    - 연결된 3개교 후보와 세 학교의 ID·재학생 수

출력 파일
---------
data/02_processed/candidate_capacity_options.csv
    - 후보–교실당 학생 수–존치학교 1개당 1행
    - 수용력·통합수요·여유인원·수용률·수용 가능 여부

data/02_processed/candidate_capacity_summary.csv
    - 후보–교실당 학생 수 1개당 1행
    - 수용 가능한 존치학교 수와 학교 목록, 후보 수준 상태

실행 방법
---------
프로젝트 루트에서:

    py scripts/03-1_evaluate_capacity_scenarios.py

주의
----
- `connector_school_key`는 CHAIN의 그래프 연결 중심학교일 뿐 존치학교가
  아니므로 수용력 평가에서 우선권을 주지 않는다.
- 세 학교를 모두 한 번씩 존치학교로 가정한다.
- 수용 불가안도 삭제하지 않고 `INSUFFICIENT_CAPACITY` 상태로 보존한다.
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

NODES_FILE = "distance_graph_nodes.csv"
CANDIDATES_FILE = "connected_three_school_candidates.csv"

OUTPUT_OPTIONS = "candidate_capacity_options.csv"
OUTPUT_SUMMARY = "candidate_capacity_summary.csv"

DEFAULT_STUDENTS_PER_CLASS = [28]
NUMERIC_TOLERANCE = 1e-9

NODES_REQUIRED = {
    "school_key",
    "school_name",
    "school_edu_office",
    "actual_enrollment",
    "classroom_count",
    "small_school_flag",
    "main_branch_type",
    "calibration_check",
}

CANDIDATES_REQUIRED = {
    "candidate_id",
    "school_edu_office",
    "component_id",
    "candidate_type",
    "total_actual_enrollment",
    "school_1_key",
    "school_1_name",
    "school_1_actual_enrollment",
    "school_2_key",
    "school_2_name",
    "school_2_actual_enrollment",
    "school_3_key",
    "school_3_name",
    "school_3_actual_enrollment",
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="연결된 3개교 후보의 존치학교별 수용력 시나리오 평가"
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
        help="거리 그래프 노드 CSV. 생략하면 data/02_processed 기본 파일 사용",
    )
    parser.add_argument(
        "--candidates-file",
        type=Path,
        default=None,
        help="연결된 3개교 후보 CSV. 생략하면 data/02_processed 기본 파일 사용",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="출력 폴더. 생략하면 data/02_processed를 사용합니다.",
    )
    parser.add_argument(
        "--students-per-class",
        type=int,
        nargs="+",
        default=DEFAULT_STUDENTS_PER_CLASS,
        help="평가할 교실당 학생 수. 기본값: 28",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"[{label}] 필수 컬럼이 없습니다: {missing}")


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def validate_students_per_class(values: list[int]) -> list[int]:
    if not values:
        raise ValueError("교실당 학생 수를 하나 이상 입력해야 합니다.")
    if any(isinstance(value, bool) or value <= 0 for value in values):
        raise ValueError("교실당 학생 수는 0보다 큰 정수여야 합니다.")
    # 큰 기준을 먼저 출력해 28명 기준을 가장 앞에서 확인할 수 있게 한다.
    return sorted(set(int(value) for value in values), reverse=True)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    project_root = args.project_root.resolve()
    processed_dir = project_root / "data" / "02_processed"
    nodes_path = (args.nodes_file or processed_dir / NODES_FILE).resolve()
    candidates_path = (
        args.candidates_file or processed_dir / CANDIDATES_FILE
    ).resolve()
    output_dir = (args.output_dir or processed_dir).resolve()
    return nodes_path, candidates_path, output_dir


def load_and_validate_inputs(
    nodes_path: Path,
    candidates_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [nodes_path, candidates_path]:
        if not path.exists():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    nodes = pd.read_csv(
        nodes_path,
        encoding="utf-8-sig",
        dtype={"school_key": "string"},
        low_memory=False,
    )
    candidate_key_dtypes = {
        "candidate_id": "string",
        "school_1_key": "string",
        "school_2_key": "string",
        "school_3_key": "string",
    }
    candidates = pd.read_csv(
        candidates_path,
        encoding="utf-8-sig",
        dtype=candidate_key_dtypes,
        low_memory=False,
    )
    require_columns(nodes, NODES_REQUIRED, NODES_FILE)
    require_columns(candidates, CANDIDATES_REQUIRED, CANDIDATES_FILE)

    nodes["school_key"] = normalize_id(nodes["school_key"])
    candidates["candidate_id"] = normalize_id(candidates["candidate_id"])
    for position in [1, 2, 3]:
        candidates[f"school_{position}_key"] = normalize_id(
            candidates[f"school_{position}_key"]
        )

    if nodes["school_key"].isna().any() or nodes.duplicated("school_key").any():
        raise ValueError("노드의 school_key에 결측 또는 중복이 있습니다.")
    if candidates["candidate_id"].isna().any() or candidates.duplicated(
        "candidate_id"
    ).any():
        raise ValueError("후보의 candidate_id에 결측 또는 중복이 있습니다.")
    if candidates.duplicated(
        ["school_1_key", "school_2_key", "school_3_key"]
    ).any():
        raise ValueError("동일한 3개교 후보가 중복되어 있습니다.")
    if not (
        (candidates["school_1_key"] < candidates["school_2_key"])
        & (candidates["school_2_key"] < candidates["school_3_key"])
    ).all():
        raise ValueError("후보의 학교키 정렬 규칙을 위반했습니다.")

    if not nodes["calibration_check"].eq("PASS").all():
        raise ValueError("노드 파일에 calibration_check가 PASS가 아닌 학교가 있습니다.")
    if not nodes["small_school_flag"].eq("Y").all():
        raise ValueError("노드 파일에 작은학교가 아닌 학교가 있습니다.")
    if not nodes["main_branch_type"].eq("본교").all():
        raise ValueError("노드 파일에 분교가 포함되어 있습니다.")

    nodes["classroom_count"] = pd.to_numeric(
        nodes["classroom_count"], errors="coerce"
    )
    nodes["actual_enrollment"] = pd.to_numeric(
        nodes["actual_enrollment"], errors="coerce"
    )
    classroom_problem = (
        nodes["classroom_count"].isna()
        | nodes["classroom_count"].le(0)
        | ~np.isclose(
            nodes["classroom_count"],
            np.round(nodes["classroom_count"]),
            atol=NUMERIC_TOLERANCE,
            rtol=0,
        )
    )
    enrollment_problem = (
        nodes["actual_enrollment"].isna()
        | nodes["actual_enrollment"].lt(0)
        | ~np.isclose(
            nodes["actual_enrollment"],
            np.round(nodes["actual_enrollment"]),
            atol=NUMERIC_TOLERANCE,
            rtol=0,
        )
    )
    if classroom_problem.any() or enrollment_problem.any():
        problems = nodes.loc[
            classroom_problem | enrollment_problem,
            ["school_key", "school_name", "classroom_count", "actual_enrollment"],
        ]
        raise ValueError(
            "학교의 일반교실수 또는 실제 재학생 수에 결측·음수·비정수 문제가 "
            f"있습니다: {problems.head(10).to_dict('records')}"
        )
    nodes["classroom_count"] = nodes["classroom_count"].round().astype(int)
    nodes["actual_enrollment"] = nodes["actual_enrollment"].round().astype(int)

    for column in [
        "total_actual_enrollment",
        "school_1_actual_enrollment",
        "school_2_actual_enrollment",
        "school_3_actual_enrollment",
    ]:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
        if candidates[column].isna().any() or candidates[column].lt(0).any():
            raise ValueError(f"후보의 {column}에 결측 또는 음수가 있습니다.")

    calculated_total = candidates[
        [
            "school_1_actual_enrollment",
            "school_2_actual_enrollment",
            "school_3_actual_enrollment",
        ]
    ].sum(axis=1)
    if not np.isclose(
        candidates["total_actual_enrollment"],
        calculated_total,
        atol=NUMERIC_TOLERANCE,
        rtol=0,
    ).all():
        raise ValueError("후보의 세 학교 재학생 합계가 total_actual_enrollment와 다릅니다.")

    # 후보 학교의 속성이 노드표와 정확히 일치하는지 검증한다.
    node_lookup = nodes.set_index("school_key")
    for position in [1, 2, 3]:
        key_column = f"school_{position}_key"
        name_column = f"school_{position}_name"
        enrollment_column = f"school_{position}_actual_enrollment"
        mapped_name = candidates[key_column].map(node_lookup["school_name"])
        mapped_office = candidates[key_column].map(node_lookup["school_edu_office"])
        mapped_enrollment = candidates[key_column].map(node_lookup["actual_enrollment"])
        if mapped_name.isna().any():
            missing_keys = candidates.loc[mapped_name.isna(), key_column].unique()[:10]
            raise ValueError(f"노드표에 없는 후보 학교키가 있습니다: {missing_keys}")
        if mapped_name.astype("string").ne(
            candidates[name_column].astype("string")
        ).any():
            raise ValueError(f"후보와 노드표의 {position}번째 학교명이 다릅니다.")
        if mapped_office.astype("string").ne(
            candidates["school_edu_office"].astype("string")
        ).any():
            raise ValueError(f"후보와 노드표의 {position}번째 교육지원청이 다릅니다.")
        if not np.isclose(
            mapped_enrollment,
            candidates[enrollment_column],
            atol=NUMERIC_TOLERANCE,
            rtol=0,
        ).all():
            raise ValueError(f"후보와 노드표의 {position}번째 재학생 수가 다릅니다.")

    return (
        nodes.sort_values(["school_edu_office", "school_key"]).reset_index(drop=True),
        candidates.sort_values("candidate_id").reset_index(drop=True),
    )


def build_capacity_options(
    nodes: pd.DataFrame,
    candidates: pd.DataFrame,
    students_per_class_values: list[int],
) -> pd.DataFrame:
    """후보의 세 학교를 각각 존치학교로 두는 모든 수용력안을 만든다."""
    node_lookup = nodes.set_index("school_key")
    option_frames: list[pd.DataFrame] = []

    for students_per_class in students_per_class_values:
        scenario_id = f"CAP{students_per_class}"
        for retained_position in [1, 2, 3]:
            closed_positions = [
                position for position in [1, 2, 3] if position != retained_position
            ]
            retained_key_column = f"school_{retained_position}_key"
            retained_name_column = f"school_{retained_position}_name"

            frame = candidates[
                [
                    "candidate_id",
                    "school_edu_office",
                    "component_id",
                    "candidate_type",
                    "total_actual_enrollment",
                    retained_key_column,
                    retained_name_column,
                    f"school_{closed_positions[0]}_key",
                    f"school_{closed_positions[0]}_name",
                    f"school_{closed_positions[1]}_key",
                    f"school_{closed_positions[1]}_name",
                ]
            ].copy()
            frame = frame.rename(
                columns={
                    "total_actual_enrollment": "total_demand",
                    retained_key_column: "retained_school_key",
                    retained_name_column: "retained_school_name",
                    f"school_{closed_positions[0]}_key": "closed_school_1_key",
                    f"school_{closed_positions[0]}_name": "closed_school_1_name",
                    f"school_{closed_positions[1]}_key": "closed_school_2_key",
                    f"school_{closed_positions[1]}_name": "closed_school_2_name",
                }
            )
            frame.insert(1, "capacity_scenario_id", scenario_id)
            frame.insert(2, "students_per_class", students_per_class)
            frame.insert(7, "retained_school_position", retained_position)

            frame["retained_classroom_count"] = frame["retained_school_key"].map(
                node_lookup["classroom_count"]
            )
            frame["retained_current_enrollment"] = frame[
                "retained_school_key"
            ].map(node_lookup["actual_enrollment"])
            frame["incoming_enrollment"] = (
                frame["total_demand"] - frame["retained_current_enrollment"]
            )
            frame["retained_capacity"] = (
                frame["retained_classroom_count"] * students_per_class
            )
            frame["capacity_margin"] = (
                frame["retained_capacity"] - frame["total_demand"]
            )
            frame["capacity_utilization_rate"] = (
                frame["total_demand"] / frame["retained_capacity"]
            )
            frame["capacity_feasible"] = frame["capacity_margin"].ge(0)
            frame["capacity_status"] = np.where(
                frame["capacity_feasible"],
                "CAPACITY_FEASIBLE",
                "INSUFFICIENT_CAPACITY",
            )
            frame.insert(
                0,
                "capacity_option_id",
                frame["candidate_id"].astype(str)
                + "_"
                + scenario_id
                + "_R"
                + str(retained_position),
            )
            option_frames.append(frame)

    options = pd.concat(option_frames, ignore_index=True)
    options = options.sort_values(
        ["students_per_class", "candidate_id", "retained_school_position"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return options


def join_sorted_unique(values: pd.Series) -> str:
    unique_values = sorted({str(value) for value in values.dropna()})
    return "|".join(unique_values)


def build_capacity_summary(options: pd.DataFrame) -> pd.DataFrame:
    """후보–교실당 학생 수 수준으로 존치 가능 학교 수를 요약한다."""
    base_columns = [
        "candidate_id",
        "capacity_scenario_id",
        "students_per_class",
        "school_edu_office",
        "component_id",
        "candidate_type",
        "total_demand",
    ]
    grouped = options.groupby(
        ["candidate_id", "capacity_scenario_id"], sort=False, dropna=False
    )
    summary = grouped.agg(
        students_per_class=("students_per_class", "first"),
        school_edu_office=("school_edu_office", "first"),
        component_id=("component_id", "first"),
        candidate_type=("candidate_type", "first"),
        total_demand=("total_demand", "first"),
        retained_option_count=("retained_school_key", "size"),
        feasible_retained_school_count=("capacity_feasible", "sum"),
        max_retained_capacity=("retained_capacity", "max"),
        max_capacity_margin=("capacity_margin", "max"),
    ).reset_index()

    feasible = options.loc[options["capacity_feasible"]]
    feasible_lists = feasible.groupby(
        ["candidate_id", "capacity_scenario_id"], sort=False
    ).agg(
        feasible_retained_school_keys=("retained_school_key", join_sorted_unique),
        feasible_retained_school_names=("retained_school_name", join_sorted_unique),
        min_feasible_capacity_utilization_rate=(
            "capacity_utilization_rate",
            "min",
        ),
    ).reset_index()
    summary = summary.merge(
        feasible_lists,
        on=["candidate_id", "capacity_scenario_id"],
        how="left",
        validate="one_to_one",
    )
    summary["feasible_retained_school_keys"] = summary[
        "feasible_retained_school_keys"
    ].fillna("")
    summary["feasible_retained_school_names"] = summary[
        "feasible_retained_school_names"
    ].fillna("")
    summary["candidate_capacity_feasible"] = summary[
        "feasible_retained_school_count"
    ].gt(0)
    summary["candidate_capacity_status"] = np.where(
        summary["candidate_capacity_feasible"],
        "CAPACITY_FEASIBLE",
        "NO_FEASIBLE_RETAINED_SCHOOL",
    )

    ordered_columns = base_columns + [
        "retained_option_count",
        "feasible_retained_school_count",
        "candidate_capacity_feasible",
        "candidate_capacity_status",
        "feasible_retained_school_keys",
        "feasible_retained_school_names",
        "max_retained_capacity",
        "max_capacity_margin",
        "min_feasible_capacity_utilization_rate",
    ]
    return summary[ordered_columns].sort_values(
        ["students_per_class", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def validate_outputs(
    candidates: pd.DataFrame,
    options: pd.DataFrame,
    summary: pd.DataFrame,
    students_per_class_values: list[int],
) -> None:
    scenario_count = len(students_per_class_values)
    expected_option_rows = len(candidates) * 3 * scenario_count
    expected_summary_rows = len(candidates) * scenario_count
    if len(options) != expected_option_rows:
        raise ValueError(
            f"수용력안 행 수가 예상과 다릅니다: 예상 {expected_option_rows:,}, "
            f"실제 {len(options):,}"
        )
    if len(summary) != expected_summary_rows:
        raise ValueError(
            f"후보 요약 행 수가 예상과 다릅니다: 예상 {expected_summary_rows:,}, "
            f"실제 {len(summary):,}"
        )
    if options["capacity_option_id"].duplicated().any():
        raise ValueError("capacity_option_id가 중복되었습니다.")
    if summary.duplicated(["candidate_id", "capacity_scenario_id"]).any():
        raise ValueError("후보–수용력 시나리오 요약이 중복되었습니다.")
    if not options.groupby(
        ["candidate_id", "capacity_scenario_id"]
    ).size().eq(3).all():
        raise ValueError("후보마다 존치학교안이 정확히 3개가 아닙니다.")

    formula_capacity = (
        options["retained_classroom_count"] * options["students_per_class"]
    )
    formula_margin = options["retained_capacity"] - options["total_demand"]
    formula_feasible = options["capacity_margin"].ge(0)
    if not np.isclose(
        options["retained_capacity"],
        formula_capacity,
        atol=NUMERIC_TOLERANCE,
        rtol=0,
    ).all():
        raise ValueError("학교 수용력 계산식 검증에 실패했습니다.")
    if not np.isclose(
        options["capacity_margin"],
        formula_margin,
        atol=NUMERIC_TOLERANCE,
        rtol=0,
    ).all():
        raise ValueError("수용 여유인원 계산식 검증에 실패했습니다.")
    if options["capacity_feasible"].ne(formula_feasible).any():
        raise ValueError("수용 가능 여부 계산식 검증에 실패했습니다.")

    option_aggregate = options.groupby(
        ["candidate_id", "capacity_scenario_id"]
    )["capacity_feasible"].sum()
    summary_aggregate = summary.set_index(
        ["candidate_id", "capacity_scenario_id"]
    )["feasible_retained_school_count"]
    if not option_aggregate.sort_index().equals(summary_aggregate.sort_index()):
        raise ValueError("상세 수용력안과 후보 요약의 가능 학교 수가 다릅니다.")

    # 교실당 학생 수가 커질수록 수용 가능한 존치학교 수가 감소해서는 안 된다.
    monotonic = summary.pivot(
        index="candidate_id",
        columns="students_per_class",
        values="feasible_retained_school_count",
    ).sort_index(axis=1)
    if monotonic.shape[1] > 1 and (monotonic.diff(axis=1).iloc[:, 1:] < 0).any().any():
        raise ValueError("교실당 학생 수 증가에 따라 수용 가능 학교 수가 감소했습니다.")


def log_results(options: pd.DataFrame, summary: pd.DataFrame) -> None:
    logging.info("교실당 학생 수별 수용력 평가 결과")
    logging.info(
        "%-10s %12s %12s %12s %12s %12s",
        "기준",
        "가능후보",
        "불가후보",
        "가능학교0",
        "가능학교1",
        "가능학교2/3",
    )
    for students_per_class in sorted(
        summary["students_per_class"].unique(), reverse=True
    ):
        rows = summary.loc[
            summary["students_per_class"].eq(students_per_class)
        ]
        counts = rows["feasible_retained_school_count"].value_counts()
        feasible_candidates = int(rows["candidate_capacity_feasible"].sum())
        infeasible_candidates = len(rows) - feasible_candidates
        logging.info(
            "%-10s %12s %12s %12s %12s %12s",
            f"{students_per_class}명",
            f"{feasible_candidates:,}",
            f"{infeasible_candidates:,}",
            f"{int(counts.get(0, 0)):,}",
            f"{int(counts.get(1, 0)):,}",
            f"{int(counts.get(2, 0) + counts.get(3, 0)):,}",
        )
        option_rows = options.loc[
            options["students_per_class"].eq(students_per_class)
        ]
        logging.info(
            "  수용 가능한 후보–존치학교안: %s",
            f"{int(option_rows['capacity_feasible'].sum()):,}",
        )


def main() -> None:
    args = parse_args()
    students_per_class_values = validate_students_per_class(
        args.students_per_class
    )
    nodes_path, candidates_path, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info("통합 가능 시나리오의 수용력 평가를 시작합니다.")
    logging.info("노드 파일: %s", nodes_path)
    logging.info("3개교 후보 파일: %s", candidates_path)
    logging.info("교실당 학생 수: %s", students_per_class_values)

    nodes, candidates = load_and_validate_inputs(nodes_path, candidates_path)
    options = build_capacity_options(
        nodes,
        candidates,
        students_per_class_values,
    )
    summary = build_capacity_summary(options)
    validate_outputs(
        candidates,
        options,
        summary,
        students_per_class_values,
    )
    log_results(options, summary)

    options_path = output_dir / OUTPUT_OPTIONS
    summary_path = output_dir / OUTPUT_SUMMARY
    options.to_csv(options_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    logging.info("03-1 수용력 평가 완료")
    logging.info("후보–존치학교 상세안: %s", options_path)
    logging.info("후보별 수용력 요약: %s", summary_path)
    logging.info(
        "주의: CAPACITY_FEASIBLE은 수용력만 통과했다는 뜻입니다. "
        "도로 이동시간과 후보 간 학교 중복은 아직 검증하지 않았습니다."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("분석이 중단되었습니다: %s", exc)
        sys.exit(1)

"""
Stage 4 확정 시나리오의 파일 정의·읽기·검증 공통 모듈

제공하는 기능
- 확정 시나리오 ID CAP28_MIN_ADDITIONAL_TIME과 다음 파일명을 정의한다.
  final_integration_scenario.csv, final_integration_schools.csv,
  final_integration_demand_assignments.csv, final_integration_manifest.json
- SHA-256 계산, 필수 컬럼 확인과 엄격한 불리언 값 변환을 제공한다.
- data/03_outputs의 manifest와 세 CSV의 해시를 대조한 뒤 파일을 읽는다.
  군집·학생 배정의 필수 컬럼, 720개 고유 군집·2,160개 고유 학교,
  존치학교 720개·폐교 대상 학교 1,440개 상태와 학생 배정의 중복·학교안 일치를 확인한다.
- 별도 학생 배정 경로를 지정해도 확정 배정 파일과 SHA-256이 같아야 한다.

사용 관계와 처리 원칙
- 04-5는 시나리오·파일명 정의, 학생 배정 검증과 해시 계산을 사용해 최종 결과를 만든다.
- Stage 5의 stage5_commute_common과 06은 load_final_scenario로 확정 결과를 읽는다.
  05-4는 불리언 변환·필수 컬럼 검사도 직접 사용한다.
- 후속 Stage는 확정 군집·존치학교를 다시 선정하거나 입력 파일에 덮어쓰지 않는다.
  반환한 DataFrame은 후속 분석을 위해 가공할 수 있으며 불변 객체는 아니다.
- 직접 실행하는 분석 스크립트가 아니며, 이 모듈 자체는 산출물을 저장하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SCENARIO_ID = "CAP28_MIN_ADDITIONAL_TIME"
SCENARIO_FILE = "final_integration_scenario.csv"
SCHOOL_FILE = "final_integration_schools.csv"
DEMAND_FILE = "final_integration_demand_assignments.csv"
MANIFEST_FILE = "final_integration_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def parse_bool(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.upper()
    if not text.isin(["TRUE", "FALSE", "1", "0"]).all():
        raise ValueError(f"Invalid boolean values in {series.name}")
    return text.isin(["TRUE", "1"])


def validate_assignments(detail: pd.DataFrame, scenario: pd.DataFrame) -> None:
    require_columns(detail, ["scenario_id", "final_cluster_id", "integration_option_id",
                             "demand_assignment_id", "current_school_key", "integrated_school_key",
                             "student_weight", "commute_analysis_eligible_flag"], DEMAND_FILE)
    if set(detail.scenario_id) != {SCENARIO_ID} or detail.demand_assignment_id.isna().any():
        raise ValueError("Not the fixed representative scenario")
    if detail.demand_assignment_id.duplicated().any():
        raise ValueError("Duplicate final demand assignment")
    pairs = detail[["final_cluster_id", "integration_option_id", "integrated_school_key"]].drop_duplicates()
    expected = scenario[["final_cluster_id", "integration_option_id", "integrated_school_key"]]
    if set(map(tuple, pairs.to_numpy())) != set(map(tuple, expected.to_numpy())):
        raise ValueError("Assignments do not match the frozen school choices")


def load_final_scenario(project_root: Path, demand_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder = project_root / "data/03_outputs"
    manifest = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest["scenario_id"] != SCENARIO_ID:
        raise ValueError("Unsupported final scenario")
    for filename in [SCENARIO_FILE, SCHOOL_FILE, DEMAND_FILE]:
        if sha256_file(folder / filename) != manifest["outputs_sha256"][filename]:
            raise ValueError(f"Frozen Stage 4 output changed: {filename}")
    scenario = pd.read_csv(folder / SCENARIO_FILE, dtype=str)
    schools = pd.read_csv(folder / SCHOOL_FILE, dtype=str)
    detail = pd.read_csv(demand_path or folder / DEMAND_FILE, dtype=str)
    require_columns(scenario, ["scenario_id", "final_cluster_id", "integration_option_id", "integrated_school_key"], SCENARIO_FILE)
    if len(scenario) != 720 or scenario.final_cluster_id.duplicated().any() or set(scenario.scenario_id) != {SCENARIO_ID}:
        raise ValueError("Expected exactly 720 unique final clusters")
    if len(schools) != 2160 or schools.school_key.duplicated().any():
        raise ValueError("Expected exactly 2160 unique scenario schools")
    if schools.school_status.value_counts().to_dict() != {"CLOSED": 1440, "RETAINED": 720}:
        raise ValueError("Invalid final school statuses")
    validate_assignments(detail, scenario)
    # 별도 경로의 학생 배정 파일도 확정 배정 파일과 내용이 완전히 같아야 한다.
    if demand_path is not None and demand_path.resolve() != (folder / DEMAND_FILE).resolve():
        if sha256_file(demand_path) != manifest["outputs_sha256"][DEMAND_FILE]:
            raise ValueError("Alternate demand input differs from frozen Stage 4 output")
    return scenario, schools, detail

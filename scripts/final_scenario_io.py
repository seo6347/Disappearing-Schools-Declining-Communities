"""Read the frozen Stage 4 scenario; downstream analysis never selects schools."""
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
    # Explicit alternate paths must carry exactly the same frozen assignments.
    if demand_path is not None and demand_path.resolve() != (folder / DEMAND_FILE).resolve():
        if sha256_file(demand_path) != manifest["outputs_sha256"][DEMAND_FILE]:
            raise ValueError("Alternate demand input differs from frozen Stage 4 output")
    return scenario, schools, detail

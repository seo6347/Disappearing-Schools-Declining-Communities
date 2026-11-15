"""
Stage 5 확정 통학 배정의 분석 준비·학생 가중 통계·출력 공통 모듈

제공하는 기능
- final_scenario_io로 Stage 4 확정 결과와 해시를 확인하고, 배정 식별자·학생 가중치·
  통학지표의 유효성과 통합 후 - 통합 전 변화량의 일치를 검증한다.
- 분석 가능 학생을 거리 변화로 분류한다. 1e-9 km 초과는 INCREASED,
  -1e-9 km 미만은 DECREASED, 그 사이는 UNCHANGED이며 경로 분석 제외는 EXCLUDED다.
- 학생 지역은 현재 배정학교, 존치학교 소재지는 통합 후 학교의 인구감소지역 구분을 쓴다.
  인구감소지역은 DECLINE, 인구감소관심지역·미해당은 NON_DECLINE,
  결측·미확인·인식하지 못한 구분은 UNKNOWN으로 처리한다.
- INCREASED, DECREASED, CHANGED, ALL_ELIGIBLE 분석 집단을 제공한다.
  CHANGED와 ALL_ELIGIBLE도 거리 변화 집단별로 나누어 통계를 계산하도록 한다.
- student_weight에 의한 평균·누적 가중치 기반 분위수·비율과 집단별 보존 검증을 제공한다.
  분모가 0인 비율과 유효한 가중치가 없는 평균·분위수는 결측값이다.
- 출력은 기본적으로 data/03_outputs/stage5/<분석명>에 두고, 모집단 옵션이 있는
  분석은 <population>을 추가한다. --output-dir도 이 하위 경로를 유지한다.
- OutputBundle은 CSV·PNG·manifest를 임시 디렉터리에 준비한 뒤 정상 종료 시
  선택 출력 경로에 파일별로 반영한다. 기존 결과 교체에는 --overwrite가 필요하다.
- manifest에 확정 입력 해시, 분석 집단·분모·지역 기준, CSV 스키마와 파일 목록을 기록한다.
- 비대화형 그래프 생성과 지정 한글 폰트 또는 맑은 고딕·Noto 한글 폰트 탐색을 지원한다.

사용 관계
- 05-1~05-4가 이 모듈을 사용하며, 공통 CLI와 통계·검증·저장 기능을 공유한다.
- 확정 입력 파일은 읽기 전용으로 사용하고 학교 선정은 수행하지 않는다.
  분석별 결과 저장은 각 스크립트가 호출하며 이 모듈은 직접 실행하지 않는다.
"""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from final_scenario_io import (DEMAND_FILE, MANIFEST_FILE, SCENARIO_ID,
                               load_final_scenario, parse_bool, require_columns)

DISTANCE_TOLERANCE_KM = 1e-9
POPULATIONS = ("INCREASED", "DECREASED", "CHANGED", "ALL_ELIGIBLE")
CATEGORIES = ("INCREASED", "DECREASED", "UNCHANGED")
REGIONS = ("DECLINE", "NON_DECLINE", "UNKNOWN")
REGION_LABELS = {"DECLINE": "인구감소지역", "NON_DECLINE": "비인구감소지역", "UNKNOWN": "미확인"}
CATEGORY_LABELS = {"INCREASED": "거리 증가", "DECREASED": "거리 감소", "UNCHANGED": "거리 동일"}
TIME_NOTE = ("네이버 Directions API 자동차 직행 편도시간 기준. "
             "실제 통학버스 시간 또는 법적 기준 위반 여부를 의미하지 않음.")
METRICS = {
    "current_distance_km": "current_road_distance_km",
    "integrated_distance_km": "integrated_road_distance_km",
    "distance_change_km": "road_distance_change_km",
    "additional_distance_burden_km": "additional_distance_burden_km",
    "current_duration_min": "current_analysis_road_duration_min",
    "integrated_duration_min": "integrated_analysis_road_duration_min",
    "duration_change_min": "road_duration_change_min",
    "additional_time_burden_min": "additional_time_burden_min",
}


def argument_parser(description: str, population: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-root", "--analysis-root", dest="project_root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, help="Must be identical to the frozen Stage 4 assignments.")
    parser.add_argument("--output-dir", type=Path,
                        help="Output parent; analysis and population namespaces are always appended.")
    parser.add_argument("--overwrite", action="store_true", help="Replace only this analysis/population's outputs.")
    parser.add_argument("--font", type=Path, help="Optional Korean font for PNG outputs.")
    if population:
        parser.add_argument("--population", choices=POPULATIONS, default=population)
    return parser


def classify_region(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().map({
        "인구감소지역": "DECLINE", "인구감소관심지역": "NON_DECLINE", "미해당": "NON_DECLINE",
    }).fillna("UNKNOWN")


def distance_categories(change: pd.Series, eligible: pd.Series) -> pd.Series:
    if not np.isfinite(change.loc[eligible]).all():
        raise ValueError("Eligible distance change contains missing or non-finite values")
    return pd.Series(np.select(
        [~eligible, change.gt(DISTANCE_TOLERANCE_KM), change.lt(-DISTANCE_TOLERANCE_KM)],
        ["EXCLUDED", "INCREASED", "DECREASED"], default="UNCHANGED"), index=change.index)


def prepare_detail(detail: pd.DataFrame) -> pd.DataFrame:
    required = ["demand_assignment_id", "census_id", "final_cluster_id", "current_school_key",
                "integrated_school_key", "student_weight", "commute_analysis_eligible_flag",
                "current_school_edu_office", "integrated_school_edu_office",
                "current_school_population_decline_type", "integrated_school_population_decline_type",
                *METRICS.values()]
    require_columns(detail, required, "frozen assignments")
    work = detail.copy()
    if work.demand_assignment_id.isna().any() or work.demand_assignment_id.duplicated().any():
        raise ValueError("Missing or duplicate demand_assignment_id")
    for column in ["census_id", "final_cluster_id", "current_school_key", "integrated_school_key"]:
        if work[column].isna().any() or work[column].astype("string").str.strip().eq("").any():
            raise ValueError(f"Missing identifier: {column}")
    for column in ["student_weight", *METRICS.values()]:
        work[column] = pd.to_numeric(work[column], errors="raise")
    work["commute_analysis_eligible_flag"] = parse_bool(work.commute_analysis_eligible_flag)
    if not np.isfinite(work.student_weight).all() or work.student_weight.le(0).any():
        raise ValueError("student_weight must be finite and positive")
    eligible = work.commute_analysis_eligible_flag
    if not np.isfinite(work.loc[eligible, list(METRICS.values())].to_numpy(float)).all():
        raise ValueError("Eligible distance/time metrics must be finite")
    for name in ["current_distance_km", "integrated_distance_km", "current_duration_min", "integrated_duration_min"]:
        if work.loc[eligible, METRICS[name]].lt(0).any():
            raise ValueError(f"Negative absolute commute metric: {name}")
    for prefix in ["distance", "duration"]:
        actual = work.loc[eligible, METRICS[f"{prefix}_change_{'km' if prefix == 'distance' else 'min'}"]]
        unit = "km" if prefix == "distance" else "min"
        expected = (work.loc[eligible, METRICS[f"integrated_{prefix}_{unit}"]]
                    - work.loc[eligible, METRICS[f"current_{prefix}_{unit}"]])
        if not np.allclose(actual, expected, rtol=1e-10, atol=1e-8):
            raise ValueError(f"Inconsistent frozen {prefix} changes")
    work["distance_change_category"] = distance_categories(work.road_distance_change_km, eligible)
    work["current_school_region_group"] = classify_region(work.current_school_population_decline_type)
    work["retained_school_location_region_group"] = classify_region(work.integrated_school_population_decline_type)
    for column in ["current_school_edu_office", "integrated_school_edu_office"]:
        work[column] = work[column].astype("string").str.strip().replace("", pd.NA).fillna("UNKNOWN")
    return work


def load_analysis(args) -> tuple[pd.DataFrame, dict]:
    root = args.project_root.expanduser().resolve()
    _, _, detail = load_final_scenario(root, args.input.expanduser().resolve() if args.input else None)
    manifest = json.loads((root / "data/03_outputs" / MANIFEST_FILE).read_text(encoding="utf-8"))
    return prepare_detail(detail), manifest["outputs_sha256"]


def population_groups(work: pd.DataFrame, population: str):
    if population not in POPULATIONS:
        raise ValueError(f"Unknown population {population}")
    categories = CATEGORIES if population == "ALL_ELIGIBLE" else (
        CATEGORIES[:2] if population == "CHANGED" else (population,))
    for category in categories:
        yield category, work.loc[work.distance_change_category.eq(category)].copy()


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    return float(np.average(values.loc[valid], weights=weights.loc[valid])) if valid.any() else np.nan


def weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    values, weights = values.loc[valid].to_numpy(float), weights.loc[valid].to_numpy(float)
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    position = np.searchsorted(np.cumsum(weights), q * weights.sum(), side="left")
    return float(values[min(position, len(values) - 1)])


def summarize(group: pd.DataFrame) -> dict:
    weights = group.student_weight
    total = float(weights.sum())
    result = {"assignment_row_count": len(group), "unique_census_point_count": group.census_id.nunique(),
              "retained_school_count": group.integrated_school_key.nunique(),
              "cluster_count": group.final_cluster_id.nunique(),
              "education_office_count": group.current_school_edu_office.replace("UNKNOWN", pd.NA).nunique(),
              "student_weight": total, "denominator_student_weight": total}
    for label, column in METRICS.items():
        result[f"weighted_mean_{label}"] = weighted_mean(group[column], weights)
    for label in ["distance_change_km", "duration_change_min"]:
        values = group[METRICS[label]]
        for q in [.1, .25, .5, .75, .9, .95, .99]:
            result[f"weighted_p{int(q * 100)}_{label}"] = weighted_quantile(values, weights, q)
        result[f"min_{label}"] = float(values.min()) if len(group) else np.nan
        result[f"max_{label}"] = float(values.max()) if len(group) else np.nan
    return result


def check(check_id: str, actual, expected, *, review: bool = False) -> dict:
    passed = bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-7))
    if not passed and not review:
        raise ValueError(f"Validation {check_id}: {actual} != {expected}")
    return {"check_id": check_id, "actual": float(actual), "expected": float(expected),
            "status": "PASS" if passed else "REVIEW"}


def partition_checks(work: pd.DataFrame) -> list[dict]:
    eligible = work.loc[work.commute_analysis_eligible_flag]
    classified = work.loc[work.distance_change_category.isin(CATEGORIES)]
    return [check("distance_categories_student_weight", classified.student_weight.sum(), eligible.student_weight.sum()),
            check("distance_categories_row_count", len(classified), len(eligible)),
            check("excluded_students_not_classified", int((~work.commute_analysis_eligible_flag & work.distance_change_category.ne("EXCLUDED")).sum()), 0)]


def output_path(args, analysis: str) -> Path:
    base = (args.output_dir.expanduser().resolve() if args.output_dir
            else args.project_root.expanduser().resolve() / "data/03_outputs/stage5")
    path = base / analysis
    return path / args.population if hasattr(args, "population") else path


class OutputBundle(AbstractContextManager):
    """선택한 출력 경로의 파일을 교체하기 전에 모든 산출물을 임시 디렉터리에 준비한다."""

    def __init__(self, destination: Path, overwrite: bool):
        self.destination = destination
        self.overwrite = overwrite
        self.tables: dict[str, pd.DataFrame] = {}

    def __enter__(self):
        if self.destination.exists() and any(self.destination.iterdir()) and not self.overwrite:
            raise FileExistsError(f"Outputs already exist: {self.destination}. Use --overwrite.")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix=f".{self.destination.name}-", dir=self.destination.parent)
        self.path = Path(self.temporary.name)
        return self

    def csv(self, name: str, table: pd.DataFrame):
        table = table.copy()
        table.insert(0, "scenario_id", SCENARIO_ID)
        table.to_csv(self.path / name, index=False, encoding="utf-8-sig")
        self.tables[name] = table

    def manifest(self, *, analysis: str, input_hashes: dict, population: str, **extra):
        payload = {
            "schema_version": 1, "scenario_id": SCENARIO_ID, "analysis": analysis,
            "population": population, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_input_sha256": input_hashes, "school_selection_performed": False,
            "weight_column": "student_weight", "rate_unit": "fraction_0_to_1",
            "distance_tolerance_km": DISTANCE_TOLERANCE_KM,
            "student_region_basis": "current_school_population_decline_type",
            "retained_school_location_basis": "integrated_school_population_decline_type",
            "region_groups": REGION_LABELS, "region_rule": {
                "DECLINE": ["인구감소지역"], "NON_DECLINE": ["인구감소관심지역", "미해당"],
                "UNKNOWN": "missing, unconfirmed, or unrecognized labels"},
            "time_interpretation": TIME_NOTE,
            "column_definitions": {
                "student_weight": "Sum of estimated student weights, not assignment rows or unique students.",
                "denominator_student_weight": "Explicit eligible population denominator for the reported rate/statistic.",
                "distance_change_category": "Distance-based subgroup; CHANGED never pools opposite-signed groups.",
                "current_school_region_group": "Student classification by current assigned school location.",
                "retained_school_location_region_group": "Retained school location; does not reclassify students.",
                "education_office_count": "Unique current assigned school education offices unless explicitly named retained.",
            },
            "csv_schemas": {name: {"row_count": len(table), "columns": list(table.columns)}
                            for name, table in self.tables.items()},
            **extra,
        }
        payload["generated_files"] = sorted([p.name for p in self.path.iterdir()] + ["analysis_manifest.json"])
        (self.path / "analysis_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self.destination.mkdir(parents=True, exist_ok=True)
                # 입력 검증과 그래프 생성이 끝나기 전에는 기존 결과를 삭제하지 않는다.
                for path in sorted(self.path.iterdir(), key=lambda p: p.name == "analysis_manifest.json"):
                    path.replace(self.destination / path.name)
        finally:
            self.temporary.cleanup()
        return False


def configure_plot(font: Path | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if font is not None:
        font = font.expanduser().resolve()
        if not font.is_file():
            raise FileNotFoundError(font)
    else:
        font = next((p for p in [Path("C:/Windows/Fonts/malgun.ttf"),
                                 Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
                     if p.is_file()), None)
    if font:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": "#F8F5EF",
                         "axes.facecolor": "#F8F5EF", "savefig.facecolor": "#F8F5EF"})
    return plt


def bar_plot(plt, destination: Path, labels, values, title: str, unit: str,
             *, national: float | None = None, horizontal: bool = False, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.subplots_adjust(left=.25 if horizontal else .10, bottom=.25, top=.82, right=.96)
    labels, values = list(labels), np.asarray(list(values), dtype=float)
    colors = ["#527D99" if v < 0 else "#3F735C" for v in values]
    if len(values):
        if horizontal:
            ax.barh(labels, values, color=colors)
            ax.invert_yaxis()
            ax.set_xlabel(unit)
        else:
            ax.bar(range(len(values)), values, color=colors)
            ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=8)
            ax.set_ylabel(unit)
    else:
        ax.text(.5, .5, "해당 학생 또는 비교 지역 없음", ha="center", transform=ax.transAxes)
    if national is not None and np.isfinite(national):
        (ax.axvline if horizontal else ax.axhline)(national, color="#2B2825", linestyle="--", label=f"전국 평균 {national:.3f}")
        ax.legend()
    ax.axvline(0, color="#D8D2CA") if horizontal else ax.axhline(0, color="#D8D2CA")
    ax.set_title(title, loc="left", fontsize=17, pad=30)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.04, .06, "student_weight 가중 집계 · 학생 지역은 현재 배정학교 소재지 기준", fontsize=9)
    fig.text(.04, .025, TIME_NOTE, fontsize=8)
    fig.savefig(destination, dpi=160)
    plt.close(fig)

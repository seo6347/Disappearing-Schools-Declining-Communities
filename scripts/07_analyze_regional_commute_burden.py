#!/usr/bin/env python3
"""07. Compare regional commute burdens and population-decline groups.

Regional unit
-------------
The source result contains complete education-office identifiers but does not
contain complete residence-region names for every census point. Therefore the
default and only lossless regional unit is ``current_school_edu_office``: the
education support office of each student's current assigned school. Results
must be described as school-affiliation regions, not residence-region rankings.

Default population
------------------
* LOWER_BOUND and ALL_STUDENTS
* moved students
* commute-analysis eligible
* one-way commute time increased
* weighted by student_weight

Top-five rankings are produced separately for additional time and road-distance
change. No arbitrary combined burden score is created. Small regional samples
are excluded with configurable student and census-point thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


IVORY = "#F8F5EF"
GREEN = "#3F735C"
WARM_CHARCOAL = "#2B2825"
WARM_GRAY = "#716C66"
GRID = "#D8D2CA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank education-office commute burdens and compare decline groups."
    )
    parser.add_argument("--analysis-root", type=Path, default=Path("."))
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit commute_bound_demand_results.csv path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: <analysis-root>/data/04_visualization/"
            "regional_commute_burden"
        ),
    )
    parser.add_argument(
        "--bound-type",
        default="LOWER_BOUND",
        choices=["LOWER_BOUND", "UPPER_BOUND"],
    )
    parser.add_argument(
        "--selection-population",
        default="ALL_STUDENTS",
        choices=["ALL_STUDENTS", "MOVED_STUDENTS"],
    )
    parser.add_argument(
        "--min-estimated-students",
        type=float,
        default=100,
        help="Minimum weighted students for Top 5 eligibility. Default 100.",
    )
    parser.add_argument(
        "--min-census-points",
        type=int,
        default=10,
        help="Minimum unique census points for Top 5 eligibility. Default 10.",
    )
    parser.add_argument(
        "--concern-as-decline",
        action="store_true",
        help=(
            "Classify 인구감소관심지역 with 인구감소지역. By default, only the "
            "official 인구감소지역 label forms the decline group."
        ),
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="Optional Korean font. Malgun Gothic is auto-detected on Windows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_input(root: Path, filename: str) -> Path:
    matches = [path for path in root.rglob(filename) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} below {root.resolve()}")

    def rank(path: Path) -> tuple[int, int, str]:
        lowered = {part.lower() for part in path.parts}
        penalty = 10 if {"upload", "tmp", "temp"} & lowered else 0
        return (penalty, len(path.parts), str(path))

    matches.sort(key=rank)
    chosen = matches[0]
    if len(matches) > 1:
        print(f"[INFO] Found {len(matches)} copies; using {chosen}")
    return chosen


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def select_population(
    df: pd.DataFrame,
    bound_type: str,
    selection_population: str,
) -> pd.DataFrame:
    require_columns(
        df,
        [
            "bound_type",
            "selection_population",
            "demand_assignment_id",
            "census_id",
            "final_cluster_id",
            "student_weight",
            "moved_student_flag",
            "commute_analysis_eligible_flag",
            "time_increase_flag",
            "road_duration_change_min",
            "road_distance_change_km",
            "current_school_edu_office",
            "current_school_population_decline_type",
        ],
    )
    mask = (
        df["bound_type"].eq(bound_type)
        & df["selection_population"].eq(selection_population)
        & normalize_bool(df["moved_student_flag"])
        & normalize_bool(df["commute_analysis_eligible_flag"])
        & normalize_bool(df["time_increase_flag"])
    )
    selected = df.loc[mask].copy()
    for column in [
        "student_weight",
        "road_duration_change_min",
        "road_distance_change_km",
    ]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.loc[
        selected["student_weight"].gt(0)
        & selected["road_duration_change_min"].gt(0)
        & selected["road_distance_change_km"].notna()
        & selected["current_school_edu_office"].notna()
    ].copy()
    if selected.empty:
        raise ValueError("No rows remain after analysis filtering.")
    if selected["demand_assignment_id"].duplicated().any():
        raise ValueError("Duplicate demand_assignment_id values remain after filtering.")
    return selected


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return float("nan")
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    values_array = values.loc[mask].to_numpy(dtype=float)
    weights_array = weights.loc[mask].to_numpy(dtype=float)
    if not len(values_array):
        return float("nan")
    order = np.argsort(values_array)
    values_array = values_array[order]
    weights_array = weights_array[order]
    cumulative = np.cumsum(weights_array) / weights_array.sum()
    index = min(np.searchsorted(cumulative, q, side="left"), len(values_array) - 1)
    return float(values_array[index])


def summarize_group(group: pd.DataFrame) -> dict[str, object]:
    weights = group["student_weight"]
    time = group["road_duration_change_min"]
    distance = group["road_distance_change_km"]
    return {
        "assignment_row_count": int(len(group)),
        "unique_census_point_count": int(group["census_id"].nunique()),
        "final_cluster_count": int(group["final_cluster_id"].nunique()),
        "estimated_students": float(weights.sum()),
        "mean_time_change_min": weighted_mean(time, weights),
        "median_time_change_min": weighted_quantile(time, weights, 0.50),
        "p90_time_change_min": weighted_quantile(time, weights, 0.90),
        "mean_distance_change_km": weighted_mean(distance, weights),
        "median_distance_change_km": weighted_quantile(distance, weights, 0.50),
        "p90_distance_change_km": weighted_quantile(distance, weights, 0.90),
    }


def regional_metrics(population: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for office, group in population.groupby("current_school_edu_office", sort=True):
        row = {"education_office": office}
        row.update(summarize_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


def classify_decline_group(series: pd.Series, concern_as_decline: bool) -> pd.Series:
    decline_labels = {"인구감소지역"}
    if concern_as_decline:
        decline_labels.add("인구감소관심지역")
    return np.where(series.isin(decline_labels), "인구감소지역", "비인구감소지역")


def decline_metrics(
    population: pd.DataFrame,
    concern_as_decline: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = population.copy()
    working["decline_group_2"] = classify_decline_group(
        working["current_school_population_decline_type"], concern_as_decline
    )

    two_rows = []
    for name, group in working.groupby("decline_group_2", sort=False):
        row = {"population_decline_group": name}
        row.update(summarize_group(group))
        two_rows.append(row)
    two_group = pd.DataFrame(two_rows)
    two_group["student_share_pct"] = (
        two_group["estimated_students"] / two_group["estimated_students"].sum() * 100
    )

    three_rows = []
    for name, group in working.groupby(
        "current_school_population_decline_type", dropna=False, sort=False
    ):
        row = {"source_population_decline_type": name}
        row.update(summarize_group(group))
        three_rows.append(row)
    three_group = pd.DataFrame(three_rows)
    three_group["student_share_pct"] = (
        three_group["estimated_students"]
        / three_group["estimated_students"].sum()
        * 100
    )
    return two_group, three_group


def resolve_font(font_arg: Path | None) -> Path | None:
    if font_arg:
        path = font_arg.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Font not found: {path}")
        return path
    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("/Library/Fonts/AppleGothic.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    ]
    return next((path for path in candidates if path.exists()), None)


def configure_style(font_path: Path | None) -> None:
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = family
    else:
        print("[WARN] Korean font not found; pass --font if needed.", file=sys.stderr)
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = IVORY
    plt.rcParams["axes.facecolor"] = IVORY
    plt.rcParams["savefig.facecolor"] = IVORY
    plt.rcParams["text.color"] = WARM_CHARCOAL
    plt.rcParams["axes.labelcolor"] = WARM_CHARCOAL
    plt.rcParams["xtick.color"] = WARM_GRAY
    plt.rcParams["ytick.color"] = WARM_GRAY


def save_figure_atomic(fig: plt.Figure, destination: Path, dpi: int = 200) -> None:
    temporary = destination.with_name(f".{destination.stem}.tmp.png")
    if temporary.exists():
        temporary.unlink()
    try:
        fig.savefig(temporary, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
        from PIL import Image

        with Image.open(temporary) as check:
            check.verify()
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def short_office_name(value: str) -> str:
    return str(value).replace("교육지원청", "").replace("전남광주통합특별시", "전남광주 ")


def plot_top5(
    top5: pd.DataFrame,
    value_column: str,
    national_value: float,
    title: str,
    unit: str,
    destination: Path,
    threshold_note: str,
) -> None:
    data = top5.sort_values(value_column, ascending=True).copy()
    labels = data["education_office"].map(short_office_name)
    values = data[value_column]

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.32, right=0.96, top=0.78, bottom=0.16)
    bars = ax.barh(labels, values, color=GREEN, height=0.62, zorder=3)
    ax.axvline(
        national_value,
        color=WARM_CHARCOAL,
        linewidth=1.4,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.set_title(title, loc="left", fontsize=21, fontweight="bold", pad=28)
    ax.text(
        0,
        1.03,
        f"전국 가중평균 {national_value:.2f}{unit} 초과 지역 · {threshold_note}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=WARM_GRAY,
    )
    ax.set_xlabel(f"학생 가중평균({unit})", fontsize=11, labelpad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelsize=10)
    maximum = max(float(values.max()), national_value)
    ax.set_xlim(0, maximum * 1.20)
    for bar, value, students in zip(bars, values, data["estimated_students"]):
        ax.text(
            value + maximum * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}{unit} · 약 {students:,.0f}명",
            va="center",
            fontsize=9.5,
            color=WARM_CHARCOAL,
        )
    ax.text(
        national_value,
        -0.12,
        "전국 평균",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        color=WARM_CHARCOAL,
    )
    fig.text(
        0.32,
        0.035,
        "주: 현재 배정학교의 교육지원청 기준. 학생 거주지역 순위가 아님. student_weight 가중평균.",
        fontsize=8.5,
        color=WARM_GRAY,
    )
    save_figure_atomic(fig, destination)
    plt.close(fig)


def plot_decline_comparison(
    table: pd.DataFrame,
    destination: Path,
    concern_as_decline: bool,
) -> None:
    order = ["비인구감소지역", "인구감소지역"]
    data = table.set_index("population_decline_group").reindex(order).reset_index()
    colors = [WARM_GRAY, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.77, bottom=0.20, wspace=0.30)
    fig.suptitle(
        "인구감소지역과 비인구감소지역의 통학부담 비교",
        x=0.09,
        y=0.94,
        ha="left",
        fontsize=22,
        fontweight="bold",
    )
    metrics = [
        ("mean_time_change_min", "편도 통학시간 증가", "분"),
        ("mean_distance_change_km", "도로 통학거리 변화", "km"),
    ]
    for ax, (column, title, unit) in zip(axes, metrics):
        bars = ax.bar(data["population_decline_group"], data[column], color=colors, width=0.60)
        ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=18)
        ax.set_ylabel(f"학생 가중평균({unit})", fontsize=10.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", length=0, labelsize=10)
        maximum = float(data[column].max())
        ax.set_ylim(0, maximum * 1.22)
        for bar, value, students in zip(bars, data[column], data["estimated_students"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + maximum * 0.035,
                f"{value:.2f}{unit}\n약 {students:,.0f}명",
                ha="center",
                va="bottom",
                fontsize=10,
                color=WARM_CHARCOAL,
            )
    definition = (
        "인구감소지역+관심지역"
        if concern_as_decline
        else "인구감소지역만 해당(관심지역은 비인구감소지역에 포함)"
    )
    fig.text(
        0.09,
        0.055,
        f"주: 현재 배정학교의 인구감소지역 분류 기준 · {definition} · student_weight 가중평균.",
        fontsize=8.5,
        color=WARM_GRAY,
    )
    save_figure_atomic(fig, destination)
    plt.close(fig)


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "national_average.csv",
        "education_office_region_metrics.csv",
        "top5_time_above_national.csv",
        "top5_distance_above_national.csv",
        "population_decline_2group.csv",
        "population_decline_3group_reference.csv",
        "population_decline_comparison_summary.csv",
        "01_top5_time_above_national.png",
        "02_top5_distance_above_national.png",
        "03_population_decline_comparison.png",
        "analysis_manifest.json",
    ]
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains outputs. Re-run with --overwrite."
        )
    for path in existing:
        path.unlink()


def main() -> int:
    args = parse_args()
    root = args.analysis_root.expanduser().resolve()
    input_path = (
        args.input.expanduser().resolve()
        if args.input
        else find_input(root, "commute_bound_demand_results.csv")
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "data" / "04_visualization" / "regional_commute_burden"
    )
    if args.min_estimated_students < 0 or args.min_census_points < 1:
        raise ValueError("Minimum sample thresholds must be non-negative/positive.")
    prepare_output(output_dir, args.overwrite)

    print(f"[LOAD] {input_path}")
    raw = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    population = select_population(
        raw, args.bound_type, args.selection_population
    )
    national = summarize_group(population)
    national_table = pd.DataFrame([{"region": "전국", **national}])

    regions = regional_metrics(population)
    regions["above_national_time"] = regions["mean_time_change_min"].gt(
        national["mean_time_change_min"]
    )
    regions["above_national_distance"] = regions["mean_distance_change_km"].gt(
        national["mean_distance_change_km"]
    )
    regions["above_national_both"] = (
        regions["above_national_time"] & regions["above_national_distance"]
    )
    regions["top5_eligible"] = (
        regions["estimated_students"].ge(args.min_estimated_students)
        & regions["unique_census_point_count"].ge(args.min_census_points)
    )

    eligible = regions.loc[regions["top5_eligible"]].copy()
    top5_time = (
        eligible.loc[eligible["above_national_time"]]
        .nlargest(5, "mean_time_change_min")
        .assign(rank_time=lambda frame: range(1, len(frame) + 1))
    )
    top5_distance = (
        eligible.loc[eligible["above_national_distance"]]
        .nlargest(5, "mean_distance_change_km")
        .assign(rank_distance=lambda frame: range(1, len(frame) + 1))
    )
    if len(top5_time) < 5 or len(top5_distance) < 5:
        raise ValueError(
            "Fewer than five eligible above-national regions remain; lower the thresholds."
        )

    decline_two, decline_three = decline_metrics(
        population, concern_as_decline=args.concern_as_decline
    )
    decline_indexed = decline_two.set_index("population_decline_group")
    decline = decline_indexed.loc["인구감소지역"]
    non_decline = decline_indexed.loc["비인구감소지역"]
    comparison = pd.DataFrame(
        [
            {
                "metric": "mean_time_change_min",
                "decline_value": decline["mean_time_change_min"],
                "non_decline_value": non_decline["mean_time_change_min"],
                "difference_decline_minus_non": decline["mean_time_change_min"]
                - non_decline["mean_time_change_min"],
                "percent_difference_vs_non": (
                    decline["mean_time_change_min"]
                    / non_decline["mean_time_change_min"]
                    - 1
                )
                * 100,
                "unit": "minutes",
            },
            {
                "metric": "mean_distance_change_km",
                "decline_value": decline["mean_distance_change_km"],
                "non_decline_value": non_decline["mean_distance_change_km"],
                "difference_decline_minus_non": decline["mean_distance_change_km"]
                - non_decline["mean_distance_change_km"],
                "percent_difference_vs_non": (
                    decline["mean_distance_change_km"]
                    / non_decline["mean_distance_change_km"]
                    - 1
                )
                * 100,
                "unit": "kilometers",
            },
        ]
    )

    configure_style(resolve_font(args.font))
    threshold_note = (
        f"추정 학생 {args.min_estimated_students:,.0f}명·고유 집계구 "
        f"{args.min_census_points:,}개 이상"
    )
    plot_top5(
        top5_time,
        "mean_time_change_min",
        float(national["mean_time_change_min"]),
        "전국 평균보다 통학시간 증가가 큰 교육지원청 Top 5",
        "분",
        output_dir / "01_top5_time_above_national.png",
        threshold_note,
    )
    plot_top5(
        top5_distance,
        "mean_distance_change_km",
        float(national["mean_distance_change_km"]),
        "전국 평균보다 통학거리 증가가 큰 교육지원청 Top 5",
        "km",
        output_dir / "02_top5_distance_above_national.png",
        threshold_note,
    )
    plot_decline_comparison(
        decline_two,
        output_dir / "03_population_decline_comparison.png",
        concern_as_decline=args.concern_as_decline,
    )

    csv_outputs = {
        "national_average.csv": national_table,
        "education_office_region_metrics.csv": regions.sort_values(
            "mean_time_change_min", ascending=False
        ),
        "top5_time_above_national.csv": top5_time,
        "top5_distance_above_national.csv": top5_distance,
        "population_decline_2group.csv": decline_two,
        "population_decline_3group_reference.csv": decline_three,
        "population_decline_comparison_summary.csv": comparison,
    }
    for filename, table in csv_outputs.items():
        table.to_csv(
            output_dir / filename,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        lineterminator="\r\n",
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "input": str(input_path),
        "regional_unit": "current_school_edu_office",
        "regional_unit_note": (
            "Current assigned school's education support office; not student residence region."
        ),
        "filters": {
            "bound_type": args.bound_type,
            "selection_population": args.selection_population,
            "moved_student_flag": True,
            "commute_analysis_eligible_flag": True,
            "time_increase_flag": True,
        },
        "weight": "student_weight",
        "minimum_top5_sample": {
            "estimated_students": args.min_estimated_students,
            "unique_census_points": args.min_census_points,
        },
        "population_decline_two_group_rule": (
            "인구감소지역+인구감소관심지역 vs others"
            if args.concern_as_decline
            else "인구감소지역 vs 인구감소관심지역+미해당"
        ),
        "national": national,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[NATIONAL] time={national['mean_time_change_min']:.3f} min, "
        f"distance={national['mean_distance_change_km']:.3f} km"
    )
    print("[TOP5 TIME]")
    print(
        top5_time[
            ["education_office", "mean_time_change_min", "mean_distance_change_km", "estimated_students"]
        ].to_string(index=False)
    )
    print("[TOP5 DISTANCE]")
    print(
        top5_distance[
            ["education_office", "mean_time_change_min", "mean_distance_change_km", "estimated_students"]
        ].to_string(index=False)
    )
    print("[DECLINE COMPARISON]")
    print(
        decline_two[
            ["population_decline_group", "estimated_students", "mean_time_change_min", "mean_distance_change_km"]
        ].to_string(index=False)
    )
    print(f"[DONE] Outputs saved to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise

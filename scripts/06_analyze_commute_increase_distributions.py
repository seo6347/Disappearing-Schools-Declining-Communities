#!/usr/bin/env python3
"""06. Analyze and visualize commute-time and distance changes after consolidation.

Default analysis population
---------------------------
* lower-bound integration option (LOWER_BOUND)
* option selected using all students (ALL_STUDENTS)
* moved students only
* route-analysis eligible rows only
* students whose one-way commute time increased

The distribution is weighted by ``student_weight``. A census point can appear
in more than one demand-assignment row when its estimated students were split
across current-school assignments; keeping those rows and applying the weights
preserves the estimated student population correctly.

Outputs
-------
* commute_increase_summary.csv
* time_increase_distribution.csv
* distance_change_distribution.csv
* 01_time_increase_distribution.png
* 02_distance_change_distribution.png
* 03_time_distance_distribution_combined.png
* analysis_manifest.json
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

TIME_EDGES = [0, 2, 4, 6, 8, 10, 15, 20, 30, np.inf]
TIME_LABELS = ["0–2", "2–4", "4–6", "6–8", "8–10", "10–15", "15–20", "20–30", "30+"]

# Negative changes are intentionally retained for students whose time rose even
# though their calculated road distance fell.
DISTANCE_EDGES = [-np.inf, 0, 1, 2, 3, 5, 10, 15, 20, np.inf]
DISTANCE_LABELS = ["0 미만", "0–1", "1–2", "2–3", "3–5", "5–10", "10–15", "15–20", "20+"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create student-weighted commute increase distributions."
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("."),
        help="Project root; the input CSV is found recursively.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit path to commute_bound_demand_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: <analysis-root>/data/04_visualization/"
            "commute_increase_distributions"
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
        "--font",
        type=Path,
        default=None,
        help="Optional Korean font (.ttf/.ttc). Malgun Gothic is auto-detected on Windows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
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
    selected["student_weight"] = pd.to_numeric(
        selected["student_weight"], errors="coerce"
    )
    selected["road_duration_change_min"] = pd.to_numeric(
        selected["road_duration_change_min"], errors="coerce"
    )
    selected["road_distance_change_km"] = pd.to_numeric(
        selected["road_distance_change_km"], errors="coerce"
    )
    selected = selected.loc[
        selected["student_weight"].gt(0)
        & selected["road_duration_change_min"].gt(0)
    ].copy()

    if selected.empty:
        raise ValueError("No rows remain after applying the analysis filters.")
    if selected["demand_assignment_id"].duplicated().any():
        duplicates = selected.loc[
            selected["demand_assignment_id"].duplicated(keep=False),
            "demand_assignment_id",
        ].head(10)
        raise ValueError(
            "Duplicate demand_assignment_id values remain after scenario filtering: "
            + ", ".join(duplicates.astype(str))
        )
    return selected


def weighted_quantile(
    values: pd.Series,
    weights: pd.Series,
    quantile: float,
) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    value_array = values.loc[mask].to_numpy(dtype=float)
    weight_array = weights.loc[mask].to_numpy(dtype=float)
    if not len(value_array):
        return float("nan")
    order = np.argsort(value_array)
    value_array = value_array[order]
    weight_array = weight_array[order]
    cumulative = np.cumsum(weight_array) / weight_array.sum()
    index = min(np.searchsorted(cumulative, quantile, side="left"), len(value_array) - 1)
    return float(value_array[index])


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return float("nan")
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def summarize_metric(
    values: pd.Series,
    weights: pd.Series,
    prefix: str,
    unit: str,
) -> list[dict[str, object]]:
    mask = values.notna() & weights.notna() & weights.gt(0)
    values = values.loc[mask]
    weights = weights.loc[mask]
    metrics = {
        "weighted_mean": weighted_mean(values, weights),
        "weighted_p10": weighted_quantile(values, weights, 0.10),
        "weighted_p25": weighted_quantile(values, weights, 0.25),
        "weighted_median": weighted_quantile(values, weights, 0.50),
        "weighted_p75": weighted_quantile(values, weights, 0.75),
        "weighted_p90": weighted_quantile(values, weights, 0.90),
        "weighted_p95": weighted_quantile(values, weights, 0.95),
        "weighted_p99": weighted_quantile(values, weights, 0.99),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }
    return [
        {"metric": f"{prefix}_{key}", "value": value, "unit": unit}
        for key, value in metrics.items()
    ]


def make_distribution(
    df: pd.DataFrame,
    value_column: str,
    edges: Sequence[float],
    labels: Sequence[str],
) -> pd.DataFrame:
    working = df.loc[df[value_column].notna()].copy()
    working["bin"] = pd.cut(
        working[value_column],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
        ordered=True,
    )
    if working["bin"].isna().any():
        raise ValueError(f"Some {value_column} values did not fit the configured bins.")

    rows = []
    total_weight = working["student_weight"].sum()
    for order, label in enumerate(labels):
        group = working.loc[working["bin"].astype("string").eq(label)]
        student_count = float(group["student_weight"].sum())
        rows.append(
            {
                "bin_order": order,
                "bin_label": label,
                "assignment_count": int(len(group)),
                "census_point_count": int(group["census_id"].nunique()),
                "estimated_students": student_count,
                "student_share_pct": (
                    student_count / total_weight * 100 if total_weight else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


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
        print(
            "[WARN] Korean font not found. Pass --font if labels render as squares.",
            file=sys.stderr,
        )
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = IVORY
    plt.rcParams["axes.facecolor"] = IVORY
    plt.rcParams["savefig.facecolor"] = IVORY
    plt.rcParams["text.color"] = WARM_CHARCOAL
    plt.rcParams["axes.labelcolor"] = WARM_CHARCOAL
    plt.rcParams["xtick.color"] = WARM_GRAY
    plt.rcParams["ytick.color"] = WARM_GRAY


def format_students(value: float, _position: int) -> str:
    return f"{value:,.0f}"


def draw_distribution_panel(
    ax: plt.Axes,
    distribution: pd.DataFrame,
    title: str,
    subtitle: str,
    x_label: str,
    negative_first_bar: bool = False,
) -> None:
    positions = np.arange(len(distribution))
    colors = [GREEN] * len(distribution)
    if negative_first_bar and colors:
        colors[0] = WARM_GRAY

    bars = ax.bar(
        positions,
        distribution["estimated_students"],
        color=colors,
        width=0.72,
        edgecolor="none",
        zorder=3,
    )
    ax.set_title(title, loc="left", fontsize=21, fontweight="bold", pad=24)
    ax.text(
        0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=WARM_GRAY,
    )
    ax.set_xticks(positions, distribution["bin_label"])
    ax.set_xlabel(x_label, fontsize=11, labelpad=13)
    ax.set_ylabel("추정 학생 수", fontsize=11, labelpad=13)
    ax.yaxis.set_major_formatter(FuncFormatter(format_students))
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelsize=9.5)
    ax.margins(x=0.025)

    maximum = float(distribution["estimated_students"].max())
    ax.set_ylim(0, maximum * 1.23 if maximum else 1)
    for bar, share in zip(bars, distribution["student_share_pct"]):
        height = bar.get_height()
        if height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + maximum * 0.025,
            f"{share:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=WARM_CHARCOAL,
        )


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


def plot_single(
    distribution: pd.DataFrame,
    title: str,
    subtitle: str,
    x_label: str,
    footnote: str,
    destination: Path,
    negative_first_bar: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.79, bottom=0.19)
    draw_distribution_panel(
        ax,
        distribution,
        title,
        subtitle,
        x_label,
        negative_first_bar=negative_first_bar,
    )
    fig.text(0.09, 0.035, footnote, fontsize=8.5, color=WARM_GRAY, ha="left")
    save_figure_atomic(fig, destination)
    plt.close(fig)


def plot_combined(
    time_distribution: pd.DataFrame,
    distance_distribution: pd.DataFrame,
    time_subtitle: str,
    distance_subtitle: str,
    footnote: str,
    destination: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.875))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.79, bottom=0.18, wspace=0.18)
    fig.suptitle(
        "통합 이후 늘어난 통학부담의 분포",
        x=0.065,
        y=0.94,
        ha="left",
        fontsize=24,
        fontweight="bold",
    )
    draw_distribution_panel(
        axes[0],
        time_distribution,
        "편도 통학시간 증가",
        time_subtitle,
        "증가 시간 구간(분)",
    )
    draw_distribution_panel(
        axes[1],
        distance_distribution,
        "도로 통학거리 변화",
        distance_subtitle,
        "거리 변화 구간(km)",
        negative_first_bar=True,
    )
    fig.text(0.065, 0.035, footnote, fontsize=8.5, color=WARM_GRAY, ha="left")
    save_figure_atomic(fig, destination)
    plt.close(fig)


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "commute_increase_summary.csv",
        "time_increase_distribution.csv",
        "distance_change_distribution.csv",
        "01_time_increase_distribution.png",
        "02_distance_change_distribution.png",
        "03_time_distance_distribution_combined.png",
        "analysis_manifest.json",
    ]
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains analysis outputs. Re-run with --overwrite."
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
        else root / "data" / "04_visualization" / "commute_increase_distributions"
    )
    prepare_output(output_dir, args.overwrite)

    print(f"[LOAD] {input_path}")
    raw = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    population = select_population(
        raw,
        bound_type=args.bound_type,
        selection_population=args.selection_population,
    )

    weights = population["student_weight"]
    time_values = population["road_duration_change_min"]
    distance_values = population["road_distance_change_km"]

    time_distribution = make_distribution(
        population,
        "road_duration_change_min",
        TIME_EDGES,
        TIME_LABELS,
    )
    distance_distribution = make_distribution(
        population,
        "road_distance_change_km",
        DISTANCE_EDGES,
        DISTANCE_LABELS,
    )

    summary_rows: list[dict[str, object]] = [
        {"metric": "assignment_row_count", "value": len(population), "unit": "rows"},
        {
            "metric": "unique_census_point_count",
            "value": population["census_id"].nunique(),
            "unit": "census_points",
        },
        {
            "metric": "final_cluster_count",
            "value": population["final_cluster_id"].nunique(),
            "unit": "clusters",
        },
        {
            "metric": "estimated_student_count",
            "value": weights.sum(),
            "unit": "students",
        },
    ]
    summary_rows.extend(summarize_metric(time_values, weights, "time_change", "minutes"))
    summary_rows.extend(
        summarize_metric(distance_values, weights, "distance_change", "kilometers")
    )
    negative_distance_students = weights.loc[distance_values.lt(0)].sum()
    summary_rows.extend(
        [
            {
                "metric": "distance_decrease_estimated_students",
                "value": negative_distance_students,
                "unit": "students",
            },
            {
                "metric": "distance_decrease_student_share_pct",
                "value": negative_distance_students / weights.sum() * 100,
                "unit": "percent",
            },
        ]
    )
    summary = pd.DataFrame(summary_rows)

    time_mean = weighted_mean(time_values, weights)
    time_median = weighted_quantile(time_values, weights, 0.50)
    time_p90 = weighted_quantile(time_values, weights, 0.90)
    distance_mean = weighted_mean(distance_values, weights)
    distance_median = weighted_quantile(distance_values, weights, 0.50)
    distance_p90 = weighted_quantile(distance_values, weights, 0.90)
    total_students = weights.sum()
    student_label = f"약 {total_students / 10000:.1f}만 명"

    time_subtitle = (
        f"{student_label} · 가중평균 {time_mean:.1f}분 · "
        f"중앙값 {time_median:.1f}분 · 상위 10% 기준 {time_p90:.1f}분"
    )
    distance_subtitle = (
        f"동일 학생 기준 · 가중평균 {distance_mean:.1f}km · "
        f"중앙값 {distance_median:.1f}km · 상위 10% 기준 {distance_p90:.1f}km"
    )
    footnote = (
        "주: 하한안·전체학생 분모 중 통합으로 학교를 옮기고 편도 통학시간이 증가했으며 "
        "경로 분석이 가능한 학생 기준. 막대 높이는 집계구 수가 아니라 student_weight를 합산한 추정 학생 수. "
        "거리 변화의 음수는 시간이 늘었지만 도로거리가 감소한 사례다."
    )

    configure_style(resolve_font(args.font))
    plot_single(
        time_distribution,
        "통합 이후 늘어난 통학시간은 얼마나 되는가",
        time_subtitle,
        "증가 시간 구간(분)",
        footnote,
        output_dir / "01_time_increase_distribution.png",
    )
    plot_single(
        distance_distribution,
        "같은 학생들의 도로 통학거리는 얼마나 달라지는가",
        distance_subtitle,
        "거리 변화 구간(km)",
        footnote,
        output_dir / "02_distance_change_distribution.png",
        negative_first_bar=True,
    )
    plot_combined(
        time_distribution,
        distance_distribution,
        time_subtitle,
        distance_subtitle,
        footnote,
        output_dir / "03_time_distance_distribution_combined.png",
    )

    summary.to_csv(
        output_dir / "commute_increase_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
        lineterminator="\r\n",
    )
    time_distribution.to_csv(
        output_dir / "time_increase_distribution.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
        lineterminator="\r\n",
    )
    distance_distribution.to_csv(
        output_dir / "distance_change_distribution.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
        lineterminator="\r\n",
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "input": str(input_path),
        "filters": {
            "bound_type": args.bound_type,
            "selection_population": args.selection_population,
            "moved_student_flag": True,
            "commute_analysis_eligible_flag": True,
            "time_increase_flag": True,
        },
        "weight": "student_weight",
        "time_metric": "road_duration_change_min",
        "distance_metric": "road_distance_change_km",
        "palette": {
            "background": IVORY,
            "key_green": GREEN,
            "warm_charcoal": WARM_CHARCOAL,
            "warm_gray": WARM_GRAY,
        },
        "counts": {
            "assignment_rows": len(population),
            "unique_census_points": int(population["census_id"].nunique()),
            "estimated_students": float(total_students),
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        "[RESULT] "
        f"{len(population):,} assignment rows, "
        f"{population['census_id'].nunique():,} census points, "
        f"{total_students:,.1f} estimated students"
    )
    print(
        "[RESULT] "
        f"time mean={time_mean:.3f} min, median={time_median:.3f}, p90={time_p90:.3f}"
    )
    print(
        "[RESULT] "
        f"distance mean={distance_mean:.3f} km, median={distance_median:.3f}, "
        f"p90={distance_p90:.3f}"
    )
    print(f"[DONE] Outputs saved to {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise

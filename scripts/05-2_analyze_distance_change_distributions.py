"""
05-2. 거리 변화 집단별 학생 가중 통학거리·통학시간 변화 분포

이 스크립트가 하는 일
1) 확정 시나리오에서 분석할 거리 변화 집단을 선택한다.
2) 집단별 요약과 거리·시간 변화 구간별 학생 가중 분포를 계산하고 그래프를 저장한다.

중요한 분석 원칙
- --population은 INCREASED, DECREASED, CHANGED, ALL_ELIGIBLE을 지원한다.
  기본값 CHANGED는 거리 증가·감소 집단을 각각 분석한다. ALL_ELIGIBLE도
  거리 증가·감소·동일 집단을 각각 분석하며 집단을 합친 평균은 계산하지 않는다.
- 경로 분석 제외 학생은 포함하지 않으며, 각 분포의 분모는 해당 집단의 학생 가중치다.
- 변화량은 통합 후 - 통합 전이다. 거리 집단 안에서도 시간 변화의 음수·0·양수를 보존한다.
- 거리 구간의 양수 경계는 1, 2, 3, 5, 10, 15, 20 km이고,
  시간 구간은 2, 4, 6, 8, 10, 15, 20, 30분이다. 대응하는 음수 구간도 만든다.
- 일반 구간은 하한 포함·상한 제외이며, 0은 독립 구간으로 분리한다.
  따라서 0에서 시작하는 양수 구간은 0을 포함하지 않는다.
  거리 변화만 절댓값 1e-9 km 이내를 분포 집계 시 0으로 처리한다.
- 빈 집단은 학생 가중치 0으로 기록하고 평균·비율은 결측값으로 유지한다.

입력 파일 (data/03_outputs)
- final_integration_demand_assignments.csv: 확정 학생 배정과 통학지표
- final_integration_scenario.csv, final_integration_schools.csv,
  final_integration_manifest.json: 확정 시나리오·학교 상태와 해시 검증

핵심 산출물 (data/03_outputs/stage5/distance_change_distributions/<population>)
- group_summary.csv: 거리 변화 집단별 요약
- distance_change_distribution.csv, time_change_distribution.csv: 집단별 구간 분포
- {집단}_distance_change_distribution.png, {집단}_time_change_distribution.png
- validation_summary.csv, analysis_manifest.json: 검증 결과와 분석 조건·입력 해시
  --output-dir 지정 시에도 분석명과 모집단 하위 경로가 추가된다.

실행 예시 (프로젝트 루트)
    python scripts/05-2_analyze_distance_change_distributions.py
    python scripts/05-2_analyze_distance_change_distributions.py --population ALL_ELIGIBLE
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stage5_commute_common import (CATEGORY_LABELS, DISTANCE_TOLERANCE_KM, OutputBundle,
    argument_parser, bar_plot, check, configure_plot, load_analysis, output_path,
    partition_checks, population_groups, ratio, summarize)

DISTANCE_CUTS = [1, 2, 3, 5, 10, 15, 20]
TIME_CUTS = [2, 4, 6, 8, 10, 15, 20, 30]


def make_distribution(group: pd.DataFrame, column: str, cuts: list[float]) -> pd.DataFrame:
    values = group[column].copy()
    if column == "road_distance_change_km":
        values = values.mask(values.abs().le(DISTANCE_TOLERANCE_KM), 0.)
    edges = [-np.inf, *[-v for v in reversed(cuts)], 0., *cuts, np.inf]
    specs = [(lo, hi, False) for lo, hi in zip(edges[:-1], edges[1:])]
    zero_position = next(i for i, (lo, _, _) in enumerate(specs) if lo == 0)
    specs.insert(zero_position, (0., 0., True))
    rows = []
    denominator = float(group.student_weight.sum())
    for order, (lo, hi, zero) in enumerate(specs):
        mask = values.eq(0) if zero else values.ge(lo) & values.lt(hi) & values.ne(0)
        selected = group.loc[mask]
        lower_label = "-∞" if np.isneginf(lo) else f"{lo:g}"
        upper_label = "∞" if np.isposinf(hi) else f"{hi:g}"
        label = "0" if zero else f"{'(' if lo == 0 or np.isneginf(lo) else '['}{lower_label},{upper_label})"
        weight = float(selected.student_weight.sum())
        rows.append({"metric": column, "bin_order": order, "bin_label": label,
                     "bin_lower": None if np.isneginf(lo) else lo,
                     "bin_upper": None if np.isposinf(hi) else hi,
                     "lower_unbounded": bool(np.isneginf(lo)), "upper_unbounded": bool(np.isposinf(hi)),
                     "lower_inclusive": bool(zero or (lo != 0 and np.isfinite(lo))), "upper_inclusive": zero,
                     "assignment_row_count": len(selected), "unique_census_point_count": selected.census_id.nunique(),
                     "student_weight": weight, "denominator_student_weight": denominator,
                     "student_rate": ratio(weight, denominator)})
    return pd.DataFrame(rows)


def analyze(work: pd.DataFrame, population: str):
    summaries, distances, times = [], [], []
    validation = partition_checks(work)
    for category, group in population_groups(work, population):
        identity = {"population": population, "distance_change_category": category}
        summaries.append({**identity, **summarize(group)})
        for metric, cuts, tables in [("road_distance_change_km", DISTANCE_CUTS, distances),
                                     ("road_duration_change_min", TIME_CUTS, times)]:
            distribution = make_distribution(group, metric, cuts)
            for key, value in reversed(list(identity.items())):
                distribution.insert(0, key, value)
            tables.append(distribution)
            validation.append(check(f"{category}_{metric}_weight", distribution.student_weight.sum(), group.student_weight.sum()))
            validation.append(check(f"{category}_{metric}_rows", distribution.assignment_row_count.sum(), len(group)))
    return pd.DataFrame(summaries), pd.concat(distances, ignore_index=True), pd.concat(times, ignore_index=True), validation


def main() -> None:
    args = argument_parser(__doc__, "CHANGED").parse_args()
    work, hashes = load_analysis(args)
    summary, distances, times, validation = analyze(work, args.population)
    destination = output_path(args, "distance_change_distributions")
    with OutputBundle(destination, args.overwrite) as output:
        output.csv("group_summary.csv", summary)
        output.csv("distance_change_distribution.csv", distances)
        output.csv("time_change_distribution.csv", times)
        output.csv("validation_summary.csv", pd.DataFrame(validation))
        plt = configure_plot(args.font)
        for category in summary.distance_change_category:
            for name, table, label in [("distance", distances, "거리 변화 구간(km)"),
                                        ("time", times, "시간 변화 구간(분)")]:
                selected = table.loc[table.distance_change_category.eq(category)]
                denominator = float(selected.denominator_student_weight.iloc[0])
                bar_plot(plt, output.path / f"{category}_{name}_change_distribution.png",
                         selected.bin_label, selected.student_weight,
                         f"{CATEGORY_LABELS[category]} 학생의 {label}", "학생 가중치",
                         subtitle=f"{category} · 집단 내 분모 {denominator:,.4f} · 음수·제로·양수 포함")
        output.manifest(analysis="distance_change_distributions", input_hashes=hashes, population=args.population,
                        aggregation_units={"group_summary.csv": ["population", "distance_change_category"],
                            "distance_change_distribution.csv": ["population", "distance_change_category", "bin_order"],
                            "time_change_distribution.csv": ["population", "distance_change_category", "bin_order"]},
                        rate_denominators="Each distance-change subgroup independently; no pooled CHANGED mean.")
    print(f"[DONE] {args.population}: {summary[['distance_change_category', 'student_weight']].to_dict('records')}; {destination}")


if __name__ == "__main__":
    main()

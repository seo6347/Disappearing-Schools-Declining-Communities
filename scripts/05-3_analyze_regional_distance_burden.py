"""
05-3. 현재 배정학교 지역·교육지원청별 통학부담 비교

이 스크립트가 하는 일
1) 확정 시나리오의 거리 변화 집단별 전국·지역·교육지원청 통학부담을 계산한다.
2) 전국 평균을 초과하는 교육지원청 상위 5개를 지표별로 선정하고 비교 그래프를 저장한다.

중요한 분석 원칙
- 학생의 지역과 교육지원청은 현재 배정학교 소재지를 기준으로 한다.
  인구감소지역은 DECLINE, 인구감소관심지역·미해당은 NON_DECLINE으로 분류하고,
  결측·미확인·인식하지 못한 지역 구분은 UNKNOWN으로 별도 집계한다.
- --population의 기본값은 INCREASED이다. DECREASED, CHANGED, ALL_ELIGIBLE도
  지원하며, 선택된 거리 증가·감소·동일 집단별로 독립 분석한다.
- 경로 분석 제외 학생은 포함하지 않는다. 평균은 Σ(값 × student_weight)
  / Σ(student_weight)이며, 지역별 학생 비율의 분모는 해당 거리 집단의 학생 가중치다.
- 상위 5개는 거리 변화와 시간 변화 각각의 학생 가중평균이 해당 집단의 전국 평균을
  엄격히 초과하는 교육지원청에서 선정한다. 학생 가중치 100 이상·고유 집계구 10개
  이상이 기본 조건이며 --min-estimated-students, --min-census-points로 조정한다.
- 순위는 가중평균 내림차순, 동률은 교육지원청명 오름차순이다.
  조건을 충족하는 교육지원청이 적으면 5개 미만을 기록한다.
- 지역별 학교·군집·교육지원청 고유 개수는 지역 간 중복될 수 있어 단순 합산하지 않는다.

입력 파일 (data/03_outputs)
- final_integration_demand_assignments.csv: 확정 학생 배정과 통학지표·학교 지역
- final_integration_scenario.csv, final_integration_schools.csv,
  final_integration_manifest.json: 확정 시나리오·학교 상태와 해시 검증

핵심 산출물 (data/03_outputs/stage5/regional_distance_burden/<population>)
- national_average.csv, education_office_metrics.csv: 집단별 전국·교육지원청 통계
- population_decline_metrics.csv: 인구감소지역·비인구감소지역·미확인 비교
- population_decline_source_metrics.csv: 원래 인구감소지역 구분별 통계
- top5_above_national.csv: 거리·시간 지표별 전국 평균 초과 교육지원청 순위
- {집단}_top5_{지표}.png, {집단}_population_decline_{지표}.png
- validation_summary.csv, analysis_manifest.json: 검증 결과와 분석 조건·입력 해시
  --output-dir 지정 시에도 분석명과 모집단 하위 경로가 추가된다.

실행 예시 (프로젝트 루트)
    python scripts/05-3_analyze_regional_distance_burden.py
    python scripts/05-3_analyze_regional_distance_burden.py --population DECREASED
"""
from __future__ import annotations

import pandas as pd

from stage5_commute_common import (CATEGORY_LABELS, REGIONS, REGION_LABELS, OutputBundle,
    argument_parser, bar_plot, check, configure_plot, load_analysis, output_path,
    partition_checks, population_groups, ratio, summarize)


def analyze(work: pd.DataFrame, population: str, min_students: float = 100, min_points: int = 10):
    if min_students < 0 or min_points < 1:
        raise ValueError("Invalid minimum sample thresholds")
    national_rows, office_rows, region_rows, source_rows, rankings = [], [], [], [], []
    validation = partition_checks(work)
    for category, group in population_groups(work, population):
        identity = {"population": population, "distance_change_category": category}
        national = summarize(group)
        national_rows.append({**identity, **national})
        current_offices = []
        for office, selected in group.groupby("current_school_edu_office", sort=True):
            row = {**identity, "current_school_edu_office": office, **summarize(selected)}
            row["top5_eligible"] = row["student_weight"] >= min_students and row["unique_census_point_count"] >= min_points
            for metric in ["distance_change_km", "duration_change_min"]:
                row[f"above_national_{metric}"] = row[f"weighted_mean_{metric}"] > national[f"weighted_mean_{metric}"]
            current_offices.append(row)
        office_rows.extend(current_offices)
        for metric in ["distance_change_km", "duration_change_min"]:
            candidates = [r for r in current_offices if r["top5_eligible"] and r[f"above_national_{metric}"]]
            candidates.sort(key=lambda r: (-r[f"weighted_mean_{metric}"], r["current_school_edu_office"]))
            for rank, row in enumerate(candidates[:5], 1):
                rankings.append({**row, "ranking_metric": metric, "rank": rank})
        for region in REGIONS:
            selected = group.loc[group.current_school_region_group.eq(region)]
            region_rows.append({**identity, "current_school_region_group": region, **summarize(selected),
                                "share_of_distance_group_students": ratio(selected.student_weight.sum(), group.student_weight.sum())})
        for source, selected in group.groupby(group.current_school_population_decline_type.fillna("UNKNOWN"), sort=True):
            source_rows.append({**identity, "current_school_population_decline_type": source, **summarize(selected)})
        validation.append(check(f"{category}_office_student_weight", sum(r['student_weight'] for r in current_offices), national['student_weight']))
        validation.append(check(f"{category}_region_student_weight", sum(r['student_weight'] for r in region_rows if r['distance_change_category'] == category), national['student_weight']))
    office_columns = ["population", "distance_change_category", "current_school_edu_office", *summarize(work.iloc[:0]),
                      "top5_eligible", "above_national_distance_change_km", "above_national_duration_change_min"]
    return {"national_average.csv": pd.DataFrame(national_rows),
            "education_office_metrics.csv": pd.DataFrame(office_rows, columns=office_columns),
            "population_decline_metrics.csv": pd.DataFrame(region_rows),
            "population_decline_source_metrics.csv": pd.DataFrame(source_rows, columns=[
                "population", "distance_change_category", "current_school_population_decline_type", *summarize(work.iloc[:0])]),
            "top5_above_national.csv": pd.DataFrame(rankings, columns=[*office_columns, "ranking_metric", "rank"]),
            "validation_summary.csv": pd.DataFrame(validation)}


def main() -> None:
    parser = argument_parser(__doc__, "INCREASED")
    parser.add_argument("--min-estimated-students", type=float, default=100)
    parser.add_argument("--min-census-points", type=int, default=10)
    args = parser.parse_args()
    work, hashes = load_analysis(args)
    tables = analyze(work, args.population, args.min_estimated_students, args.min_census_points)
    destination = output_path(args, "regional_distance_burden")
    with OutputBundle(destination, args.overwrite) as output:
        for name, table in tables.items():
            output.csv(name, table)
        plt = configure_plot(args.font)
        for _, national in tables["national_average.csv"].iterrows():
            category = national.distance_change_category
            regions = tables["population_decline_metrics.csv"]
            regions = regions.loc[regions.distance_change_category.eq(category)]
            for metric, label, unit in [("distance_change_km", "거리 변화", "km"), ("duration_change_min", "시간 변화", "분")]:
                ranked = tables["top5_above_national.csv"]
                ranked = ranked.loc[ranked.distance_change_category.eq(category) & ranked.ranking_metric.eq(metric)]
                bar_plot(plt, output.path / f"{category}_top5_{metric}.png", ranked.current_school_edu_office,
                         ranked[f"weighted_mean_{metric}"], f"{CATEGORY_LABELS[category]} 학생: 전국 평균 초과 교육지원청", unit,
                         national=national[f"weighted_mean_{metric}"], horizontal=True,
                         subtitle=f"{label} · 학생 가중치 {args.min_estimated_students:g} 이상·집계구 {args.min_census_points}개 이상")
                bar_plot(plt, output.path / f"{category}_population_decline_{metric}.png",
                         regions.current_school_region_group.map(REGION_LABELS), regions[f"weighted_mean_{metric}"],
                         f"{CATEGORY_LABELS[category]} 학생: 현재 배정학교 지역별 {label}", unit,
                         subtitle="인구감소관심지역·미해당은 비인구감소지역 · 미확인은 별도 집계")
        output.manifest(analysis="regional_distance_burden", input_hashes=hashes, population=args.population,
                        aggregation_units={"national_average.csv": ["distance_change_category"],
                            "education_office_metrics.csv": ["distance_change_category", "current_school_edu_office"],
                            "population_decline_metrics.csv": ["distance_change_category", "current_school_region_group"],
                            "population_decline_source_metrics.csv": ["distance_change_category", "current_school_population_decline_type"],
                            "top5_above_national.csv": ["distance_change_category", "ranking_metric", "rank"]},
                        minimum_top5_sample={"student_weight": args.min_estimated_students, "census_points": args.min_census_points},
                        rate_denominators="Means are local subgroup means; regional shares use the corresponding distance group.",
                        unique_count_note="Student-region school/cluster/office counts may overlap across regions; do not sum them.")
    print(f"[DONE] {args.population}; {destination}")


if __name__ == "__main__":
    main()

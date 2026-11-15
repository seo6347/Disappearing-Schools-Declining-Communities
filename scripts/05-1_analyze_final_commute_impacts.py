"""
05-1. 확정 시나리오의 전국 통학부담과 거리 변화 집단별 요약

이 스크립트가 하는 일
- Stage 4의 확정 학생 배정을 읽어 전국 통학거리·통학시간 변화와
  거리 증가·감소·동일 집단별 학생 가중 통계를 산출하고 집단 분할을 검증한다.

중요한 분석 원칙
- 전체 분석 가능 학생을 사용하며 학교 이동 여부로 제한하지 않는다.
  commute_analysis_eligible_flag가 거짓인 학생은 제외 가중치로 별도 기록한다.
- 거리 변화는 통합 후 - 통합 전이며, 절댓값 1e-9 km 이내는 거리 동일로 분류한다.
- 집단별 학생 비율의 분모는 전체 분석 가능 학생 가중치이고,
  집단별 평균·분위수는 해당 집단의 student_weight로 계산한다.
- 존치학교를 다시 선정하지 않으며 --population 옵션은 제공하지 않는다.

입력 파일 (data/03_outputs)
- final_integration_demand_assignments.csv: 확정 학생 배정과 통학지표
- final_integration_scenario.csv, final_integration_schools.csv,
  final_integration_manifest.json: 확정 시나리오·학교 상태와 해시 검증

핵심 산출물 (data/03_outputs/stage5/commute_impacts)
- demand_results.csv: 제외 대상을 포함한 배정별 거리·지역 분류
- national_summary.csv, distance_group_summary.csv: 전국·거리 변화 집단별 요약
- validation_summary.csv, analysis_manifest.json: 검증 결과와 분석 조건·입력 해시

실행 예시 (프로젝트 루트)
    python scripts/05-1_analyze_final_commute_impacts.py
"""
from __future__ import annotations

import pandas as pd

from stage5_commute_common import (CATEGORIES, DISTANCE_TOLERANCE_KM, OutputBundle,
    argument_parser, check, distance_categories, load_analysis, output_path,
    partition_checks, prepare_detail, ratio, summarize, weighted_mean, weighted_quantile)


def analyze(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = prepare_detail(detail)
    eligible = work.loc[work.commute_analysis_eligible_flag]
    total = float(work.student_weight.sum())
    available = float(eligible.student_weight.sum())
    row = {"population": "ALL_ELIGIBLE", **summarize(eligible),
           "student_weight_total": total, "analysis_eligible_student_weight": available,
           "analysis_excluded_student_weight": float(work.loc[~work.commute_analysis_eligible_flag, "student_weight"].sum()),
           "analysis_student_coverage_rate": ratio(available, total)}
    for category in CATEGORIES:
        weight = float(eligible.loc[eligible.distance_change_category.eq(category), "student_weight"].sum())
        row[f"{category.lower()}_student_weight"] = weight
        row[f"{category.lower()}_student_rate"] = ratio(weight, available)
    return work, pd.DataFrame([row])


def main() -> None:
    args = argument_parser(__doc__).parse_args()
    work, hashes = load_analysis(args)
    result, national = analyze(work)
    eligible_weight = float(result.loc[result.commute_analysis_eligible_flag, "student_weight"].sum())
    groups = []
    for category in CATEGORIES:
        group = result.loc[result.distance_change_category.eq(category)]
        groups.append({"population": "ALL_ELIGIBLE", "distance_change_category": category,
                       **summarize(group), "all_eligible_student_weight": eligible_weight,
                       "share_of_all_eligible_students": ratio(group.student_weight.sum(), eligible_weight)})
    validation = partition_checks(result)
    validation.append(check("category_shares_sum", sum(row["student_weight"] for row in groups), eligible_weight))
    destination = output_path(args, "commute_impacts")
    with OutputBundle(destination, args.overwrite) as output:
        output.csv("demand_results.csv", result.drop(columns=["scenario_id"], errors="ignore"))
        output.csv("national_summary.csv", national)
        output.csv("distance_group_summary.csv", pd.DataFrame(groups))
        output.csv("validation_summary.csv", pd.DataFrame(validation))
        output.manifest(analysis="commute_impacts", input_hashes=hashes, population="ALL_ELIGIBLE",
                        aggregation_units={"demand_results.csv": ["demand_assignment_id"],
                                           "national_summary.csv": ["population"],
                                           "distance_group_summary.csv": ["distance_change_category"]},
                        rate_denominators="Category shares: all eligible students; subgroup means: that subgroup only.")
    print(f"[DONE] {len(result)} frozen assignments; eligible weight={eligible_weight:.10f}; {destination}")


if __name__ == "__main__":
    main()

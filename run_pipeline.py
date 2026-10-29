"""Run five analysis stages using archived route data, without API credentials."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def execute(root: Path, filename: str, *options: str) -> None:
    log = root / "logs" / (Path(filename).stem + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {filename} (log: {log.relative_to(root)})", flush=True)
    command = [sys.executable, str(ROOT / "scripts" / filename), *options]
    with log.open("w", encoding="utf-8") as out:
        result = subprocess.run(command, cwd=root, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8").splitlines()[-25:])
        raise RuntimeError(f"{filename} failed (exit {result.returncode}).\n{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "1", "2", "3", "4", "5"], default="all")
    parser.add_argument("--project-root", type=Path, default=ROOT,
                        help="Data/output root; defaults to the folder containing this runner.")
    parser.add_argument("--overwrite", action="store_true", help="Replace Step 06/07 generated outputs.")
    parser.add_argument("--font", type=Path, help="Optional Korean font file for Step 06/07 plots.")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    raw = root / "data" / "01_raw"
    required = ["elementary_master.csv", "school_area_master.csv", "census_demand_points.csv",
                "elementary_polygon.gpkg", "census_demand_points.gpkg",
                "api_routes/school_road_routes.csv",
                "team_route_results/naver_commute_routes_B01_of_02.csv",
                "team_route_results/naver_commute_routes_B02_of_02.csv"]
    missing = [str(raw / p) for p in required if not (raw / p).is_file()]
    if missing:
        raise FileNotFoundError("Missing prepared input files:\n" + "\n".join(missing))
    base = ["--project-root", str(root)]
    stages = range(1, 6) if args.stage == "all" else [int(args.stage)]
    for stage in stages:
        print(f"\n[STAGE {stage}/5]", flush=True)
        if stage == 1:
            execute(root, "01_estimate_current_student_distribution.py", *base)
            execute(root, "01-1_analyze_no_zone_sensitivity.py", *base)
        elif stage == 2:
            execute(root, "02-1_analyze_distance_graph_sensitivity.py", *base)
            execute(root, "02-2_build_adjacency_graph_and_triples.py", *base)
        elif stage == 3:
            # Retain the original early CAP28/CAP19 comparison. Downstream uses CAP28.
            execute(root, "03-1_evaluate_capacity_scenarios.py", *base, "--students-per-class", "28", "19")
            processed = root / "data" / "02_processed"
            processed.mkdir(parents=True, exist_ok=True)
            # Use the completed historical API results; never request new routes here.
            shutil.copy2(raw / "api_routes" / "school_road_routes.csv", processed / "school_road_routes.csv")
            # Without --execute the original collector validates/joins cached results only.
            execute(root, "03-2_fetch_school_road_routes.py", *base)
            import pandas as pd
            cached = pd.read_csv(processed / "school_road_routes.csv", dtype=str)
            if not cached["route_status"].eq("SUCCESS").all():
                raise ValueError("Archived school routes do not cover the generated requests; no API fallback is allowed.")
            for filename in ["03-3_analyze_road_access_sensitivity.py", "03-4_finalize_road_access_options.py",
                             "03-5_select_nonoverlapping_clusters.py"]:
                execute(root, filename, *base)
        elif stage == 4:
            # Batch count must match the two historical collection files.
            execute(root, "04-1_prepare_commute_route_requests.py", *base, "--scenarios", "CAP28", "--batch-count", "2")
            # 04-2 is deliberately not run: 04-3 reads archived team API results.
            for filename in ["04-3_merge_team_commute_routes.py", "04-4_calculate_commute_time_changes.py",
                             "04-5_select_commute_bounds.py"]:
                execute(root, filename, *base)
        else:
            execute(root, "05_analyze_school_voids.py", *base)
            opts = ["--analysis-root", str(root), "--input", str(root / "data/03_outputs/commute_bound_demand_results.csv")]
            if args.overwrite:
                opts.append("--overwrite")
            if args.font:
                opts.extend(["--font", str(args.font.expanduser().resolve())])
            for filename in ["06_analyze_commute_increase_distributions.py", "07_analyze_regional_commute_burden.py"]:
                execute(root, filename, *opts)
    print("\n[DONE] Requested analysis stages completed. No live API collection was requested.", flush=True)


if __name__ == "__main__":
    main()

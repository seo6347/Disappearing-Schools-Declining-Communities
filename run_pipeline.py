"""Run six analysis stages using archived route data, without API credentials."""
from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--stage", choices=["all", "1", "2", "3", "4", "5", "6"], default="all")
    parser.add_argument("--project-root", type=Path, default=ROOT,
                        help="Data/output root; defaults to the folder containing this runner.")
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs only within each selected Stage 5 analysis/population.")
    parser.add_argument("--font", type=Path, help="Optional Korean font file for Stage 5 supplementary plots.")
    parser.add_argument("--source-requests", type=Path,
                        help="Original collection request CSV whose SHA-256 is recorded in archived commute results.")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    raw = root / "data" / "01_raw"
    required = ["elementary_master.csv", "school_area_master.csv", "census_demand_points.csv",
                "elementary_polygon.gpkg", "census_demand_points.gpkg",
                "api_routes/school_road_routes.csv",
                "team_route_results/naver_commute_routes_B01_of_02.csv",
                "team_route_results/naver_commute_routes_B02_of_02.csv"]
    # Influence analysis reads frozen choices, not the route collection inputs.
    # Infrastructure analysis additionally needs the original population/geography.
    checked_inputs = (
        [] if args.stage == "5" else
        ["elementary_master.csv", "census_demand_points.csv", "census_demand_points.gpkg"]
        if args.stage == "6" else required
    )
    missing = [str(raw / p) for p in checked_inputs if not (raw / p).is_file()]
    if missing:
        raise FileNotFoundError("Missing prepared input files:\n" + "\n".join(missing))
    base = ["--project-root", str(root)]
    stages = range(1, 7) if args.stage == "all" else [int(args.stage)]
    for stage in stages:
        print(f"\n[STAGE {stage}/6]", flush=True)
        if stage == 1:
            execute(root, "01_estimate_current_student_distribution.py", *base)
        elif stage == 2:
            execute(root, "02-1_analyze_distance_graph_sensitivity.py", *base)
            execute(root, "02-2_build_adjacency_graph_and_triples.py", *base)
        elif stage == 3:
            # Evaluate the supported CAP28 capacity assumption.
            execute(root, "03-1_evaluate_capacity_scenarios.py", *base, "--students-per-class", "28")
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
            source_requests = (
                args.source_requests.expanduser().resolve() if args.source_requests
                else raw / "team_route_results" / "commute_route_requests_source.csv"
            )
            if not source_requests.is_file() and not args.source_requests:
                source_requests = root / "data/02_processed/commute_route_requests_source.csv"
                if not source_requests.is_file():
                    # Preserve an existing, authenticated collection input before 04-1 replaces it.
                    previous_requests = root / "data/03_outputs/commute_route_requests.csv"
                    if previous_requests.is_file():
                        import pandas as pd
                        source_hash = hashlib.sha256(previous_requests.read_bytes()).hexdigest()
                        for batch in required[-2:]:
                            hashes = pd.read_csv(raw / batch, usecols=["source_request_file_sha256"])
                            if not hashes["source_request_file_sha256"].eq(source_hash).all():
                                raise ValueError("Existing request file does not match the archived collection source hash.")
                        source_requests.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(previous_requests, source_requests)
            if not source_requests.is_file():
                raise FileNotFoundError("Preserve the original collection input and provide --source-requests before Stage 4.")
            if source_requests.resolve() == (root / "data/03_outputs/commute_route_requests.csv").resolve():
                raise ValueError("--source-requests must be a preserved copy outside the generated request output path.")
            # Batch count must match the two historical collection files.
            execute(root, "04-1_prepare_commute_route_requests.py", *base, "--scenarios", "CAP28", "--batch-count", "2")
            # 04-2 is deliberately not run: 04-3 reads archived team API results.
            for filename in ["04-3_merge_team_commute_routes.py", "04-4_calculate_commute_time_changes.py",
                             "04-5_select_final_retained_schools.py"]:
                options = ["--source-requests", str(source_requests)] if filename.startswith("04-3") else []
                execute(root, filename, *base, *options)
        elif stage == 5:
            opts = list(base)
            if args.overwrite:
                opts.append("--overwrite")
            if args.font:
                opts.extend(["--font", str(args.font.expanduser().resolve())])
            for filename in ["05-1_analyze_final_commute_impacts.py",
                             "05-2_analyze_distance_change_distributions.py",
                             "05-3_analyze_regional_distance_burden.py",
                             "05-4_analyze_commute_time_thresholds.py"]:
                execute(root, filename, *opts)
        else:
            execute(root, "06_analyze_school_voids.py", *base)
    print("\n[DONE] Requested analysis stages completed. No live API collection was requested.", flush=True)


if __name__ == "__main__":
    main()

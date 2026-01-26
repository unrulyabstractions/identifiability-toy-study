import argparse
import datetime
import os
import sys
from pathlib import Path

from identifiability_toy_study.common.schemas import (
    ExperimentConfig,
    ExperimentResult,
)
from identifiability_toy_study.common.utils import setup_logging
from identifiability_toy_study.experiment import run_experiment
from identifiability_toy_study.persistence import (
    get_all_runs,
    load_results,
    save_results,
)
from identifiability_toy_study.profiler import print_profile
from identifiability_toy_study.visualization import visualize_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for data/model. Use cpu, mps (Apple Silicon), or cuda:0 (NVIDIA)",
    )
    parser.add_argument(
        "--spd-device",
        type=str,
        default="cpu",
        help="Device for SPD decomposition. CPU is fastest for small models",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Re-run visualization on existing runs without training",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization step",
    )
    args = parser.parse_args()

    # Output path
    output_dir = Path("runs")
    os.makedirs(output_dir, exist_ok=True)

    if args.analysis_only:
        # Re-run visualization on all runs
        runs = get_all_runs(output_dir)
        if not runs:
            print("No runs found")
            sys.exit(1)
        print(f"Found {len(runs)} runs to analyze")
        for run_dir in runs:
            print(f"Re-analyzing {run_dir.name}...")
            result = load_results(run_dir)
            if not result.trials:
                print("  No trials found, skipping")
                continue
            print(f"  Loaded {len(result.trials)} trials")
            result.viz_paths = visualize_experiment(result, run_dir=str(run_dir))
            print("  Visualization complete")
        print(f"Finished analyzing {len(runs)} runs")
        sys.exit(0)

    # We identify run by time
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Run path
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)

    # File paths
    log_path = os.path.join(run_dir, "log.txt")

    # Logger
    logger = setup_logging(log_path, args.debug)

    # Run Experiment with many trials
    cfg = ExperimentConfig(
        debug=args.debug,
        device=args.device,
        spd_device=args.spd_device,
    )

    result: ExperimentResult = run_experiment(cfg, logger=logger)
    save_results(result, run_dir=run_dir)  # Save before just in case viz crashes

    if not args.no_viz:
        result.viz_paths = visualize_experiment(result, run_dir=run_dir)
        # Save again but with viz_paths
        save_results(result, run_dir=run_dir)

    # Print summary
    logger.info(f"\n\n\n run_{timestamp} \n")
    summary = result.print_summary()
    logger.info(summary)

    # Print profiling
    print_profile()

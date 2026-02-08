import argparse
import datetime
import os
import sys
from pathlib import Path

from src.experiment import run_experiment
from src.infra import print_profile, profile, profile_fn, setup_logging
from src.persistence import (
    get_all_runs,
    load_results,
    save_results,
)
from src.schemas import (
    ExperimentConfig,
    ExperimentResult,
)
from src.spd import SpdResults, load_spd_results, run_spd, save_spd_results
from src.viz import visualize_experiment, visualize_spd_experiment


def get_args():
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
        "--spd-only",
        action="store_true",
        help="Re-run SPD on existing runs without training",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization step",
    )
    parser.add_argument(
        "--spd",
        action="store_true",
        help="Enable SPD decomposition analysis (disabled by default)",
    )
    args = parser.parse_args()

    # args.spd_only should turn on args.spd
    args.spd = args.spd or args.spd_only

    return args


@profile_fn("rerun_all")
def rerun_all(output_dir, process_fn, process_name=""):
    """Re-run processing on all runs."""
    runs = get_all_runs(output_dir)
    if not runs:
        print("No runs found")
        sys.exit(1)
    print(f"Found {len(runs)} runs to re-run {process_name}")
    for run_dir in runs:
        print(f"Re-analyzing {run_dir.name}...")
        result = load_results(str(run_dir))
        if not result.trials:
            print("  No trials found, skipping")
            continue
        print(f"  Loaded {len(result.trials)} trials")
        process_fn(result, run_dir)
        print("  Processing complete")
    print(f"Finished analyzing {len(runs)} runs")


@profile_fn("do_viz")
def do_viz(result, run_dir, spd):
    visualize_experiment(result, run_dir=run_dir)
    if spd:
        spd_result = load_spd_results(run_dir)
        if spd_result:
            visualize_spd_experiment(spd_result, run_dir=run_dir)


@profile_fn("do_spd")
def do_spd(result, run_dir, viz, spd_device):
    spd_result: SpdResults = run_spd(result, run_dir=run_dir, device=spd_device)
    save_spd_results(spd_result, run_dir=run_dir)
    if viz:
        visualize_spd_experiment(spd_result, run_dir=run_dir)
    return spd_result


def rerun_pipeline(output_dir, args) -> bool:
    did_rerun_pipeline = False
    if args.spd_only:
        do_fx = lambda *a: do_spd(*a, viz=not args.no_viz, spd_device=args.spd_device)
        rerun_all(output_dir, do_fx, "spd")
        did_rerun_pipeline = True
    if args.analysis_only:
        do_fx = lambda *a: do_viz(*a, spd=args.spd)
        rerun_all(output_dir, do_fx, "analysis")
        did_rerun_pipeline = True
    return did_rerun_pipeline


def print_summary(experiment_result, spd_result, logger):
    logger.info("\n\n\n\n")
    logger.info("experiment_result")
    logger.info("\n\n\n\n")
    summary = experiment_result.print_summary()
    logger.info(summary)
    if spd_result:
        logger.info("\n\n\n\n")
        logger.info("spd_result")
        logger.info("\n\n\n\n")
        spd_summary = spd_result.print_summary()
        logger.info(spd_summary)


def exit_fx():
    # Print profiling
    print_profile()
    sys.exit(0)


if __name__ == "__main__":
    args = get_args()

    # Output path
    output_dir = Path("runs")
    os.makedirs(output_dir, exist_ok=True)

    # Alternative use of main to iterate on analysis/spd
    if rerun_pipeline(output_dir, args):
        exit_fx()

    # We identify run by time
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Run path
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)

    # File paths
    log_path = os.path.join(run_dir, "log.txt")

    # Logger
    logger = setup_logging(log_path, args.debug)

    # Default config
    cfg = ExperimentConfig(
        debug=args.debug,
        device=args.device,
    )

    # Run Experiment with many trials
    with profile("run_experiment"):
        experiment_result: ExperimentResult = run_experiment(cfg, logger=logger)
        save_results(
            experiment_result, run_dir=run_dir
        )  # Save before just in case viz crashes

    # Other experiments + viz
    spd_result = None
    if args.spd:
        spd_result = do_spd(
            experiment_result, run_dir, viz=not args.no_viz, spd_device=args.spd_device
        )
    if not args.no_viz:
        do_viz(experiment_result, run_dir, spd=args.spd)

    print_summary(experiment_result, spd_result, logger)

    exit_fx()

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Any

from src.experiment import build_trial_settings
from src.experiment_config import ExperimentConfig, set_test_mode_global
from src.infra import print_profile, profile_fn, setup_logging
from src.infra.profiler import Trace
from src.persistence import get_all_runs, load_results
from src.pipeline import (
    do_spd_on_experiment,
    do_viz_on_experiment,
    run_iteratively,
    run_monolith,
)


def get_args() -> Any:
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
        "--test",
        action="store_true",
        help="Enable fast test mode (smaller datasets, fewer epochs)",
    )
    parser.add_argument(
        "--test-gates",
        type=int,
        default=0,
        help="Test gate configuration: -1=ALL, 0=XOR, 1=OR+AND, 2=XOR+XOR, 3=ID+IMP, 4=ID+MAJ, 5=XOR+MAJ",
    )
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
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run directory to process (for --analysis-only or --spd-only)",
    )
    parser.add_argument(
        "--trial",
        type=str,
        default=None,
        help="Specific trial ID to process across runs (for --analysis-only or --spd-only)",
    )
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="Force iterative mode (save after each trial)",
    )
    args = parser.parse_args()

    # args.spd_only should turn on args.spd
    args.spd = args.spd or args.spd_only

    return args


def get_target_runs(output_dir: Path, run_filter: str | None) -> list[Path]:
    """Get run directories, optionally filtered to a specific run."""
    all_runs = get_all_runs(output_dir)
    if run_filter is None:
        return all_runs
    # Match by exact name or by suffix (e.g., "run_20260208-141100" or just the directory name)
    for run_dir in all_runs:
        if run_dir.name == run_filter or run_dir.name == f"run_{run_filter}":
            return [run_dir]
    # Also check if it's a full path
    run_path = Path(run_filter)
    if run_path.exists() and run_path.is_dir():
        return [run_path]
    run_path = output_dir / run_filter
    if run_path.exists() and run_path.is_dir():
        return [run_path]
    print(f"Run directory not found: {run_filter}")
    return []


def filter_result_trials(result, trial_filter: str | None):
    """Filter trials in a result to only include matching trial IDs."""
    if trial_filter is None:
        return result
    # Filter to only matching trials
    filtered_trials = {
        tid: trial for tid, trial in result.trials.items() if tid == trial_filter
    }
    result.trials = filtered_trials
    return result


@profile_fn("rerun_all")
def rerun_all(
    output_dir: Path,
    process_fn: Any,
    process_name: str = "",
    run_filter: str | None = None,
    trial_filter: str | None = None,
) -> None:
    """Re-run processing on runs, optionally filtered by run and/or trial."""
    runs = get_target_runs(output_dir, run_filter)
    if not runs:
        print("No runs found")
        sys.exit(1)

    filter_desc = ""
    if run_filter:
        filter_desc += f" (run: {run_filter})"
    if trial_filter:
        filter_desc += f" (trial: {trial_filter})"

    print(f"Found {len(runs)} run(s) to re-run {process_name}{filter_desc}")

    total_trials_processed = 0
    for run_dir in runs:
        print(f"Re-analyzing {run_dir.name}...")
        result = load_results(str(run_dir))
        result = filter_result_trials(result, trial_filter)
        if not result.trials:
            print("  No matching trials found, skipping")
            continue
        print(f"  Processing {len(result.trials)} trial(s)")
        process_fn(result, run_dir)
        total_trials_processed += len(result.trials)
        print("  Processing complete")

    print(
        f"Finished analyzing {total_trials_processed} trial(s) across {len(runs)} run(s)"
    )


def rerun_pipeline(output_dir: Path, args: Any) -> bool:
    """Handle --analysis-only and --spd-only modes."""
    did_rerun_pipeline = False
    if args.spd_only:
        do_fx = lambda result, run_dir: do_spd_on_experiment(
            result, str(run_dir), viz=not args.no_viz, spd_device=args.spd_device
        )
        rerun_all(
            output_dir, do_fx, "spd", run_filter=args.run, trial_filter=args.trial
        )
        did_rerun_pipeline = True
    if args.analysis_only:
        do_fx = lambda result, run_dir: do_viz_on_experiment(
            result, str(run_dir), spd=args.spd
        )
        rerun_all(
            output_dir, do_fx, "analysis", run_filter=args.run, trial_filter=args.trial
        )
        did_rerun_pipeline = True
    return did_rerun_pipeline


def exit_fx() -> None:
    """Print profiling and exit."""
    print_profile()
    sys.exit(0)


def run_single_experiment(args: Any, output_dir: Path) -> None:
    """Run a single experiment with the current test mode settings."""
    # We identify run by time (with microseconds to avoid collisions in parallel runs)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")

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

    # Determine execution mode based on trial count
    trial_settings, _ = build_trial_settings(cfg, logger)
    n_trials = len(trial_settings)

    use_iterative = args.iterative or n_trials > cfg.max_num_trials_in_monolith
    mode_name = "iterative" if use_iterative else "monolith"
    logger.info(f"\nRunning {n_trials} trials in {mode_name} mode\n")

    if use_iterative:
        run_iteratively(
            cfg=cfg,
            run_dir=run_dir,
            logger=logger,
            spd=args.spd,
            no_viz=args.no_viz,
            spd_device=args.spd_device,
        )
    else:
        run_monolith(
            cfg=cfg,
            run_dir=run_dir,
            logger=logger,
            spd=args.spd,
            no_viz=args.no_viz,
            spd_device=args.spd_device,
        )


# All available test gate configurations
ALL_TEST_GATE_CONFIGS = [0, 1, 2, 3, 4, 5]


def main() -> None:
    args = get_args()

    # Enable debug tracing if --debug flag is set
    if args.debug:
        Trace.enable()
        print("[DEBUG] Detailed tracing enabled - expect verbose output")

    # Output path
    output_dir = Path("runs")
    os.makedirs(output_dir, exist_ok=True)

    # Alternative use of main to iterate on analysis/spd
    if rerun_pipeline(output_dir, args):
        exit_fx()

    # Determine which test configs to run
    if args.test and args.test_gates == -1:
        # Run all test configurations
        test_configs = ALL_TEST_GATE_CONFIGS
        print(f"Running ALL {len(test_configs)} test configurations: {test_configs}")
        for i, config_idx in enumerate(test_configs):
            print(f"\n{'=' * 60}")
            print(
                f"  Test Configuration {i + 1}/{len(test_configs)}: --test-gates {config_idx}"
            )
            print(f"{'=' * 60}\n")
            set_test_mode_global(True, config_idx)
            run_single_experiment(args, output_dir)
            print_profile()
    else:
        # Run single configuration
        if args.test:
            set_test_mode_global(True, args.test_gates)
        run_single_experiment(args, output_dir)

    exit_fx()


if __name__ == "__main__":
    main()

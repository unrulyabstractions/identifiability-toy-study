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
    regenerate_from_models,
    run_iteratively,
    run_monolith,
)
from src.viz.viz_config import VizConfig


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
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="GATES",
        help="Enable fast test mode. Optional gate config: -1=ALL, 0=XOR, 1=OR+AND, 2=XOR+XOR, 3=ID+IMP, 4=ID+MAJ, 5=XOR+MAJ",
    )
    parser.add_argument(
        "--viz-only",
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
        help="Skip visualization step (deprecated, use --viz 0)",
    )
    parser.add_argument(
        "--viz",
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4],
        metavar="LEVEL",
        help="Visualization level: 0=none, 1=summary, 2=+top subcircuit, 3=+top 5, 4=all (default)",
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
        help="Specific run directory to process (for --viz-only or --spd-only)",
    )
    parser.add_argument(
        "--trial",
        type=str,
        default=None,
        help="Specific trial ID to process across runs (for --viz-only or --spd-only)",
    )
    parser.add_argument(
        "--rename",
        type=str,
        default=None,
        metavar="DIR_NAME",
        help="Custom name for run directory (instead of timestamp)",
    )
    parser.add_argument(
        "--skip-observational",
        action="store_true",
        help="Skip observational faithfulness analysis",
    )
    parser.add_argument(
        "--skip-interventional",
        action="store_true",
        help="Skip interventional faithfulness analysis",
    )
    parser.add_argument(
        "--skip-counterfactual",
        action="store_true",
        help="Skip counterfactual faithfulness analysis",
    )
    parser.add_argument(
        "--only-observational",
        action="store_true",
        help="Run only observational faithfulness analysis (skip interventional and counterfactual)",
    )
    parser.add_argument(
        "--only-interventional",
        action="store_true",
        help="Run only interventional faithfulness analysis (skip observational and counterfactual)",
    )
    parser.add_argument(
        "--only-counterfactual",
        action="store_true",
        help="Run only counterfactual faithfulness analysis (skip observational and interventional)",
    )
    parser.add_argument(
        "--load-mlp",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help="Regenerate faithfulness analysis from saved models in RUN_DIR (e.g., runs/counter)",
    )
    args = parser.parse_args()

    # args.spd_only should turn on args.spd
    args.spd = args.spd or args.spd_only

    # Handle --only-* flags (mutually exclusive with each other)
    only_count = sum([args.only_observational, args.only_interventional, args.only_counterfactual])
    if only_count > 1:
        parser.error("Cannot specify multiple --only-* flags")

    if args.only_observational:
        args.skip_interventional = True
        args.skip_counterfactual = True
    elif args.only_interventional:
        args.skip_observational = True
        args.skip_counterfactual = True
    elif args.only_counterfactual:
        args.skip_observational = True
        args.skip_interventional = True

    # Handle --no-viz (deprecated) and --viz
    # --no-viz is equivalent to --viz 0
    # If neither specified, default to --viz 4 (all)
    if args.no_viz and args.viz is None:
        args.viz = 0
    elif args.viz is None:
        args.viz = 4  # Default to full visualization

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
    """Handle --viz-only, --spd-only, and --load-mlp modes."""
    did_rerun_pipeline = False
    viz_config = VizConfig.from_int(args.viz)

    if args.load_mlp:
        # Regenerate faithfulness analysis from saved models
        from src.experiment_config import FaithfulnessConfig
        faith_config = FaithfulnessConfig(
            skip_observational=args.skip_observational,
            skip_interventional=args.skip_interventional,
            skip_counterfactual=args.skip_counterfactual,
        )
        regenerate_from_models(
            run_dir=args.load_mlp,
            device=args.device,
            faith_config=faith_config,
            viz_config=viz_config,
            trial_filter=args.trial,
        )
        did_rerun_pipeline = True

    if args.spd_only:
        do_fx = lambda result, run_dir: do_spd_on_experiment(
            result, str(run_dir), viz=viz_config.level > 0, spd_device=args.spd_device
        )
        rerun_all(
            output_dir, do_fx, "spd", run_filter=args.run, trial_filter=args.trial
        )
        did_rerun_pipeline = True
    if args.viz_only:
        do_fx = lambda result, run_dir: do_viz_on_experiment(
            result, str(run_dir), spd=args.spd, viz_config=viz_config
        )
        rerun_all(
            output_dir, do_fx, "viz", run_filter=args.run, trial_filter=args.trial
        )
        did_rerun_pipeline = True
    return did_rerun_pipeline


def exit_fx() -> None:
    """Print profiling and exit."""
    print_profile()
    sys.exit(0)


def run_single_experiment(args: Any, output_dir: Path) -> None:
    """Run a single experiment with the current test mode settings."""
    # Run path - use custom name or timestamp
    if args.rename:
        run_dir = os.path.join(output_dir, args.rename)
        if os.path.exists(run_dir):
            # Rename old folder to {name}_old_{SHORT_TIMESTAMP}
            short_timestamp = datetime.datetime.now().strftime("%H%M%S")
            old_dir = os.path.join(output_dir, f"{args.rename}_old_{short_timestamp}")
            os.rename(run_dir, old_dir)
            print(f"Renamed existing '{args.rename}' to '{args.rename}_old_{short_timestamp}'")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
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

    # Apply skip flags to faithfulness config
    cfg.base_trial.faithfulness_config.skip_observational = args.skip_observational
    cfg.base_trial.faithfulness_config.skip_interventional = args.skip_interventional
    cfg.base_trial.faithfulness_config.skip_counterfactual = args.skip_counterfactual

    # Visualization config
    viz_config = VizConfig.from_int(args.viz)

    # Determine execution mode based on trial count
    trial_settings, _ = build_trial_settings(cfg, logger)
    n_trials = len(trial_settings)

    use_iterative = n_trials > cfg.max_num_trials_in_monolith
    mode_name = "iterative" if use_iterative else "monolith"
    logger.info(f"\nRunning {n_trials} trials in {mode_name} mode\n")
    logger.info(f"Visualization level: {args.viz} ({viz_config.level.name})\n")

    if use_iterative:
        run_iteratively(
            cfg=cfg,
            run_dir=run_dir,
            logger=logger,
            spd=args.spd,
            viz_config=viz_config,
            spd_device=args.spd_device,
        )
    else:
        run_monolith(
            cfg=cfg,
            run_dir=run_dir,
            logger=logger,
            spd=args.spd,
            viz_config=viz_config,
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
    # args.test is None (not test mode), -1 (all), or 0-5 (specific config)
    if args.test is not None and args.test == -1:
        # Run all test configurations
        test_configs = ALL_TEST_GATE_CONFIGS
        print(f"Running ALL {len(test_configs)} test configurations: {test_configs}")
        for i, config_idx in enumerate(test_configs):
            print(f"\n{'=' * 60}")
            print(
                f"  Test Configuration {i + 1}/{len(test_configs)}: --test {config_idx}"
            )
            print(f"{'=' * 60}\n")
            set_test_mode_global(True, config_idx)
            run_single_experiment(args, output_dir)
            print_profile()
    else:
        # Run single configuration
        if args.test is not None:
            set_test_mode_global(True, args.test)
        run_single_experiment(args, output_dir)

    exit_fx()


if __name__ == "__main__":
    main()

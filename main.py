import argparse
import datetime
import os
import sys
from pathlib import Path

from src.common.schemas import (
    ExperimentConfig,
    ExperimentResult,
    SPDConfig,
    generate_spd_sweep_configs,
)
from src.common.utils import setup_logging
from src.experiment import run_experiment
from src.persistence import (
    get_all_runs,
    load_results,
    save_results,
)
from src.common.profiler import print_profile
from src.viz import visualize_experiment

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
    parser.add_argument(
        "--no-spd-sweep",
        action="store_true",
        help="Disable SPD parameter sweep (only run single config)",
    )
    parser.add_argument(
        "--spd",
        action="store_true",
        help="Enable SPD decomposition analysis (disabled by default)",
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
        run_spd=args.spd,
    )

    # SPD sweep is enabled by default
    # Configs designed for speed + variety based on SPD paper recommendations:
    # - Paper uses 40k steps, but for 2->3->1 networks 1000 is enough
    # - Paper recommends importance_coeff=3e-3 and p=1.0 (L1)
    # - Varying p (sparsity shape) and importance_coeff (sparsity strength) matters most
    if not args.no_spd_sweep:
        base_config_id = cfg.base_trial.spd_config.get_config_id()
        sweep_configs = generate_spd_sweep_configs(
            base_config=cfg.base_trial.spd_config,
            # Fewer components for tiny network, one larger for comparison
            n_components_list=[8, 20],
            # Extended range including paper's recommended 3e-3
            importance_coeff_list=[1e-4, 1e-3, 3e-3],
            # Fast iteration - 1000 steps is enough for toy models
            steps_list=[1000],
            # Vary sparsity shape: 0.5=extreme, 1.0=L1 (paper default), 2.0=L2
            importance_p_list=[0.5, 1.0, 2.0],
        )
        # Exclude configs that match the base config
        cfg.base_trial.spd_sweep_configs = [
            c for c in sweep_configs if c.get_config_id() != base_config_id
        ]
        logger.info(f"SPD sweep: {len(cfg.base_trial.spd_sweep_configs) + 1} total configs")

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

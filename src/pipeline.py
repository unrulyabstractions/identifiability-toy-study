"""Pipeline orchestration for experiment execution.

Provides two modes:
- Monolith: Run all trials, save once at the end
- Iterative: Run and save each trial individually (for long experiments)
"""

from pathlib import Path
from typing import Any, Iterator

from src.experiment import experiment_run, run_experiment
from src.experiment_config import ExperimentConfig
from src.infra import print_profile, profile_fn
from src.infra.profiler import Trace
from src.persistence import load_results, save_results, save_training_data
from src.schemas import ExperimentResult, TrialResult
from src.spd import (
    SpdResults,
    load_spd_results,
    run_spd,
    run_trial_on_spd_results,
    save_spd_results,
)
from src.viz import save_per_gate_data, visualize_experiment, visualize_spd_experiment
from src.viz.viz_config import VizConfig, VizLevel


@profile_fn("do_viz_on_experiment")
def do_viz_on_experiment(
    result: ExperimentResult,
    run_dir: str,
    spd: bool,
    viz_config: VizConfig | None = None,
) -> None:
    """Run visualization on a complete experiment result."""
    if viz_config is None:
        viz_config = VizConfig()

    # ALWAYS save per-gate JSON data (independent of viz level)
    # Pass viz_config to control circuit.png generation
    save_per_gate_data(result, run_dir, viz_config=viz_config)

    # Skip PNG visualization if level is NONE
    if viz_config.skip_all_viz:
        return

    visualize_experiment(result, run_dir=run_dir, viz_config=viz_config)
    if spd:
        spd_result = load_spd_results(run_dir)
        if spd_result:
            visualize_spd_experiment(spd_result, run_dir=run_dir)


@profile_fn("do_spd_on_experiment")
def do_spd_on_experiment(
    result: ExperimentResult, run_dir: str, viz: bool, spd_device: str
) -> SpdResults:
    """Run SPD decomposition on a complete experiment result."""
    spd_result = run_spd(result, run_dir=run_dir, device=spd_device)
    save_spd_results(spd_result, run_dir=run_dir)

    # Run trial analysis on SPD-discovered subcircuits
    run_trial_on_spd_results(spd_result, result, run_dir=run_dir, device=spd_device)

    if viz:
        visualize_spd_experiment(spd_result, run_dir=run_dir)
    return spd_result


def do_viz_on_trial(
    trial_result: TrialResult,
    run_dir: str,
    spd: bool,
    viz_config: VizConfig | None = None,
) -> None:
    """Run visualization on a single trial result.

    Creates a temporary ExperimentResult wrapper for compatibility with existing viz.
    """
    if viz_config is None:
        viz_config = VizConfig()

    # Wrap trial in experiment result for viz compatibility
    from src.experiment_config import ExperimentConfig
    temp_result = ExperimentResult(config=ExperimentConfig())
    temp_result.trials[trial_result.trial_id] = trial_result

    # ALWAYS save per-gate JSON data (independent of viz level)
    # Pass viz_config to control circuit.png generation
    save_per_gate_data(temp_result, run_dir, viz_config=viz_config)

    # Skip PNG visualization if level is NONE
    if viz_config.skip_all_viz:
        return

    visualize_experiment(temp_result, run_dir=run_dir, viz_config=viz_config)
    if spd:
        spd_result = load_spd_results(run_dir)
        if spd_result:
            visualize_spd_experiment(spd_result, run_dir=run_dir)


def do_spd_on_trial(
    trial_result: TrialResult, run_dir: str, viz: bool, spd_device: str
) -> SpdResults | None:
    """Run SPD decomposition on a single trial result."""
    # Wrap trial in experiment result for SPD compatibility
    from src.experiment_config import ExperimentConfig
    temp_result = ExperimentResult(config=ExperimentConfig())
    temp_result.trials[trial_result.trial_id] = trial_result

    spd_result = run_spd(temp_result, run_dir=run_dir, device=spd_device)
    save_spd_results(spd_result, run_dir=run_dir)

    run_trial_on_spd_results(spd_result, temp_result, run_dir=run_dir, device=spd_device)

    if viz:
        visualize_spd_experiment(spd_result, run_dir=run_dir)
    return spd_result


def save_experiment_results(result: ExperimentResult, run_dir: str) -> None:
    """Save complete experiment results."""
    save_results(result, run_dir=run_dir)


def save_trial_result(trial_result: TrialResult, run_dir: str, cfg: ExperimentConfig) -> None:
    """Save a single trial result incrementally.

    OPTIMIZED: Saves only this trial's data without loading all previous trials.
    Run-level summaries are regenerated at the end of the experiment.
    """
    from src.persistence.save import save_single_trial

    run_dir = Path(run_dir)

    # Save just this trial (no O(n^2) loading of all previous trials)
    save_single_trial(trial_result, run_dir, cfg)


def print_experiment_summary(
    experiment_result: ExperimentResult, spd_result: SpdResults | None, logger: Any
) -> None:
    """Print summary of experiment results."""
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


@profile_fn("run_monolith")
def run_monolith(
    cfg: ExperimentConfig,
    run_dir: str,
    logger: Any,
    spd: bool = False,
    viz_config: VizConfig | None = None,
    spd_device: str = "cpu",
) -> ExperimentResult:
    """Run all trials at once, save at the end.

    Best for small experiments where total runtime is short.
    """
    if viz_config is None:
        viz_config = VizConfig()

    experiment_result, master_data = run_experiment(cfg, logger=logger)
    save_training_data(master_data, run_dir, gate_names=cfg.target_logic_gates)
    save_experiment_results(experiment_result, run_dir=run_dir)

    spd_result = None
    if spd:
        spd_result = do_spd_on_experiment(
            experiment_result, run_dir, viz=not viz_config.skip_all_viz, spd_device=spd_device
        )
    if not viz_config.skip_all_viz:
        do_viz_on_experiment(experiment_result, run_dir, spd=spd, viz_config=viz_config)

    print_experiment_summary(experiment_result, spd_result, logger)
    return experiment_result


@profile_fn("run_iteratively")
def run_iteratively(
    cfg: ExperimentConfig,
    run_dir: str,
    logger: Any,
    spd: bool = False,
    viz_config: VizConfig | None = None,
    spd_device: str = "cpu",
) -> ExperimentResult:
    """Run trials one at a time, saving after each.

    Best for long experiments where you want incremental saves.
    Results are saved after each trial completes, so partial progress
    is preserved if the experiment is interrupted.

    MEMORY OPTIMIZED: Trials are saved and then cleared from memory.
    Final result is loaded from disk at the end.
    """
    import gc

    if viz_config is None:
        viz_config = VizConfig()

    trial_iterator, master_data = experiment_run(cfg, logger=logger)
    save_training_data(master_data, run_dir, gate_names=cfg.target_logic_gates)

    trial_count = 0
    for trial_result in trial_iterator:
        trial_count += 1

        # Save this trial incrementally (without loading all previous trials)
        save_trial_result(trial_result, run_dir, cfg)

        # Optionally run SPD per trial
        if spd:
            do_spd_on_trial(trial_result, run_dir, viz=not viz_config.skip_all_viz, spd_device=spd_device)

        # ALWAYS run viz (saves JSON data regardless of viz level, only skips PNGs)
        do_viz_on_trial(trial_result, run_dir, spd=spd, viz_config=viz_config)

        # Clear trial from memory after processing
        del trial_result
        gc.collect()

    # Load final results from disk for summary (memory-efficient: only loads once)
    logger.info(f"Loading {trial_count} trials from disk for final summary...")
    experiment_result = load_results(str(run_dir))

    # Generate run-level summaries only (trials already saved individually)
    from src.persistence.save import save_run_summaries_only
    save_run_summaries_only(experiment_result, run_dir=run_dir, logger=logger)

    # Final summary
    spd_result = load_spd_results(run_dir) if spd else None
    print_experiment_summary(experiment_result, spd_result, logger)
    return experiment_result


@profile_fn("regenerate_from_models")
def regenerate_from_models(
    run_dir: str,
    device: str = "cpu",
    faith_config: Any = None,
    viz_config: VizConfig | None = None,
    trial_filter: str | None = None,
) -> None:
    """Regenerate everything from saved models (skip training only).

    Loads models from disk and re-runs the full analysis pipeline:
    - Circuit enumeration
    - Subcircuit metrics computation
    - Faithfulness analysis (observational, interventional, counterfactual)
    - Visualization
    - Slice analysis

    This is equivalent to running the full experiment except for MLP training.

    Args:
        run_dir: Path to the run directory (e.g., "runs/counter")
        device: Device for computation
        faith_config: FaithfulnessConfig controlling which analyses to run
        viz_config: VizConfig for visualization
        trial_filter: Optional trial ID to process only one trial
    """
    import json
    import torch
    from pathlib import Path

    from src.experiment_config import DataParams, ExperimentConfig, FaithfulnessConfig, ModelParams, TrainParams, TrialSetup
    from src.model import MLP
    from src.persistence.save import save_single_trial
    from src.schemas import ExperimentResult
    from src.training import generate_trial_data
    from src.trial.trial_executor import run_trial_from_saved_model

    if faith_config is None:
        faith_config = FaithfulnessConfig()
    if viz_config is None:
        viz_config = VizConfig()

    run_path = Path(run_dir)
    trials_dir = run_path / "trials"

    if not trials_dir.exists():
        print(f"No trials directory found in {run_dir}")
        return

    # Load experiment config
    config_path = run_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            exp_config_data = json.load(f)
        exp_config = ExperimentConfig()  # Use defaults, override as needed
    else:
        exp_config = ExperimentConfig()

    # Apply faithfulness config overrides
    exp_config.base_trial.faithfulness_config = faith_config

    # Get trial directories
    trial_dirs = [d for d in trials_dir.iterdir() if d.is_dir()]
    if trial_filter:
        trial_dirs = [d for d in trial_dirs if d.name == trial_filter]

    if not trial_dirs:
        print(f"No matching trials found in {run_dir}")
        return

    print(f"Found {len(trial_dirs)} trial(s) to regenerate")

    # Collect all trial results for final experiment result
    all_trial_results = {}

    for trial_dir in sorted(trial_dirs):
        trial_id = trial_dir.name
        print(f"\n{'=' * 60}")
        print(f"Processing trial: {trial_id}")
        print("=" * 60)

        # Load the model
        model_path = trial_dir / "all_gates" / "model.pt"
        if not model_path.exists():
            print(f"  No model found at {model_path}, skipping")
            continue

        print(f"  Loading model...")
        model = MLP.load_from_file(str(model_path), device=device)

        # Load trial setup from setup.json
        setup_path = trial_dir / "setup.json"
        if not setup_path.exists():
            print(f"  No setup.json found, skipping")
            continue

        with open(setup_path) as f:
            setup_data = json.load(f)

        # Reconstruct TrialSetup
        setup = TrialSetup(
            model_params=ModelParams(
                width=setup_data["model_params"]["width"],
                depth=setup_data["model_params"]["depth"],
                logic_gates=setup_data["model_params"]["logic_gates"],
            ),
            train_params=TrainParams(),
            seed=setup_data.get("seed", 42),
            faithfulness_config=faith_config,
        )

        # Generate test data for the gates
        gate_names = setup.model_params.logic_gates
        print(f"  Generating data for gates: {gate_names}")
        data = generate_trial_data(
            data_params=DataParams(),
            logic_gates=gate_names,
            device=device,
        )

        # Run the full trial pipeline (skip training)
        print(f"  Running full analysis pipeline...")
        trial_result = run_trial_from_saved_model(
            model=model,
            setup=setup,
            data=data,
            device=device,
            logger=None,
            debug=False,
        )

        # Save the trial result
        print(f"  Saving trial result...")
        save_single_trial(trial_result, run_path, exp_config)

        all_trial_results[trial_id] = trial_result

        # Run visualization for this trial
        print(f"  Running visualization...")
        do_viz_on_trial(trial_result, str(run_path), spd=False, viz_config=viz_config)

        print(f"  Done with trial {trial_id}")

    # Create experiment result and save run-level summaries
    if all_trial_results:
        experiment_result = ExperimentResult(config=exp_config)
        experiment_result.trials = all_trial_results

        from src.persistence.save import save_run_summaries_only
        print(f"\nSaving run-level summaries...")
        save_run_summaries_only(experiment_result, run_dir=str(run_path), logger=None)

    print("\n" + "=" * 60)
    print("Regeneration complete!")
    print("=" * 60)

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
    save_per_gate_data(result, run_dir)

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
    save_per_gate_data(temp_result, run_dir)

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

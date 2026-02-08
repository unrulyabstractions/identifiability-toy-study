"""
SPD experiment-level entry point.

Contains run_spd() for running SPD analysis on all trials in an experiment.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from src.infra import profile_fn

from .spd_trial import run_spd_trial
from .spd_types import SPDConfig, SpdResults

if TYPE_CHECKING:
    from src.schemas import ExperimentResult


@profile_fn("SPD Experiment")
def run_spd(
    experiment_result: "ExperimentResult",
    run_dir: str | Path,
    device: str = "cpu",
    spd_config: SPDConfig | None = None,
    spd_sweep_configs: list[SPDConfig] | None = None,
) -> SpdResults:
    """
    Run SPD analysis on all trials in an experiment.

    This is the main entry point for SPD analysis, called AFTER run_experiment().

    Args:
        experiment_result: ExperimentResult from run_experiment()
        run_dir: Directory where experiment results are saved
        device: Device for SPD computation (cpu recommended for small models)
        spd_config: SPD configuration (uses default if None)
        spd_sweep_configs: Optional list of additional configs to sweep

    Returns:
        SpdResults containing per-trial SPD analysis
    """
    run_dir = Path(run_dir)

    # Use default config if not provided
    if spd_config is None:
        spd_config = SPDConfig()

    spd_results = SpdResults(
        config=spd_config,
        sweep_configs=spd_sweep_configs or [],
    )

    n_trials = len(experiment_result.trials)
    for trial_idx, (trial_id, trial_result) in enumerate(
        experiment_result.trials.items()
    ):
        print(f"[SPD] Trial {trial_idx + 1}/{n_trials}: {trial_id[:8]}...")

        spd_trial_result = run_spd_trial(
            trial_result=trial_result,
            spd_config=spd_config,
            spd_sweep_configs=spd_sweep_configs,
            device=device,
        )

        spd_results.per_trial[trial_id] = spd_trial_result

    return spd_results

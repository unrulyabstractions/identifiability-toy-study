"""
SPD trial-level analysis.

Contains run_spd_trial() for running SPD on a single trial.
Logic extracted from trial/phases.py:spd_phase().
"""

from typing import TYPE_CHECKING

from src.infra import profile

from .decomposition import decompose_mlp
from .spd_types import SPDConfig, SpdTrialResult
from .subcircuits import estimate_spd_subcircuits

if TYPE_CHECKING:
    from src.model import MLP
    from src.schemas import TrialResult

import torch


def run_spd_trial(
    trial_result: "TrialResult",
    spd_config: SPDConfig,
    spd_sweep_configs: list[SPDConfig] | None = None,
    device: str = "cpu",
) -> SpdTrialResult:
    """
    Run SPD analysis on a single trial.

    Args:
        trial_result: The completed trial result containing model and test data
        spd_config: Primary SPD configuration
        spd_sweep_configs: Optional list of additional configs to sweep
        device: Device for SPD computation

    Returns:
        SpdTrialResult with decomposed models and subcircuit estimates
    """
    model = trial_result.model
    x = trial_result.test_x
    y_pred = trial_result.test_y_pred
    gate_names = trial_result.setup.model_params.logic_gates
    input_size = x.shape[1] if x is not None else 2

    spd_trial_result = SpdTrialResult(trial_id=trial_result.trial_id)

    if model is None or x is None or y_pred is None:
        return spd_trial_result

    # Collect all configs to run
    spd_configs_to_run = [spd_config]
    if spd_sweep_configs:
        spd_configs_to_run.extend(spd_sweep_configs)

    # Run decomposition for each config
    for config_idx, config in enumerate(spd_configs_to_run):
        config_id = config.get_config_id()
        print(f"    SPD config {config_idx + 1}/{len(spd_configs_to_run)}: {config_id}")

        with profile(f"spd_mlp_{config_id}"):
            decomposed = decompose_mlp(x, y_pred, model, device, config)

        if config_idx == 0:
            spd_trial_result.decomposed_model = decomposed
        spd_trial_result.decomposed_models_sweep[config_id] = decomposed

    # SPD Subcircuit estimation for each config
    for config_idx, config in enumerate(spd_configs_to_run):
        config_id = config.get_config_id()
        decomposed = spd_trial_result.decomposed_models_sweep[config_id]
        print(
            f"    SPD subcircuit {config_idx + 1}/{len(spd_configs_to_run)}: {config_id}"
        )

        with profile(f"spd_mlp_sc_{config_id}"):
            estimate = estimate_spd_subcircuits(
                decomposed_model=decomposed,
                n_inputs=input_size,
                gate_names=gate_names,
                device=device,
            )

        if config_idx == 0:
            spd_trial_result.spd_subcircuit_estimate = estimate
        spd_trial_result.spd_subcircuit_estimates_sweep[config_id] = estimate

    return spd_trial_result

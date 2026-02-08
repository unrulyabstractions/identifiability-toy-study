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
    from src.schemas import TrialResult


def run_spd_trial(
    trial_result: "TrialResult",
    spd_config: SPDConfig,
    device: str = "cpu",
) -> SpdTrialResult:
    """
    Run SPD analysis on a single trial.

    Args:
        trial_result: The completed trial result containing model and test data
        spd_config: SPD configuration
        device: Device for SPD computation

    Returns:
        SpdTrialResult with decomposed model and subcircuit estimate
    """
    model = trial_result.model
    x = trial_result.test_x
    y_pred = trial_result.test_y_pred
    gate_names = trial_result.setup.model_params.logic_gates
    input_size = x.shape[1] if x is not None else 2

    spd_trial_result = SpdTrialResult(trial_id=trial_result.trial_id)

    if model is None or x is None or y_pred is None:
        return spd_trial_result

    config_id = spd_config.get_config_id()
    print(f"    SPD config: {config_id}")

    # Run decomposition
    with profile(f"spd_mlp_{config_id}"):
        decomposed = decompose_mlp(x, y_pred, model, device, spd_config)

    spd_trial_result.decomposed_model = decomposed

    # SPD Subcircuit estimation
    print(f"    SPD subcircuit: {config_id}")
    with profile(f"spd_mlp_sc_{config_id}"):
        estimate = estimate_spd_subcircuits(
            decomposed_model=decomposed,
            n_inputs=input_size,
            gate_names=gate_names,
            device=device,
        )

    spd_trial_result.spd_subcircuit_estimate = estimate

    return spd_trial_result

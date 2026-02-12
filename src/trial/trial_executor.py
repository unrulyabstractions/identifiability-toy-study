"""Main trial runner - orchestrates the full trial pipeline."""

from typing import Any

from src.domain import get_max_n_inputs, resolve_gate
from src.experiment_config import TrialSetup
from src.infra import ParallelConfig, get_eval_device, set_seeds, update_status_fx
from src.infra.profiler import trace, traced
from src.math import calculate_match_rate, logits_to_binary
from src.schemas import TrialData, TrialResult
from src.training import train_model
from src.training_analysis import do_training_analysis

from .gate_analysis import analyze_gate
from .phases import (
    compute_activations_phase,
    decision_boundary_phase,
    enumerate_circuits_phase,
    precompute_masks_phase,
)


def run_trial(
    setup: TrialSetup,
    data: TrialData,
    device: str,
    logger: Any = None,
    debug: bool = False,
    precomputed_circuits: tuple = None,
) -> TrialResult:
    """Run a complete training and analysis trial.

    This is the main entry point for running a single trial. It:
    1. Trains an MLP model on the provided data
    2. Computes activations and metrics
    3. Uses pre-computed subcircuits (or enumerates if not provided)
    4. Runs robustness and faithfulness analysis on best subcircuits

    Note: SPD decomposition is now run separately via src.spd.run_spd()

    Args:
        setup: Trial configuration (model params, training params, etc.)
        data: Training, validation, and test data
        device: Main compute device (e.g., "cpu", "cuda", "mps")
        logger: Optional logger for status updates
        debug: Enable debug output
        precomputed_circuits: Optional tuple of (subcircuits, subcircuit_structures)
            pre-computed at experiment level

    Returns:
        TrialResult containing model, metrics, and analysis results
    """
    # This also sets deterministic behavior
    set_seeds(setup.seed)
    parallel_config = ParallelConfig()

    trial_result = TrialResult(
        setup=setup,
    )

    trial_metrics = trial_result.metrics
    update_status = update_status_fx(trial_result, logger, device=device)

    gate_names = setup.model_params.logic_gates
    trace("run_trial starting", gates=gate_names, seed=setup.seed, device=device)

    # ===== Train Model =====
    update_status("STARTED_MLP_TRAINING")
    input_size = get_max_n_inputs(gate_names)  # Use max to support mixed gate sizes
    output_size = len(gate_names)

    # train_model has @profile_fn("Train Model") directly in helpers.py
    model, avg_loss, val_acc, training_record = train_model(
        train_params=setup.train_params,
        model_params=setup.model_params,
        data=data,
        device=device,
        logger=logger,
        debug=debug,
        input_size=input_size,
        output_size=output_size,
    )
    if model is None:
        return trial_result
    trial_result.model = model
    gate_models = model.separate_into_k_mlps()
    update_status("ENDED_MLP_TRAINING")

    # ===== Compute Activations =====
    act_data = compute_activations_phase(model, data, device, input_size)
    x = act_data["x"]
    y_gt = act_data["y_gt"]
    y_pred = act_data["y_pred"]
    activations = act_data["activations"]

    trial_result.test_x = x
    trial_result.test_y = y_gt
    trial_result.test_y_pred = y_pred.detach().clone()
    trial_result.activations = activations
    trial_result.canonical_activations = act_data["canonical_activations"]
    trial_result.mean_activations_by_range = act_data["mean_activations_by_range"]
    trial_result.layer_weights = act_data["layer_weights"]
    trial_result.layer_biases = act_data["layer_biases"]

    # y_gt is already 0/1 labels (not logits), y_pred is logits (threshold at 0)
    bit_gt = y_gt.float()  # [n_samples, n_gates] - already 0/1
    bit_pred = logits_to_binary(y_pred)  # [n_samples, n_gates] - threshold logits at 0

    trial_metrics.avg_loss = avg_loss
    trial_metrics.val_acc = val_acc
    trial_metrics.test_acc = calculate_match_rate(bit_gt, bit_pred).item()

    # ===== Circuit Finding =====
    update_status("STARTED_CIRCUITS")
    if precomputed_circuits is not None:
        # Use pre-computed circuits from experiment level
        subcircuits, subcircuit_structures = precomputed_circuits
    else:
        # Fall back to computing circuits (for standalone trial runs)
        subcircuits, subcircuit_structures = enumerate_circuits_phase(
            model, parallel_config
        )
    update_status("FINISHED_CIRCUITS")

    # ===== Training Analysis (Epiplexity Estimation) =====
    update_status("STARTED_TRAINING_ANALYSIS")
    training_analysis = do_training_analysis(
        training_record=training_record,
        model=model,
        x=data.train.x,
        y=data.train.y,
        device=device,
    )
    trial_metrics.training_analysis = training_analysis
    update_status("FINISHED_TRAINING_ANALYSIS")

    trial_result.subcircuits = [
        {"idx": idx, **s.to_dict()} for idx, s in enumerate(subcircuits)
    ]
    trial_result.subcircuit_structure_analysis = [
        {"idx": idx, **s.to_dict()} for idx, s in enumerate(subcircuit_structures)
    ]
    update_status("FINISHED_CIRCUIT_FINDING")

    # Determine evaluation device
    eval_device = get_eval_device(parallel_config, device)

    # Precompute circuit masks
    # Compute n_inputs per gate for proper input masking with mixed gate sizes
    gate_n_inputs_list = [resolve_gate(name).n_inputs for name in gate_names]
    precomputed_masks_per_gate = {}
    if parallel_config.precompute_masks:
        precomputed_masks_per_gate = precompute_masks_phase(
            subcircuits,
            model,
            gate_names,
            eval_device,
            output_size,
            gate_n_inputs_list=gate_n_inputs_list,
        )

    # ===== Gate Analysis =====
    update_status("STARTED_GATE_ANALYSIS")
    trace("gate_analysis starting", n_gates=len(gate_models), gates=gate_names)

    for gate_idx in range(len(gate_models)):
        gate_name = gate_names[gate_idx]
        gate_model = gate_models[gate_idx]

        with traced("analyze_gate", gate_idx=gate_idx, gate_name=gate_name):
            analyze_gate(
                gate_idx=gate_idx,
                gate_name=gate_name,
                gate_model=gate_model,
                gate_models=gate_models,
                model=model,
                y_pred=y_pred,
                bit_gt=bit_gt,
                bit_pred=bit_pred,
                x=x,
                activations=activations,
                subcircuits=subcircuits,
                subcircuit_structures=subcircuit_structures,
                precomputed_masks_per_gate=precomputed_masks_per_gate,
                parallel_config=parallel_config,
                setup=setup,
                device=device,
                eval_device=eval_device,
                update_status=update_status,
                trial_metrics=trial_metrics,
            )

    update_status("FINISHED_GATE_ANALYSIS")

    # ===== Decision Boundary Data =====
    update_status("STARTED_DECISION_BOUNDARY")
    db_data, subcircuit_db_data = decision_boundary_phase(
        model=model,
        gate_models=gate_models,
        gate_names=gate_names,
        subcircuits=subcircuits,
        trial_metrics=trial_metrics,
        device=device,
    )
    trial_result.decision_boundary_data = db_data
    trial_result.subcircuit_decision_boundary_data = subcircuit_db_data
    update_status("FINISHED_DECISION_BOUNDARY")

    update_status("SUCCESSFUL_TRIAL")
    trace("run_trial complete", trial_id=trial_result.trial_id)
    return trial_result

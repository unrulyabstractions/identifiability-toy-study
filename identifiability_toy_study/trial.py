from typing import Any

import torch
import torch.nn as nn

from .causal_analysis import (
    CleanCorruptedPair,
    calculate_faithfulness_metrics,
    calculate_robustness_metrics,
    create_clean_corrupted_data,
    filter_subcircuits,
)
from .common.circuit import (
    enumerate_all_valid_circuit,
)
from .common.helpers import calculate_match_rate, train_model, update_status_fx
from .common.logic_gates import ALL_LOGIC_GATES
from .common.schemas import (
    GateMetrics,
    SubcircuitMetrics,
    TrialData,
    TrialResult,
    TrialSetup,
)
from .common.utils import (
    set_seeds,
)
from .parameter_decomposition import decompose_mlp


def run_trial(
    setup: TrialSetup,
    data: TrialData,
    device: str,
    spd_device: str = "cpu",
    logger: Any = None,
    debug: bool = False,
) -> TrialResult:
    # This also sets deterministic behavior
    set_seeds(setup.seed)

    trial_result = TrialResult(
        setup=setup,
    )

    trial_metrics = trial_result.metrics
    per_gate_metrics = trial_metrics.per_gate_metrics
    per_gate_bests = trial_metrics.per_gate_bests
    per_gate_bests_robust = trial_metrics.per_gate_bests_robust
    per_gate_bests_faith = trial_metrics.per_gate_bests_faith

    update_status = update_status_fx(trial_result, logger, device=device)

    # ===== Train Model =====
    update_status("STARTED_MLP_TRAINING")
    gate_names = setup.model_params.logic_gates
    input_size = ALL_LOGIC_GATES[gate_names[0]].n_inputs
    output_size = len(gate_names)

    model, avg_loss, val_acc = train_model(
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

    x = data.test.x.contiguous()
    y_gt = data.test.y.contiguous()
    y_pred = model(data.test.x)
    activations = model(data.test.x, return_activations=True)

    # Store test data and activations for persistence
    trial_result.test_x = x
    trial_result.test_y = y_gt
    trial_result.test_y_pred = y_pred.detach()
    trial_result.activations = activations

    # Pre-compute canonical activations for visualization (no model runs needed later)
    canonical_inputs = {
        "0_0": torch.tensor([[0.0, 0.0]], device=device),
        "0_1": torch.tensor([[0.0, 1.0]], device=device),
        "1_0": torch.tensor([[1.0, 0.0]], device=device),
        "1_1": torch.tensor([[1.0, 1.0]], device=device),
    }
    trial_result.canonical_activations = {
        label: model(inp, return_activations=True)
        for label, inp in canonical_inputs.items()
    }

    # Store layer weights for visualization
    trial_result.layer_weights = [
        layer[0].weight.detach().cpu() for layer in model.layers
    ]

    bit_gt = torch.round(y_gt)
    bit_pred = torch.round(y_pred)

    trial_metrics.avg_loss = avg_loss
    trial_metrics.val_acc = val_acc
    trial_metrics.test_acc = calculate_match_rate(torch.round(y_pred), y_gt).item()

    # ===== Parameter Decomposition =====
    update_status("STARTED_SPD_TRAINING_MLP")
    spd_config = setup.spd_config
    trial_result.decomposed_model = decompose_mlp(
        x, y_pred, model, spd_device, spd_config
    )
    update_status("ENDED_SPD_TRAINING_MLP")

    # ===== Find Circuits =====
    update_status("STARTED_CIRCUIT_FINDING")
    subcircuits = enumerate_all_valid_circuit(model, use_tqdm=True)
    subcircuit_structures = [s.analyze_structure() for s in subcircuits]

    trial_result.subcircuits = [
        {"idx": idx, **s.to_dict()} for idx, s in enumerate(subcircuits)
    ]
    trial_result.subcircuit_structure_analysis = [
        {"idx": idx, **s.to_dict()} for idx, s in enumerate(subcircuit_structures)
    ]
    update_status("FINISHED_CIRCUIT_FINDING")

    # Gate subcircuit metrics
    def run_gate_trial(gate_idx: int) -> None:
        gate_name = gate_names[gate_idx]
        gate_model = gate_models[gate_idx]
        y_gate = y_pred[..., [gate_idx]]
        bit_gate_gt = bit_gt[..., [gate_idx]]
        bit_gate = bit_pred[..., [gate_idx]]

        subcircuit_models = [gate_model.separate_subcircuit(sub) for sub in subcircuits]

        # ===== Calculate Gate Metrics =====
        update_status(f"STARTED_GATE_METRICS:{gate_idx}")
        gate_acc = calculate_match_rate(bit_gate, bit_gate_gt).item()

        subcircuit_metrics = []
        for subcircuit_idx, subcircuit_model in enumerate(subcircuit_models):
            # Predict
            y_circuit = subcircuit_model(x)
            bit_circuit = torch.round(y_circuit)

            # Compute
            accuracy = calculate_match_rate(bit_circuit, bit_gate_gt).item()
            logit_similarity = 1 - nn.MSELoss()(y_gate, y_circuit).item()
            bit_similarity = calculate_match_rate(bit_circuit, bit_gate).item()

            subcircuit_metric = SubcircuitMetrics(
                idx=subcircuit_idx,
                accuracy=accuracy,
                logit_similarity=logit_similarity,
                bit_similarity=bit_similarity,
            )
            subcircuit_metrics.append(subcircuit_metric)

        per_gate_metrics[gate_name] = GateMetrics(
            test_acc=gate_acc,
            subcircuit_metrics=subcircuit_metrics,
        )

        per_gate_bests[gate_name] = filter_subcircuits(
            setup.constraints,
            per_gate_metrics[gate_name].subcircuit_metrics,
        )

        best_indices = per_gate_bests[gate_name]

        # ===== Robustness Analysis =====
        update_status(f"STARTED_ROBUSTNESS:{gate_idx}")
        robusts = per_gate_bests_robust[gate_name]
        for subcircuit_idx in best_indices:
            subcircuit_model = subcircuit_models[subcircuit_idx]
            robustness = calculate_robustness_metrics(
                subcircuit=subcircuit_model,
                full_model=gate_model,
                n_samples_per_base=100,
                device=device,
            )
            robusts.append(robustness)
        update_status(f"FINISHED_ROBUSTNESS:{gate_idx}")

        # ===== Create Counterfactual Pairs (reused across subcircuits) =====
        counterfactual_pairs = create_clean_corrupted_data(
            x=x,
            y=y_gate,
            activations=activations,
            n_pairs=10,
        )

        # ===== Faithfulness Analysis =====
        update_status(f"STARTED_FAITH:{gate_idx}")
        faiths = per_gate_bests_faith[gate_name]
        for subcircuit_idx in best_indices:
            subcircuit_model = subcircuit_models[subcircuit_idx]
            faithfulness = calculate_faithfulness_metrics(
                x=x,
                y=y_gate,
                model=gate_model,
                activations=activations,
                subcircuit=subcircuit_model,
                structure=subcircuit_structures[subcircuit_idx],
                counterfactual_pairs=counterfactual_pairs,
                config=setup.faithfulness_config,
                device=device,
            )
            faiths.append(faithfulness)
        update_status(f"FINISHED_FAITH:{gate_idx}")

        # ===== Parameter Decomposition =====
        update_status(f"STARTED_SPD_TRAINING_GATE:{gate_idx}")
        # Decompose full single-gate model
        trial_result.decomposed_gate_models[gate_name] = decompose_mlp(
            x, y_gate, gate_model, spd_device, spd_config
        )
        update_status(f"FINISHED_SPD_TRAINING_GATE:{gate_idx}")

        # Decompose best subcircuits (limited by max_subcircuits)
        decomposed_dict = trial_result.decomposed_subcircuits[gate_name]

        best_indices = per_gate_bests[gate_name]
        if spd_config.max_subcircuits:
            best_indices = best_indices[: spd_config.max_subcircuits]

        for subcircuit_idx in best_indices:
            update_status(
                f"STARTED_SPD_TRAINING_SUBCIRCUIT:{gate_idx}:{subcircuit_idx}"
            )
            decomposed_dict[subcircuit_idx] = decompose_mlp(
                x, y_gate, subcircuit_models[subcircuit_idx], spd_device, spd_config
            )
            # Track which subcircuits were decomposed (for JSON serialization)
            trial_result.decomposed_subcircuit_indices[gate_name].append(subcircuit_idx)
            update_status(
                f"FINISHED_SPD_TRAINING_SUBCIRCUIT:{gate_idx}:{subcircuit_idx}"
            )

    update_status("STARTED_GATE_ANALYSIS")
    for gate_idx in range(len(gate_models)):
        run_gate_trial(gate_idx)
    update_status("FINISHED_GATE_ANALYSIS")

    update_status("SUCCESSFUL_TRIAL")
    return trial_result

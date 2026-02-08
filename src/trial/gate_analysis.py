"""Gate analysis - per-gate metric computation, edge optimization, and faithfulness."""

from ..causal import create_clean_corrupted_data, filter_subcircuits
from ..common.batched_eval import batch_compute_metrics, batch_evaluate_edge_variants
from ..common.metrics import calculate_match_rate
from ..common.schemas import GateMetrics, SubcircuitMetrics
from .phases import (
    faithfulness_phase,
    robustness_phase,
)


def analyze_gate(
    gate_idx,
    gate_name,
    gate_model,
    gate_models,
    model,
    y_pred,
    bit_gt,
    bit_pred,
    x,
    activations,
    subcircuits,
    subcircuit_structures,
    precomputed_masks_per_gate,
    parallel_config,
    setup,
    device,
    eval_device,
    update_status,
    trial_metrics,
):
    """
    Analyze a single gate: compute metrics, find best subcircuits, run robustness/faithfulness.

    Args:
        gate_idx: Index of the gate being analyzed
        gate_name: Name of the gate (e.g., "XOR", "AND")
        gate_model: The separated single-gate model
        gate_models: All separated gate models
        model: Full multi-gate model
        y_pred: Full model predictions [batch, n_gates]
        bit_gt: Ground truth binary values [batch, n_gates]
        bit_pred: Predicted binary values [batch, n_gates]
        x: Input data [batch, input_size]
        activations: Pre-computed activations
        subcircuits: List of all subcircuits
        subcircuit_structures: Structure analysis for each subcircuit
        precomputed_masks_per_gate: Precomputed masks keyed by gate_idx
        parallel_config: Parallelization configuration
        setup: Trial setup configuration
        device: Main compute device
        eval_device: Device for batched evaluation
        update_status: Status update callback
        trial_metrics: Metrics object to update

    Returns:
        None (updates trial_metrics in place)
    """
    per_gate_metrics = trial_metrics.per_gate_metrics
    per_gate_bests = trial_metrics.per_gate_bests
    per_gate_bests_robust = trial_metrics.per_gate_bests_robust
    per_gate_bests_faith = trial_metrics.per_gate_bests_faith

    y_gate = y_pred[..., [gate_idx]]  # [n_samples, 1] - model logits for this gate
    bit_gate_gt = bit_gt[..., [gate_idx]]  # [n_samples, 1] - binary ground truth
    bit_gate = bit_pred[..., [gate_idx]]  # [n_samples, 1] - binary model prediction

    print(f"\n{'~' * 60}")
    print(f"  Gate {gate_idx}: {gate_name}")
    print("~" * 60)

    update_status(f"STARTED_GATE_METRICS:{gate_idx}")
    gate_acc = calculate_match_rate(bit_gate, bit_gate_gt).item()

    # Move data to eval device
    x_eval = x.to(eval_device) if eval_device != device else x
    y_gate_eval = y_gate.to(eval_device) if eval_device != device else y_gate
    bit_gate_gt_eval = (
        bit_gate_gt.to(eval_device) if eval_device != device else bit_gate_gt
    )

    precomputed = (
        precomputed_masks_per_gate.get(gate_idx)
        if parallel_config.precompute_masks
        else None
    )

    # batch_compute_metrics has @profile_fn("Gate Metrics (Batched)") directly
    accuracies, logit_sims, bit_sims, best_sims = batch_compute_metrics(
        model=model,
        circuits=subcircuits,
        x=x_eval,
        y_target=bit_gate_gt_eval,
        y_pred=y_gate_eval,
        gate_idx=gate_idx,
        precomputed_masks=precomputed,
        eval_device=eval_device,
    )

    subcircuit_metrics = [
        SubcircuitMetrics(
            idx=idx,
            accuracy=float(accuracies[idx]),
            logit_similarity=float(logit_sims[idx]),
            bit_similarity=float(bit_sims[idx]),
            best_similarity=float(best_sims[idx]),
        )
        for idx in range(len(subcircuits))
    ]

    per_gate_metrics[gate_name] = GateMetrics(
        test_acc=gate_acc,
        subcircuit_metrics=subcircuit_metrics,
    )

    per_gate_bests[gate_name] = filter_subcircuits(
        setup.constraints,
        per_gate_metrics[gate_name].subcircuit_metrics,
        subcircuits,
        subcircuit_structures,
        max_subcircuits=setup.faithfulness_config.max_subcircuits_per_gate,
    )

    best_indices = per_gate_bests[gate_name]
    best_circuits_for_gate = [subcircuits[idx] for idx in best_indices]

    # batch_evaluate_edge_variants has @profile_fn("Edge Variants") directly
    edge_results = batch_evaluate_edge_variants(
        model=model,
        base_circuits=best_circuits_for_gate,
        x=x_eval,
        y_target=bit_gate_gt_eval,
        y_pred=y_gate_eval,
        gate_idx=gate_idx,
        eval_device=eval_device,
    )

    optimized_circuits = {
        best_indices[orig_idx]: opt_circuit
        for orig_idx, opt_circuit, _, _, _ in edge_results
    }

    best_subcircuit_models = {
        idx: gate_model.separate_subcircuit(optimized_circuits[idx], gate_idx=gate_idx)
        for idx in best_indices
    }

    counterfactual_pairs = create_clean_corrupted_data(
        x=x,
        y=y_gate,
        activations=activations,
        n_pairs=setup.faithfulness_config.n_counterfactual_pairs,
    )

    update_status(f"STARTED_ROBUSTNESS:{gate_idx}")
    robustness_results = robustness_phase(
        best_indices, best_subcircuit_models, gate_model, device, parallel_config
    )
    per_gate_bests_robust[gate_name].extend(robustness_results)
    update_status(f"FINISHED_ROBUSTNESS:{gate_idx}")

    update_status(f"STARTED_FAITH:{gate_idx}")
    faithfulness_results = faithfulness_phase(
        best_indices,
        best_subcircuit_models,
        gate_model,
        x,
        y_gate,
        activations,
        subcircuit_structures,
        counterfactual_pairs,
        setup.faithfulness_config,
        device,
        parallel_config,
    )
    per_gate_bests_faith[gate_name].extend(faithfulness_results)
    update_status(f"FINISHED_FAITH:{gate_idx}")

"""Phase functions for trial execution.

Each phase is a distinct step in the trial pipeline:
- Computing activations
- SPD decomposition
- Enumerating circuits
- Precomputing masks
- Robustness analysis
- Faithfulness analysis

Note: Some phases like train_model, batch_compute_metrics, and
batch_evaluate_edge_variants are now decorated with @profile_fn directly
in their source modules (helpers.py and batched_eval.py).
"""

import torch

from ..causal import calculate_faithfulness_metrics, calculate_observational_metrics
from ..common.batched_eval import (
    adapt_masks_for_gate,
    batch_compute_metrics,
    batch_evaluate_edge_variants,
    precompute_circuit_masks_base,
)
from ..common.circuit import enumerate_all_valid_circuit
from ..common.helpers import train_model
from ..common.parallelization import ParallelTasks
from ..common.profiler import profile, profile_fn
from ..spd_internal.decomposition import decompose_mlp
from ..spd_internal.subcircuits import estimate_spd_subcircuits

# Re-export functions that have @profile_fn directly on them
# These are called from gate_analysis.py and runner.py
__all__ = [
    "batch_compute_metrics",
    "batch_evaluate_edge_variants",
    "compute_activations_phase",
    "enumerate_circuits_phase",
    "faithfulness_phase",
    "precompute_masks_phase",
    "robustness_phase",
    "spd_phase",
    "train_model",
]


@profile_fn("Compute Activations")
def compute_activations_phase(model, data, device, input_size):
    """Compute and store activations for test data."""
    x = data.test.x.contiguous()
    y_gt = data.test.y.contiguous()

    with torch.inference_mode():
        y_pred = model(data.test.x)
        activations = model(data.test.x, return_activations=True)

        # Pre-compute canonical activations for visualization
        canonical_inputs = {
            "0_0": torch.tensor([[0.0, 0.0]], device=device),
            "0_1": torch.tensor([[0.0, 1.0]], device=device),
            "1_0": torch.tensor([[1.0, 0.0]], device=device),
            "1_1": torch.tensor([[1.0, 1.0]], device=device),
        }
        canonical_activations = {
            label: [a.clone() for a in model(inp, return_activations=True)]
            for label, inp in canonical_inputs.items()
        }

        # Pre-compute mean activations for different input ranges
        n_samples = 100
        input_ranges = {
            "0_1": (0.0, 1.0),
            "-1_0": (-1.0, 0.0),
            "-2_2": (-2.0, 2.0),
            "-100_100": (-100.0, 100.0),
        }
        mean_activations_by_range = {}
        for label, (low, high) in input_ranges.items():
            random_inputs = (
                torch.rand(n_samples, input_size, device=device) * (high - low) + low
            )
            all_activations = model(random_inputs, return_activations=True)
            mean_activations = [
                a.mean(dim=0, keepdim=True).clone() for a in all_activations
            ]
            mean_activations_by_range[label] = mean_activations

    # Store layer weights and biases
    layer_weights = [layer[0].weight.detach().cpu() for layer in model.layers]
    layer_biases = [layer[0].bias.detach().cpu() for layer in model.layers]

    return {
        "x": x,
        "y_gt": y_gt,
        "y_pred": y_pred,
        "activations": [a.clone() for a in activations],
        "canonical_activations": canonical_activations,
        "mean_activations_by_range": mean_activations_by_range,
        "layer_weights": layer_weights,
        "layer_biases": layer_biases,
    }


@profile_fn("SPD Decomposition")
def spd_phase(
    setup, trial_result, model, x, y_pred, spd_device, input_size, gate_names
):
    """Run SPD decomposition for all configs."""
    spd_configs_to_run = [setup.spd_config]
    if setup.spd_sweep_configs:
        spd_configs_to_run.extend(setup.spd_sweep_configs)

    for config_idx, spd_config in enumerate(spd_configs_to_run):
        config_id = spd_config.get_config_id()
        print(f"    SPD config {config_idx + 1}/{len(spd_configs_to_run)}: {config_id}")

        with profile(f"spd_mlp_{config_id}"):
            decomposed = decompose_mlp(x, y_pred, model, spd_device, spd_config)

        if config_idx == 0:
            trial_result.decomposed_model = decomposed
        trial_result.decomposed_models_sweep[config_id] = decomposed

    # SPD Subcircuit estimation
    for config_idx, spd_config in enumerate(spd_configs_to_run):
        config_id = spd_config.get_config_id()
        decomposed = trial_result.decomposed_models_sweep[config_id]
        print(
            f"    SPD subcircuit {config_idx + 1}/{len(spd_configs_to_run)}: {config_id}"
        )

        with profile(f"spd_mlp_sc_{config_id}"):
            estimate = estimate_spd_subcircuits(
                decomposed_model=decomposed,
                n_inputs=input_size,
                gate_names=gate_names,
                device=spd_device,
            )

        if config_idx == 0:
            trial_result.spd_subcircuit_estimate = estimate
        trial_result.spd_subcircuit_estimates_sweep[config_id] = estimate


@profile_fn("Enumerate Circuits")
def enumerate_circuits_phase(model, parallel_config):
    """Enumerate all valid circuits and analyze structure."""
    with profile("circuit_enum"):
        subcircuits = enumerate_all_valid_circuit(model, use_tqdm=True)

    with profile("structure_analysis"):
        if parallel_config.enable_parallel_structure:
            with ParallelTasks(
                max_workers=parallel_config.max_workers_structure
            ) as tasks:
                futures = [tasks.submit(s.analyze_structure) for s in subcircuits]
            subcircuit_structures = [f.result() for f in futures]
        else:
            subcircuit_structures = [s.analyze_structure() for s in subcircuits]

    return subcircuits, subcircuit_structures


@profile_fn("Precompute Circuit Masks")
def precompute_masks_phase(subcircuits, model, gate_names, eval_device, output_size):
    """Precompute circuit masks for all gates."""
    precomputed_base_masks = precompute_circuit_masks_base(
        subcircuits, len(model.layers), device=eval_device
    )
    precomputed_masks_per_gate = {}
    for gate_idx in range(len(gate_names)):
        precomputed_masks_per_gate[gate_idx] = adapt_masks_for_gate(
            precomputed_base_masks, gate_idx, output_size
        )
    return precomputed_masks_per_gate


@profile_fn("Robustness Analysis")
def robustness_phase(
    best_indices, best_subcircuit_models, gate_model, device, parallel_config
):
    """Compute robustness metrics for best subcircuits."""

    def compute_single_robustness(subcircuit_idx):
        subcircuit_model = best_subcircuit_models[subcircuit_idx]
        return calculate_observational_metrics(
            subcircuit=subcircuit_model,
            full_model=gate_model,
            n_samples_per_base=200,
            device=device,
        )

    if parallel_config.enable_parallel_robustness and len(best_indices) > 1:
        with ParallelTasks(max_workers=min(4, len(best_indices))) as tasks:
            futures = [
                tasks.submit(compute_single_robustness, idx) for idx in best_indices
            ]
        return [f.result() for f in futures]
    else:
        results = []
        for subcircuit_idx in best_indices:
            with profile("calc_robustness"):
                results.append(compute_single_robustness(subcircuit_idx))
        return results


@profile_fn("Faithfulness Analysis")
def faithfulness_phase(
    best_indices,
    best_subcircuit_models,
    gate_model,
    x,
    y_gate,
    activations,
    subcircuit_structures,
    counterfactual_pairs,
    config,
    device,
    parallel_config,
):
    """Compute faithfulness metrics for best subcircuits."""

    def compute_single_faithfulness(subcircuit_idx):
        subcircuit_model = best_subcircuit_models[subcircuit_idx]
        return calculate_faithfulness_metrics(
            x=x,
            y=y_gate,
            model=gate_model,
            activations=activations,
            subcircuit=subcircuit_model,
            structure=subcircuit_structures[subcircuit_idx],
            counterfactual_pairs=counterfactual_pairs,
            config=config,
            device=device,
        )

    if parallel_config.enable_parallel_faithfulness and len(best_indices) > 1:
        with ParallelTasks(max_workers=min(4, len(best_indices))) as tasks:
            futures = [
                tasks.submit(compute_single_faithfulness, idx) for idx in best_indices
            ]
        return [f.result() for f in futures]
    else:
        results = []
        for subcircuit_idx in best_indices:
            with profile("calc_faithfulness"):
                results.append(compute_single_faithfulness(subcircuit_idx))
        return results

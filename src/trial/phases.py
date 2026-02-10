"""Phase functions for trial execution.

Each phase is a distinct step in the trial pipeline:
- Computing activations
- Enumerating circuits
- Precomputing masks
- Faithfulness analysis (includes observational, interventional, counterfactual)

Note: SPD decomposition is now run separately via src.spd.run_spd()

Note: Some phases like train_model, batch_compute_metrics, and
batch_evaluate_edge_variants are now decorated with @profile_fn directly
in their source modules (helpers.py and batched_eval.py).
"""

import torch

from src.analysis import calculate_faithfulness_metrics
from src.circuit import (
    adapt_masks_for_gate,
    batch_compute_metrics,
    batch_evaluate_edge_variants,
    enumerate_all_valid_circuit,
    precompute_circuit_masks_base,
)
from src.infra import ParallelTasks, profile, profile_fn
from src.training import train_model

# Re-export functions that have @profile_fn directly on them
# These are called from gate_analysis.py and trial_executor.py
__all__ = [
    "batch_compute_metrics",
    "batch_evaluate_edge_variants",
    "compute_activations_phase",
    "enumerate_circuits_phase",
    "faithfulness_phase",
    "precompute_masks_phase",
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


@profile_fn("Faithfulness Analysis")
def faithfulness_phase(
    subcircuit_keys,
    subcircuit_models,
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
    """Compute complete faithfulness metrics (observational + interventional + counterfactual).

    For each subcircuit, computes:
    - Observational: How well subcircuit matches gate model under perturbations
    - Interventional: How well subcircuit matches under activation patching
    - Counterfactual: 2x2 patching matrix (sufficiency, completeness, necessity, independence)

    Args:
        subcircuit_keys: List of keys identifying subcircuits. Can be:
            - Integers (legacy: node pattern indices)
            - Tuples of (node_idx, edge_var_idx) for hierarchical structure
        subcircuit_models: Dict mapping keys to subcircuit models
        subcircuit_structures: Dict mapping keys to circuit structures
    """

    def compute_single_faithfulness(key):
        subcircuit_model = subcircuit_models[key]
        structure = subcircuit_structures[key]
        return calculate_faithfulness_metrics(
            x=x,
            y=y_gate,
            model=gate_model,
            activations=activations,
            subcircuit=subcircuit_model,
            structure=structure,
            counterfactual_pairs=counterfactual_pairs,
            config=config,
            device=device,
        )

    if parallel_config.enable_parallel_faithfulness and len(subcircuit_keys) > 1:
        with ParallelTasks(max_workers=min(4, len(subcircuit_keys))) as tasks:
            futures = [
                tasks.submit(compute_single_faithfulness, key)
                for key in subcircuit_keys
            ]
        return [f.result() for f in futures]
    else:
        results = []
        for key in subcircuit_keys:
            with profile("calc_faithfulness"):
                results.append(compute_single_faithfulness(key))
        return results

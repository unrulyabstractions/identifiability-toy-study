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
from src.domain import generate_canonical_inputs
from src.circuit import (
    adapt_masks_for_gate,
    batch_compute_metrics,
    batch_evaluate_edge_variants,
    enumerate_all_valid_circuit,
    parse_subcircuit_idx,
    precompute_circuit_masks_base,
)
from src.infra import ParallelTasks, profile, profile_fn
from src.infra.profiler import trace, traced, trace_progress
from src.training import train_model

# Re-export functions that have @profile_fn directly on them
# These are called from gate_analysis.py and trial_executor.py
__all__ = [
    "batch_compute_metrics",
    "batch_evaluate_edge_variants",
    "compute_activations_phase",
    "decision_boundary_phase",
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
        canonical_inputs = generate_canonical_inputs(input_size, device)
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
def precompute_masks_phase(
    subcircuits, model, gate_names, eval_device, output_size, gate_n_inputs_list=None
):
    """Precompute circuit masks for all gates.

    Args:
        subcircuits: List of subcircuits
        model: Model to compute masks for
        gate_names: List of gate names
        eval_device: Device for evaluation
        output_size: Number of output gates
        gate_n_inputs_list: Optional list of n_inputs per gate (for mixed input sizes)
    """
    precomputed_base_masks = precompute_circuit_masks_base(
        subcircuits, len(model.layers), device=eval_device
    )
    precomputed_masks_per_gate = {}
    for gate_idx in range(len(gate_names)):
        gate_n_inputs = gate_n_inputs_list[gate_idx] if gate_n_inputs_list else None
        precomputed_masks_per_gate[gate_idx] = adapt_masks_for_gate(
            precomputed_base_masks, gate_idx, output_size, gate_n_inputs=gate_n_inputs
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
            - Integers (flat indices from make_subcircuit_idx)
            - Tuples of (node_mask_idx, edge_variant_rank) for hierarchical structure
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

    trace("faithfulness_phase starting", n_subcircuits=len(subcircuit_keys))

    if parallel_config.enable_parallel_faithfulness and len(subcircuit_keys) > 1:
        with ParallelTasks(max_workers=min(4, len(subcircuit_keys))) as tasks:
            futures = [
                tasks.submit(compute_single_faithfulness, key)
                for key in subcircuit_keys
            ]
        return [f.result() for f in futures]
    else:
        results = []
        for i, key in enumerate(subcircuit_keys):
            trace_progress(i + 1, len(subcircuit_keys), "faithfulness subcircuits", every=5)
            with traced("calc_faithfulness", key=key):
                results.append(compute_single_faithfulness(key))
        trace("faithfulness_phase complete", n_results=len(results))
        return results


@profile_fn("Decision Boundary Data")
def decision_boundary_phase(
    model,
    gate_models,
    gate_names,
    subcircuits,
    trial_metrics,
    device,
    setup,
):
    """Generate decision boundary data for visualization.

    Pre-computes all data needed for decision boundary plots so no
    model inference is needed during visualization.

    Args:
        model: Full model
        gate_models: List of single-gate models
        gate_names: List of gate names
        subcircuits: List of subcircuit dicts
        trial_metrics: Metrics with per_gate_bests containing best subcircuit keys
        device: Compute device
        setup: Trial setup with model params (for width/depth)

    Returns:
        Tuple of (gate_db_data, subcircuit_db_data) where:
        - gate_db_data: dict mapping gate_name -> decision boundary data
        - subcircuit_db_data: dict mapping gate_name -> {subcircuit_idx: data}
    """
    from src.circuit import Circuit
    from src.domain import resolve_gate
    from src.visualization.decision_boundary import (
        generate_grid_data,
        generate_monte_carlo_data,
    )

    # Get architecture parameters for subcircuit indexing
    width = setup.model_params.width
    depth = setup.model_params.depth

    # Get model's expected input size (may differ from individual gate n_inputs)
    model_n_inputs = model.layers[0][0].in_features

    gate_db_data = {}
    subcircuit_db_data = {}

    for gate_idx, gate_name in enumerate(gate_names):
        gate = resolve_gate(gate_name)
        n_inputs = gate.n_inputs
        gate_model = gate_models[gate_idx]

        # Generate data for this gate's separated model
        # We visualize only n_inputs dimensions but pad to model_n_inputs for model
        # (e.g., XOR visualizes 2D but model may expect 3 inputs due to MAJORITY)
        if n_inputs <= 2:
            gate_db_data[gate_name] = generate_grid_data(
                model=gate_model,
                n_inputs=n_inputs,
                gate_idx=0,  # gate_model has single output
                device=device,
                resolution=400,  # Higher resolution for smoother contours
                model_n_inputs=model_n_inputs,
            )
        else:
            gate_db_data[gate_name] = generate_monte_carlo_data(
                model=gate_model,
                n_inputs=n_inputs,
                gate_idx=0,  # gate_model has single output
                device=device,
                model_n_inputs=model_n_inputs,
            )

        # Generate data for best subcircuits
        per_gate_bests = trial_metrics.per_gate_bests
        if gate_name not in per_gate_bests:
            continue

        subcircuit_db_data[gate_name] = {}
        gate_model = gate_models[gate_idx]

        # Get stored edge-masked circuits for this gate
        gate_circuits = trial_metrics.per_gate_circuits.get(gate_name, {})

        for subcircuit_idx in per_gate_bests[gate_name]:
            # Parse flat index to get node pattern and edge variant rank
            node_mask_idx, edge_variant_rank = parse_subcircuit_idx(width, depth, subcircuit_idx)

            # Look up the edge-masked circuit (stored during gate analysis)
            sc_data = gate_circuits.get(subcircuit_idx)

            # Fallback to base circuit if edge-masked not available
            if sc_data is None:
                for sc in subcircuits:
                    if isinstance(sc, dict):
                        if sc.get("idx") == node_mask_idx:
                            sc_data = sc
                            break
                    else:
                        sc_data = sc.to_dict() if hasattr(sc, "to_dict") else None
                        break

            if sc_data is None:
                continue

            try:
                circuit = Circuit.from_dict(sc_data)
                subcircuit_model = gate_model.separate_subcircuit(circuit, gate_idx=0)

                if n_inputs <= 2:
                    data = generate_grid_data(
                        model=subcircuit_model,
                        n_inputs=n_inputs,
                        gate_idx=0,  # Subcircuit model has single output
                        device=device,
                        resolution=400,  # Higher resolution for smoother contours
                        model_n_inputs=model_n_inputs,
                    )
                else:
                    data = generate_monte_carlo_data(
                        model=subcircuit_model,
                        n_inputs=n_inputs,
                        gate_idx=0,
                        device=device,
                        model_n_inputs=model_n_inputs,
                    )

                subcircuit_db_data[gate_name][subcircuit_idx] = data
            except Exception:
                pass  # Skip failures silently

    return gate_db_data, subcircuit_db_data

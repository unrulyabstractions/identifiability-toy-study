"""Run trial-like analysis on SPD-discovered subcircuits.

After SPD discovers clusters of components that implement functions,
this module runs the same analysis pipeline as regular trials on those
SPD-discovered subcircuits. This validates SPD's discoveries against
the standard circuit analysis methods.

Key approach:
1. Group SPD clusters by which gate function they implement
2. Reconstruct weight matrices from the selected components
3. Create an MLP with those weights (SPD subcircuit model)
4. Run observational, interventional, and counterfactual analysis
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.analysis import (
    CleanCorruptedPair,
    create_clean_corrupted_data,
    calculate_faithfulness_metrics,
)
from src.circuit import CircuitStructure
from src.domain import get_base_gate_name, resolve_gate
from src.experiment_config import FaithfulnessConfig
from src.infra import profile_fn
from src.model import MLP
from src.schemas import FaithfulnessMetrics

if TYPE_CHECKING:
    from src.model import DecomposedMLP
    from src.schemas import ExperimentResult, TrialResult
    from .types import SpdResults, SpdTrialResult, SPDSubcircuitEstimate


@dataclass
class SPDGateAnalysis:
    """Analysis results for a single gate's SPD subcircuit."""
    gate_name: str
    gate_idx: int
    cluster_indices: list[int]  # Which clusters implement this gate
    component_indices: list[int]  # All components for this gate
    reconstructed_model: MLP | None = None
    faithfulness: FaithfulnessMetrics | None = None


@dataclass
class SPDTrialAnalysis:
    """Complete SPD trial analysis results."""
    trial_id: str
    gate_analyses: dict[str, SPDGateAnalysis] = field(default_factory=dict)


def extract_gate_components(
    estimate: "SPDSubcircuitEstimate",
    gate_names: list[str],
) -> dict[str, list[int]]:
    """Map each gate to SPD components that implement it.

    Args:
        estimate: SPD subcircuit estimate with cluster assignments and functions
        gate_names: List of gate names in the model

    Returns:
        Dict mapping gate_name -> list of component indices
    """
    gate_to_components = {name: [] for name in gate_names}

    # Build cluster -> components mapping
    cluster_to_components = {}
    for comp_idx, cluster_idx in enumerate(estimate.cluster_assignments):
        cluster_to_components.setdefault(cluster_idx, []).append(comp_idx)

    # Map clusters to gates based on function mapping
    for cluster_idx, func_str in estimate.cluster_functions.items():
        # Parse function string like "XOR (0.92)" to get base gate name
        if func_str in ("UNKNOWN", "INACTIVE", "EMPTY"):
            continue

        # Extract gate name from "GATE_NAME (score)" format
        base_func = func_str.split(" (")[0] if " (" in func_str else func_str

        # Find which gate(s) in the model this matches
        for gate_name in gate_names:
            gate_base = get_base_gate_name(gate_name)
            if gate_base == base_func:
                components = cluster_to_components.get(cluster_idx, [])
                gate_to_components[gate_name].extend(components)

    return gate_to_components


def reconstruct_weights_for_components(
    decomposed: "DecomposedMLP",
    component_indices: list[int],
) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    """Reconstruct weight matrices using only specified components.

    For each layer, computes: W = V[:, indices] @ U[indices, :]

    Args:
        decomposed: The decomposed MLP with U, V matrices
        component_indices: Which global component indices to include

    Returns:
        Dict mapping layer name -> (weight, bias)
    """
    if decomposed.component_model is None:
        return {}

    component_model = decomposed.component_model
    layer_weights = {}

    # Track which components belong to which layer
    layer_component_ranges = {}
    offset = 0
    for module_name in sorted(component_model.components.keys()):
        comp = component_model.components[module_name]
        n_comp = comp.C
        layer_component_ranges[module_name] = (offset, offset + n_comp)
        offset += n_comp

    # Reconstruct each layer's weights
    for module_name, components in component_model.components.items():
        layer_start, layer_end = layer_component_ranges[module_name]

        # Get local indices for this layer
        local_indices = [
            idx - layer_start
            for idx in component_indices
            if layer_start <= idx < layer_end
        ]

        if not local_indices:
            # No components for this layer - use zeros
            weight = torch.zeros_like(components.weight)
        else:
            # Reconstruct: W = (V[:, indices] @ U[indices, :]).T
            V = components.V  # [d_in, C]
            U = components.U  # [C, d_out]
            local_idx_tensor = torch.tensor(local_indices, device=V.device)
            V_subset = V[:, local_idx_tensor]  # [d_in, k]
            U_subset = U[local_idx_tensor, :]  # [k, d_out]
            weight = (V_subset @ U_subset).T  # [d_out, d_in]

        bias = components.bias if hasattr(components, 'bias') else None
        layer_weights[module_name] = (weight, bias)

    return layer_weights


def create_mlp_from_weights(
    target_model: MLP,
    layer_weights: dict[str, tuple[torch.Tensor, torch.Tensor | None]],
    gate_idx: int = 0,
) -> MLP:
    """Create a new MLP with the given layer weights.

    Args:
        target_model: Original MLP to copy structure from
        layer_weights: Dict mapping layer name -> (weight, bias)
        gate_idx: Which output gate to use (for multi-gate models)

    Returns:
        New MLP with the specified weights (output_size=1 for the gate)
    """
    # Create single-output MLP for this gate
    submodel = MLP(
        hidden_sizes=target_model.hidden_sizes,
        input_size=target_model.input_size,
        output_size=1,
        activation=target_model.activation,
        device=target_model.device,
        debug=target_model.debug,
    )

    with torch.no_grad():
        for layer_idx in range(len(target_model.layers)):
            layer_name = f"layers.{layer_idx}.0"

            if layer_name in layer_weights:
                weight, bias = layer_weights[layer_name]

                # Handle final layer: slice to get single output
                if layer_idx == len(target_model.layers) - 1:
                    weight = weight[gate_idx:gate_idx+1, :]
                    if bias is not None:
                        bias = bias[gate_idx:gate_idx+1]

                dst = submodel.layers[layer_idx][0]
                dst.weight.copy_(weight)
                if bias is not None:
                    dst.bias.copy_(bias)

    return submodel


def create_full_circuit_structure(model: MLP) -> CircuitStructure:
    """Create a CircuitStructure representing the full circuit (no masking)."""
    layer_sizes = model.layer_sizes

    # Full node masks (all ones)
    node_masks = [torch.ones(size) for size in layer_sizes]

    # Full edge masks (all ones)
    edge_masks = []
    for i in range(len(layer_sizes) - 1):
        edge_masks.append(torch.ones(layer_sizes[i+1], layer_sizes[i]))

    return CircuitStructure(
        layer_sizes=layer_sizes,
        n_active_nodes=[int(m.sum().item()) for m in node_masks],
        n_total_nodes=sum(layer_sizes),
        n_active_edges=[int(m.sum().item()) for m in edge_masks],
        n_total_edges=sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1)),
        connectivity_score=1.0,
        has_isolated_nodes=False,
        layer_utilization=[1.0] * len(layer_sizes),
    )


@profile_fn("Trial on SPD Results")
def run_trial_on_spd_results(
    spd_results: "SpdResults",
    experiment_result: "ExperimentResult",
    run_dir: str | Path,
    device: str = "cpu",
) -> dict[str, SPDTrialAnalysis]:
    """Run trial-like analysis on SPD-discovered subcircuits.

    For each trial's SPD decomposition, this:
    1. Groups SPD components by which gate they implement
    2. Reconstructs weight matrices from those components
    3. Creates MLPs with the reconstructed weights
    4. Runs observational, interventional, and counterfactual analysis

    Args:
        spd_results: Results from run_spd()
        experiment_result: Original experiment results with trained models
        run_dir: Directory for saving results
        device: Compute device

    Returns:
        Dict mapping trial_id -> SPDTrialAnalysis
    """
    run_dir = Path(run_dir)
    results = {}

    for trial_id, spd_trial in spd_results.per_trial.items():
        if spd_trial.decomposed_model is None:
            continue

        trial_result = experiment_result.trials.get(trial_id)
        if trial_result is None:
            continue

        print(f"[SPD Trial Analysis] {trial_id[:8]}...")

        analysis = analyze_spd_trial(
            spd_trial=spd_trial,
            trial_result=trial_result,
            run_dir=run_dir / "trials" / trial_id / "spd_analysis",
            device=device,
        )
        results[trial_id] = analysis

    return results


def analyze_spd_trial(
    spd_trial: "SpdTrialResult",
    trial_result: "TrialResult",
    run_dir: Path,
    device: str = "cpu",
) -> SPDTrialAnalysis:
    """Analyze a single trial's SPD results.

    Args:
        spd_trial: SPD trial result with decomposed model and cluster estimates
        trial_result: Original trial result with model and data
        run_dir: Output directory
        device: Compute device

    Returns:
        SPDTrialAnalysis with results for each gate
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    decomposed = spd_trial.decomposed_model
    estimate = spd_trial.spd_subcircuit_estimate
    model = trial_result.model
    gate_names = trial_result.setup.model_params.logic_gates
    x = trial_result.test_x
    y = trial_result.test_y

    if decomposed is None or estimate is None or model is None or x is None:
        return SPDTrialAnalysis(trial_id=spd_trial.trial_id)

    analysis = SPDTrialAnalysis(trial_id=spd_trial.trial_id)

    # Get activations for counterfactual analysis
    with torch.inference_mode():
        activations = model(x, return_activations=True)

    # Map gates to their SPD components
    gate_to_components = extract_gate_components(estimate, gate_names)

    # Get separated gate models for comparison
    gate_models = model.separate_into_k_mlps()

    # Analyze each gate
    for gate_idx, gate_name in enumerate(gate_names):
        component_indices = gate_to_components[gate_name]

        gate_analysis = SPDGateAnalysis(
            gate_name=gate_name,
            gate_idx=gate_idx,
            cluster_indices=[],  # Could compute which clusters these came from
            component_indices=component_indices,
        )

        if not component_indices:
            print(f"    {gate_name}: No SPD components found")
            analysis.gate_analyses[gate_name] = gate_analysis
            continue

        # Reconstruct weights from SPD components
        layer_weights = reconstruct_weights_for_components(decomposed, component_indices)

        if not layer_weights:
            print(f"    {gate_name}: Could not reconstruct weights")
            analysis.gate_analyses[gate_name] = gate_analysis
            continue

        # Create MLP with reconstructed weights
        spd_subcircuit_model = create_mlp_from_weights(
            target_model=model,
            layer_weights=layer_weights,
            gate_idx=gate_idx,
        )
        gate_analysis.reconstructed_model = spd_subcircuit_model

        # Get gate-specific data
        gate_model = gate_models[gate_idx]
        y_gate = trial_result.test_y_pred[..., [gate_idx]] if trial_result.test_y_pred is not None else None

        if y_gate is None:
            with torch.inference_mode():
                y_gate = model(x)[..., [gate_idx]]

        # Skip faithfulness for high-input gates (>2 inputs)
        n_inputs = x.shape[1]
        if n_inputs > 2:
            print(f"    {gate_name}: {len(component_indices)} components (skipping faithfulness for {n_inputs}-input)")
            analysis.gate_analyses[gate_name] = gate_analysis
            continue

        # Create counterfactual pairs
        counterfactual_pairs = create_clean_corrupted_data(
            x=x,
            y=y_gate,
            activations=activations,
            n_pairs=5,  # Small number for SPD analysis
        )

        # Create structure for full circuit (SPD subcircuit has no internal masking)
        structure = create_full_circuit_structure(spd_subcircuit_model)

        # Run faithfulness analysis
        try:
            faithfulness = calculate_faithfulness_metrics(
                x=x,
                y=y_gate,
                model=gate_model,
                activations=activations,
                subcircuit=spd_subcircuit_model,
                structure=structure,
                counterfactual_pairs=counterfactual_pairs,
                config=FaithfulnessConfig(
                    n_interventions_per_patch=5,
                    n_counterfactual_pairs=5,
                ),
                device=device,
            )
            gate_analysis.faithfulness = faithfulness

            print(f"    {gate_name}: {len(component_indices)} components, "
                  f"sufficiency={faithfulness.counterfactual.sufficiency:.3f}, "
                  f"necessity={faithfulness.counterfactual.necessity:.3f}")
        except Exception as e:
            print(f"    {gate_name}: Faithfulness analysis failed: {e}")

        analysis.gate_analyses[gate_name] = gate_analysis

    return analysis

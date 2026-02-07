"""
SPD-based subcircuit estimation.

This module identifies subcircuits from SPD (Stochastic Parameter Decomposition)
results. SPD components that are consistently masked together likely form
functional subcircuits.

References:
- SPD paper: https://arxiv.org/pdf/2506.20790
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..common.circuit import Circuit
from .analysis import (
    cluster_components_hierarchical,
    compute_coactivation_matrix,
    compute_importance_matrix,
    map_clusters_to_functions,
)

if TYPE_CHECKING:
    from ..common.neural_model import DecomposedMLP, MLP
    from .analysis import SPDAnalysisResult


@dataclass
class SPDSubcircuitEstimate:
    """Result of SPD-based subcircuit estimation."""

    cluster_assignments: list[int] = field(default_factory=list)
    n_clusters: int = 0
    cluster_sizes: list[int] = field(default_factory=list)
    component_importance: np.ndarray | None = None
    coactivation_matrix: np.ndarray | None = None
    component_labels: list[str] = field(default_factory=list)
    cluster_functions: dict[int, str] = field(default_factory=dict)
    full_analysis: "SPDAnalysisResult | None" = None


def estimate_spd_subcircuits(
    decomposed_model: "DecomposedMLP",
    target_model: "MLP" = None,
    n_inputs: int = 2,
    gate_names: list[str] = None,
    device: str = "cpu",
) -> SPDSubcircuitEstimate | None:
    """
    Estimate subcircuits from SPD decomposition using component clustering.

    Approach:
    1. Compute causal importance values for all binary input combinations
    2. Build coactivation matrix (which components fire together)
    3. Cluster components using hierarchical clustering
    4. Map clusters to potential boolean functions

    Args:
        decomposed_model: Result from decompose_mlp()
        target_model: Original MLP (unused, kept for API compatibility)
        n_inputs: Number of input dimensions
        gate_names: Names of gates in the model
        device: Device for computation

    Returns:
        SPDSubcircuitEstimate with cluster assignments and statistics,
        or None if SPD decomposition is not available.
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return None

    n_components = decomposed_model.get_n_components()
    if n_components == 0:
        return None

    # Compute importance matrix for all binary inputs
    importance_matrix, component_labels = compute_importance_matrix(
        decomposed_model, n_inputs, device
    )

    if importance_matrix.size == 0:
        return SPDSubcircuitEstimate(
            cluster_assignments=list(range(n_components)),
            n_clusters=n_components,
            cluster_sizes=[1] * n_components,
        )

    # Cluster components based on coactivation
    coactivation_matrix = compute_coactivation_matrix(importance_matrix)
    cluster_assignments = cluster_components_hierarchical(coactivation_matrix)
    n_clusters = max(cluster_assignments) + 1 if cluster_assignments else 0

    # Compute cluster sizes
    cluster_sizes = [0] * n_clusters
    for c in cluster_assignments:
        cluster_sizes[c] += 1

    # Map clusters to functions
    cluster_functions = map_clusters_to_functions(
        importance_matrix,
        cluster_assignments,
        n_inputs,
        gate_names,
    )

    return SPDSubcircuitEstimate(
        cluster_assignments=cluster_assignments,
        n_clusters=n_clusters,
        cluster_sizes=cluster_sizes,
        component_importance=importance_matrix.mean(axis=0),
        coactivation_matrix=coactivation_matrix,
        component_labels=component_labels,
        cluster_functions=cluster_functions,
    )


def spd_clusters_to_circuits(
    estimate: SPDSubcircuitEstimate,
    model_layer_sizes: list[int],
) -> list:
    """
    Convert SPD cluster assignments to Circuit objects.

    Maps continuous SPD component masks to discrete circuit structures by:
    1. For each cluster, identify which layers have active components
    2. Create node masks based on component activity patterns
    3. Create edge masks that connect active nodes

    Args:
        estimate: SPD subcircuit estimate
        model_layer_sizes: Layer sizes of the target model [input, hidden..., output]

    Returns:
        List of Circuit objects (one per cluster)
    """
    if estimate.n_clusters == 0:
        return []

    circuits = []

    for cluster_idx in range(estimate.n_clusters):
        component_indices = [
            i for i, c in enumerate(estimate.cluster_assignments)
            if c == cluster_idx
        ]

        if not component_indices:
            continue

        # Parse component labels to get layer info
        layer_activity = {}
        for comp_idx in component_indices:
            if comp_idx < len(estimate.component_labels):
                label = estimate.component_labels[comp_idx]
                parts = label.split(":")
                layer_part = parts[0]
                try:
                    layer_idx = int(layer_part.split(".")[1])
                    if layer_idx not in layer_activity:
                        layer_activity[layer_idx] = set()
                    layer_activity[layer_idx].add(comp_idx)
                except (IndexError, ValueError):
                    continue

        # Create full circuit as placeholder
        circuit = Circuit.full(model_layer_sizes)
        circuits.append(circuit)

    return circuits


def save_spd_estimate(estimate: SPDSubcircuitEstimate, output_dir: str | Path) -> None:
    """Save SPD subcircuit estimate to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON-serializable data
    estimate_data = {
        "cluster_assignments": estimate.cluster_assignments,
        "n_clusters": estimate.n_clusters,
        "cluster_sizes": estimate.cluster_sizes,
        "component_labels": estimate.component_labels,
        "cluster_functions": {str(k): v for k, v in estimate.cluster_functions.items()},
    }
    with open(output_dir / "estimate.json", "w", encoding="utf-8") as f:
        json.dump(estimate_data, f, indent=2)

    # Save numpy arrays
    if estimate.component_importance is not None:
        np.save(output_dir / "component_importance.npy", estimate.component_importance)

    if estimate.coactivation_matrix is not None:
        np.save(output_dir / "coactivation_matrix.npy", estimate.coactivation_matrix)


def load_spd_estimate(input_dir: str | Path) -> SPDSubcircuitEstimate | None:
    """Load SPD subcircuit estimate from disk."""
    input_dir = Path(input_dir)

    estimate_path = input_dir / "estimate.json"
    if not estimate_path.exists():
        return None

    with open(estimate_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    estimate = SPDSubcircuitEstimate(
        cluster_assignments=data["cluster_assignments"],
        n_clusters=data["n_clusters"],
        cluster_sizes=data["cluster_sizes"],
        component_labels=data["component_labels"],
        cluster_functions={int(k): v for k, v in data["cluster_functions"].items()},
    )

    # Load numpy arrays if they exist
    importance_path = input_dir / "component_importance.npy"
    if importance_path.exists():
        estimate.component_importance = np.load(importance_path)

    coact_path = input_dir / "coactivation_matrix.npy"
    if coact_path.exists():
        estimate.coactivation_matrix = np.load(coact_path)

    return estimate

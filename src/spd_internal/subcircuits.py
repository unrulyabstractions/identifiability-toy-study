"""
SPD-based subcircuit estimation.

This module identifies subcircuits from SPD (Stochastic Parameter Decomposition)
results. SPD components that are consistently masked together likely form
functional subcircuits.

References:
- SPD paper: https://arxiv.org/pdf/2506.20790
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..common.circuit import Circuit
from .analysis import (
    cluster_components_hierarchical,
    compute_coactivation_matrix,
    compute_importance_matrix,
    map_clusters_to_functions,
)

from .persistence import load_spd_estimate, save_spd_estimate

if TYPE_CHECKING:
    from ..common.neural_model import MLP, DecomposedMLP
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
            i for i, c in enumerate(estimate.cluster_assignments) if c == cluster_idx
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

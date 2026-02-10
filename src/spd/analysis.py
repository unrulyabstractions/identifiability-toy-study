"""SPD analysis: orchestration of importance, clustering, and function mapping.

After SPD decomposes weights into components, we need to understand what each
component does. This module answers: "Which components implement the same
function, and what function is that?"

The analysis pipeline:
    1. Compute importance matrix
       - For each possible input (00, 01, 10, 11), measure how much each
         component contributes to the output (causal importance, CI)
       - Result: [n_inputs, n_components] matrix of CI values in [0, 1]

    2. Compute coactivation matrix
       - Components that fire together (both have high CI on same inputs)
         are likely implementing the same function
       - Result: [n_components, n_components] matrix of co-firing counts

    3. Cluster components
       - Use hierarchical clustering on coactivation patterns
       - Components with similar activation patterns get grouped together
       - Result: cluster_assignments list (component_idx -> cluster_idx)

    4. Map clusters to functions
       - Compare each cluster's activation pattern to known boolean gates
         (XOR activates on 01 and 10, AND activates only on 11, etc.)
       - Use Jaccard similarity to find best match
       - Result: cluster_functions dict (cluster_idx -> "XOR (0.95)")

Key insight: A cluster that activates on exactly the inputs where XOR=1
is likely implementing XOR. This lets us identify functional subcircuits.

This module provides the orchestration layer. The actual implementations are in:
- importance.py: compute_importance_matrix, compute_coactivation_matrix
- clustering.py: detect_dead_components, cluster_components_hierarchical, map_clusters_to_functions
- evaluation.py: analyze_cluster_robustness, analyze_cluster_faithfulness, analyze_all_clusters
"""

from typing import TYPE_CHECKING

import numpy as np

from .clustering import (
    cluster_components_hierarchical,
    detect_dead_components,
    map_clusters_to_functions,
)
from .evaluation import (
    analyze_all_clusters,
    analyze_cluster_faithfulness,
    analyze_cluster_robustness,
)
from .importance import compute_coactivation_matrix, compute_importance_matrix
from .types import ClusterInfo, SPDAnalysisResult, SPDSubcircuitEstimate
from .validation import compute_validation_metrics

if TYPE_CHECKING:
    from src.model import DecomposedMLP


# Re-export all functions for backward compatibility
__all__ = [
    # Importance (from importance.py)
    "compute_importance_matrix",
    "compute_coactivation_matrix",
    # Clustering (from clustering.py)
    "detect_dead_components",
    "cluster_components_hierarchical",
    "map_clusters_to_functions",
    # Evaluation (from evaluation.py)
    "analyze_cluster_robustness",
    "analyze_cluster_faithfulness",
    "analyze_all_clusters",
    # Orchestration (this module)
    "run_spd_analysis",
    "estimate_spd_subcircuits",
]


# =============================================================================
# Main Analysis Entry Point
# =============================================================================


def run_spd_analysis(
    decomposed_model: "DecomposedMLP",
    target_model=None,
    n_inputs: int = 2,
    gate_names: list[str] = None,
    n_clusters: int = None,
    device: str = "cpu",
) -> SPDAnalysisResult:
    """Run complete SPD analysis: clustering, function mapping, and metrics.

    This is the main entry point for analyzing an SPD decomposition. It:
    1. Computes validation metrics (MMCS, ML2R, faithfulness loss)
    2. Detects dead components
    3. Builds the importance and coactivation matrices
    4. Clusters components by coactivation patterns
    5. Maps clusters to boolean functions

    Args:
        decomposed_model: Trained SPD decomposition
        target_model: Original MLP (for robustness/faithfulness tests)
        n_inputs: Number of input dimensions
        gate_names: Names of gates in the model
        n_clusters: Target number of clusters (None = auto)
        device: Compute device

    Returns:
        SPDAnalysisResult with all analysis data
    """
    result = SPDAnalysisResult()

    if decomposed_model is None or decomposed_model.component_model is None:
        return result

    # Validation metrics
    validation_metrics = compute_validation_metrics(decomposed_model)
    result.mmcs = validation_metrics["mmcs"]
    result.ml2r = validation_metrics["ml2r"]
    result.faithfulness_loss = validation_metrics["faithfulness_loss"]

    # Dead component detection
    alive_labels, dead_labels = detect_dead_components(decomposed_model)
    result.n_alive_components = len(alive_labels)
    result.n_dead_components = len(dead_labels)
    result.dead_component_labels = dead_labels

    # Importance matrix
    importance_matrix, component_labels = compute_importance_matrix(
        decomposed_model, n_inputs, device
    )

    if importance_matrix.size == 0:
        return result

    result.importance_matrix = importance_matrix
    result.component_labels = component_labels
    result.n_components = len(component_labels)

    # Count layers
    layer_names = set(label.split(":")[0] for label in component_labels)
    result.n_layers = len(layer_names)

    # Clustering
    result.coactivation_matrix = compute_coactivation_matrix(importance_matrix)
    result.cluster_assignments = cluster_components_hierarchical(
        result.coactivation_matrix, n_clusters=n_clusters
    )
    result.n_clusters = max(result.cluster_assignments) + 1 if result.cluster_assignments else 0

    # Function mapping
    cluster_functions = map_clusters_to_functions(
        importance_matrix, result.cluster_assignments, n_inputs, gate_names
    )

    # Build cluster info
    for cluster_idx in range(result.n_clusters):
        component_indices = [
            i for i, c in enumerate(result.cluster_assignments) if c == cluster_idx
        ]
        cluster_labels = [component_labels[i] for i in component_indices]

        if component_indices:
            mean_imp = importance_matrix[:, component_indices].mean()
        else:
            mean_imp = 0.0

        cluster_info = ClusterInfo(
            cluster_idx=cluster_idx,
            component_indices=component_indices,
            component_labels=cluster_labels,
            mean_importance=float(mean_imp),
            function_mapping=cluster_functions.get(cluster_idx, ""),
        )
        result.clusters.append(cluster_info)

    return result


def estimate_spd_subcircuits(
    decomposed_model: "DecomposedMLP",
    n_inputs: int = 2,
    gate_names: list[str] = None,
    device: str = "cpu",
) -> SPDSubcircuitEstimate | None:
    """Estimate subcircuits from SPD decomposition using component clustering.

    This is a lighter-weight alternative to run_spd_analysis() that focuses
    specifically on subcircuit discovery. It returns just the cluster
    assignments and function mappings without the full analysis artifacts.

    Args:
        decomposed_model: Trained SPD decomposition
        n_inputs: Number of input bits (2 for boolean gates)
        gate_names: Names of gates to match against
        device: Compute device

    Returns:
        SPDSubcircuitEstimate with cluster assignments and statistics
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return None

    n_components = decomposed_model.get_n_components()
    if n_components == 0:
        return None

    importance_matrix, component_labels = compute_importance_matrix(
        decomposed_model, n_inputs, device
    )

    if importance_matrix.size == 0:
        return SPDSubcircuitEstimate(
            cluster_assignments=list(range(n_components)),
            n_clusters=n_components,
            cluster_sizes=[1] * n_components,
        )

    coactivation_matrix = compute_coactivation_matrix(importance_matrix)
    cluster_assignments = cluster_components_hierarchical(coactivation_matrix)
    n_clusters = max(cluster_assignments) + 1 if cluster_assignments else 0

    cluster_sizes = [0] * n_clusters
    for c in cluster_assignments:
        cluster_sizes[c] += 1

    cluster_functions = map_clusters_to_functions(
        importance_matrix, cluster_assignments, n_inputs, gate_names
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

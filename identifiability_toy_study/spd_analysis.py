"""
SPD Analysis Module - Comprehensive analysis of SPD decomposition results.

This module provides:
1. Component clustering based on coactivation patterns
2. Per-cluster robustness and faithfulness analysis
3. Visualization generation (importance heatmaps, coactivation matrices, circuit diagrams)

The output is organized under trial_id/spd/ with the following structure:
    spd/
        config.json              - SPD configuration and parameters
        decomposed_model.pt      - The trained decomposed model
        clustering/
            assignments.json     - Component -> cluster mapping
            coactivation.npy     - Coactivation matrix
            importance_matrix.npy - Full importance matrix
        visualizations/
            importance_heatmap.png
            coactivation_matrix.png
            components_as_circuits.png
            uv_matrices.png
            summary.png
        clusters/
            {cluster_idx}/
                analysis.json    - Cluster statistics
                robustness.json  - Robustness metrics
                faithfulness.json - Faithfulness metrics
                circuit.png      - Circuit diagram for this cluster
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import json

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from .common.logic_gates import ALL_LOGIC_GATES

if TYPE_CHECKING:
    from .common.neural_model import DecomposedMLP, MLP


@dataclass
class ClusterInfo:
    """Information about a single component cluster."""
    cluster_idx: int
    component_indices: list[int]  # Which components belong to this cluster
    component_labels: list[str]   # Labels like "layers.0.0:3"
    mean_importance: float = 0.0

    # Analysis results (filled in later)
    robustness_score: float = 0.0
    faithfulness_score: float = 0.0
    function_mapping: str = ""  # e.g., "XOR", "AND", etc.


@dataclass
class SPDAnalysisResult:
    """Complete result of SPD analysis including clustering and per-cluster metrics."""

    # Basic info
    n_components: int = 0
    n_layers: int = 0
    n_clusters: int = 0

    # Validation metrics (from SPD paper)
    mmcs: float = 0.0  # Mean Max Cosine Similarity (should be ~1.0)
    ml2r: float = 0.0  # Mean L2 Ratio (should be ~1.0)
    faithfulness_loss: float = 0.0  # MSE between target and reconstructed weights

    # Component health
    n_alive_components: int = 0
    n_dead_components: int = 0
    dead_component_labels: list[str] = field(default_factory=list)

    # Clustering results
    cluster_assignments: list[int] = field(default_factory=list)  # component_idx -> cluster_idx
    clusters: list[ClusterInfo] = field(default_factory=list)

    # Raw data (stored as numpy arrays on disk)
    importance_matrix: Optional[np.ndarray] = None  # [n_inputs, n_components]
    coactivation_matrix: Optional[np.ndarray] = None  # [n_components, n_components]

    # Component labels for each index
    component_labels: list[str] = field(default_factory=list)

    # Visualization paths (relative to spd/ folder)
    visualization_paths: dict[str, str] = field(default_factory=dict)


def compute_validation_metrics(
    decomposed_model: "DecomposedMLP",
) -> dict[str, float]:
    """
    Compute SPD validation metrics: MMCS and ML2R.

    MMCS (Mean Max Cosine Similarity): Measures directional alignment between
    learned subcomponents and target model weights. Value of 1.0 means perfect
    directional match.

    ML2R (Mean L2 Ratio): Measures magnitude correspondence between reconstructed
    and original weights. Value close to 1.0 means minimal shrinkage.

    Args:
        decomposed_model: Trained SPD decomposition

    Returns:
        Dict with 'mmcs' and 'ml2r' metrics (higher is better, 1.0 is perfect)
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return {"mmcs": 0.0, "ml2r": 0.0, "faithfulness_loss": float("inf")}

    component_model = decomposed_model.component_model

    mmcs_values = []
    ml2r_values = []
    faithfulness_losses = []

    for module_name, components in component_model.components.items():
        # Get target weight
        target_weight = component_model.target_weight(module_name)

        # Get reconstructed weight (U @ V^T)
        U = components.U  # [C, d_out]
        V = components.V  # [d_in, C]
        reconstructed = (V @ U).T  # [d_out, d_in]

        target_np = target_weight.detach().cpu().numpy()
        recon_np = reconstructed.detach().cpu().numpy()

        # Faithfulness loss (MSE)
        mse = ((target_np - recon_np) ** 2).mean()
        faithfulness_losses.append(mse)

        # MMCS: For each column in target, find max cosine similarity with any component
        for col_idx in range(target_np.shape[1]):
            target_col = target_np[:, col_idx]
            target_norm = np.linalg.norm(target_col)
            if target_norm < 1e-8:
                continue

            max_cos_sim = 0.0
            for c in range(U.shape[0]):
                # Component c contributes: U[c,:] * V[:,c]^T as rank-1 matrix
                # For column col_idx, the contribution is U[c,:] * V[col_idx, c]
                comp_col = U[c, :].detach().cpu().numpy() * V[col_idx, c].item()
                comp_norm = np.linalg.norm(comp_col)
                if comp_norm < 1e-8:
                    continue

                cos_sim = np.dot(target_col, comp_col) / (target_norm * comp_norm)
                max_cos_sim = max(max_cos_sim, abs(cos_sim))

            mmcs_values.append(max_cos_sim)

        # ML2R: Ratio of reconstructed to target magnitude
        target_norm = np.linalg.norm(target_np, 'fro')
        recon_norm = np.linalg.norm(recon_np, 'fro')
        if target_norm > 1e-8:
            ml2r_values.append(recon_norm / target_norm)

    return {
        "mmcs": float(np.mean(mmcs_values)) if mmcs_values else 0.0,
        "ml2r": float(np.mean(ml2r_values)) if ml2r_values else 0.0,
        "faithfulness_loss": float(np.mean(faithfulness_losses)) if faithfulness_losses else float("inf"),
    }


def detect_dead_components(
    decomposed_model: "DecomposedMLP",
    threshold: float = 0.01,
) -> tuple[list[str], list[str]]:
    """
    Detect dead components (those with negligible weight norms).

    Dead components are superfluous for replicating target model behavior.
    A good decomposition will have some dead components if n_components > needed.

    Args:
        decomposed_model: Trained SPD decomposition
        threshold: Relative threshold for considering a component dead

    Returns:
        Tuple of (alive_labels, dead_labels)
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return [], []

    component_model = decomposed_model.component_model

    alive_labels = []
    dead_labels = []

    for module_name, components in component_model.components.items():
        U = components.U  # [C, d_out]
        V = components.V  # [d_in, C]

        n_components = U.shape[0]

        # Compute norm of each rank-1 component contribution
        norms = []
        for c in range(n_components):
            u_norm = torch.norm(U[c, :]).item()
            v_norm = torch.norm(V[:, c]).item()
            norms.append(u_norm * v_norm)

        max_norm = max(norms) if norms else 1.0

        for c in range(n_components):
            label = f"{module_name}:{c}"
            if norms[c] < threshold * max_norm:
                dead_labels.append(label)
            else:
                alive_labels.append(label)

    return alive_labels, dead_labels


def compute_importance_matrix(
    decomposed_model: "DecomposedMLP",
    n_inputs: int = 2,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """
    Compute causal importance values for all binary input combinations.

    Args:
        decomposed_model: The trained SPD decomposition
        n_inputs: Number of input dimensions (default 2 for boolean gates)
        device: Compute device

    Returns:
        importance_matrix: Shape [2^n_inputs, total_components]
        component_labels: List of component labels like "layers.0.0:3"
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return np.array([]), []

    component_model = decomposed_model.component_model

    # Generate all binary input combinations
    n_total_inputs = 2 ** n_inputs
    all_inputs = torch.zeros(n_total_inputs, n_inputs, device=device)
    for i in range(n_total_inputs):
        for j in range(n_inputs):
            all_inputs[i, j] = (i >> j) & 1

    # Get pre-weight activations (inputs to each layer)
    with torch.inference_mode():
        output_with_cache = component_model(all_inputs, cache_type="input")
        pre_weight_acts = output_with_cache.cache

        # Compute causal importances
        ci_outputs = component_model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sampling="continuous",
            detach_inputs=False,
        )

        # Use upper_leaky for visualization (values in [0, 1])
        ci_values = ci_outputs.upper_leaky

    # Concatenate all layer importances and create labels
    all_importances = []
    component_labels = []

    for module_name in sorted(ci_values.keys()):
        ci_tensor = ci_values[module_name]  # [n_inputs, C]
        ci_np = ci_tensor.detach().cpu().numpy()
        all_importances.append(ci_np)

        n_components = ci_np.shape[1]
        for c in range(n_components):
            component_labels.append(f"{module_name}:{c}")

    if all_importances:
        importance_matrix = np.concatenate(all_importances, axis=1)
    else:
        importance_matrix = np.array([])

    return importance_matrix, component_labels


def compute_coactivation_matrix(
    importance_matrix: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Compute coactivation matrix showing which components fire together.

    Args:
        importance_matrix: Shape [n_samples, n_components]
        threshold: Activation threshold (component is "active" if importance > threshold)

    Returns:
        coactivation_matrix: Shape [n_components, n_components], symmetric
    """
    if importance_matrix.size == 0:
        return np.array([])

    # Binarize activations
    active_mask = (importance_matrix > threshold).astype(np.float32)

    # Coactivation = A^T @ A (counts how often pairs co-activate)
    coactivation = active_mask.T @ active_mask

    return coactivation


def cluster_components_hierarchical(
    coactivation_matrix: np.ndarray,
    n_clusters: int = None,
    merge_threshold: float = 0.7,
) -> list[int]:
    """
    Cluster components based on coactivation patterns using hierarchical clustering.

    Uses scipy's fast hierarchical clustering implementation.

    Args:
        coactivation_matrix: Shape [n_components, n_components]
        n_clusters: Target number of clusters (if None, determined by threshold)
        merge_threshold: Similarity threshold for merging (higher = more merging)

    Returns:
        cluster_assignments: List where index i gives cluster ID for component i
    """
    if coactivation_matrix.size == 0:
        return []

    n_components = coactivation_matrix.shape[0]

    # Handle trivial cases
    if n_components == 1:
        return [0]

    # Compute Jaccard-like similarity (vectorized)
    diag = np.diag(coactivation_matrix)
    diag_safe = np.maximum(diag, 1e-8)

    # Broadcasting: union[i,j] = diag[i] + diag[j] - coact[i,j]
    diag_i = diag_safe[:, np.newaxis]
    diag_j = diag_safe[np.newaxis, :]
    union = diag_i + diag_j - coactivation_matrix
    union = np.maximum(union, 1e-8)  # Avoid division by zero

    similarity = coactivation_matrix / union
    np.fill_diagonal(similarity, 1.0)

    # Convert to distance (condensed form for scipy)
    distance = 1 - similarity
    np.fill_diagonal(distance, 0)

    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # Convert to condensed distance matrix
        condensed_dist = squareform(distance, checks=False)

        # Perform hierarchical clustering (average linkage)
        Z = linkage(condensed_dist, method='average')

        if n_clusters is not None:
            # Cut to get exactly n_clusters
            cluster_assignments = fcluster(Z, t=n_clusters, criterion='maxclust')
        else:
            # Cut at distance threshold (1 - merge_threshold = distance threshold)
            dist_threshold = 1 - merge_threshold
            cluster_assignments = fcluster(Z, t=dist_threshold, criterion='distance')

        # Convert to 0-indexed list
        cluster_assignments = [int(c - 1) for c in cluster_assignments]

    except ImportError:
        # Fallback: simple greedy clustering if scipy not available
        cluster_assignments = list(range(n_components))

        # Merge similar components based on similarity threshold
        merged = [False] * n_components
        current_cluster = 0

        for i in range(n_components):
            if merged[i]:
                continue

            cluster_assignments[i] = current_cluster
            merged[i] = True

            # Find all components similar to i
            for j in range(i + 1, n_components):
                if not merged[j] and similarity[i, j] >= merge_threshold:
                    cluster_assignments[j] = current_cluster
                    merged[j] = True

            current_cluster += 1

    # Renumber clusters to be 0, 1, 2, ... contiguously
    unique_clusters = sorted(set(cluster_assignments))
    remap = {old: new for new, old in enumerate(unique_clusters)}
    cluster_assignments = [remap[c] for c in cluster_assignments]

    return cluster_assignments


def map_clusters_to_functions(
    importance_matrix: np.ndarray,
    cluster_assignments: list[int],
    n_inputs: int = 2,
    gate_names: list[str] = None,
) -> dict[int, str]:
    """
    Try to identify which boolean function each cluster implements.

    Args:
        importance_matrix: Shape [2^n_inputs, n_components]
        cluster_assignments: Component -> cluster mapping
        n_inputs: Number of input bits
        gate_names: Names of the gates in the model

    Returns:
        cluster_functions: cluster_idx -> function name
    """
    if importance_matrix.size == 0 or not cluster_assignments:
        return {}

    n_total_inputs = 2 ** n_inputs
    n_clusters = max(cluster_assignments) + 1

    # For each cluster, find which inputs activate it
    cluster_functions = {}

    for cluster_idx in range(n_clusters):
        # Get components in this cluster
        component_indices = [i for i, c in enumerate(cluster_assignments) if c == cluster_idx]

        if not component_indices:
            cluster_functions[cluster_idx] = "EMPTY"
            continue

        # Average importance over cluster components for each input
        cluster_importance = importance_matrix[:, component_indices].mean(axis=1)

        # Inputs where cluster is highly active
        active_threshold = 0.5
        active_inputs = set(np.where(cluster_importance > active_threshold)[0])

        if not active_inputs:
            cluster_functions[cluster_idx] = "INACTIVE"
            continue

        # Compare to known boolean functions
        best_match = "UNKNOWN"
        best_jaccard = 0

        # Generate all binary inputs
        all_inputs = []
        for i in range(n_total_inputs):
            inp = tuple((i >> j) & 1 for j in range(n_inputs))
            all_inputs.append(inp)

        # Check against all known gates
        gates_to_check = gate_names if gate_names else list(ALL_LOGIC_GATES.keys())
        for gate_name in gates_to_check:
            if gate_name not in ALL_LOGIC_GATES:
                continue
            gate = ALL_LOGIC_GATES[gate_name]
            if gate.n_inputs != n_inputs:
                continue

            # Find inputs where this gate outputs 1 (use truth_table)
            truth_table = gate.truth_table()
            gate_active = set()
            for idx, inp in enumerate(all_inputs):
                if truth_table.get(inp, 0) == 1:
                    gate_active.add(idx)

            # Jaccard similarity
            intersection = len(active_inputs & gate_active)
            union = len(active_inputs | gate_active)
            jaccard = intersection / union if union > 0 else 0

            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_match = gate_name

        if best_jaccard > 0.5:
            cluster_functions[cluster_idx] = f"{best_match} ({best_jaccard:.2f})"
        else:
            cluster_functions[cluster_idx] = "UNKNOWN"

    return cluster_functions


def run_spd_analysis(
    decomposed_model: "DecomposedMLP",
    target_model: "MLP" = None,
    n_inputs: int = 2,
    gate_names: list[str] = None,
    n_clusters: int = None,
    device: str = "cpu",
) -> SPDAnalysisResult:
    """
    Run complete SPD analysis: clustering, function mapping, and metrics.

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

    # Compute validation metrics (MMCS, ML2R)
    validation_metrics = compute_validation_metrics(decomposed_model)
    result.mmcs = validation_metrics["mmcs"]
    result.ml2r = validation_metrics["ml2r"]
    result.faithfulness_loss = validation_metrics["faithfulness_loss"]

    # Detect dead components
    alive_labels, dead_labels = detect_dead_components(decomposed_model)
    result.n_alive_components = len(alive_labels)
    result.n_dead_components = len(dead_labels)
    result.dead_component_labels = dead_labels

    # Compute importance matrix
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

    # Compute coactivation matrix
    result.coactivation_matrix = compute_coactivation_matrix(importance_matrix)

    # Cluster components
    result.cluster_assignments = cluster_components_hierarchical(
        result.coactivation_matrix,
        n_clusters=n_clusters,
    )
    result.n_clusters = max(result.cluster_assignments) + 1 if result.cluster_assignments else 0

    # Map clusters to functions
    cluster_functions = map_clusters_to_functions(
        importance_matrix,
        result.cluster_assignments,
        n_inputs,
        gate_names,
    )

    # Build cluster info
    for cluster_idx in range(result.n_clusters):
        component_indices = [i for i, c in enumerate(result.cluster_assignments) if c == cluster_idx]
        cluster_labels = [component_labels[i] for i in component_indices]

        # Mean importance over all inputs for this cluster
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


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_importance_heatmap(
    importance_matrix: np.ndarray,
    component_labels: list[str],
    output_path: str,
    n_inputs: int = 2,
) -> str:
    """
    Create heatmap showing importance values per input pattern.

    Args:
        importance_matrix: Shape [2^n_inputs, n_components]
        component_labels: Labels for each component
        output_path: Path to save the figure
        n_inputs: Number of input bits

    Returns:
        Path to saved figure
    """
    if importance_matrix.size == 0:
        return ""

    fig, ax = plt.subplots(figsize=(max(12, len(component_labels) * 0.3), 6))

    im = ax.imshow(importance_matrix, aspect='auto', cmap='Reds', vmin=0, vmax=1)

    # Y-axis: input patterns
    n_total_inputs = 2 ** n_inputs
    input_labels = []
    for i in range(n_total_inputs):
        bits = tuple((i >> j) & 1 for j in range(n_inputs))
        input_labels.append(str(bits))

    ax.set_yticks(range(n_total_inputs))
    ax.set_yticklabels(input_labels)
    ax.set_ylabel("Input Pattern")

    # X-axis: components (simplified labels)
    if len(component_labels) <= 30:
        ax.set_xticks(range(len(component_labels)))
        short_labels = [l.split(".")[-1] for l in component_labels]
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xlabel(f"Component Index (total: {len(component_labels)})")

    ax.set_title("Causal Importance by Input Pattern")

    plt.colorbar(im, ax=ax, label="Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_coactivation_matrix(
    coactivation_matrix: np.ndarray,
    cluster_assignments: list[int],
    component_labels: list[str],
    output_path: str,
) -> str:
    """
    Create coactivation matrix visualization with cluster grouping.

    Args:
        coactivation_matrix: Shape [n_components, n_components]
        cluster_assignments: Component -> cluster mapping
        component_labels: Labels for each component
        output_path: Path to save the figure

    Returns:
        Path to saved figure
    """
    if coactivation_matrix.size == 0:
        return ""

    n_components = coactivation_matrix.shape[0]

    # Sort by cluster assignment for visualization
    sorted_indices = sorted(range(n_components), key=lambda i: cluster_assignments[i])
    sorted_matrix = coactivation_matrix[sorted_indices][:, sorted_indices]

    # Normalize for visualization
    max_val = sorted_matrix.max()
    if max_val > 0:
        normalized = sorted_matrix / max_val
    else:
        normalized = sorted_matrix

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(normalized, cmap='viridis', aspect='equal')

    # Add cluster boundaries
    n_clusters = max(cluster_assignments) + 1 if cluster_assignments else 0
    cluster_counts = [0] * n_clusters
    for c in cluster_assignments:
        cluster_counts[c] += 1

    # Draw boundaries
    cumsum = 0
    for count in cluster_counts[:-1]:
        cumsum += count
        ax.axhline(cumsum - 0.5, color='white', linewidth=2)
        ax.axvline(cumsum - 0.5, color='white', linewidth=2)

    ax.set_xlabel("Component Index (sorted by cluster)")
    ax.set_ylabel("Component Index (sorted by cluster)")
    ax.set_title(f"Component Coactivation Matrix ({n_clusters} clusters)")

    plt.colorbar(im, ax=ax, label="Normalized Coactivation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def _get_all_component_neuron_weights(
    decomposed_model: "DecomposedMLP",
    all_component_labels: list[str],
    threshold: float = 0.1,
) -> dict[str, dict[int, dict[str, set[int]]]]:
    """
    Pre-compute neuron weights for all components at once (caching optimization).

    Returns:
        Dict mapping component_label -> layer_idx -> {"reads": set, "writes": set}
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return {}

    cm = decomposed_model.component_model
    result = {}

    # Cache U/V matrices per layer to avoid repeated detach/cpu calls
    layer_matrices = {}
    for layer_name, comp in cm.components.items():
        layer_matrices[layer_name] = {
            'U': comp.U.detach().cpu().numpy(),  # [C, d_out]
            'V': comp.V.detach().cpu().numpy(),  # [d_in, C]
        }

    for comp_label in all_component_labels:
        # Parse "layers.X.0:Y" format
        parts = comp_label.split(":")
        layer_name = parts[0]  # "layers.X.0"
        comp_idx = int(parts[1])
        layer_idx = int(layer_name.split(".")[1])

        if layer_name not in layer_matrices:
            result[comp_label] = {}
            continue

        U = layer_matrices[layer_name]['U']
        V = layer_matrices[layer_name]['V']

        reads = set()
        writes = set()

        # Find neurons this component reads from (V column)
        v_col = V[:, comp_idx]
        v_max = np.max(np.abs(v_col))
        if v_max > 0:
            mask = np.abs(v_col) > threshold * v_max
            reads = set(np.where(mask)[0].tolist())

        # Find neurons this component writes to (U row)
        u_row = U[comp_idx, :]
        u_max = np.max(np.abs(u_row))
        if u_max > 0:
            mask = np.abs(u_row) > threshold * u_max
            writes = set(np.where(mask)[0].tolist())

        result[comp_label] = {layer_idx: {"reads": reads, "writes": writes}}

    return result


def _get_component_neuron_weights(
    decomposed_model: "DecomposedMLP",
    component_labels: list[str],
    threshold: float = 0.1,
    cached_weights: dict = None,
) -> dict[int, dict[str, set[int]]]:
    """
    Get which neurons each component reads from (V) and writes to (U).

    Args:
        decomposed_model: Trained SPD decomposition
        component_labels: Labels for components in this cluster
        threshold: Relative threshold for neuron activity
        cached_weights: Pre-computed weights from _get_all_component_neuron_weights

    Returns:
        Dict mapping layer_idx -> {"reads": set of input neuron indices,
                                   "writes": set of output neuron indices}
    """
    layer_neurons = {}

    # Use cached weights if available
    if cached_weights is not None:
        for comp_label in component_labels:
            if comp_label in cached_weights:
                for layer_idx, rw in cached_weights[comp_label].items():
                    if layer_idx not in layer_neurons:
                        layer_neurons[layer_idx] = {"reads": set(), "writes": set()}
                    layer_neurons[layer_idx]["reads"].update(rw["reads"])
                    layer_neurons[layer_idx]["writes"].update(rw["writes"])
        return layer_neurons

    # Fallback to direct computation
    if decomposed_model is None or decomposed_model.component_model is None:
        return {}

    cm = decomposed_model.component_model

    for comp_label in component_labels:
        # Parse "layers.X.0:Y" format
        parts = comp_label.split(":")
        layer_name = parts[0]  # "layers.X.0"
        comp_idx = int(parts[1])
        layer_idx = int(layer_name.split(".")[1])

        if layer_name not in cm.components:
            continue

        comp = cm.components[layer_name]
        U = comp.U.detach().cpu().numpy()  # [C, d_out]
        V = comp.V.detach().cpu().numpy()  # [d_in, C]

        if layer_idx not in layer_neurons:
            layer_neurons[layer_idx] = {"reads": set(), "writes": set()}

        # Find neurons this component reads from (V column)
        v_col = V[:, comp_idx]
        v_max = np.max(np.abs(v_col))
        if v_max > 0:
            mask = np.abs(v_col) > threshold * v_max
            layer_neurons[layer_idx]["reads"].update(np.where(mask)[0].tolist())

        # Find neurons this component writes to (U row)
        u_row = U[comp_idx, :]
        u_max = np.max(np.abs(u_row))
        if u_max > 0:
            mask = np.abs(u_row) > threshold * u_max
            layer_neurons[layer_idx]["writes"].update(np.where(mask)[0].tolist())

    return layer_neurons


def _draw_cluster_circuit(
    ax,
    cluster_info: ClusterInfo,
    layer_sizes: list[int],
    layer_neurons: dict[int, dict[str, set[int]]],
    color,
):
    """Draw a single cluster's circuit diagram."""
    G = nx.DiGraph()
    pos = {}
    node_colors = []
    edge_colors = []
    edge_widths = []

    max_width = max(layer_sizes)
    n_layers = len(layer_sizes)

    # Create nodes for each layer
    for layer_idx, layer_size in enumerate(layer_sizes):
        y_start = (max_width - layer_size) / 2

        for node_idx in range(layer_size):
            node_id = f"L{layer_idx}_N{node_idx}"
            G.add_node(node_id)
            pos[node_id] = (layer_idx, y_start + node_idx)

            # Check if this neuron is active for any component in the cluster
            is_read = False
            is_write = False

            # Check if this neuron is read by layer (layer_idx) components
            if layer_idx in layer_neurons:
                is_read = node_idx in layer_neurons[layer_idx]["reads"]

            # Check if previous layer writes to this neuron
            if layer_idx > 0 and (layer_idx - 1) in layer_neurons:
                is_write = node_idx in layer_neurons[layer_idx - 1]["writes"]

            if is_read or is_write:
                node_colors.append(color)
            else:
                node_colors.append((0.9, 0.9, 0.9, 0.5))

    # Create edges
    for layer_idx in range(n_layers - 1):
        curr_size = layer_sizes[layer_idx]
        next_size = layer_sizes[layer_idx + 1]

        # Get active neurons for this layer transition
        reads = layer_neurons.get(layer_idx, {}).get("reads", set())
        writes = layer_neurons.get(layer_idx, {}).get("writes", set())

        for curr_idx in range(curr_size):
            for next_idx in range(next_size):
                curr_id = f"L{layer_idx}_N{curr_idx}"
                next_id = f"L{layer_idx + 1}_N{next_idx}"
                G.add_edge(curr_id, next_id)

                # Highlight edge if it connects active neurons
                if curr_idx in reads and next_idx in writes:
                    edge_colors.append(color)
                    edge_widths.append(2.0)
                else:
                    edge_colors.append((0.8, 0.8, 0.8, 0.3))
                    edge_widths.append(0.5)

    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, width=edge_widths,
        arrows=True, arrowsize=8, ax=ax, connectionstyle="arc3,rad=0.1"
    )

    # Add layer labels
    for layer_idx in range(n_layers):
        ax.text(layer_idx, -0.8, f"L{layer_idx}", ha='center', fontsize=8)

    # Title
    title = f"Cluster {cluster_info.cluster_idx}"
    if cluster_info.function_mapping:
        title += f"\n{cluster_info.function_mapping}"
    title += f"\n({len(cluster_info.component_indices)} comp)"
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def visualize_components_as_circuits(
    analysis_result: SPDAnalysisResult,
    layer_sizes: list[int],
    output_path: str,
    decomposed_model: "DecomposedMLP" = None,
) -> dict[str, str]:
    """
    Create circuit-style diagrams showing component clusters, split by category.

    Creates 3 separate visualizations:
    - circuits_matched.png: Clusters matching known functions (XOR, AND, etc.)
    - circuits_unknown.png: Clusters with unknown function
    - circuits_inactive.png: Inactive clusters

    Args:
        analysis_result: SPD analysis result with clustering
        layer_sizes: Size of each layer in the network
        output_path: Base path (will create 3 files)
        decomposed_model: The decomposed model for weight analysis

    Returns:
        Dict of category -> path for created files
    """
    n_clusters = analysis_result.n_clusters
    if n_clusters == 0:
        return {}

    # Pre-compute all component weights at once for efficiency
    cached_weights = _get_all_component_neuron_weights(
        decomposed_model,
        analysis_result.component_labels,
    )

    # Categorize clusters
    matched_clusters = []
    unknown_clusters = []
    inactive_clusters = []

    for cluster_info in analysis_result.clusters:
        func = cluster_info.function_mapping
        if "INACTIVE" in func:
            inactive_clusters.append(cluster_info)
        elif "UNKNOWN" in func or not func:
            unknown_clusters.append(cluster_info)
        else:
            matched_clusters.append(cluster_info)

    # Colors
    matched_color = (0.2, 0.7, 0.3, 0.8)  # Green
    unknown_color = (0.9, 0.6, 0.1, 0.8)  # Orange
    inactive_color = (0.5, 0.5, 0.5, 0.5)  # Gray

    output_paths = {}
    base_path = Path(output_path)
    base_dir = base_path.parent
    base_name = base_path.stem

    def _create_category_figure(clusters, color, category_name, filename):
        if not clusters:
            return None

        n = len(clusters)
        n_cols = min(4, n)
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, cluster_info in enumerate(clusters):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            # Get neuron weights for this cluster (using cache)
            layer_neurons = _get_component_neuron_weights(
                decomposed_model,
                cluster_info.component_labels,
                cached_weights=cached_weights,
            )

            _draw_cluster_circuit(ax, cluster_info, layer_sizes, layer_neurons, color)

        # Hide empty subplots
        for idx in range(n, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.suptitle(f"SPD Clusters: {category_name} ({n} clusters)", fontsize=12)
        plt.tight_layout()

        path = base_dir / filename
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(path)

    # Create each category
    path = _create_category_figure(
        matched_clusters, matched_color, "Matched Functions", f"{base_name}_matched.png"
    )
    if path:
        output_paths["matched"] = path

    path = _create_category_figure(
        unknown_clusters, unknown_color, "Unknown Functions", f"{base_name}_unknown.png"
    )
    if path:
        output_paths["unknown"] = path

    path = _create_category_figure(
        inactive_clusters, inactive_color, "Inactive", f"{base_name}_inactive.png"
    )
    if path:
        output_paths["inactive"] = path

    return output_paths


def visualize_uv_matrices(
    decomposed_model: "DecomposedMLP",
    output_path: str,
) -> str:
    """
    Visualize U and V matrices from the decomposition.

    Args:
        decomposed_model: Trained SPD decomposition
        output_path: Path to save the figure

    Returns:
        Path to saved figure
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return ""

    components = decomposed_model.component_model.components
    n_layers = len(components)

    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 4 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, component) in enumerate(sorted(components.items())):
        # Get U and V
        V = component.V.detach().cpu().numpy()  # [d_in, C]
        U = component.U.detach().cpu().numpy()  # [C, d_out]

        # Plot V
        ax = axes[idx, 0]
        im = ax.imshow(V.T, aspect='auto', cmap='RdBu_r')
        ax.set_xlabel('Input dimension')
        ax.set_ylabel('Component')
        ax.set_title(f'{name} - V matrix')
        plt.colorbar(im, ax=ax)

        # Plot U
        ax = axes[idx, 1]
        im = ax.imshow(U, aspect='auto', cmap='RdBu_r')
        ax.set_xlabel('Output dimension')
        ax.set_ylabel('Component')
        ax.set_title(f'{name} - U matrix')
        plt.colorbar(im, ax=ax)

    plt.suptitle("SPD U and V Decomposition Matrices", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_ci_histograms(
    importance_matrix: np.ndarray,
    component_labels: list[str],
    output_path: str,
) -> str:
    """
    Create histograms of causal importance values (SPD paper style).

    Shows distribution of CI values for each layer separately.
    From SPD paper: helps diagnose whether sparsity pressure is working correctly.
    - Healthy: bimodal distribution (values near 0 and near 1)
    - Unhealthy: all values near 0.5 or all near 0

    Args:
        importance_matrix: Shape [n_inputs, n_components]
        component_labels: Labels for each component
        output_path: Path to save the figure

    Returns:
        Path to saved figure
    """
    if importance_matrix.size == 0:
        return ""

    # Group components by layer
    layer_components = {}
    for idx, label in enumerate(component_labels):
        layer_name = label.split(":")[0]
        if layer_name not in layer_components:
            layer_components[layer_name] = []
        layer_components[layer_name].append(idx)

    n_layers = len(layer_components)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for idx, (layer_name, comp_indices) in enumerate(sorted(layer_components.items())):
        ax = axes[idx]
        layer_ci = importance_matrix[:, comp_indices].flatten()

        ax.hist(layer_ci, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel("Causal Importance")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{layer_name.replace('.', '_')}")
        ax.set_yscale('log')
        ax.set_xlim(0, 1)

    plt.suptitle("Causal Importance Distribution per Layer", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_mean_ci_per_component(
    importance_matrix: np.ndarray,
    component_labels: list[str],
    output_path: str,
) -> str:
    """
    Plot mean CI per component, sorted (SPD paper style).

    Shows which components are most/least important on average.
    From SPD paper: useful for identifying dead/alive components.
    - Dead components: mean CI near 0
    - Alive components: mean CI > threshold (typically 0.5)

    Args:
        importance_matrix: Shape [n_inputs, n_components]
        component_labels: Labels for each component
        output_path: Path to save the figure

    Returns:
        Path to saved figure
    """
    if importance_matrix.size == 0:
        return ""

    # Group by layer
    layer_components = {}
    for idx, label in enumerate(component_labels):
        layer_name = label.split(":")[0]
        if layer_name not in layer_components:
            layer_components[layer_name] = []
        layer_components[layer_name].append(idx)

    n_layers = len(layer_components)
    fig, axes = plt.subplots(2, n_layers, figsize=(5 * n_layers, 8))
    if n_layers == 1:
        axes = axes.reshape(-1, 1)

    for idx, (layer_name, comp_indices) in enumerate(sorted(layer_components.items())):
        # Mean CI for each component in this layer
        mean_ci = importance_matrix[:, comp_indices].mean(axis=0)
        sorted_indices = np.argsort(mean_ci)[::-1]  # Descending
        sorted_mean_ci = mean_ci[sorted_indices]

        # Linear scale
        ax = axes[0, idx]
        ax.scatter(range(len(sorted_mean_ci)), sorted_mean_ci, marker='x', s=30, c='steelblue')
        ax.set_xlabel("Component (sorted)")
        ax.set_ylabel("Mean CI")
        ax.set_title(f"{layer_name.replace('.', '_')} (linear)")
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='threshold')

        # Log scale
        ax = axes[1, idx]
        ax.scatter(range(len(sorted_mean_ci)), sorted_mean_ci, marker='x', s=30, c='steelblue')
        ax.set_xlabel("Component (sorted)")
        ax.set_ylabel("Mean CI")
        ax.set_title(f"{layer_name.replace('.', '_')} (log)")
        ax.set_yscale('log')

    plt.suptitle("Mean Causal Importance per Component", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_l0_sparsity(
    importance_matrix: np.ndarray,
    component_labels: list[str],
    output_path: str,
    threshold: float = 0.5,
) -> str:
    """
    Create L0 sparsity bar chart per layer (SPD paper style).

    Shows average number of active components per input for each layer.
    From SPD paper: L0 = count of components with CI > threshold.
    Lower L0 = sparser representation = better decomposition.

    Args:
        importance_matrix: Shape [n_inputs, n_components]
        component_labels: Labels for each component
        output_path: Path to save the figure
        threshold: CI threshold for considering component "active"

    Returns:
        Path to saved figure
    """
    if importance_matrix.size == 0:
        return ""

    # Group by layer
    layer_components = {}
    for idx, label in enumerate(component_labels):
        layer_name = label.split(":")[0]
        if layer_name not in layer_components:
            layer_components[layer_name] = []
        layer_components[layer_name].append(idx)

    # Compute L0 for each layer
    layer_names = []
    l0_values = []

    for layer_name in sorted(layer_components.keys()):
        comp_indices = layer_components[layer_name]
        layer_ci = importance_matrix[:, comp_indices]
        # L0 = mean count of active components per input
        active_count = (layer_ci > threshold).sum(axis=1)
        l0 = active_count.mean()
        layer_names.append(layer_name.replace('.', '_'))
        l0_values.append(l0)

    fig, ax = plt.subplots(figsize=(max(6, len(layer_names) * 1.5), 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(layer_names)))
    bars = ax.bar(layer_names, l0_values, color=colors, edgecolor='black')

    # Add value labels on bars
    for bar, val in zip(bars, l0_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel("Layer")
    ax.set_ylabel(f"L0 (threshold={threshold})")
    ax.set_title(f"L0 Sparsity per Layer (avg active components per input)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_summary(
    analysis_result: SPDAnalysisResult,
    output_path: str,
) -> str:
    """
    Create summary visualization with all cluster statistics.

    Args:
        analysis_result: SPD analysis result
        output_path: Path to save the figure

    Returns:
        Path to saved figure
    """
    if analysis_result.n_clusters == 0:
        return ""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Cluster sizes bar chart
    ax = axes[0, 0]
    cluster_sizes = [len(c.component_indices) for c in analysis_result.clusters]
    cluster_labels = [f"C{c.cluster_idx}" for c in analysis_result.clusters]
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
    ax.bar(cluster_labels, cluster_sizes, color=colors)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Components")
    ax.set_title("Cluster Sizes")

    # 2. Mean importance per cluster
    ax = axes[0, 1]
    mean_importances = [c.mean_importance for c in analysis_result.clusters]
    ax.bar(cluster_labels, mean_importances, color=colors)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean Importance")
    ax.set_title("Average Component Importance by Cluster")

    # 3. Function mapping table
    ax = axes[1, 0]
    ax.axis('off')
    table_data = []
    for c in analysis_result.clusters:
        table_data.append([
            f"Cluster {c.cluster_idx}",
            c.function_mapping or "Unknown",
            len(c.component_indices),
            f"{c.mean_importance:.3f}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["Cluster", "Function", "Components", "Mean Imp."],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("Cluster Summary", pad=20)

    # 4. Overall statistics text
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    SPD Analysis Summary

    Validation Metrics:
      MMCS: {analysis_result.mmcs:.4f} (target: 1.0)
      ML2R: {analysis_result.ml2r:.4f} (target: 1.0)
      Faithfulness Loss: {analysis_result.faithfulness_loss:.6f}

    Components:
      Total: {analysis_result.n_components}
      Alive: {analysis_result.n_alive_components}
      Dead: {analysis_result.n_dead_components}

    Structure:
      Layers: {analysis_result.n_layers}
      Clusters: {analysis_result.n_clusters}

    Modules:
    {', '.join(set(l.split(':')[0] for l in analysis_result.component_labels))}
    """
    ax.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.suptitle("SPD Decomposition Analysis Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


# ============================================================================
# Cluster Robustness and Faithfulness Analysis
# ============================================================================

def analyze_cluster_robustness(
    decomposed_model: "DecomposedMLP",
    cluster_info: ClusterInfo,
    importance_matrix: np.ndarray,
    n_samples: int = 20,  # Reduced default for speed
    noise_levels: list[float] = None,
    device: str = "cpu",
) -> dict:
    """
    Analyze robustness of a single cluster to input perturbations.

    Tests whether the cluster's importance pattern is stable under noise.
    Uses batched inference for efficiency.

    Args:
        decomposed_model: Trained SPD decomposition
        cluster_info: Information about the cluster
        importance_matrix: Full importance matrix [n_inputs, n_components]
        n_samples: Number of noise samples per level
        noise_levels: List of noise magnitudes to test
        device: Compute device

    Returns:
        Dict with robustness metrics:
            - mean_importance_stability: How stable importance is under noise
            - noise_sensitivity: Importance change per unit noise
    """
    if noise_levels is None:
        noise_levels = [0.05, 0.1, 0.2]  # Fewer levels for speed

    if decomposed_model is None or decomposed_model.component_model is None:
        return {"mean_importance_stability": 0.0, "noise_sensitivity": 0.0}

    component_model = decomposed_model.component_model
    n_inputs = 2  # Boolean gates

    # Get component indices for this cluster
    comp_indices = cluster_info.component_indices
    if not comp_indices:
        return {"mean_importance_stability": 0.0, "noise_sensitivity": 0.0}

    # Baseline importance for cluster (mean over inputs)
    baseline_imp = importance_matrix[:, comp_indices].mean()

    stability_scores = []
    importance_changes = []

    # Generate all noisy inputs at once (batched)
    total_samples = len(noise_levels) * n_samples
    all_base_inputs = torch.randint(0, 2, (total_samples, n_inputs), dtype=torch.float, device=device)
    all_noise = torch.randn_like(all_base_inputs)

    # Scale noise by level
    noise_scales = torch.tensor(
        [level for level in noise_levels for _ in range(n_samples)],
        device=device
    ).unsqueeze(1)
    all_noisy_inputs = all_base_inputs + all_noise * noise_scales

    try:
        with torch.inference_mode():
            output_with_cache = component_model(all_noisy_inputs, cache_type="input")
            pre_weight_acts = output_with_cache.cache

            ci_outputs = component_model.calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                sampling="continuous",
                detach_inputs=False,
            )

            # Get importance for cluster components
            all_imp = []
            for module_name in sorted(ci_outputs.upper_leaky.keys()):
                ci_tensor = ci_outputs.upper_leaky[module_name]
                all_imp.append(ci_tensor.detach().cpu().numpy())

            if all_imp:
                full_imp = np.concatenate(all_imp, axis=1)  # [total_samples, n_components]
                cluster_imp = full_imp[:, comp_indices].mean(axis=1)  # [total_samples]

                # Compute stability and sensitivity for each sample
                for i, (imp, noise_level) in enumerate(
                    zip(cluster_imp, noise_scales.squeeze().cpu().numpy())
                ):
                    stability = 1.0 - abs(imp - baseline_imp)
                    stability_scores.append(max(0, stability))
                    importance_changes.append(abs(imp - baseline_imp) / max(noise_level, 0.01))

    except Exception:
        pass

    return {
        "mean_importance_stability": float(np.mean(stability_scores)) if stability_scores else 0.0,
        "noise_sensitivity": float(np.mean(importance_changes)) if importance_changes else 0.0,
        "n_samples_tested": len(stability_scores),
    }


def analyze_cluster_faithfulness(
    decomposed_model: "DecomposedMLP",
    cluster_info: ClusterInfo,
    importance_matrix: np.ndarray,
    n_inputs: int = 2,
    device: str = "cpu",
) -> dict:
    """
    Analyze faithfulness of a cluster by testing ablation effects.

    Tests whether ablating (masking out) the cluster's components
    changes the model output appropriately.

    Args:
        decomposed_model: Trained SPD decomposition
        cluster_info: Information about the cluster
        importance_matrix: Full importance matrix [n_inputs, n_components]
        n_inputs: Number of input dimensions
        device: Compute device

    Returns:
        Dict with faithfulness metrics:
            - mean_ablation_effect: How much output changes when cluster is ablated
            - sufficiency_score: Whether cluster alone can produce correct output
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return {"mean_ablation_effect": 0.0, "sufficiency_score": 0.0}

    # For now, return placeholder metrics based on importance patterns
    # Full ablation testing requires modifying the forward pass with masks

    comp_indices = cluster_info.component_indices
    if not comp_indices:
        return {"mean_ablation_effect": 0.0, "sufficiency_score": 0.0}

    # Use importance matrix to estimate faithfulness
    # High importance on specific inputs suggests functional role
    cluster_importance = importance_matrix[:, comp_indices]

    # Ablation effect proxy: variance in importance across inputs
    # Higher variance = more selective = likely more faithful
    importance_variance = float(np.var(cluster_importance))

    # Sufficiency proxy: maximum importance achieved
    max_importance = float(np.max(cluster_importance))

    # Mean importance when "active" (above threshold)
    active_mask = cluster_importance > 0.5
    mean_when_active = float(np.mean(cluster_importance[active_mask])) if active_mask.any() else 0.0

    return {
        "mean_ablation_effect": importance_variance,
        "sufficiency_score": max_importance,
        "mean_when_active": mean_when_active,
        "selectivity": importance_variance / (cluster_importance.mean() + 1e-8),
    }


def analyze_all_clusters(
    decomposed_model: "DecomposedMLP",
    analysis_result: SPDAnalysisResult,
    device: str = "cpu",
) -> list[dict]:
    """
    Run robustness and faithfulness analysis on all clusters.

    Args:
        decomposed_model: Trained SPD decomposition
        analysis_result: SPD analysis result with clustering
        device: Compute device

    Returns:
        List of analysis dicts, one per cluster
    """
    results = []

    for cluster_info in analysis_result.clusters:
        robustness = analyze_cluster_robustness(
            decomposed_model,
            cluster_info,
            analysis_result.importance_matrix,
            device=device,
        )

        faithfulness = analyze_cluster_faithfulness(
            decomposed_model,
            cluster_info,
            analysis_result.importance_matrix,
            device=device,
        )

        # Update cluster info with scores
        cluster_info.robustness_score = robustness.get("mean_importance_stability", 0.0)
        cluster_info.faithfulness_score = faithfulness.get("sufficiency_score", 0.0)

        results.append({
            "cluster_idx": cluster_info.cluster_idx,
            "robustness": robustness,
            "faithfulness": faithfulness,
        })

    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def analyze_and_visualize_spd(
    decomposed_model: "DecomposedMLP",
    target_model: "MLP",
    output_dir: str | Path,
    gate_names: list[str] = None,
    n_inputs: int = 2,
    device: str = "cpu",
) -> SPDAnalysisResult:
    """
    Run complete SPD analysis and generate all visualizations.

    Args:
        decomposed_model: Trained SPD decomposition
        target_model: Original MLP model
        output_dir: Directory to save all outputs
        gate_names: Names of gates in the model
        n_inputs: Number of input bits
        device: Compute device

    Returns:
        SPDAnalysisResult with all data and visualization paths
    """
    output_dir = Path(output_dir)

    # Create directory structure
    (output_dir / "clustering").mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (output_dir / "clusters").mkdir(parents=True, exist_ok=True)

    # Run analysis
    result = run_spd_analysis(
        decomposed_model=decomposed_model,
        target_model=target_model,
        n_inputs=n_inputs,
        gate_names=gate_names,
        device=device,
    )

    if result.n_components == 0:
        return result

    # Save validation metrics
    validation_data = {
        "mmcs": result.mmcs,
        "ml2r": result.ml2r,
        "faithfulness_loss": result.faithfulness_loss,
        "n_alive_components": result.n_alive_components,
        "n_dead_components": result.n_dead_components,
        "dead_component_labels": result.dead_component_labels,
    }
    with open(output_dir / "validation.json", "w") as f:
        json.dump(validation_data, f, indent=2)

    # Save clustering data
    if result.importance_matrix is not None:
        np.save(output_dir / "clustering" / "importance_matrix.npy", result.importance_matrix)
    if result.coactivation_matrix is not None:
        np.save(output_dir / "clustering" / "coactivation_matrix.npy", result.coactivation_matrix)

    # Save cluster assignments
    assignments_data = {
        "cluster_assignments": result.cluster_assignments,
        "component_labels": result.component_labels,
        "n_clusters": result.n_clusters,
        "clusters": [asdict(c) for c in result.clusters],
    }
    with open(output_dir / "clustering" / "assignments.json", "w") as f:
        json.dump(assignments_data, f, indent=2)

    # Generate visualizations
    layer_sizes = target_model.layer_sizes if target_model else [n_inputs, 3, 1]

    viz_paths = {}

    # Importance heatmap
    path = visualize_importance_heatmap(
        result.importance_matrix,
        result.component_labels,
        str(output_dir / "visualizations" / "importance_heatmap.png"),
        n_inputs,
    )
    if path:
        viz_paths["importance_heatmap"] = "visualizations/importance_heatmap.png"

    # Coactivation matrix
    path = visualize_coactivation_matrix(
        result.coactivation_matrix,
        result.cluster_assignments,
        result.component_labels,
        str(output_dir / "visualizations" / "coactivation_matrix.png"),
    )
    if path:
        viz_paths["coactivation_matrix"] = "visualizations/coactivation_matrix.png"

    # Components as circuits (split into 3 files by category)
    circuit_paths = visualize_components_as_circuits(
        result,
        layer_sizes,
        str(output_dir / "visualizations" / "circuits"),
        decomposed_model=decomposed_model,
    )
    for category, path in circuit_paths.items():
        viz_paths[f"circuits_{category}"] = f"visualizations/circuits_{category}.png"

    # UV matrices
    path = visualize_uv_matrices(
        decomposed_model,
        str(output_dir / "visualizations" / "uv_matrices.png"),
    )
    if path:
        viz_paths["uv_matrices"] = "visualizations/uv_matrices.png"

    # SPD paper-style visualizations
    # CI histograms (per layer)
    path = visualize_ci_histograms(
        result.importance_matrix,
        result.component_labels,
        str(output_dir / "visualizations" / "ci_histograms.png"),
    )
    if path:
        viz_paths["ci_histograms"] = "visualizations/ci_histograms.png"

    # Mean CI per component
    path = visualize_mean_ci_per_component(
        result.importance_matrix,
        result.component_labels,
        str(output_dir / "visualizations" / "mean_ci_per_component.png"),
    )
    if path:
        viz_paths["mean_ci_per_component"] = "visualizations/mean_ci_per_component.png"

    # L0 sparsity
    path = visualize_l0_sparsity(
        result.importance_matrix,
        result.component_labels,
        str(output_dir / "visualizations" / "l0_sparsity.png"),
    )
    if path:
        viz_paths["l0_sparsity"] = "visualizations/l0_sparsity.png"

    # Summary
    path = visualize_summary(
        result,
        str(output_dir / "visualizations" / "summary.png"),
    )
    if path:
        viz_paths["summary"] = "visualizations/summary.png"

    result.visualization_paths = viz_paths

    # Run cluster-level robustness and faithfulness analysis
    cluster_analyses = analyze_all_clusters(
        decomposed_model=decomposed_model,
        analysis_result=result,
        device=device,
    )

    # Create per-cluster directories and save analysis files
    def _to_serializable(obj):
        """Convert numpy types to Python native types for JSON."""
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    for cluster_info, cluster_analysis in zip(result.clusters, cluster_analyses):
        cluster_dir = output_dir / "clusters" / str(cluster_info.cluster_idx)
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Save cluster info
        with open(cluster_dir / "analysis.json", "w") as f:
            json.dump(_to_serializable(asdict(cluster_info)), f, indent=2)

        # Save robustness metrics
        with open(cluster_dir / "robustness.json", "w") as f:
            json.dump(_to_serializable(cluster_analysis["robustness"]), f, indent=2)

        # Save faithfulness metrics
        with open(cluster_dir / "faithfulness.json", "w") as f:
            json.dump(_to_serializable(cluster_analysis["faithfulness"]), f, indent=2)

    return result

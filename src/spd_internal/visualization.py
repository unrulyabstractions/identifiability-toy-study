"""
SPD Visualization Module - All visualization functions for SPD analysis.

This module provides functions for visualizing SPD decomposition results
including importance heatmaps, coactivation matrices, circuit diagrams,
and summary statistics.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .schemas import ClusterInfo, SPDAnalysisResult

if TYPE_CHECKING:
    from src.model import DecomposedMLP


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

    im = ax.imshow(importance_matrix, aspect="auto", cmap="Reds", vmin=0, vmax=1)

    # Y-axis: input patterns
    n_total_inputs = 2**n_inputs
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
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xlabel(f"Component Index (total: {len(component_labels)})")

    ax.set_title("Causal Importance by Input Pattern")

    plt.colorbar(im, ax=ax, label="Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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

    im = ax.imshow(normalized, cmap="viridis", aspect="equal")

    # Add cluster boundaries
    n_clusters = max(cluster_assignments) + 1 if cluster_assignments else 0
    cluster_counts = [0] * n_clusters
    for c in cluster_assignments:
        cluster_counts[c] += 1

    # Draw boundaries
    cumsum = 0
    for count in cluster_counts[:-1]:
        cumsum += count
        ax.axhline(cumsum - 0.5, color="white", linewidth=2)
        ax.axvline(cumsum - 0.5, color="white", linewidth=2)

    ax.set_xlabel("Component Index (sorted by cluster)")
    ax.set_ylabel("Component Index (sorted by cluster)")
    ax.set_title(f"Component Coactivation Matrix ({n_clusters} clusters)")

    plt.colorbar(im, ax=ax, label="Normalized Coactivation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
            "U": comp.U.detach().cpu().numpy(),  # [C, d_out]
            "V": comp.V.detach().cpu().numpy(),  # [d_in, C]
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

        U = layer_matrices[layer_name]["U"]
        V = layer_matrices[layer_name]["V"]

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
        G,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=8,
        ax=ax,
        connectionstyle="arc3,rad=0.1",
    )

    # Add layer labels
    for layer_idx in range(n_layers):
        ax.text(layer_idx, -0.8, f"L{layer_idx}", ha="center", fontsize=8)

    # Title
    title = f"Cluster {cluster_info.cluster_idx}"
    if cluster_info.function_mapping:
        title += f"\n{cluster_info.function_mapping}"
    title += f"\n({len(cluster_info.component_indices)} comp)"
    ax.set_title(title, fontsize=9)
    ax.axis("off")


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
            axes[row, col].axis("off")

        plt.suptitle(f"SPD Clusters: {category_name} ({n} clusters)", fontsize=12)
        plt.tight_layout()

        path = base_dir / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
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
        im = ax.imshow(V.T, aspect="auto", cmap="RdBu_r")
        ax.set_xlabel("Input dimension")
        ax.set_ylabel("Component")
        ax.set_title(f"{name} - V matrix")
        plt.colorbar(im, ax=ax)

        # Plot U
        ax = axes[idx, 1]
        im = ax.imshow(U, aspect="auto", cmap="RdBu_r")
        ax.set_xlabel("Output dimension")
        ax.set_ylabel("Component")
        ax.set_title(f"{name} - U matrix")
        plt.colorbar(im, ax=ax)

    plt.suptitle("SPD U and V Decomposition Matrices", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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

        ax.hist(layer_ci, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_xlabel("Causal Importance")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{layer_name.replace('.', '_')}")
        ax.set_yscale("log")
        ax.set_xlim(0, 1)

    plt.suptitle("Causal Importance Distribution per Layer", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
        ax.scatter(
            range(len(sorted_mean_ci)), sorted_mean_ci, marker="x", s=30, c="steelblue"
        )
        ax.set_xlabel("Component (sorted)")
        ax.set_ylabel("Mean CI")
        ax.set_title(f"{layer_name.replace('.', '_')} (linear)")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="threshold")

        # Log scale
        ax = axes[1, idx]
        ax.scatter(
            range(len(sorted_mean_ci)), sorted_mean_ci, marker="x", s=30, c="steelblue"
        )
        ax.set_xlabel("Component (sorted)")
        ax.set_ylabel("Mean CI")
        ax.set_title(f"{layer_name.replace('.', '_')} (log)")
        ax.set_yscale("log")

    plt.suptitle("Mean Causal Importance per Component", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
        layer_names.append(layer_name.replace(".", "_"))
        l0_values.append(l0)

    fig, ax = plt.subplots(figsize=(max(6, len(layer_names) * 1.5), 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(layer_names)))
    bars = ax.bar(layer_names, l0_values, color=colors, edgecolor="black")

    # Add value labels on bars
    for bar, val in zip(bars, l0_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel(f"L0 (threshold={threshold})")
    ax.set_title("L0 Sparsity per Layer (avg active components per input)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
    ax.axis("off")
    table_data = []
    for c in analysis_result.clusters:
        table_data.append(
            [
                f"Cluster {c.cluster_idx}",
                c.function_mapping or "Unknown",
                len(c.component_indices),
                f"{c.mean_importance:.3f}",
            ]
        )

    table = ax.table(
        cellText=table_data,
        colLabels=["Cluster", "Function", "Components", "Mean Imp."],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("Cluster Summary", pad=20)

    # 4. Overall statistics text
    ax = axes[1, 1]
    ax.axis("off")
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
    {", ".join(set(l.split(":")[0] for l in analysis_result.component_labels))}
    """
    ax.text(
        0.05,
        0.5,
        stats_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    plt.suptitle("SPD Decomposition Analysis Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path

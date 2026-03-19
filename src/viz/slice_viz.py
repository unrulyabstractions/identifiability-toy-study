"""Visualization for slice analysis.

Creates scatter plots and other visualizations for ObservationalSlice,
InterventionalSlice, and CounterfactualSlice.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .constants import get_dpi
from src.circuit import Circuit, compare_circuits


def visualize_accuracy_vs_sparsity(
    ranking_result,
    output_dir: str,
    title_prefix: str = "",
) -> dict[str, str]:
    """Create scatter plots showing subcircuit scores vs edge sparsity for each node pattern.

    X-axis: Edge sparsity (fraction of edges pruned)
    Y-axis: Metric score
    Each point is labeled with its subcircuit_idx.

    Args:
        ranking_result: RankingResult with subcircuits data
        output_dir: Directory to save plots
        title_prefix: Prefix for plot titles

    Returns:
        Dict mapping node_pattern to file path
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # Group subcircuits by node pattern
    by_node = {}
    for sc in ranking_result.subcircuits:
        np_id = sc.node_pattern
        if np_id not in by_node:
            by_node[np_id] = []
        by_node[np_id].append(sc)

    for node_pattern, subcircuits in by_node.items():
        if len(subcircuits) < 2:
            continue  # Skip if only one point

        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort by sparsity for consistent ordering
        subcircuits_sorted = sorted(subcircuits, key=lambda x: x.sparsity)

        sparsities = [sc.sparsity for sc in subcircuits_sorted]
        scores = [sc.score for sc in subcircuits_sorted]
        accuracies = [sc.accuracy for sc in subcircuits_sorted]

        # Color by accuracy
        scatter = ax.scatter(
            sparsities, scores,
            c=accuracies, cmap='RdYlGn', s=120, alpha=0.8,
            edgecolors='black', linewidths=0.5
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Accuracy", fontsize=10)

        # Label each point with subcircuit_idx
        for sc in subcircuits_sorted:
            ax.annotate(
                str(sc.subcircuit_idx),
                (sc.sparsity, sc.score),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=8,
                ha='center',
                color='darkblue'
            )

        # Labels and title
        ax.set_xlabel("Edge Sparsity (fraction of edges pruned)", fontsize=11)
        ax.set_ylabel(ranking_result.metric_name, fontsize=12)
        title = f"Node {node_pattern}: {ranking_result.metric_name} vs Edge Sparsity"
        if title_prefix:
            title = f"{title_prefix} - {title}"
        ax.set_title(title, fontsize=12, fontweight='bold')

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(output_dir, f"{node_pattern}_accuracy.png")
        plt.savefig(path, dpi=get_dpi(), bbox_inches='tight')
        plt.close(fig)
        paths[node_pattern] = path

    return paths


def visualize_score_vs_sparsity(
    ranking_result,
    output_dir: str,
    score_name: str,
    title_prefix: str = "",
) -> dict[str, str]:
    """Create scatter plots of score vs sparsity for each node pattern.

    Args:
        ranking_result: RankingResult with subcircuits data
        output_dir: Directory to save plots
        score_name: Name of the score being plotted
        title_prefix: Prefix for plot titles

    Returns:
        Dict mapping node_pattern to file path
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # Group subcircuits by node pattern
    by_node = {}
    for sc in ranking_result.subcircuits:
        np_id = sc.node_pattern
        if np_id not in by_node:
            by_node[np_id] = []
        by_node[np_id].append(sc)

    for node_pattern, subcircuits in by_node.items():
        if len(subcircuits) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        sparsities = [sc.sparsity for sc in subcircuits]
        scores = [sc.score for sc in subcircuits]
        accuracies = [sc.accuracy for sc in subcircuits]

        # Color by accuracy
        scatter = ax.scatter(
            sparsities, scores,
            c=accuracies, cmap='RdYlGn', s=80, alpha=0.7,
            edgecolors='black', linewidths=0.5
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Accuracy", fontsize=10)

        ax.set_xlabel("Edge Sparsity", fontsize=12)
        ax.set_ylabel(score_name, fontsize=12)
        title = f"Node {node_pattern}: {score_name} vs Sparsity"
        if title_prefix:
            title = f"{title_prefix} - {title}"
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(output_dir, f"{node_pattern}_{score_name.lower().replace(' ', '_')}.png")
        plt.savefig(path, dpi=get_dpi(), bbox_inches='tight')
        plt.close(fig)
        paths[node_pattern] = path

    return paths


def visualize_node_pattern_comparison(
    ranking_result,
    output_path: str,
    title: str = "Node Pattern Comparison",
) -> str:
    """Create bar chart comparing node patterns (max score only).

    Args:
        ranking_result: RankingResult with node_patterns data
        output_path: Path to save the plot
        title: Plot title

    Returns:
        Path to saved file
    """
    if not ranking_result.node_patterns:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    # Sort by max_score descending
    sorted_patterns = sorted(ranking_result.node_patterns, key=lambda p: p.max_score, reverse=True)
    patterns = [p.node_pattern for p in sorted_patterns]
    max_scores = [p.max_score for p in sorted_patterns]

    x = np.arange(len(patterns))
    width = 0.6

    # Single bar for max score
    bars = ax.bar(x, max_scores, width, color='steelblue', edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Node Pattern", fontsize=12)
    ax.set_ylabel("Max Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(patterns, rotation=45, ha='right', fontsize=10)

    # Zoom y-axis to 0.75-1.0 range
    ax.set_ylim(0.75, 1.0)

    # Finer grid
    ax.set_yticks(np.arange(0.75, 1.01, 0.05))
    ax.grid(True, alpha=0.4, axis='y', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.3, which='minor')
    ax.set_yticks(np.arange(0.75, 1.01, 0.025), minor=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=get_dpi(), bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_edge_variant_jaccard(
    circuits: dict[int, Circuit],
    output_path: str,
    node_pattern: int,
    title_prefix: str = "",
) -> str:
    """Create Jaccard similarity heatmap for edge variants within a node pattern.

    Args:
        circuits: Dict mapping subcircuit_idx to Circuit object
        output_path: Path to save the plot
        node_pattern: Node pattern ID for title
        title_prefix: Prefix for title

    Returns:
        Path to saved file
    """
    if len(circuits) < 2:
        return None

    sc_ids = sorted(circuits.keys())
    n = len(sc_ids)

    # Build Jaccard matrix
    jaccard_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                jaccard_matrix[i][j] = 1.0
            else:
                comparison = compare_circuits(circuits[sc_ids[i]], circuits[sc_ids[j]])
                jaccard_matrix[i][j] = comparison["jaccard"]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(jaccard_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')

    # Short labels
    short_labels = [str(sc_id)[-4:] for sc_id in sc_ids]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.set_xlabel("Subcircuit (last 4 digits)", fontsize=11)
    ax.set_ylabel("Subcircuit (last 4 digits)", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Jaccard Similarity", fontsize=11)

    # Annotate
    for i in range(n):
        for j in range(n):
            val = jaccard_matrix[i][j]
            color = 'white' if val > 0.55 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                   fontsize=8, color=color, fontweight='bold' if i == j else 'normal')

    title = f"Node {node_pattern}: Edge Variant Jaccard Similarity"
    if title_prefix:
        title = f"{title_prefix} - {title}"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=get_dpi(), bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_subset_relationships(
    circuits: dict[int, Circuit],
    output_path: str,
    node_pattern: int,
    title_prefix: str = "",
) -> str:
    """Create nested circles diagram showing subset relationships between edge variants.

    Args:
        circuits: Dict mapping subcircuit_idx to Circuit object
        output_path: Path to save the plot
        node_pattern: Node pattern ID for title
        title_prefix: Prefix for title

    Returns:
        Path to saved file
    """
    if len(circuits) < 2:
        return None

    sc_ids = sorted(circuits.keys())
    n = len(sc_ids)

    # Build subset matrix and get sparsities
    subset_matrix = np.zeros((n, n))
    sparsities = []

    for i in range(n):
        _, edge_sparsity, _ = circuits[sc_ids[i]].sparsity()
        sparsities.append(edge_sparsity)
        for j in range(n):
            if i != j:
                comparison = compare_circuits(circuits[sc_ids[i]], circuits[sc_ids[j]])
                if comparison["a_subset_of_b"]:
                    subset_matrix[i][j] = 1

    # Find connected components
    visited = [False] * n
    groups = []

    def dfs(idx, group):
        if visited[idx]:
            return
        visited[idx] = True
        group.append(idx)
        for j in range(n):
            if idx != j and (subset_matrix[idx][j] or subset_matrix[j][idx]):
                dfs(j, group)

    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)
            groups.append(group)

    # Sort groups and within groups by sparsity
    groups = sorted(groups, key=lambda g: -len(g))
    for group in groups:
        group.sort(key=lambda idx: sparsities[idx])

    # Create figure
    n_groups = len(groups)
    fig_width = max(10, 3.5 * n_groups)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    group_colors = plt.cm.Set3(np.linspace(0, 1, max(n_groups, 8)))

    # Position each group
    group_width = 2.2
    total_width = n_groups * group_width
    start_x = -total_width / 2 + group_width / 2

    for g_idx, group in enumerate(groups):
        center_x = start_x + g_idx * group_width
        center_y = 0

        n_in_group = len(group)
        base_radius = 0.9

        for depth, idx in enumerate(group):
            sc_id = sc_ids[idx]
            sparsity = sparsities[idx]

            radius = base_radius * (1 - depth * 0.15)
            if radius < 0.12:
                radius = 0.12

            alpha = min(0.25 + 0.12 * depth, 0.9)
            color = group_colors[g_idx % len(group_colors)]

            circle = mpatches.Circle(
                (center_x, center_y),
                radius=radius,
                fill=True,
                facecolor=(color[0], color[1], color[2], alpha),
                edgecolor='darkblue',
                linewidth=2.0 if depth == 0 else 1.2,
            )
            ax.add_patch(circle)

            # Label
            if n_in_group <= 4:
                label_y = center_y + radius * 0.5 - depth * 0.18
            else:
                label_y = center_y + radius * 0.55 - depth * 0.14

            # Short label
            short_id = str(sc_id)[-4:]
            ax.text(center_x, label_y, short_id,
                   ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   color='black',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                            edgecolor='none', alpha=0.75))

        # Sparsity range for group
        group_sparsities = [sparsities[idx] for idx in group]
        min_sp, max_sp = min(group_sparsities), max(group_sparsities)
        ax.text(center_x, -1.15, f"Sparsity:\n{min_sp:.0%}-{max_sp:.0%}",
               ha='center', va='top', fontsize=9, color='gray')

    ax.set_xlim(start_x - 1.3, start_x + n_groups * group_width)
    ax.set_ylim(-1.5, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    title = f"Node {node_pattern}: Edge Variant Subsets"
    if title_prefix:
        title = f"{title_prefix} - {title}"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Legend
    ax.text(0.02, 0.02,
           "Nested circles = subset relationships\nOuter = superset, Inner = subset",
           transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=get_dpi(), bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_node_pattern_jaccard(
    node_circuits: dict[int, Circuit],
    output_path: str,
    title_prefix: str = "",
) -> str:
    """Create Jaccard similarity heatmap for node patterns.

    Args:
        node_circuits: Dict mapping node_pattern_idx to Circuit
        output_path: Path to save the plot
        title_prefix: Prefix for title

    Returns:
        Path to saved file
    """
    if len(node_circuits) < 2:
        return None

    node_ids = sorted(node_circuits.keys())
    n = len(node_ids)

    # Build Jaccard matrix
    jaccard_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                jaccard_matrix[i][j] = 1.0
            else:
                comparison = compare_circuits(node_circuits[node_ids[i]], node_circuits[node_ids[j]])
                jaccard_matrix[i][j] = comparison["jaccard"]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(jaccard_matrix, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(node_ids, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(node_ids, fontsize=11)
    ax.set_xlabel("Node Pattern", fontsize=12)
    ax.set_ylabel("Node Pattern", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Jaccard Similarity", fontsize=11)

    # Annotate all cells
    for i in range(n):
        for j in range(n):
            val = jaccard_matrix[i][j]
            color = 'white' if val > 0.55 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                   fontsize=9, color=color, fontweight='bold' if i == j else 'normal')

    title = "Node Pattern Jaccard Similarity"
    if title_prefix:
        title = f"{title_prefix}: {title}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=get_dpi(), bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_node_pattern_subsets(
    node_circuits: dict[int, Circuit],
    output_path: str,
    title_prefix: str = "",
) -> str:
    """Create nested circles diagram showing subset relationships between node patterns.

    Draws circles where smaller (inner) circles represent subsets of larger (outer) ones.
    Uses a Venn-diagram-like layout with nested circles for related patterns.

    Args:
        node_circuits: Dict mapping node_pattern_idx to Circuit (full edges)
        output_path: Path to save the plot
        title_prefix: Prefix for title

    Returns:
        Path to saved file
    """
    if len(node_circuits) < 2:
        return None

    node_ids = sorted(node_circuits.keys())
    n = len(node_ids)

    # Build subset matrix and get sparsities
    subset_matrix = np.zeros((n, n))
    node_sparsities = []

    for i in range(n):
        node_sparsity, _, _ = node_circuits[node_ids[i]].sparsity()
        node_sparsities.append(node_sparsity)
        for j in range(n):
            if i != j:
                comparison = compare_circuits(node_circuits[node_ids[i]], node_circuits[node_ids[j]])
                if comparison["a_subset_of_b"]:
                    subset_matrix[i][j] = 1

    # Find connected components (groups of related patterns)
    visited = [False] * n
    groups = []

    def dfs(idx, group):
        if visited[idx]:
            return
        visited[idx] = True
        group.append(idx)
        for j in range(n):
            if idx != j and (subset_matrix[idx][j] or subset_matrix[j][idx]):
                dfs(j, group)

    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)
            groups.append(group)

    # Sort groups by size (largest first) and sort within groups by sparsity
    groups = sorted(groups, key=lambda g: -len(g))
    for group in groups:
        group.sort(key=lambda idx: node_sparsities[idx])  # Less sparse (more nodes) first

    # Create figure
    n_groups = len(groups)
    fig_width = max(10, 4 * n_groups)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    # Colors for different groups
    group_colors = plt.cm.Set2(np.linspace(0, 1, max(n_groups, 8)))

    # Position each group
    group_width = 2.5
    total_width = n_groups * group_width
    start_x = -total_width / 2 + group_width / 2

    for g_idx, group in enumerate(groups):
        center_x = start_x + g_idx * group_width
        center_y = 0

        # Draw nested circles for this group
        n_in_group = len(group)
        base_radius = 1.0

        for depth, idx in enumerate(group):
            node_id = node_ids[idx]
            sparsity = node_sparsities[idx]

            # Radius decreases with depth (more sparse = smaller)
            radius = base_radius * (1 - depth * 0.18)
            if radius < 0.15:
                radius = 0.15

            # Color intensity based on depth
            alpha = min(0.3 + 0.15 * depth, 0.9)
            color = group_colors[g_idx % len(group_colors)]

            circle = mpatches.Circle(
                (center_x, center_y),
                radius=radius,
                fill=True,
                facecolor=(color[0], color[1], color[2], alpha),
                edgecolor='black',
                linewidth=2.5 if depth == 0 else 1.5,
                linestyle='-'
            )
            ax.add_patch(circle)

            # Label position - spiral arrangement at circle edges
            import math
            # Spread labels around the circle, starting from top and going clockwise
            angle = math.pi / 2 - depth * (math.pi / (n_in_group + 1))
            label_x = center_x + (radius + 0.12) * math.cos(angle)
            label_y = center_y + (radius + 0.12) * math.sin(angle)
            label = f"{node_id}"
            # Adjust alignment based on position
            ha = 'left' if label_x > center_x else 'right'
            va = 'bottom' if label_y > center_y else 'top'
            if abs(label_x - center_x) < 0.1:
                ha = 'center'
            ax.text(label_x, label_y, label,
                   ha=ha, va=va,
                   fontsize=10, fontweight='bold',
                   color='black',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                            edgecolor='none', alpha=0.8))

        # Group label below
        ax.text(center_x, -1.3, f"Group {g_idx + 1}\n({len(group)} patterns)",
               ha='center', va='top', fontsize=10, style='italic', color='gray')

    # Set limits
    ax.set_xlim(start_x - 1.5, start_x + n_groups * group_width)
    ax.set_ylim(-1.6, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    title = "Node Pattern Subset Relationships"
    if title_prefix:
        title = f"{title_prefix}: {title}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=get_dpi(), bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_slice(
    slice_obj,
    output_dir: str,
    gate_name: str = "",
    trial=None,
) -> dict[str, str]:
    """Visualize all rankings in a slice.

    Works with ObservationalSlice, InterventionalSlice, or CounterfactualSlice.

    Args:
        slice_obj: Slice object with ranking results
        output_dir: Base output directory
        gate_name: Gate name for title prefix
        trial: Optional trial object with circuit data for subset visualizations

    Returns:
        Dict of paths to generated visualizations
    """
    paths = {}

    # Determine slice type and get rankings
    if hasattr(slice_obj, 'rank_accuracy'):  # ObservationalSlice
        slice_dir = os.path.join(output_dir, "observational", "viz")
        rankings = [
            (slice_obj.rank_accuracy, "accuracy"),
            (slice_obj.rank_noise, "noise"),
            (slice_obj.rank_ood, "ood"),
        ]
    elif hasattr(slice_obj, 'rank_in_circuit_id'):  # InterventionalSlice
        slice_dir = os.path.join(output_dir, "interventional", "viz")
        rankings = [
            (slice_obj.rank_in_circuit_id, "in_circuit_id"),
            (slice_obj.rank_in_circuit_ood, "in_circuit_ood"),
            (slice_obj.rank_out_circuit_id, "out_circuit_id"),
            (slice_obj.rank_out_circuit_ood, "out_circuit_ood"),
            (slice_obj.rank_overall, "overall"),
        ]
    elif hasattr(slice_obj, 'rank_sufficiency'):  # CounterfactualSlice
        slice_dir = os.path.join(output_dir, "counterfactual", "viz")
        rankings = [
            (slice_obj.rank_sufficiency, "sufficiency"),
            (slice_obj.rank_completeness, "completeness"),
            (slice_obj.rank_necessity, "necessity"),
            (slice_obj.rank_independence, "independence"),
            (slice_obj.rank_overall, "overall"),
        ]
    else:
        return paths

    os.makedirs(slice_dir, exist_ok=True)

    for ranking, name in rankings:
        if not ranking:
            continue

        # Accuracy vs sparsity scatter plots per node pattern
        scatter_dir = os.path.join(slice_dir, name)
        scatter_paths = visualize_accuracy_vs_sparsity(
            ranking, scatter_dir, title_prefix=gate_name
        )
        paths[f"{name}_scatter"] = scatter_paths

        # Node pattern comparison bar chart
        comparison_path = os.path.join(slice_dir, f"{name}_comparison.png")
        visualize_node_pattern_comparison(
            ranking, comparison_path,
            title=f"{gate_name}: {ranking.metric_name} by Node Pattern"
        )
        paths[f"{name}_comparison"] = comparison_path

    # Generate subset relationship visualizations if trial data available
    if trial is not None and hasattr(slice_obj, 'rank_accuracy'):
        # Only do this for observational slice (once per gate)
        subset_dir = os.path.join(output_dir, "observational", "viz", "subsets")
        os.makedirs(subset_dir, exist_ok=True)

        per_gate_circuits = trial.metrics.per_gate_circuits.get(gate_name, {})
        if per_gate_circuits:
            # Group circuits by node pattern
            from src.circuit import parse_subcircuit_idx
            width = trial.setup.model_params.width
            depth = trial.setup.model_params.depth

            by_node = {}
            for sc_idx, circuit_dict in per_gate_circuits.items():
                node_idx, _ = parse_subcircuit_idx(width, depth, sc_idx)
                if node_idx not in by_node:
                    by_node[node_idx] = {}
                by_node[node_idx][sc_idx] = Circuit.from_dict(circuit_dict)

            # Generate subset visualization for each node pattern (edge variants)
            for node_pattern, circuits in by_node.items():
                if len(circuits) >= 2:
                    # Jaccard heatmap
                    jaccard_path = os.path.join(subset_dir, f"{node_pattern}_jaccard.png")
                    visualize_edge_variant_jaccard(
                        circuits, jaccard_path, node_pattern, title_prefix=gate_name
                    )
                    paths[f"jaccard_{node_pattern}"] = jaccard_path

                    # Subset circles
                    subset_path = os.path.join(subset_dir, f"{node_pattern}_subsets.png")
                    visualize_subset_relationships(
                        circuits, subset_path, node_pattern, title_prefix=gate_name
                    )
                    paths[f"subsets_{node_pattern}"] = subset_path

            # Generate node pattern comparison (comparing different node masks)
            # Use the least sparse (most edges) circuit for each node pattern
            if len(by_node) >= 2:
                node_representatives = {}
                for node_pattern, circuits in by_node.items():
                    # Pick circuit with lowest edge sparsity (most edges)
                    best_circuit = min(
                        circuits.values(),
                        key=lambda c: c.sparsity()[1]  # edge sparsity
                    )
                    node_representatives[node_pattern] = best_circuit

                # Jaccard heatmap (separate plot)
                jaccard_path = os.path.join(subset_dir, "node_patterns_jaccard.png")
                visualize_node_pattern_jaccard(
                    node_representatives, jaccard_path, title_prefix=gate_name
                )
                paths["node_pattern_jaccard"] = jaccard_path

                # Subset circles (separate plot)
                subset_path = os.path.join(subset_dir, "node_patterns_subsets.png")
                visualize_node_pattern_subsets(
                    node_representatives, subset_path, title_prefix=gate_name
                )
                paths["node_pattern_subsets"] = subset_path

    return paths

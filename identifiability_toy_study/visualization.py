"""Visualization for circuits and SPD decompositions.

IMPORTANT: This module does NOT run any models. All visualizations use
pre-computed data stored in tensors.pt and results.json.

Look at visualize_experiment for the main entry point.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from .common.circuit import Circuit
from .common.neural_model import MLP, DecomposedMLP
from .common.schemas import (
    ExperimentResult,
    FaithfulnessMetrics,
    InterventionSample,
    ProfilingData,
    RobustnessMetrics,
)

# ------------------ CONSTANTS ------------------

# Plot styling
COLORS = {
    "gate": "steelblue",
    "subcircuit": "coral",
    "agreement": "purple",
    "mse": "teal",
    "correct": "green",
    "incorrect": "red",
    # Faithfulness-specific
    "in_circuit": "#2E7D32",  # Dark green
    "out_circuit": "#C62828",  # Dark red
    "faithfulness": "#6A1B9A",  # Purple
    "counterfactual": "#EF6C00",  # Orange
}
MARKERS = {"gate": "^", "subcircuit": "v", "agreement": "o", "mse": "s"}
JITTER = {"correct": 1.05, "incorrect": -0.05, "gate_correct": 1.05, "sc_correct": 0.95}


# ------------------ HELPERS ------------------


def _activation_to_color(val: float, vmin: float, vmax: float) -> tuple:
    """RdYlGn colormap: red=neg, yellow=0, green=pos."""
    normalized = (val - vmin) / (vmax - vmin)
    normalized = min(max(normalized, 0), 1)
    return plt.cm.RdYlGn(normalized)


def _text_color_for_background(bg_color: tuple) -> str:
    """Contrasting text color based on luminance."""
    r, g, b = bg_color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


def _symmetric_range(activations: list) -> tuple[float, float]:
    """Color range centered at 0."""
    all_vals = [act[0].detach().cpu().numpy() for act in activations]
    vmin, vmax = min(v.min() for v in all_vals), max(v.max() for v in all_vals)
    abs_max = max(abs(vmin), abs(vmax), 0.1)
    return -abs_max, abs_max


def _get_spd_component_weights(decomposed: DecomposedMLP) -> np.ndarray | None:
    """Normalized weight magnitudes per component."""
    components = getattr(decomposed.component_model, "components", {})
    if not components:
        return None

    n_components = getattr(next(iter(components.values())), "C", 0)
    if n_components == 0:
        return None

    weights = []
    for i in range(n_components):
        total = 0.0
        for c in components.values():
            if hasattr(c, "V") and c.V is not None:
                total += c.V[:, i].abs().sum().item()
            if hasattr(c, "U") and c.U is not None:
                total += c.U[i, :].abs().sum().item()
        weights.append(total)

    if not weights or all(w == 0 for w in weights):
        return None

    weights = np.array(weights)
    return weights / weights.max() if weights.max() > 0 else weights


# ------------------ CIRCUIT GRAPH ------------------


def _build_graph(activations, circuit, weights_per_layer, vmin, vmax):
    """Build networkx graph from pre-computed activations and circuit mask."""
    G = nx.DiGraph()
    pos, node_colors, labels, text_colors = {}, [], {}, {}

    max_width = max(act.shape[-1] for act in activations)

    # Nodes
    for layer_idx, layer_act in enumerate(activations):
        n = layer_act.shape[-1]
        y_off = -(max_width - n) / 2

        for node_idx in range(n):
            name = f"({layer_idx},{node_idx})"
            G.add_node(name)
            pos[name] = (layer_idx, y_off - node_idx)

            val = layer_act[0, node_idx].item()
            labels[name] = f"{val:.2f}"

            active = (
                layer_idx >= len(circuit.node_masks)
                or circuit.node_masks[layer_idx][node_idx] == 1
            )
            if active:
                color = _activation_to_color(val, vmin, vmax)
                node_colors.append(color)
                text_colors[name] = _text_color_for_background(color)
            else:
                node_colors.append("#d3d3d3")
                text_colors[name] = "gray"

    # Edges (store weight for thickness calculation)
    edge_labels = {}
    edge_weights = {}
    for layer_idx, mask in enumerate(circuit.edge_masks):
        w = weights_per_layer[layer_idx]
        for out_idx, row in enumerate(mask):
            for in_idx, active in enumerate(row):
                e = (f"({layer_idx},{in_idx})", f"({layer_idx + 1},{out_idx})")
                weight_val = w[out_idx, in_idx]
                G.add_edge(*e, active=active, weight=weight_val)
                if active == 1:
                    edge_labels[e] = f"{weight_val:.2f}"
                    edge_weights[e] = abs(weight_val)

    return G, pos, node_colors, labels, text_colors, edge_labels, edge_weights


def _draw_graph(
    ax, G, pos, node_colors, labels, text_colors, edge_labels, edge_weights
):
    """Draw graph on axis with edge thickness proportional to weight magnitude."""
    active = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 1]
    inactive = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 0]

    # Compute edge widths (linear scale: min 0.5, max 4)
    if edge_weights:
        max_w = max(edge_weights.values()) if edge_weights.values() else 1.0
        max_w = max(max_w, 0.01)
        active_widths = [0.5 + 3.5 * (edge_weights.get(e, 0) / max_w) for e in active]
    else:
        active_widths = [2] * len(active)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=900,
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=active,
        edge_color="#333333",
        width=active_widths,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=inactive,
        edge_color="#cccccc",
        width=0.5,
        style="dashed",
        arrows=True,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    for node, (x, y) in pos.items():
        ax.text(
            x,
            y,
            labels[node],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=text_colors[node],
        )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=6,
        font_color="#666666",
        alpha=0.8,
        label_pos=0.3,
        bbox=dict(
            boxstyle="round,pad=0.1", facecolor="white", alpha=0.7, edgecolor="none"
        ),
        ax=ax,
    )
    ax.axis("off")


def _draw_circuit_from_data(ax, activations, circuit, weights, title):
    """Draw a single circuit using pre-computed activations (no model run)."""
    vmin, vmax = _symmetric_range(activations)
    G, pos, colors, node_labels, text_colors, edge_labels, edge_weights = _build_graph(
        activations, circuit, weights, vmin, vmax
    )
    _draw_graph(ax, G, pos, colors, node_labels, text_colors, edge_labels, edge_weights)

    output = activations[-1][0, 0].item()
    ax.set_title(f"{title} -> {output:.3f}", fontsize=10, fontweight="bold")


# ------------------ PUBLIC FUNCTIONS ------------------


def visualize_circuit_activations_from_data(
    canonical_activations: dict[str, list[torch.Tensor]],
    layer_weights: list[torch.Tensor],
    circuit: Circuit,
    output_dir: str,
    filename: str = "circuit_activations.png",
    gate_name: str = "",
) -> str:
    """
    2x2 grid: circuit activations for (0,0), (0,1), (1,0), (1,1) inputs.

    Uses pre-computed activations - NO model execution.
    """
    labels_map = {
        "0_0": "(0, 0)",
        "0_1": "(0, 1)",
        "1_0": "(1, 0)",
        "1_1": "(1, 1)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    weights = [w.numpy() for w in layer_weights]

    for i, (key, label) in enumerate(labels_map.items()):
        activations = canonical_activations.get(key, [])
        if activations:
            _draw_circuit_from_data(
                axes[i], activations, circuit, weights, f"Input: {label}"
            )
        else:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[i].axis("off")

    if gate_name:
        fig.suptitle(
            f"{gate_name} - Circuit Activations", fontsize=14, fontweight="bold"
        )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------ ROBUSTNESS VISUALIZATION ------------------


def _bin_samples_by_magnitude(
    samples: list, n_bins: int = 20, log_scale: bool = False
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Bin samples by noise magnitude and compute stats per bin.

    Args:
        samples: List of RobustnessSample
        n_bins: Number of bins
        log_scale: Use logarithmic binning (for OOD scale factors)

    Returns: (bin_centers, gate_acc, sc_acc, bit_agree, mse_mean)
    """
    if not samples:
        return [], [], [], [], []

    # Sort by magnitude
    sorted_samples = sorted(samples, key=lambda s: s.noise_magnitude)
    magnitudes = [s.noise_magnitude for s in sorted_samples]

    # Create bins (linear or logarithmic)
    min_mag, max_mag = min(magnitudes), max(magnitudes)
    if log_scale and min_mag > 0:
        bin_edges = np.logspace(np.log10(min_mag), np.log10(max_mag), n_bins + 1)
    else:
        bin_edges = np.linspace(min_mag, max_mag, n_bins + 1)

    bin_centers = []
    gate_accs = []
    sc_accs = []
    bit_agrees = []
    mse_means = []

    for i in range(n_bins):
        bin_samples = [
            s
            for s in sorted_samples
            if bin_edges[i] <= s.noise_magnitude < bin_edges[i + 1]
        ]
        if not bin_samples:
            continue

        # For log scale, use geometric mean for bin center
        if log_scale:
            bin_centers.append(np.sqrt(bin_edges[i] * bin_edges[i + 1]))
        else:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        gate_accs.append(
            sum(1 for s in bin_samples if s.gate_correct) / len(bin_samples)
        )
        sc_accs.append(
            sum(1 for s in bin_samples if s.subcircuit_correct) / len(bin_samples)
        )
        bit_agrees.append(
            sum(1 for s in bin_samples if s.agreement_bit) / len(bin_samples)
        )
        mse_means.append(sum(s.mse for s in bin_samples) / len(bin_samples))

    return bin_centers, gate_accs, sc_accs, bit_agrees, mse_means


def _save_figure(fig, output_dir: str, filename: str) -> str:
    """Save figure and return path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_accuracy_scatter(
    samples: list,
    ax,
    xlabel: str,
    title: str,
    use_log_scale: bool = False,
    use_abs_magnitude: bool = False,
) -> None:
    """Plot accuracy scatter for gate model vs subcircuit."""
    mags = [abs(s.noise_magnitude) if use_abs_magnitude else s.noise_magnitude for s in samples]
    gate_y = [JITTER["correct"] if s.gate_correct else JITTER["incorrect"] for s in samples]
    sc_y = [JITTER["sc_correct"] if s.subcircuit_correct else JITTER["incorrect"] + 0.1 for s in samples]

    ax.scatter(mags, gate_y, alpha=0.5, s=20, c=COLORS["gate"], marker=MARKERS["gate"], label="Gate Model")
    ax.scatter(mags, sc_y, alpha=0.5, s=20, c=COLORS["subcircuit"], marker=MARKERS["subcircuit"], label="Subcircuit")

    ax.axhline(y=1, color=COLORS["correct"], linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=0, color=COLORS["incorrect"], linestyle="--", alpha=0.5, linewidth=1)

    if use_log_scale:
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
    else:
        ax.grid(alpha=0.3)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Correct (1) / Incorrect (0)", fontsize=12)
    ax.set_ylim(-0.2, 1.2)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best")


def _plot_agreement_scatter(
    samples: list,
    ax1,
    ax2,
    xlabel: str,
    title: str,
    use_log_scale: bool = False,
    use_abs_magnitude: bool = False,
) -> None:
    """Plot agreement and MSE scatter on dual y-axis."""
    mags = [abs(s.noise_magnitude) if use_abs_magnitude else s.noise_magnitude for s in samples]
    agree_y = [JITTER["correct"] if s.agreement_bit else JITTER["incorrect"] for s in samples]
    mses = [s.mse for s in samples]

    # Agreement on left axis
    ax1.scatter(mags, agree_y, alpha=0.5, s=20, c=COLORS["agreement"], marker=MARKERS["agreement"], label="Bit Agreement")
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Agree (1) / Disagree (0)", fontsize=12, color=COLORS["agreement"])
    ax1.tick_params(axis="y", labelcolor=COLORS["agreement"])
    ax1.set_ylim(-0.2, 1.2)
    ax1.axhline(y=1, color=COLORS["correct"], linestyle="--", alpha=0.5, linewidth=1)
    ax1.axhline(y=0, color=COLORS["incorrect"], linestyle="--", alpha=0.5, linewidth=1)

    # MSE on right axis
    ax2.scatter(mags, mses, alpha=0.5, s=20, c=COLORS["mse"], marker=MARKERS["mse"], label="MSE")
    ax2.set_ylabel("MSE", fontsize=12, color=COLORS["mse"])
    ax2.tick_params(axis="y", labelcolor=COLORS["mse"])
    ax2.set_ylim(bottom=0)

    if use_log_scale:
        ax1.set_xscale("log")
        ax1.grid(alpha=0.3, which="both")
    else:
        ax1.grid(alpha=0.3)

    ax1.set_title(title, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")


def _group_samples_by_input(samples: list) -> dict[str, list]:
    """Group samples by their base input combination."""
    base_to_key = {
        (0.0, 0.0): "0_0",
        (0.0, 1.0): "0_1",
        (1.0, 0.0): "1_0",
        (1.0, 1.0): "1_1",
    }
    grouped = {k: [] for k in base_to_key.values()}
    for s in samples:
        key = base_to_key.get((s.base_input[0], s.base_input[1]))
        if key:
            grouped[key].append(s)
    return grouped


def _plot_robustness_set(
    noise_samples: list,
    ood_samples: list,
    output_dir: str,
    prefix: str,
    input_label: str = "",
) -> dict[str, str]:
    """Plot a complete set of robustness graphs for given samples."""
    paths = {}
    label_suffix = f" [{input_label}]" if input_label else ""

    # --- Noise Plots ---
    if noise_samples:
        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_accuracy_scatter(
            noise_samples, ax,
            xlabel="Noise Magnitude",
            title=f"{prefix}Accuracy vs Noise{label_suffix}",
        )
        fname = f"noise_accuracy_{input_label}.png" if input_label else "noise_accuracy.png"
        paths["noise_accuracy"] = _save_figure(fig, output_dir, fname)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        _plot_agreement_scatter(
            noise_samples, ax1, ax2,
            xlabel="Noise Magnitude",
            title=f"{prefix}Agreement vs Noise{label_suffix}",
        )
        fname = f"noise_agreement_{input_label}.png" if input_label else "noise_agreement.png"
        paths["noise_agreement"] = _save_figure(fig, output_dir, fname)

    # --- OOD Plots ---
    if ood_samples:
        pos_samples = [s for s in ood_samples if s.noise_magnitude > 0]
        neg_samples = [s for s in ood_samples if s.noise_magnitude < 0]

        if pos_samples:
            fig, ax = plt.subplots(figsize=(10, 5))
            _plot_accuracy_scatter(
                pos_samples, ax,
                xlabel="Scale Factor",
                title=f"{prefix}Positive OOD Accuracy{label_suffix}",
                use_log_scale=True,
            )
            fname = f"ood_positive_accuracy_{input_label}.png" if input_label else "ood_positive_accuracy.png"
            paths["ood_positive_accuracy"] = _save_figure(fig, output_dir, fname)

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()
            _plot_agreement_scatter(
                pos_samples, ax1, ax2,
                xlabel="Scale Factor",
                title=f"{prefix}Positive OOD Agreement{label_suffix}",
                use_log_scale=True,
            )
            fname = f"ood_positive_agreement_{input_label}.png" if input_label else "ood_positive_agreement.png"
            paths["ood_positive_agreement"] = _save_figure(fig, output_dir, fname)

        if neg_samples:
            fig, ax = plt.subplots(figsize=(10, 5))
            _plot_accuracy_scatter(
                neg_samples, ax,
                xlabel="|Scale| (negative)",
                title=f"{prefix}Negative OOD Accuracy{label_suffix}",
                use_log_scale=True,
                use_abs_magnitude=True,
            )
            fname = f"ood_negative_accuracy_{input_label}.png" if input_label else "ood_negative_accuracy.png"
            paths["ood_negative_accuracy"] = _save_figure(fig, output_dir, fname)

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()
            _plot_agreement_scatter(
                neg_samples, ax1, ax2,
                xlabel="|Scale| (negative)",
                title=f"{prefix}Negative OOD Agreement{label_suffix}",
                use_log_scale=True,
                use_abs_magnitude=True,
            )
            fname = f"ood_negative_agreement_{input_label}.png" if input_label else "ood_negative_agreement.png"
            paths["ood_negative_agreement"] = _save_figure(fig, output_dir, fname)

    return paths


def visualize_robustness_curves(
    robustness: RobustnessMetrics,
    output_dir: str,
    gate_name: str = "",
) -> dict[str, str]:
    """
    Visualize robustness metrics as scatter plots.

    Creates robustness/stats/ folder with:
    - noise_accuracy.png, noise_agreement.png
    - ood_positive_accuracy.png, ood_negative_accuracy.png
    - ood_positive_agreement.png, ood_negative_agreement.png

    Also creates stats/per_input/ with same plots per input combination.
    """
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    prefix = f"{gate_name} - " if gate_name else ""

    # Aggregate plots (all inputs)
    paths = _plot_robustness_set(
        robustness.noise_samples,
        robustness.ood_samples,
        stats_dir,
        prefix,
    )

    # Per-input plots
    per_input_dir = os.path.join(stats_dir, "per_input")
    os.makedirs(per_input_dir, exist_ok=True)

    noise_by_input = _group_samples_by_input(robustness.noise_samples)
    ood_by_input = _group_samples_by_input(robustness.ood_samples)

    paths["per_input"] = {}
    for input_key in ["0_0", "0_1", "1_0", "1_1"]:
        input_paths = _plot_robustness_set(
            noise_by_input.get(input_key, []),
            ood_by_input.get(input_key, []),
            per_input_dir,
            prefix,
            input_label=input_key,
        )
        if input_paths:
            paths["per_input"][input_key] = input_paths

    return paths


def _draw_graph_with_output_highlight(
    ax, G, pos, node_colors, labels, text_colors, edge_labels, edge_weights,
    output_correct: bool
):
    """Draw graph with highlighted output node border (green=correct, red=incorrect)."""
    active = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 1]
    inactive = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 0]

    # Compute edge widths
    if edge_weights:
        max_w = max(edge_weights.values()) if edge_weights.values() else 1.0
        max_w = max(max_w, 0.01)
        active_widths = [0.5 + 3.5 * (edge_weights.get(e, 0) / max_w) for e in active]
    else:
        active_widths = [2] * len(active)

    # Find output node (last layer, single node)
    nodes = list(G.nodes())
    output_node = max(nodes, key=lambda n: (n[0], n[1]))  # Highest layer, highest idx

    # Draw non-output nodes
    non_output = [n for n in nodes if n != output_node]
    non_output_colors = [node_colors[nodes.index(n)] for n in non_output]
    nx.draw_networkx_nodes(
        G, pos, nodelist=non_output, node_color=non_output_colors,
        node_size=900, edgecolors="black", linewidths=1.5, ax=ax,
    )

    # Draw output node with thick colored border
    output_color = node_colors[nodes.index(output_node)]
    border_color = "#4CAF50" if output_correct else "#F44336"
    nx.draw_networkx_nodes(
        G, pos, nodelist=[output_node], node_color=[output_color],
        node_size=900, edgecolors=border_color, linewidths=4, ax=ax,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, edgelist=active, edge_color="#333333", width=active_widths,
        arrows=True, arrowstyle="-|>", arrowsize=15,
        connectionstyle="arc3,rad=0.1", ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=inactive, edge_color="#cccccc", width=0.5,
        style="dashed", arrows=True, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1", ax=ax,
    )

    # Draw labels
    for node, (x, y) in pos.items():
        ax.text(x, y, labels[node], ha="center", va="center",
                fontsize=9, fontweight="bold", color=text_colors[node])

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=6, font_color="#666666",
        alpha=0.8, label_pos=0.3,
        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7, edgecolor="none"),
        ax=ax,
    )
    ax.axis("off")


def visualize_robustness_circuit_samples(
    robustness: RobustnessMetrics,
    subcircuit_model: MLP,
    gate_model: MLP,
    circuit: Circuit,
    layer_weights: list[torch.Tensor],
    output_dir: str,
    n_samples_per_grid: int = 8,
) -> dict[str, str]:
    """
    Visualize circuit diagrams comparing subcircuit vs full model under noise.

    Creates one PNG per base input per category (e.g., 0_0_noise.png).
    Each row shows: [subcircuit | full circuit] for one sample.
    Samples are balanced between correct/incorrect and sorted by noise magnitude.
    Output nodes have thick border: green=correct, red=incorrect.
    """
    circuit_viz_dir = os.path.join(output_dir, "circuit_viz")
    os.makedirs(circuit_viz_dir, exist_ok=True)
    paths = {}

    base_to_key = {
        (0.0, 0.0): "0_0",
        (0.0, 1.0): "0_1",
        (1.0, 0.0): "1_0",
        (1.0, 1.0): "1_1",
    }

    weights = [w.numpy() for w in layer_weights]

    # Create full circuit for gate model comparison
    layer_sizes = [layer_weights[0].shape[1]] + [w.shape[0] for w in layer_weights]
    full_circuit = Circuit.full(layer_sizes)

    # Group samples by base input (separate ood into positive and negative)
    samples_by_base: dict[str, dict[str, list]] = {
        k: {"noise": [], "ood_positive": [], "ood_negative": []} for k in base_to_key.values()
    }

    for sample in robustness.noise_samples:
        base_key = base_to_key.get((sample.base_input[0], sample.base_input[1]))
        if base_key:
            samples_by_base[base_key]["noise"].append(sample)

    for sample in robustness.ood_samples:
        base_key = base_to_key.get((sample.base_input[0], sample.base_input[1]))
        if base_key:
            if sample.noise_magnitude > 0:
                samples_by_base[base_key]["ood_positive"].append(sample)
            else:
                samples_by_base[base_key]["ood_negative"].append(sample)

    # Create one visualization per base input per category
    for base_key, by_category in samples_by_base.items():
        for category in ["noise", "ood_positive", "ood_negative"]:
            all_samples = by_category[category]
            if not all_samples:
                continue

            # Prioritize disagreeing samples (gate vs subcircuit differ)
            # For ood_negative, sort by absolute value
            sort_key = (lambda s: abs(s.noise_magnitude)) if category == "ood_negative" else (lambda s: s.noise_magnitude)
            disagree = sorted(
                [s for s in all_samples if not s.agreement_bit],
                key=sort_key
            )
            agree = sorted(
                [s for s in all_samples if s.agreement_bit],
                key=sort_key
            )

            # Take all disagreeing samples first, then fill with agreeing
            if len(disagree) >= n_samples_per_grid:
                # Evenly distributed from disagreeing samples
                d_idx = np.linspace(0, len(disagree) - 1, n_samples_per_grid, dtype=int)
                samples = [disagree[i] for i in d_idx]
            else:
                # Take all disagreeing, fill rest with agreeing
                samples = list(disagree)
                remaining = n_samples_per_grid - len(samples)
                if remaining > 0 and agree:
                    a_idx = np.linspace(0, len(agree) - 1, min(remaining, len(agree)), dtype=int)
                    samples.extend([agree[i] for i in a_idx])

            # Sort final selection by magnitude (abs for negative OOD)
            samples = sorted(samples, key=sort_key)
            n_samples = len(samples)

            if n_samples == 0:
                continue

            # Layout: n_samples rows x 2 cols (subcircuit | full)
            fig, axes = plt.subplots(n_samples, 2, figsize=(8, n_samples * 3))
            if n_samples == 1:
                axes = axes.reshape(1, -1)

            gt = int(all_samples[0].ground_truth)

            for i, sample in enumerate(samples):
                input_tensor = torch.tensor(
                    [sample.input_values], dtype=torch.float32
                )

                # Highlight disagreeing rows with red background
                if not sample.agreement_bit:
                    for col in range(2):
                        axes[i, col].set_facecolor("#FFEEEE")

                # Left: subcircuit
                with torch.no_grad():
                    sc_acts = subcircuit_model(input_tensor, return_activations=True)
                vmin, vmax = _symmetric_range(sc_acts)
                G, pos, colors, labels, text_colors, edge_labels, edge_w = (
                    _build_graph(sc_acts, circuit, weights, vmin, vmax)
                )
                _draw_graph_with_output_highlight(
                    axes[i, 0], G, pos, colors, labels, text_colors,
                    edge_labels, edge_w, sample.subcircuit_correct
                )

                # Right: full model
                with torch.no_grad():
                    full_acts = gate_model(input_tensor, return_activations=True)
                vmin, vmax = _symmetric_range(full_acts)
                G, pos, colors, labels, text_colors, edge_labels, edge_w = (
                    _build_graph(full_acts, full_circuit, weights, vmin, vmax)
                )
                _draw_graph_with_output_highlight(
                    axes[i, 1], G, pos, colors, labels, text_colors,
                    edge_labels, edge_w, sample.gate_correct
                )

                # Add "DISAGREE" label for disagreeing rows
                if not sample.agreement_bit:
                    axes[i, 0].text(
                        0.02, 0.98, "⚠ DISAGREE", transform=axes[i, 0].transAxes,
                        fontsize=8, color="red", fontweight="bold",
                        verticalalignment="top"
                    )

            # Column headers
            axes[0, 0].set_title("Subcircuit", fontsize=10, fontweight="bold")
            axes[0, 1].set_title("Full", fontsize=10, fontweight="bold")

            # Main title: (base_input) -> expected_output
            base_str = base_key.replace("_", ",")
            category_label = {
                "noise": "noise",
                "ood_positive": "ood: scale > 1",
                "ood_negative": "ood: scale < 0",
            }.get(category, category)
            fig.suptitle(
                f"({base_str}) → {gt}  [{category_label}]",
                fontsize=14, fontweight="bold"
            )
            plt.tight_layout()

            filename = f"{base_key}_{category}.png"
            path = os.path.join(circuit_viz_dir, filename)
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths[filename] = path

    return paths


def visualize_robustness_summary(
    robustness: RobustnessMetrics,
    output_dir: str,
    gate_name: str = "",
) -> str:
    """
    Visualize robustness summary as a single overview chart.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    prefix = f"{gate_name} - " if gate_name else ""

    # Top-left: Noise stats summary
    ax = axes[0, 0]
    categories = ["Gate\nAcc", "SC\nAcc", "Bit\nAgree"]
    values = [
        robustness.noise_gate_accuracy,
        robustness.noise_subcircuit_accuracy,
        robustness.noise_agreement_bit,
    ]
    colors = ["steelblue", "coral", "purple"]
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Rate")
    ax.set_title(
        f"Noise Summary (MSE: {robustness.noise_mse_mean:.4f})", fontweight="bold"
    )
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)

    # Top-right: OOD stats summary
    ax = axes[0, 1]
    values = [
        robustness.ood_gate_accuracy,
        robustness.ood_subcircuit_accuracy,
        robustness.ood_agreement_bit,
    ]
    ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Rate")
    ax.set_title(f"OOD Summary (MSE: {robustness.ood_mse_mean:.4f})", fontweight="bold")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)

    # Bottom-left: Noise scatter by magnitude
    ax = axes[1, 0]
    if robustness.noise_samples:
        bin_centers, gate_accs, sc_accs, bit_agrees, _ = _bin_samples_by_magnitude(
            robustness.noise_samples, n_bins=15
        )
        ax.plot(
            bin_centers,
            gate_accs,
            "o-",
            color="steelblue",
            label="Gate Acc",
            markersize=4,
        )
        ax.plot(bin_centers, sc_accs, "s-", color="coral", label="SC Acc", markersize=4)
        ax.plot(
            bin_centers,
            bit_agrees,
            "^-",
            color="purple",
            label="Bit Agree",
            markersize=4,
        )
        ax.set_xlabel("Noise Magnitude")
        ax.set_ylabel("Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
    ax.set_title("Noise: Metrics vs Magnitude", fontweight="bold")

    # Bottom-right: OOD scatter by magnitude
    ax = axes[1, 1]
    if robustness.ood_samples:
        bin_centers, gate_accs, sc_accs, bit_agrees, _ = _bin_samples_by_magnitude(
            robustness.ood_samples, n_bins=15
        )
        ax.plot(
            bin_centers,
            gate_accs,
            "o-",
            color="steelblue",
            label="Gate Acc",
            markersize=4,
        )
        ax.plot(bin_centers, sc_accs, "s-", color="coral", label="SC Acc", markersize=4)
        ax.plot(
            bin_centers,
            bit_agrees,
            "^-",
            color="purple",
            label="Bit Agree",
            markersize=4,
        )
        ax.set_xlabel("Perturbation Magnitude")
        ax.set_ylabel("Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
    ax.set_title("OOD: Metrics vs Magnitude", fontweight="bold")

    fig.suptitle(
        f"{prefix}Robustness Analysis (Agreement: {robustness.overall_robustness:.1%})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------ FAITHFULNESS VISUALIZATION ------------------


def _format_patch_label(patch_key: str) -> str:
    """Convert patch key to clean label like 'L1[0]'."""
    import re
    layer_m = re.search(r"layers=\((\d+),?\)", patch_key)
    idx_m = re.search(r"indices=\(([^)]*)\)", patch_key)
    layer = layer_m.group(1) if layer_m else "?"
    idx = idx_m.group(1).replace(",", "").strip() if idx_m else ""
    return f"L{layer}[{idx}]" if idx else f"L{layer}"


def _plot_patch_agreement_scatter(
    samples: list[InterventionSample],
    ax,
    xlabel: str,
    title: str,
) -> None:
    """Plot agreement scatter for intervention samples."""
    if not samples:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
        ax.set_title(title)
        return

    iv_mags = [sum(abs(v) for v in s.intervention_values) / max(len(s.intervention_values), 1) for s in samples]
    agree_y = [JITTER["correct"] if s.bit_agreement else JITTER["incorrect"] for s in samples]

    ax.scatter(iv_mags, agree_y, alpha=0.5, s=30, c=COLORS["agreement"], marker=MARKERS["agreement"],
               label="Intervention")
    ax.axhline(y=1, color=COLORS["correct"], linestyle="--", alpha=0.5, linewidth=1, label="Agree")
    ax.axhline(y=0, color=COLORS["incorrect"], linestyle="--", alpha=0.5, linewidth=1, label="Disagree")
    ax.grid(alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Agree (1) / Disagree (0)", fontsize=12)
    ax.set_ylim(-0.2, 1.2)

    # Add summary stats
    n_agree = sum(1 for s in samples if s.bit_agreement)
    agree_rate = n_agree / len(samples) * 100
    ax.set_title(f"{title}\n(Agreement: {agree_rate:.0f}%)", fontweight="bold")


def _plot_patch_mse_scatter(
    samples: list[InterventionSample],
    ax,
    xlabel: str,
    title: str,
) -> None:
    """Plot MSE scatter for intervention samples."""
    if not samples:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
        ax.set_title(title)
        return

    iv_mags = [sum(abs(v) for v in s.intervention_values) / max(len(s.intervention_values), 1) for s in samples]
    mses = [s.mse for s in samples]

    ax.scatter(iv_mags, mses, alpha=0.6, s=40, c=COLORS["mse"], marker=MARKERS["mse"])

    # Add trend line if enough points
    if len(iv_mags) >= 3:
        import numpy as np
        z = np.polyfit(iv_mags, mses, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(iv_mags), max(iv_mags), 50)
        ax.plot(x_line, p(x_line), "--", color=COLORS["mse"], alpha=0.5, linewidth=1.5, label="Trend")

    ax.grid(alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("MSE (gate vs subcircuit)", fontsize=12)

    # Add summary stats
    mean_mse = sum(mses) / len(mses)
    ax.set_title(f"{title}\n(Mean MSE: {mean_mse:.2e})", fontweight="bold")


def _plot_patch_stats_set(
    stats: dict,
    output_dir: str,
    prefix: str,
    circuit_type: str,  # "in_circuit" or "out_circuit"
) -> dict[str, str]:
    """Plot individual samples for each patch (not aggregated stats)."""
    paths = {}

    if not stats:
        return paths

    os.makedirs(output_dir, exist_ok=True)

    # Collect ALL samples across patches for combined view
    all_samples = []
    for patch_key, patch_stats in stats.items():
        patch_label = _format_patch_label(patch_key)
        for s in patch_stats.samples:
            all_samples.append((patch_label, s))

    if not all_samples:
        return paths

    # 1. Combined scatter: all samples colored by patch
    color = COLORS["in_circuit"] if circuit_type == "in_circuit" else COLORS["out_circuit"]
    circuit_label = "In-Circuit" if circuit_type == "in_circuit" else "Out-Circuit"

    # Agreement scatter - all samples
    fig, ax = plt.subplots(figsize=(10, 6))
    patch_labels = list(set(pl for pl, _ in all_samples))
    cmap = plt.cm.get_cmap('tab10', len(patch_labels))

    for idx, pl in enumerate(patch_labels):
        patch_samples = [s for label, s in all_samples if label == pl]
        iv_mags = [sum(abs(v) for v in s.intervention_values) / max(len(s.intervention_values), 1) for s in patch_samples]
        agree_y = [1.05 if s.bit_agreement else -0.05 for s in patch_samples]
        ax.scatter(iv_mags, agree_y, alpha=0.6, s=40, c=[cmap(idx)], label=pl)

    ax.axhline(y=1, color=COLORS["correct"], linestyle="--", alpha=0.5)
    ax.axhline(y=0, color=COLORS["incorrect"], linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean |Intervention Value|")
    ax.set_ylabel("Agree (1) / Disagree (0)")
    ax.set_ylim(-0.2, 1.2)
    ax.legend(loc="upper right", fontsize=8)
    n_agree = sum(1 for _, s in all_samples if s.bit_agreement)
    ax.set_title(f"{prefix}{circuit_label} Samples (n={len(all_samples)}, agree={n_agree})", fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    paths["agreement_samples"] = _save_figure(fig, output_dir, f"{circuit_type}_agreement_samples.png")

    # MSE scatter - all samples
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, pl in enumerate(patch_labels):
        patch_samples = [s for label, s in all_samples if label == pl]
        iv_mags = [sum(abs(v) for v in s.intervention_values) / max(len(s.intervention_values), 1) for s in patch_samples]
        mses = [s.mse for s in patch_samples]
        ax.scatter(iv_mags, mses, alpha=0.6, s=40, c=[cmap(idx)], label=pl)

    ax.set_xlabel("Mean |Intervention Value|")
    ax.set_ylabel("MSE (gate vs subcircuit)")
    ax.legend(loc="upper right", fontsize=8)
    mean_mse = sum(s.mse for _, s in all_samples) / len(all_samples)
    ax.set_title(f"{prefix}{circuit_label} MSE (mean={mean_mse:.2e})", fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    paths["mse_samples"] = _save_figure(fig, output_dir, f"{circuit_type}_mse_samples.png")

    # 2. Per-patch individual sample plots
    paths["per_patch"] = {}
    for patch_key, patch_stats in stats.items():
        samples = patch_stats.samples
        if not samples:
            continue

        patch_label = _format_patch_label(patch_key)
        patch_folder = patch_label.replace("[", "_").replace("]", "")

        patch_dir = os.path.join(output_dir, patch_folder)
        os.makedirs(patch_dir, exist_ok=True)
        patch_paths = {}

        # Agreement scatter
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_patch_agreement_scatter(samples, ax, "Mean |IV|", f"{patch_label}")
        patch_paths["agreement"] = _save_figure(fig, patch_dir, "agreement.png")

        # MSE scatter
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_patch_mse_scatter(samples, ax, "Mean |IV|", f"{patch_label}")
        patch_paths["mse"] = _save_figure(fig, patch_dir, "mse.png")

        paths["per_patch"][patch_folder] = patch_paths

    return paths


def visualize_faithfulness_stats(
    faithfulness: FaithfulnessMetrics,
    output_dir: str,
    gate_name: str = "",
) -> dict[str, str]:
    """
    Visualize faithfulness metrics as scatter plots and bar charts.

    Creates faithfulness/stats/ folder with:
    - in_circuit_patch_similarity.png
    - out_circuit_patch_similarity.png
    - counterfactual_faithfulness.png
    - per_patch/{patch_name}/ subfolders with agreement.png, mse.png
    """
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    prefix = f"{gate_name} - " if gate_name else ""

    paths = {}

    # In-circuit stats
    if faithfulness.in_circuit_stats:
        in_paths = _plot_patch_stats_set(
            faithfulness.in_circuit_stats,
            os.path.join(stats_dir, "in_circuit"),
            prefix,
            "in_circuit",
        )
        paths["in_circuit"] = in_paths

    # Out-circuit stats
    if faithfulness.out_circuit_stats:
        out_paths = _plot_patch_stats_set(
            faithfulness.out_circuit_stats,
            os.path.join(stats_dir, "out_circuit"),
            prefix,
            "out_circuit",
        )
        paths["out_circuit"] = out_paths

    # Counterfactual faithfulness - separate visualizations for out-circuit and in-circuit
    has_out = bool(faithfulness.out_counterfactual_effects)
    has_in = bool(faithfulness.in_counterfactual_effects)

    if has_out or has_in:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Left: Out-circuit faithfulness (tests sufficiency)
        ax = axes[0]
        if has_out:
            out_scores = [c.faithfulness_score for c in faithfulness.out_counterfactual_effects]
            bars = ax.bar(range(len(out_scores)), out_scores, color=COLORS["out_circuit"], alpha=0.8)
            ax.axhline(y=1.0, color=COLORS["correct"], linestyle="--", alpha=0.5, label="Perfect (1.0)")
            ax.set_xlabel("Counterfactual Pair Index")
            ax.set_ylabel("Faithfulness Score")
            out_mean = sum(out_scores) / len(out_scores) if out_scores else 0
            ax.set_title(f"{prefix}Out-Circuit (Sufficiency)\nMean: {out_mean:.3f}", fontweight="bold")
            ax.legend()
            ax.set_ylim(-0.2, 1.3)
            for i, (bar, score) in enumerate(zip(bars, out_scores)):
                if abs(score - 1.0) > 0.02:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f"{score:.2f}", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(0.5, 0.5, "No out-circuit data", ha="center", va="center")
            ax.axis("off")

        # Middle: In-circuit faithfulness (tests necessity)
        ax = axes[1]
        if has_in:
            in_scores = [c.faithfulness_score for c in faithfulness.in_counterfactual_effects]
            bars = ax.bar(range(len(in_scores)), in_scores, color=COLORS["in_circuit"], alpha=0.8)
            ax.axhline(y=0.0, color=COLORS["correct"], linestyle="--", alpha=0.5, label="Expected (0.0)")
            ax.set_xlabel("Counterfactual Pair Index")
            ax.set_ylabel("Faithfulness Score")
            in_mean = sum(in_scores) / len(in_scores) if in_scores else 0
            ax.set_title(f"{prefix}In-Circuit (Necessity)\nMean: {in_mean:.3f}", fontweight="bold")
            ax.legend()
            ax.set_ylim(-0.2, 1.3)
            for i, (bar, score) in enumerate(zip(bars, in_scores)):
                if abs(score) > 0.02:
                    y_pos = bar.get_height() + 0.02 if score >= 0 else bar.get_height() - 0.08
                    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                           f"{score:.2f}", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(0.5, 0.5, "No in-circuit data", ha="center", va="center")
            ax.axis("off")

        # Right: Summary comparison
        ax = axes[2]
        out_stayed = sum(1 for c in faithfulness.out_counterfactual_effects if not c.output_changed_to_corrupted) if has_out else 0
        out_changed = len(faithfulness.out_counterfactual_effects) - out_stayed if has_out else 0
        in_stayed = sum(1 for c in faithfulness.in_counterfactual_effects if not c.output_changed_to_corrupted) if has_in else 0
        in_changed = len(faithfulness.in_counterfactual_effects) - in_stayed if has_in else 0

        x = [0, 1]
        width = 0.35
        ax.bar([i - width/2 for i in x], [out_stayed, in_stayed], width, label="Stayed Clean", color=COLORS["correct"], alpha=0.8)
        ax.bar([i + width/2 for i in x], [out_changed, in_changed], width, label="Changed", color=COLORS["counterfactual"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Out-Circuit\n(expect: stay)", "In-Circuit\n(expect: change)"])
        ax.set_ylabel("Count")
        ax.set_title(f"{prefix}Response Summary", fontweight="bold")
        ax.legend()

        # Add counts on bars
        for i, (stay, chg) in enumerate([(out_stayed, out_changed), (in_stayed, in_changed)]):
            ax.text(i - width/2, stay + 0.2, str(stay), ha="center", fontweight="bold", fontsize=9)
            ax.text(i + width/2, chg + 0.2, str(chg), ha="center", fontweight="bold", fontsize=9)

        plt.tight_layout()
        paths["counterfactual"] = _save_figure(fig, stats_dir, "counterfactual_faithfulness.png")

    # Legacy: combined counterfactual effects (backwards compatibility)
    elif faithfulness.counterfactual_effects:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        scores = [c.faithfulness_score for c in faithfulness.counterfactual_effects]
        bars = ax.bar(range(len(scores)), scores, color=COLORS["faithfulness"], alpha=0.8)
        ax.axhline(y=1.0, color=COLORS["correct"], linestyle="--", alpha=0.5, label="Perfect (1.0)")
        ax.set_xlabel("Counterfactual Pair Index")
        ax.set_ylabel("Faithfulness Score")
        mean_score = sum(scores) / len(scores)
        ax.set_title(f"{prefix}Counterfactual Faithfulness (Legacy)\n(Mean: {mean_score:.3f})", fontweight="bold")
        ax.legend()

        ax = axes[1]
        changed = sum(1 for c in faithfulness.counterfactual_effects if c.output_changed_to_corrupted)
        not_changed = len(faithfulness.counterfactual_effects) - changed
        ax.bar(["Changed", "Stayed"], [changed, not_changed],
               color=[COLORS["counterfactual"], COLORS["in_circuit"]], alpha=0.8)
        ax.set_ylabel("Count")
        ax.set_title(f"{prefix}Counterfactual Response (Legacy)", fontweight="bold")
        plt.tight_layout()
        paths["counterfactual"] = _save_figure(fig, stats_dir, "counterfactual_faithfulness.png")

    return paths


def visualize_faithfulness_summary(
    faithfulness: FaithfulnessMetrics,
    output_dir: str,
    gate_name: str = "",
) -> str:
    """
    Visualize faithfulness summary as a single overview chart.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    prefix = f"{gate_name} - " if gate_name else ""

    # Top-left: In-circuit vs Out-circuit similarity comparison
    ax = axes[0, 0]
    categories = ["In-Circuit\nSimilarity", "Out-Circuit\nSimilarity", "Faithfulness\nScore"]
    values = [
        faithfulness.mean_in_circuit_similarity,
        faithfulness.mean_out_circuit_similarity,
        faithfulness.mean_faithfulness_score,
    ]
    colors = [COLORS["in_circuit"], COLORS["out_circuit"], COLORS["faithfulness"]]
    ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_ylim(0, 1.5)
    ax.axhline(y=1.0, color=COLORS["correct"], linestyle="--", alpha=0.5)
    ax.set_ylabel("Score")
    ax.set_title("Faithfulness Metrics Summary", fontweight="bold")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # Top-right: Per-patch in-circuit similarity breakdown
    ax = axes[0, 1]
    if faithfulness.in_circuit_stats:
        patch_keys = list(faithfulness.in_circuit_stats.keys())[:10]  # Limit to 10
        sims = [faithfulness.in_circuit_stats[k].mean_bit_similarity for k in patch_keys]
        labels = [_format_patch_label(k) for k in patch_keys]
        n_bars = len(patch_keys)
        x_pos = range(len(patch_keys))
        ax.bar(x_pos, sims, color=COLORS["in_circuit"], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45 if n_bars > 5 else 0, ha="right" if n_bars > 5 else "center", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color=COLORS["correct"], linestyle="--", alpha=0.5)
        ax.set_ylabel("Mean Bit Similarity")
        ax.set_title("In-Circuit Patches", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No in-circuit patches", ha="center", va="center")
        ax.axis("off")

    # Bottom-left: Per-patch out-circuit similarity breakdown
    ax = axes[1, 0]
    if faithfulness.out_circuit_stats:
        patch_keys = list(faithfulness.out_circuit_stats.keys())[:10]
        sims = [faithfulness.out_circuit_stats[k].mean_bit_similarity for k in patch_keys]
        labels = [_format_patch_label(k) for k in patch_keys]
        n_bars = len(patch_keys)
        bar_width = min(0.8, 0.3 + 0.1 * n_bars)  # Narrower for few bars
        x_pos = range(len(patch_keys))
        ax.bar(x_pos, sims, color=COLORS["out_circuit"], alpha=0.8, width=bar_width)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45 if n_bars > 3 else 0, ha="right" if n_bars > 3 else "center")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color=COLORS["correct"], linestyle="--", alpha=0.5)
        ax.set_ylabel("Mean Bit Similarity")
        ax.set_title("Out-Circuit Patches", fontweight="bold")
        # Adjust x-axis limits to prevent bar from spanning full width
        ax.set_xlim(-0.5, max(0.5, n_bars - 0.5))
    else:
        ax.text(0.5, 0.5, "No out-circuit patches", ha="center", va="center")
        ax.axis("off")

    # Bottom-right: Counterfactual faithfulness histogram (separate in/out)
    ax = axes[1, 1]
    has_out = bool(faithfulness.out_counterfactual_effects)
    has_in = bool(faithfulness.in_counterfactual_effects)
    if has_out or has_in:
        out_scores = [c.faithfulness_score for c in faithfulness.out_counterfactual_effects] if has_out else []
        in_scores = [c.faithfulness_score for c in faithfulness.in_counterfactual_effects] if has_in else []

        if out_scores:
            ax.hist(out_scores, bins=10, color=COLORS["out_circuit"], alpha=0.6, edgecolor="black", label="Out-circuit (sufficiency)")
        if in_scores:
            ax.hist(in_scores, bins=10, color=COLORS["in_circuit"], alpha=0.6, edgecolor="black", label="In-circuit (necessity)")

        ax.axvline(x=1.0, color=COLORS["correct"], linestyle="--", linewidth=2, label="Perfect sufficiency (1.0)")
        ax.axvline(x=0.0, color="gray", linestyle="--", linewidth=2, label="Perfect necessity (0.0)")
        ax.set_xlabel("Faithfulness Score")
        ax.set_ylabel("Count")
        ax.set_title("Counterfactual Distribution (In vs Out)", fontweight="bold")
        ax.legend(fontsize=7)
    elif faithfulness.counterfactual_effects:
        # Legacy fallback
        scores = [c.faithfulness_score for c in faithfulness.counterfactual_effects]
        ax.hist(scores, bins=10, color=COLORS["faithfulness"], alpha=0.8, edgecolor="black")
        ax.axvline(x=1.0, color=COLORS["correct"], linestyle="--", linewidth=2, label="Perfect (1.0)")
        ax.set_xlabel("Faithfulness Score")
        ax.set_ylabel("Count")
        ax.set_title("Counterfactual Distribution (Legacy)", fontweight="bold")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No counterfactual data", ha="center", va="center")
        ax.axis("off")

    fig.suptitle(
        f"{prefix}Faithfulness Analysis (Overall: {faithfulness.overall_faithfulness:.2f})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_graph_with_intervention(
    activations, circuit, weights_per_layer, vmin, vmax,
    patch_layer: int = -1, patch_indices: list[int] = None,
    intervention_values: list[float] = None
):
    """Build graph with intervention highlighted (orange border on patched neurons)."""
    G = nx.DiGraph()
    pos, node_colors, labels, text_colors = {}, [], {}, {}
    patch_indices = patch_indices or []
    intervention_values = intervention_values or []

    max_width = max(act.shape[-1] for act in activations)
    patched_nodes = set()

    # Nodes
    for layer_idx, layer_act in enumerate(activations):
        n = layer_act.shape[-1]
        y_off = -(max_width - n) / 2

        for node_idx in range(n):
            name = f"({layer_idx},{node_idx})"
            G.add_node(name)
            pos[name] = (layer_idx, y_off - node_idx)

            # Check if this node is being patched
            is_patched = (layer_idx == patch_layer and node_idx in patch_indices)
            if is_patched:
                patched_nodes.add(name)
                # Use intervention value if available
                iv_idx = patch_indices.index(node_idx) if node_idx in patch_indices else -1
                if iv_idx >= 0 and iv_idx < len(intervention_values):
                    val = intervention_values[iv_idx]
                else:
                    val = layer_act[0, node_idx].item()
            else:
                val = layer_act[0, node_idx].item()

            labels[name] = f"{val:.2f}"

            active = (
                layer_idx >= len(circuit.node_masks)
                or circuit.node_masks[layer_idx][node_idx] == 1
            )
            if active:
                color = _activation_to_color(val, vmin, vmax)
                node_colors.append(color)
                text_colors[name] = _text_color_for_background(color)
            else:
                node_colors.append("#d3d3d3")
                text_colors[name] = "gray"

    # Edges
    edge_labels = {}
    edge_weights = {}
    for layer_idx, mask in enumerate(circuit.edge_masks):
        w = weights_per_layer[layer_idx]
        for out_idx, row in enumerate(mask):
            for in_idx, active in enumerate(row):
                e = (f"({layer_idx},{in_idx})", f"({layer_idx + 1},{out_idx})")
                weight_val = w[out_idx, in_idx]
                G.add_edge(*e, active=active, weight=weight_val)
                if active == 1:
                    edge_labels[e] = f"{weight_val:.2f}"
                    edge_weights[e] = abs(weight_val)

    return G, pos, node_colors, labels, text_colors, edge_labels, edge_weights, patched_nodes


def _draw_graph_with_intervention_highlight(
    ax, G, pos, node_colors, labels, text_colors, edge_labels, edge_weights,
    output_correct: bool, patched_nodes: set
):
    """Draw graph with patched nodes highlighted with orange border."""
    active = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 1]
    inactive = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 0]

    # Edge widths
    if edge_weights:
        max_w = max(edge_weights.values()) if edge_weights.values() else 1.0
        max_w = max(max_w, 0.01)
        active_widths = [0.5 + 3.5 * (edge_weights.get(e, 0) / max_w) for e in active]
    else:
        active_widths = [1.0] * len(active)

    # Draw inactive edges
    nx.draw_networkx_edges(G, pos, edgelist=inactive, edge_color="#e0e0e0", width=0.5, ax=ax)

    # Draw active edges
    nx.draw_networkx_edges(G, pos, edgelist=active, edge_color="black", width=active_widths, ax=ax)

    # Draw nodes (non-patched)
    non_patched = [n for n in G.nodes() if n not in patched_nodes]
    non_patched_colors = [node_colors[list(G.nodes()).index(n)] for n in non_patched]
    nx.draw_networkx_nodes(G, pos, nodelist=non_patched, node_color=non_patched_colors,
                           node_size=700, ax=ax, edgecolors="black", linewidths=1)

    # Draw patched nodes with orange border
    if patched_nodes:
        patched_list = list(patched_nodes)
        patched_colors = [node_colors[list(G.nodes()).index(n)] for n in patched_list]
        nx.draw_networkx_nodes(G, pos, nodelist=patched_list, node_color=patched_colors,
                               node_size=700, ax=ax, edgecolors="#FF6600", linewidths=3)

    # Labels
    for node, (x, y) in pos.items():
        ax.text(x, y, labels[node], fontsize=7, ha="center", va="center",
               color=text_colors[node], fontweight="bold")

    # Output node border (last layer)
    last_layer = max(int(n.split(",")[0][1:]) for n in G.nodes())
    output_nodes = [n for n in G.nodes() if n.startswith(f"({last_layer},")]
    border_color = "green" if output_correct else "red"
    for node in output_nodes:
        x, y = pos[node]
        circle = plt.Circle((x, y), 0.15, fill=False, color=border_color, linewidth=3)
        ax.add_patch(circle)

    ax.axis("off")


def _apply_intervention_to_activations(
    model: MLP,
    input_tensor: torch.Tensor,
    patch_layer: int,
    patch_indices: list[int],
    intervention_values: list[float],
) -> list[torch.Tensor]:
    """Run model with intervention applied at specified layer/indices."""
    with torch.no_grad():
        acts = model(input_tensor, return_activations=True)
        # Apply intervention to the patched layer
        if 0 <= patch_layer < len(acts):
            for j, idx in enumerate(patch_indices):
                if j < len(intervention_values) and idx < acts[patch_layer].shape[-1]:
                    acts[patch_layer][0, idx] = intervention_values[j]
    return acts


def _find_changed_nodes(acts1: list[torch.Tensor], acts2: list[torch.Tensor], threshold: float = 0.1) -> set:
    """Find nodes where activation changed significantly between two forward passes."""
    changed = set()
    for layer_idx in range(min(len(acts1), len(acts2))):
        a1 = acts1[layer_idx][0]
        a2 = acts2[layer_idx][0]
        for node_idx in range(min(len(a1), len(a2))):
            if abs(a1[node_idx].item() - a2[node_idx].item()) > threshold:
                changed.add(f"({layer_idx},{node_idx})")
    return changed


def _draw_graph_with_changed_highlight(
    ax, G, pos, node_colors, labels, text_colors, edge_labels, edge_weights,
    output_correct: bool, changed_nodes: set, highlight_color: str = "#9C27B0"
):
    """Draw graph with changed nodes highlighted with colored border (purple by default)."""
    active = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 1]
    inactive = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 0]

    # Edge widths
    if edge_weights:
        max_w = max(edge_weights.values()) if edge_weights.values() else 1.0
        max_w = max(max_w, 0.01)
        active_widths = [0.5 + 3.5 * (edge_weights.get(e, 0) / max_w) for e in active]
    else:
        active_widths = [1.0] * len(active)

    # Draw inactive edges
    nx.draw_networkx_edges(G, pos, edgelist=inactive, edge_color="#e0e0e0", width=0.5, ax=ax)

    # Draw active edges
    nx.draw_networkx_edges(G, pos, edgelist=active, edge_color="black", width=active_widths, ax=ax)

    # Draw nodes (non-changed)
    non_changed = [n for n in G.nodes() if n not in changed_nodes]
    non_changed_colors = [node_colors[list(G.nodes()).index(n)] for n in non_changed]
    nx.draw_networkx_nodes(G, pos, nodelist=non_changed, node_color=non_changed_colors,
                           node_size=700, ax=ax, edgecolors="black", linewidths=1)

    # Draw changed nodes with colored border
    if changed_nodes:
        changed_list = [n for n in changed_nodes if n in G.nodes()]
        changed_colors = [node_colors[list(G.nodes()).index(n)] for n in changed_list]
        nx.draw_networkx_nodes(G, pos, nodelist=changed_list, node_color=changed_colors,
                               node_size=700, ax=ax, edgecolors=highlight_color, linewidths=3)

    # Labels
    for node, (x, y) in pos.items():
        ax.text(x, y, labels[node], fontsize=7, ha="center", va="center",
               color=text_colors[node], fontweight="bold")

    # Output node border (last layer)
    last_layer = max(int(n.split(",")[0][1:]) for n in G.nodes())
    output_nodes = [n for n in G.nodes() if n.startswith(f"({last_layer},")]
    border_color = "green" if output_correct else "red"
    for node in output_nodes:
        x, y = pos[node]
        circle = plt.Circle((x, y), 0.15, fill=False, color=border_color, linewidth=3)
        ax.add_patch(circle)

    ax.axis("off")


def visualize_faithfulness_circuit_samples(
    faithfulness: FaithfulnessMetrics,
    subcircuit_model: MLP,
    gate_model: MLP,
    circuit: Circuit,
    layer_weights: list[torch.Tensor],
    output_dir: str,
    n_samples_per_grid: int = 8,
) -> dict[str, str]:
    """
    Visualize circuit diagrams showing interventions.

    For in_circuit/out_circuit:
    - Each row: [Subcircuit + intervention | Full + same intervention]
    - Both models get the SAME intervention applied

    For counterfactual:
    - 4 circuits per sample:
      1. Full + clean input
      2. Full + corrupted input
      3. Subcircuit + clean input
      4. Subcircuit + intervention (patched activations)
    - Changed nodes circled in purple
    - Faithfulness score in title
    """
    circuit_viz_dir = os.path.join(output_dir, "circuit_viz")
    os.makedirs(circuit_viz_dir, exist_ok=True)
    paths = {}

    weights = [w.numpy() for w in layer_weights]
    layer_sizes = [layer_weights[0].shape[1]] + [w.shape[0] for w in layer_weights]
    full_circuit = Circuit.full(layer_sizes)

    def visualize_patch_circuits(stats: dict, circuit_type: str, out_dir: str) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        type_paths = {}

        for patch_key, patch_stats in stats.items():
            samples = patch_stats.samples
            if not samples:
                continue

            patch_label = _format_patch_label(patch_key)
            patch_folder = patch_label.replace("[", "_").replace("]", "")

            # Get patch info from first sample
            patch_layer = samples[0].patch_layer
            patch_indices = samples[0].patch_indices

            # Select samples (prioritize disagreeing)
            disagree = [s for s in samples if not s.bit_agreement]
            agree = [s for s in samples if s.bit_agreement]
            selected = (disagree[:n_samples_per_grid] if len(disagree) >= n_samples_per_grid
                       else disagree + agree[:n_samples_per_grid - len(disagree)])

            if not selected:
                continue

            n_samples = len(selected)
            fig, axes = plt.subplots(n_samples, 2, figsize=(10, n_samples * 3))
            if n_samples == 1:
                axes = axes.reshape(1, -1)

            for i, sample in enumerate(selected):
                if not sample.bit_agreement:
                    for col in range(2):
                        axes[i, col].set_facecolor("#FFEEEE")

                iv = sample.intervention_values
                iv_val = iv[0] if iv else 0.0
                sc_out = sample.subcircuit_output
                gate_out = sample.gate_output

                # Simple circuit drawing - no labels on nodes, just structure
                def draw_simple_circuit(ax, circ, output_val):
                    """Draw circuit structure with patched node highlighted, output colored."""
                    G = nx.DiGraph()
                    pos_dict = {}
                    node_colors_list = []
                    patched_set = set()

                    layer_sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]
                    max_width = max(layer_sizes)
                    output_layer = len(layer_sizes) - 1

                    for layer_idx, n_nodes in enumerate(layer_sizes):
                        y_off = -(max_width - n_nodes) / 2
                        for node_idx in range(n_nodes):
                            name = f"({layer_idx},{node_idx})"
                            G.add_node(name)
                            pos_dict[name] = (layer_idx, y_off - node_idx)

                            active = (layer_idx >= len(circ.node_masks) or
                                     circ.node_masks[layer_idx][node_idx] == 1)
                            is_patched = (layer_idx == patch_layer and node_idx in patch_indices)
                            is_output = (layer_idx == output_layer)

                            if is_patched:
                                patched_set.add(name)
                                node_colors_list.append(_activation_to_color(iv_val, -1, 1))
                            elif is_output:
                                node_colors_list.append(_activation_to_color(output_val, -1, 1))
                            elif active:
                                node_colors_list.append("#FFFFCC")
                            else:
                                node_colors_list.append("#d3d3d3")

                    for layer_idx, mask in enumerate(circ.edge_masks):
                        for out_idx, row in enumerate(mask):
                            for in_idx, active in enumerate(row):
                                G.add_edge(f"({layer_idx},{in_idx})", f"({layer_idx + 1},{out_idx})",
                                          active=active)

                    active_edges = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 1]
                    inactive_edges = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 0]

                    nx.draw_networkx_edges(G, pos_dict, edgelist=inactive_edges,
                                          edge_color="#e0e0e0", width=0.5, ax=ax)
                    nx.draw_networkx_edges(G, pos_dict, edgelist=active_edges,
                                          edge_color="black", width=1.5, ax=ax)

                    # Draw non-patched, non-output nodes
                    regular = [n for n in G.nodes() if n not in patched_set and
                              not n.startswith(f"({output_layer},")]
                    reg_colors = [node_colors_list[list(G.nodes()).index(n)] for n in regular]
                    nx.draw_networkx_nodes(G, pos_dict, nodelist=regular, node_color=reg_colors,
                                          node_size=500, ax=ax, edgecolors="black", linewidths=1)

                    # Draw patched nodes with orange border
                    if patched_set:
                        p_list = list(patched_set)
                        p_colors = [node_colors_list[list(G.nodes()).index(n)] for n in p_list]
                        nx.draw_networkx_nodes(G, pos_dict, nodelist=p_list, node_color=p_colors,
                                              node_size=500, ax=ax, edgecolors="#FF6600", linewidths=4)

                    # Draw output node with result border
                    out_nodes = [n for n in G.nodes() if n.startswith(f"({output_layer},")]
                    out_colors = [node_colors_list[list(G.nodes()).index(n)] for n in out_nodes]
                    border = "green" if sample.bit_agreement else "red"
                    nx.draw_networkx_nodes(G, pos_dict, nodelist=out_nodes, node_color=out_colors,
                                          node_size=500, ax=ax, edgecolors=border, linewidths=4)

                    ax.axis("off")

                # Draw both circuits
                draw_simple_circuit(axes[i, 0], circuit, sc_out)
                draw_simple_circuit(axes[i, 1], full_circuit, gate_out)

                # Add text annotations instead of node labels
                axes[i, 0].text(0.5, -0.05, f"IV:{iv_val:.2f} → {sc_out:.2f}",
                               transform=axes[i, 0].transAxes, ha="center", fontsize=9, fontweight="bold")
                axes[i, 1].text(0.5, -0.05, f"IV:{iv_val:.2f} → {gate_out:.2f}",
                               transform=axes[i, 1].transAxes, ha="center", fontsize=9, fontweight="bold")

            n_agree = sum(1 for s in selected if s.bit_agreement)
            axes[0, 0].set_title("Subcircuit", fontsize=10, fontweight="bold")
            axes[0, 1].set_title("Full Model", fontsize=10, fontweight="bold")
            fig.suptitle(f"{patch_label} | {circuit_type.replace('_', '-')} | agree: {n_agree}/{n_samples}",
                        fontsize=11, fontweight="bold")
            plt.tight_layout()

            path = os.path.join(out_dir, f"{patch_folder}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            type_paths[patch_folder] = path

        return type_paths

    # In-circuit and out-circuit
    if faithfulness.in_circuit_stats:
        paths["in_circuit"] = visualize_patch_circuits(
            faithfulness.in_circuit_stats, "in_circuit",
            os.path.join(circuit_viz_dir, "in_circuit")
        )

    if faithfulness.out_circuit_stats:
        paths["out_circuit"] = visualize_patch_circuits(
            faithfulness.out_circuit_stats, "out_circuit",
            os.path.join(circuit_viz_dir, "out_circuit")
        )

    # Helper to get out-circuit node names from circuit
    def get_out_circuit_nodes(circ: Circuit) -> set:
        """Get node names for out-of-circuit neurons (excluding input/output layers)."""
        out_nodes = set()
        for layer_idx in range(1, len(circ.node_masks) - 1):  # Skip input/output
            for node_idx, active in enumerate(circ.node_masks[layer_idx]):
                if active == 0:  # Out of circuit
                    out_nodes.add(f"({layer_idx},{node_idx})")
        return out_nodes

    def get_in_circuit_nodes(circ: Circuit) -> set:
        """Get node names for in-circuit neurons (excluding input/output layers)."""
        in_nodes = set()
        for layer_idx in range(1, len(circ.node_masks) - 1):  # Skip input/output
            for node_idx, active in enumerate(circ.node_masks[layer_idx]):
                if active == 1:  # In circuit
                    in_nodes.add(f"({layer_idx},{node_idx})")
        return in_nodes

    def draw_counterfactual_circuit(ax, acts, subcircuit, patched_nodes, vmin, vmax, title, output_correct):
        """Draw circuit showing in-circuit vs out-circuit nodes, with patched nodes highlighted."""
        G = nx.DiGraph()
        pos_dict = {}
        node_colors_list = []
        labels_dict = {}
        text_colors_dict = {}

        layer_sizes_local = [weights[0].shape[1]] + [w.shape[0] for w in weights]
        max_width = max(layer_sizes_local)
        output_layer = len(layer_sizes_local) - 1

        # Build node membership sets for subcircuit
        in_circuit_set = set()
        out_circuit_set = set()
        for layer_idx in range(1, len(subcircuit.node_masks) - 1):  # Skip input/output
            for node_idx, active in enumerate(subcircuit.node_masks[layer_idx]):
                name = f"({layer_idx},{node_idx})"
                if active == 1:
                    in_circuit_set.add(name)
                else:
                    out_circuit_set.add(name)

        for layer_idx, n_nodes in enumerate(layer_sizes_local):
            y_off = -(max_width - n_nodes) / 2
            for node_idx in range(n_nodes):
                name = f"({layer_idx},{node_idx})"
                G.add_node(name)
                pos_dict[name] = (layer_idx, y_off - node_idx)

                val = acts[layer_idx][0, node_idx].item() if layer_idx < len(acts) else 0
                color = _activation_to_color(val, vmin, vmax)
                node_colors_list.append(color)
                labels_dict[name] = f"{val:.2f}"
                text_colors_dict[name] = _text_color_for_background(color)

        # Draw edges based on subcircuit structure
        for layer_idx, mask in enumerate(subcircuit.edge_masks):
            for out_idx, row in enumerate(mask):
                for in_idx, active in enumerate(row):
                    G.add_edge(f"({layer_idx},{in_idx})", f"({layer_idx + 1},{out_idx})", active=active)

        active_edges = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 1]
        inactive_edges = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 0]

        nx.draw_networkx_edges(G, pos_dict, edgelist=inactive_edges, edge_color="#e0e0e0", width=0.5, ax=ax)
        nx.draw_networkx_edges(G, pos_dict, edgelist=active_edges, edge_color="black", width=1.5, ax=ax)

        # Input nodes (layer 0)
        input_nodes = [n for n in G.nodes() if n.startswith("(0,")]
        input_colors = [node_colors_list[list(G.nodes()).index(n)] for n in input_nodes]
        nx.draw_networkx_nodes(G, pos_dict, nodelist=input_nodes, node_color=input_colors,
                              node_size=600, ax=ax, edgecolors="black", linewidths=1)

        # Out-circuit nodes (thin gray border) - NOT in subcircuit
        out_list = [n for n in out_circuit_set if n in G.nodes() and n not in patched_nodes]
        if out_list:
            out_colors = [node_colors_list[list(G.nodes()).index(n)] for n in out_list]
            nx.draw_networkx_nodes(G, pos_dict, nodelist=out_list, node_color=out_colors,
                                  node_size=600, ax=ax, edgecolors="#888888", linewidths=1)

        # In-circuit nodes (thick black border) - IN subcircuit
        in_list = [n for n in in_circuit_set if n in G.nodes() and n not in patched_nodes]
        if in_list:
            in_colors = [node_colors_list[list(G.nodes()).index(n)] for n in in_list]
            nx.draw_networkx_nodes(G, pos_dict, nodelist=in_list, node_color=in_colors,
                                  node_size=600, ax=ax, edgecolors="black", linewidths=2.5)

        # Patched nodes (purple border) - activations being changed
        patched_list = [n for n in patched_nodes if n in G.nodes() and not n.startswith(f"({output_layer},")]
        if patched_list:
            p_colors = [node_colors_list[list(G.nodes()).index(n)] for n in patched_list]
            nx.draw_networkx_nodes(G, pos_dict, nodelist=patched_list, node_color=p_colors,
                                  node_size=600, ax=ax, edgecolors="#9C27B0", linewidths=4)  # Purple

        # Output node
        out_nodes = [n for n in G.nodes() if n.startswith(f"({output_layer},")]
        out_colors_list = [node_colors_list[list(G.nodes()).index(n)] for n in out_nodes]
        border = "green" if output_correct else "red"
        nx.draw_networkx_nodes(G, pos_dict, nodelist=out_nodes, node_color=out_colors_list,
                              node_size=600, ax=ax, edgecolors=border, linewidths=3)

        # Labels
        for node, (x, y) in pos_dict.items():
            ax.text(x, y, labels_dict[node], fontsize=6, ha="center", va="center",
                   color=text_colors_dict[node], fontweight="bold")

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    # Visualize OUT-circuit counterfactuals (tests sufficiency)
    if faithfulness.out_counterfactual_effects:
        cf_dir = os.path.join(circuit_viz_dir, "counterfactual_out")
        os.makedirs(cf_dir, exist_ok=True)
        out_circuit_nodes = get_out_circuit_nodes(circuit)

        for i, effect in enumerate(faithfulness.out_counterfactual_effects[:6]):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            clean_input = torch.tensor([effect.clean_input], dtype=torch.float32)
            corrupt_input = torch.tensor([effect.corrupted_input], dtype=torch.float32)

            with torch.no_grad():
                clean_acts = gate_model(clean_input, return_activations=True)
                corrupt_acts = gate_model(corrupt_input, return_activations=True)

            all_acts = clean_acts + corrupt_acts
            vmin, vmax = _symmetric_range(all_acts)

            # 1. Clean input - no patching
            draw_counterfactual_circuit(axes[0], clean_acts, circuit, set(), vmin, vmax,
                                       f"Clean Input → {effect.expected_clean_output:.2f}", True)

            # 2. Corrupted input - no patching
            draw_counterfactual_circuit(axes[1], corrupt_acts, circuit, set(), vmin, vmax,
                                       f"Corrupted Input → {effect.expected_corrupted_output:.2f}", True)

            # 3. Clean input + Patch out-circuit with corrupted activations
            patched_output_correct = not effect.output_changed_to_corrupted  # Faithful if NOT changed
            draw_counterfactual_circuit(axes[2], clean_acts, circuit, out_circuit_nodes, vmin, vmax,
                                       f"Clean + Patch(out) → {effect.actual_output:.2f}", patched_output_correct)

            clean_str = ",".join(f"{v:.0f}" for v in effect.clean_input)
            corrupt_str = ",".join(f"{v:.0f}" for v in effect.corrupted_input)
            fig.suptitle(f"Out-Circuit Counterfactual #{i} | ({clean_str})→({corrupt_str}) | Faith: {effect.faithfulness_score:.2f}",
                        fontsize=11, fontweight="bold")
            fig.text(0.5, 0.02, "Thick border=in-circuit | Thin gray=out-circuit | Purple=patched (activations changed) | Green/Red=output",
                    ha="center", fontsize=8)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(os.path.join(cf_dir, f"out_cf_{i}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        paths["counterfactual_out"] = cf_dir

    # Visualize IN-circuit counterfactuals (tests necessity)
    if faithfulness.in_counterfactual_effects:
        cf_dir = os.path.join(circuit_viz_dir, "counterfactual_in")
        os.makedirs(cf_dir, exist_ok=True)
        in_circuit_nodes = get_in_circuit_nodes(circuit)

        for i, effect in enumerate(faithfulness.in_counterfactual_effects[:6]):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            clean_input = torch.tensor([effect.clean_input], dtype=torch.float32)
            corrupt_input = torch.tensor([effect.corrupted_input], dtype=torch.float32)

            with torch.no_grad():
                clean_acts = gate_model(clean_input, return_activations=True)
                corrupt_acts = gate_model(corrupt_input, return_activations=True)

            all_acts = clean_acts + corrupt_acts
            vmin, vmax = _symmetric_range(all_acts)

            # 1. Clean input - no patching
            draw_counterfactual_circuit(axes[0], clean_acts, circuit, set(), vmin, vmax,
                                       f"Clean Input → {effect.expected_clean_output:.2f}", True)

            # 2. Corrupted input - no patching
            draw_counterfactual_circuit(axes[1], corrupt_acts, circuit, set(), vmin, vmax,
                                       f"Corrupted Input → {effect.expected_corrupted_output:.2f}", True)

            # 3. Clean input + Patch in-circuit with corrupted activations
            patched_output_correct = effect.output_changed_to_corrupted  # Expected to change if necessary
            draw_counterfactual_circuit(axes[2], clean_acts, circuit, in_circuit_nodes, vmin, vmax,
                                       f"Clean + Patch(in) → {effect.actual_output:.2f}", patched_output_correct)

            clean_str = ",".join(f"{v:.0f}" for v in effect.clean_input)
            corrupt_str = ",".join(f"{v:.0f}" for v in effect.corrupted_input)
            fig.suptitle(f"In-Circuit Counterfactual #{i} | ({clean_str})→({corrupt_str}) | Faith: {effect.faithfulness_score:.2f}",
                        fontsize=11, fontweight="bold")
            fig.text(0.5, 0.02, "Thick border=in-circuit | Thin gray=out-circuit | Purple=patched (activations changed) | Green/Red=output",
                    ha="center", fontsize=8)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(os.path.join(cf_dir, f"in_cf_{i}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        paths["counterfactual_in"] = cf_dir

    # Legacy: combined counterfactual effects (backwards compatibility)
    if faithfulness.counterfactual_effects and not faithfulness.out_counterfactual_effects:
        cf_dir = os.path.join(circuit_viz_dir, "counterfactual")
        os.makedirs(cf_dir, exist_ok=True)
        out_circuit_nodes = get_out_circuit_nodes(circuit)

        for i, effect in enumerate(faithfulness.counterfactual_effects[:6]):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            clean_input = torch.tensor([effect.clean_input], dtype=torch.float32)
            corrupt_input = torch.tensor([effect.corrupted_input], dtype=torch.float32)

            with torch.no_grad():
                clean_acts = gate_model(clean_input, return_activations=True)
                corrupt_acts = gate_model(corrupt_input, return_activations=True)

            all_acts = clean_acts + corrupt_acts
            vmin, vmax = _symmetric_range(all_acts)

            draw_counterfactual_circuit(axes[0], clean_acts, circuit, set(), vmin, vmax,
                                       f"Clean → {effect.expected_clean_output:.2f}", True)
            draw_counterfactual_circuit(axes[1], corrupt_acts, circuit, set(), vmin, vmax,
                                       f"Corrupted → {effect.expected_corrupted_output:.2f}", True)
            patched_output_correct = not effect.output_changed_to_corrupted
            draw_counterfactual_circuit(axes[2], clean_acts, circuit, out_circuit_nodes, vmin, vmax,
                                       f"Clean+Patch → {effect.actual_output:.2f}", patched_output_correct)

            clean_str = ",".join(f"{v:.0f}" for v in effect.clean_input)
            corrupt_str = ",".join(f"{v:.0f}" for v in effect.corrupted_input)
            fig.suptitle(f"Counterfactual #{i} | ({clean_str})→({corrupt_str}) | Faith: {effect.faithfulness_score:.2f}",
                        fontsize=11, fontweight="bold")
            fig.text(0.5, 0.02, "Thick border=in-circuit | Thin gray=out-circuit | Purple=patched", ha="center", fontsize=8)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(os.path.join(cf_dir, f"cf_{i}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        paths["counterfactual"] = cf_dir

    return paths


# ------------------ SPD VISUALIZATION ------------------


def visualize_spd_components(
    decomposed: DecomposedMLP | None, output_dir: str, gate_name: str = ""
) -> str | None:
    """Bar chart: SPD component importances."""
    if decomposed is None or decomposed.component_model is None:
        return None

    weights = _get_spd_component_weights(decomposed)
    if weights is None:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(weights)), weights, color="steelblue", alpha=0.8)
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Relative Weight Magnitude")
    prefix = f"{gate_name} - " if gate_name else ""
    ax.set_title(f"{prefix}SPD Component Importance")
    ax.set_xticks(range(len(weights)))
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "spd_components.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------ PROFILING VISUALIZATION ------------------


def visualize_profiling_timeline(
    profiling: ProfilingData,
    output_dir: str,
    filename: str = "timeline.png",
) -> str:
    """
    Visualize profiling events as a timeline/waterfall chart.

    Shows each phase as a horizontal bar with duration.
    """
    events = profiling.events
    if not events:
        return ""

    fig, ax = plt.subplots(figsize=(14, max(6, len(events) * 0.3)))

    # Build timeline bars from events
    bars = []
    for i, event in enumerate(events):
        bars.append(
            {
                "label": event.status,
                "start": event.timestamp_ms - event.elapsed_ms,
                "duration": event.elapsed_ms,
                "timestamp": event.timestamp_ms,
            }
        )

    # Sort by start time
    bars.sort(key=lambda x: x["start"])

    # Color mapping for different phases
    color_map = {
        "STARTED": "#4CAF50",  # Green
        "ENDED": "#2196F3",  # Blue
        "FINISHED": "#2196F3",  # Blue
        "SUCCESSFUL": "#8BC34A",  # Light green
        "GATE": "#FF9800",  # Orange
        "SPD": "#9C27B0",  # Purple
        "ROBUSTNESS": "#00BCD4",  # Cyan
        "FAITH": "#E91E63",  # Pink
    }

    def get_color(label: str) -> str:
        for key, color in color_map.items():
            if key in label:
                return color
        return "#757575"  # Gray default

    # Plot bars
    y_positions = range(len(bars))
    for i, bar in enumerate(bars):
        color = get_color(bar["label"])
        ax.barh(
            i, bar["duration"], left=bar["start"], color=color, alpha=0.8, height=0.6
        )

        # Add duration label
        duration_text = f"{bar['duration']:.0f}ms"
        ax.text(
            bar["start"] + bar["duration"] + 50,
            i,
            duration_text,
            va="center",
            fontsize=8,
            color="#666666",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([b["label"] for b in bars], fontsize=8)
    ax.set_xlabel("Time (ms)")
    ax.set_title(
        f"Trial Timeline (Total: {profiling.total_duration_ms:.0f}ms)",
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_profiling_phases(
    profiling: ProfilingData,
    output_dir: str,
    filename: str = "phases.png",
) -> str:
    """
    Visualize aggregated phase durations as a bar chart.

    Shows total time spent in each major phase.
    """
    phase_durations = profiling.phase_durations_ms
    if not phase_durations:
        return ""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort phases by duration (descending)
    sorted_phases = sorted(phase_durations.items(), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in sorted_phases]
    durations = [p[1] for p in sorted_phases]

    # Color by phase type
    colors = []
    for label in labels:
        if "SPD" in label:
            colors.append("#9C27B0")  # Purple
        elif "MLP" in label:
            colors.append("#4CAF50")  # Green
        elif "GATE" in label:
            colors.append("#FF9800")  # Orange
        elif "ROBUSTNESS" in label:
            colors.append("#00BCD4")  # Cyan
        elif "FAITH" in label:
            colors.append("#E91E63")  # Pink
        elif "CIRCUIT" in label:
            colors.append("#3F51B5")  # Indigo
        else:
            colors.append("#757575")  # Gray

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, durations, color=colors, alpha=0.8)

    # Add duration labels on bars
    for i, (label, duration) in enumerate(zip(labels, durations)):
        ax.text(
            i,
            duration + max(durations) * 0.02,
            f"{duration:.0f}ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Duration (ms)")
    ax.set_title(
        f"Phase Durations (Total: {profiling.total_duration_ms:.0f}ms)",
        fontweight="bold",
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_profiling_summary(
    profiling: ProfilingData,
    output_dir: str,
    filename: str = "summary.png",
) -> str:
    """
    Visualize profiling summary with pie chart of major phases.
    """
    phase_durations = profiling.phase_durations_ms
    if not phase_durations:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Group phases into categories
    categories = {
        "MLP Training": 0.0,
        "SPD Decomposition": 0.0,
        "Circuit Finding": 0.0,
        "Gate Analysis": 0.0,
        "Other": 0.0,
    }

    for phase, duration in phase_durations.items():
        if "MLP_TRAINING" in phase:
            categories["MLP Training"] += duration
        elif "SPD" in phase:
            categories["SPD Decomposition"] += duration
        elif "CIRCUIT" in phase:
            categories["Circuit Finding"] += duration
        elif "GATE" in phase or "ROBUSTNESS" in phase or "FAITH" in phase:
            categories["Gate Analysis"] += duration
        else:
            categories["Other"] += duration

    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v > 0}

    # Pie chart
    colors = ["#4CAF50", "#9C27B0", "#3F51B5", "#FF9800", "#757575"]
    axes[0].pie(
        categories.values(),
        labels=categories.keys(),
        autopct="%1.1f%%",
        colors=colors[: len(categories)],
        startangle=90,
    )
    axes[0].set_title("Time Distribution by Category", fontweight="bold")

    # Summary stats
    total_ms = profiling.total_duration_ms
    total_secs = total_ms / 1000
    axes[1].axis("off")
    summary_text = (
        f"Device: {profiling.device}\n\n"
        f"Total Duration: {total_secs:.2f}s ({total_ms:.0f}ms)\n\n"
        f"Phases:\n"
    )
    for cat, duration in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = (duration / total_ms * 100) if total_ms > 0 else 0
        summary_text += f"  • {cat}: {duration:.0f}ms ({pct:.1f}%)\n"

    axes[1].text(
        0.1,
        0.9,
        summary_text,
        transform=axes[1].transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------ MAIN FUNCTION ------------------


def visualize_experiment(result: ExperimentResult, run_dir: str | Path) -> dict:
    """
    Generate all visualizations for experiment using pre-computed data.

    IMPORTANT: This function does NOT run any models. All data comes from:
    - trial.canonical_activations: Pre-computed activations for binary inputs
    - trial.layer_weights: Weight matrices from the trained model
    - trial.metrics: Robustness and faithfulness results

    Returns paths dict.
    """
    os.makedirs(run_dir, exist_ok=True)

    viz_paths = {}

    for trial_id, trial in result.trials.items():
        subcircuits = [Circuit.from_dict(s) for s in trial.subcircuits]
        viz_paths[trial_id] = {}
        trial_dir = os.path.join(run_dir, trial_id)

        # --- profiling/ (timing visualizations) ---
        if trial.profiling and trial.profiling.events:
            profiling_dir = os.path.join(trial_dir, "profiling")
            os.makedirs(profiling_dir, exist_ok=True)
            viz_paths[trial_id]["profiling"] = {}

            if path := visualize_profiling_timeline(trial.profiling, profiling_dir):
                viz_paths[trial_id]["profiling"]["timeline"] = path
            if path := visualize_profiling_phases(trial.profiling, profiling_dir):
                viz_paths[trial_id]["profiling"]["phases"] = path
            if path := visualize_profiling_summary(trial.profiling, profiling_dir):
                viz_paths[trial_id]["profiling"]["summary"] = path

        # Extract pre-computed data
        canonical_activations = trial.canonical_activations or {}
        layer_weights = trial.layer_weights or []
        gate_names = trial.setup.model_params.logic_gates

        if not canonical_activations or not layer_weights:
            continue

        # Full circuit (all edges/nodes active)
        layer_sizes = [layer_weights[0].shape[1]] + [w.shape[0] for w in layer_weights]
        full_circuit = Circuit.full(layer_sizes)

        # --- all_gates/ (full multi-gate model) ---
        folder = os.path.join(trial_dir, "all_gates")
        os.makedirs(folder, exist_ok=True)
        viz_paths[trial_id]["all_gates"] = {}
        all_gates_label = "All Gates (" + ", ".join(gate_names) + ")"

        act_path = visualize_circuit_activations_from_data(
            canonical_activations,
            layer_weights,
            full_circuit,
            folder,
            gate_name=all_gates_label,
        )
        viz_paths[trial_id]["all_gates"]["activations"] = act_path

        if trial.decomposed_model:
            if path := visualize_spd_components(
                trial.decomposed_model, folder, gate_name=all_gates_label
            ):
                viz_paths[trial_id]["all_gates"]["spd"] = path

        # --- Per-gate visualization ---
        for gate_idx, gname in enumerate(gate_names):
            folder = os.path.join(trial_dir, gname, "full")
            os.makedirs(folder, exist_ok=True)
            viz_paths[trial_id].setdefault(gname, {})["full"] = {}

            gate_label = f"{gname} (Full)"

            # Circuit activations (using full circuit, same activations)
            act_path = visualize_circuit_activations_from_data(
                canonical_activations,
                layer_weights,
                full_circuit,
                folder,
                gate_name=gate_label,
            )
            viz_paths[trial_id][gname]["full"]["activations"] = act_path

            # SPD components
            if gname in trial.decomposed_gate_models:
                decomposed = trial.decomposed_gate_models[gname]
                if path := visualize_spd_components(
                    decomposed, folder, gate_name=gate_label
                ):
                    viz_paths[trial_id][gname]["full"]["spd"] = path

        # --- Subcircuit visualization ---
        # Get gate models for creating subcircuit models
        gate_models = trial.model.separate_into_k_mlps() if trial.model else []

        for gate_idx, gname in enumerate(gate_names):
            # Use per_gate_bests to iterate over all best subcircuits (not just decomposed ones)
            best_indices = trial.metrics.per_gate_bests.get(gname, [])
            if not best_indices:
                continue

            gate_model = gate_models[gate_idx] if gate_idx < len(gate_models) else None
            bests_robust = trial.metrics.per_gate_bests_robust.get(gname, [])
            decomposed_indices = trial.decomposed_subcircuit_indices.get(gname, [])

            for i, sc_idx in enumerate(best_indices):
                circuit = subcircuits[sc_idx]
                folder = os.path.join(trial_dir, gname, str(sc_idx))
                os.makedirs(folder, exist_ok=True)
                viz_paths[trial_id].setdefault(gname, {})[sc_idx] = {}

                sc_label = f"{gname} (SC #{sc_idx})"

                # Static circuit structure
                path = os.path.join(folder, "circuit.png")
                circuit.visualize(file_path=path, node_size="small")
                viz_paths[trial_id][gname][sc_idx]["circuit"] = path

                # Circuit activations
                act_path = visualize_circuit_activations_from_data(
                    canonical_activations,
                    layer_weights,
                    circuit,
                    folder,
                    gate_name=sc_label,
                )
                viz_paths[trial_id][gname][sc_idx]["activations"] = act_path

                # Robustness visualization in robustness/ subfolder
                if i < len(bests_robust):
                    robustness_data = bests_robust[i]
                    robustness_dir = os.path.join(folder, "robustness")
                    os.makedirs(robustness_dir, exist_ok=True)
                    viz_paths[trial_id][gname][sc_idx]["robustness"] = {}

                    # Summary overview
                    summary_path = visualize_robustness_summary(
                        robustness_data, robustness_dir, gate_name=sc_label
                    )
                    viz_paths[trial_id][gname][sc_idx]["robustness"]["summary"] = (
                        summary_path
                    )

                    # Stats curves (accuracy and agreement vs level)
                    stats_paths = visualize_robustness_curves(
                        robustness_data, robustness_dir, gate_name=sc_label
                    )
                    viz_paths[trial_id][gname][sc_idx]["robustness"]["stats"] = (
                        stats_paths
                    )

                    # Circuit visualizations for sample inputs
                    if gate_model is not None:
                        subcircuit_model = gate_model.separate_subcircuit(circuit)
                        circuit_viz_paths = visualize_robustness_circuit_samples(
                            robustness_data,
                            subcircuit_model,
                            gate_model,
                            circuit,
                            layer_weights,
                            robustness_dir,
                        )
                    else:
                        circuit_viz_paths = {}
                    viz_paths[trial_id][gname][sc_idx]["robustness"]["circuit_viz"] = (
                        circuit_viz_paths
                    )

                # Faithfulness visualization in faithfulness/ subfolder
                bests_faith = trial.metrics.per_gate_bests_faith.get(gname, [])
                if i < len(bests_faith):
                    faithfulness_data = bests_faith[i]
                    faithfulness_dir = os.path.join(folder, "faithfulness")
                    os.makedirs(faithfulness_dir, exist_ok=True)
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"] = {}

                    # Summary overview
                    summary_path = visualize_faithfulness_summary(
                        faithfulness_data, faithfulness_dir, gate_name=sc_label
                    )
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"]["summary"] = (
                        summary_path
                    )

                    # Stats (in/out circuit patches, counterfactual)
                    stats_paths = visualize_faithfulness_stats(
                        faithfulness_data, faithfulness_dir, gate_name=sc_label
                    )
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"]["stats"] = (
                        stats_paths
                    )

                    # Circuit visualizations for interventions
                    if gate_model is not None:
                        subcircuit_model = gate_model.separate_subcircuit(circuit)
                        circuit_viz_paths = visualize_faithfulness_circuit_samples(
                            faithfulness_data,
                            subcircuit_model,
                            gate_model,
                            circuit,
                            layer_weights,
                            faithfulness_dir,
                        )
                    else:
                        circuit_viz_paths = {}
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"]["circuit_viz"] = (
                        circuit_viz_paths
                    )

                # SPD
                if gname in trial.decomposed_subcircuits:
                    if sc_idx in trial.decomposed_subcircuits[gname]:
                        decomposed = trial.decomposed_subcircuits[gname][sc_idx]
                        if path := visualize_spd_components(
                            decomposed, folder, gate_name=sc_label
                        ):
                            viz_paths[trial_id][gname][sc_idx]["spd"] = path

    return viz_paths

"""Visualization for circuits and SPD decompositions.

##############################################################################
#                                                                            #
#   ██████╗ ███████╗ █████╗ ██████╗     ██████╗ ███╗   ██╗██╗  ██╗   ██╗██╗  #
#   ██╔══██╗██╔════╝██╔══██╗██╔══██╗   ██╔═══██╗████╗  ██║██║  ╚██╗ ██╔╝██║  #
#   ██████╔╝█████╗  ███████║██║  ██║   ██║   ██║██╔██╗ ██║██║   ╚████╔╝ ██║  #
#   ██╔══██╗██╔══╝  ██╔══██║██║  ██║   ██║   ██║██║╚██╗██║██║    ╚██╔╝  ╚═╝  #
#   ██║  ██║███████╗██║  ██║██████╔╝   ╚██████╔╝██║ ╚████║███████╗██║   ██╗  #
#   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝     ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝   ╚═╝  #
#                                                                            #
#   THIS MODULE DOES NOT RUN ANY MODELS!                                     #
#   ALL DATA MUST BE PRE-COMPUTED IN trial.py OR causal_analysis.py          #
#                                                                            #
#   If you need activations, add them to the relevant schema class and       #
#   compute them during analysis, NOT during visualization.                  #
#                                                                            #
#   Running models here causes ~5000 forward passes and kills performance!   #
#                                                                            #
##############################################################################

Look at visualize_experiment for the main entry point.
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

# Configure matplotlib backend BEFORE importing pyplot (critical for batch rendering)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from .common.circuit import Circuit
from .common.neural_model import DecomposedMLP
from .common.schemas import (
    ExperimentResult,
    FaithfulnessMetrics,
    ProfilingData,
    RobustnessMetrics,
)
from .profiler import profile

# Configure matplotlib for performance
plt.rcParams["path.simplify"] = True
plt.rcParams["path.simplify_threshold"] = 1.0
plt.rcParams["agg.path.chunksize"] = 10000
plt.rcParams["text.usetex"] = False

# Global styling - clean academic aesthetic with monospace fonts
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["DejaVu Sans Mono", "Courier New", "monospace"]
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.titlepad"] = 18  # More spacing between title and plot
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["figure.titlesize"] = 13
plt.rcParams["figure.titleweight"] = "bold"
plt.rcParams["hatch.linewidth"] = 3.0  # Thick hatch lines for visibility at small sizes


# ------------------ CONSTANTS ------------------

# Plot styling
COLORS = {
    "gate": "steelblue",
    "subcircuit": "coral",
    "agreement": "purple",
    "mse": "teal",
    "correct": "green",
    "incorrect": "red",
    # Faithfulness-specific (pastel)
    "in_circuit": "#77DD77",  # Pastel green
    "out_circuit": "#DA70D6",  # Pastel orchid
    "faithfulness": "#DA70D6",  # Pastel orchid (purple)
    "counterfactual": "#EF6C00",  # Orange
}
MARKERS = {"gate": "^", "subcircuit": "v", "agreement": "o", "mse": "s"}
JITTER = {"correct": 1.05, "incorrect": -0.05, "gate_correct": 1.05, "sc_correct": 0.95}


# ------------------ LAYOUT CACHE ------------------


class GraphLayoutCache:
    """Cache graph layouts by structure to avoid recomputation.

    Positions depend only on layer sizes, not activation values.
    Computing positions once and reusing saves significant time.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
        return cls._instance

    def get_positions(self, layer_sizes: tuple) -> dict:
        """Get or compute node positions for given layer structure."""
        key = tuple(layer_sizes)
        if key not in self._cache:
            pos = {}
            max_width = max(layer_sizes)
            for layer_idx, n_nodes in enumerate(layer_sizes):
                y_offset = -(max_width - n_nodes) / 2
                for node_idx in range(n_nodes):
                    name = f"({layer_idx},{node_idx})"
                    pos[name] = (layer_idx, y_offset - node_idx)
            self._cache[key] = pos
        return self._cache[key]

    def clear(self):
        self._cache.clear()


# Global singleton
_layout_cache = GraphLayoutCache()


# ------------------ HELPERS ------------------


def _activation_to_color(val: float, vmin: float = -2, vmax: float = 2) -> tuple:
    """
    Pastel color gradient for activation values:
    - 0.5 = beige/cream
    - 1.0 = light mint green
    - >1.0 = green -> teal (more saturated)
    - 0.0 = light peach/orange
    - -1.0 = coral/salmon
    - <-1.0 = deeper coral -> rose
    """
    # Pastel colors (RGB tuples, 0-1 range)
    colors = {
        "deep_rose": (0.85, 0.45, 0.55),      # < -1.0
        "coral": (0.95, 0.65, 0.60),           # -1.0
        "peach": (0.98, 0.82, 0.70),           # 0.0
        "cream": (0.98, 0.95, 0.80),           # 0.5
        "mint": (0.75, 0.92, 0.78),            # 1.0
        "teal": (0.55, 0.82, 0.78),            # > 1.0
    }

    def lerp(c1, c2, t):
        return tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(3))

    if val <= -1.0:
        # Deep rose to coral
        t = min(1.0, (-1.0 - val) / 1.0)  # How far below -1
        rgb = lerp(colors["coral"], colors["deep_rose"], t)
    elif val <= 0.0:
        # Coral to peach
        t = (val + 1.0) / 1.0  # -1 -> 0 maps to 0 -> 1
        rgb = lerp(colors["coral"], colors["peach"], t)
    elif val <= 0.5:
        # Peach to cream
        t = val / 0.5  # 0 -> 0.5 maps to 0 -> 1
        rgb = lerp(colors["peach"], colors["cream"], t)
    elif val <= 1.0:
        # Cream to mint
        t = (val - 0.5) / 0.5  # 0.5 -> 1 maps to 0 -> 1
        rgb = lerp(colors["cream"], colors["mint"], t)
    else:
        # Mint to teal
        t = min(1.0, (val - 1.0) / 1.0)  # How far above 1
        rgb = lerp(colors["mint"], colors["teal"], t)

    return (*rgb, 1.0)


def _text_color_for_background(bg_color: tuple) -> str:
    """Contrasting text color based on luminance."""
    r, g, b = bg_color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


def _symmetric_range(activations: list) -> tuple[float, float]:
    """Color range centered at 0."""
    all_vals = [
        act[0].detach().cpu().numpy()
        if isinstance(act, torch.Tensor)
        else np.array(act[0] if isinstance(act[0], list) else act)
        for act in activations
    ]
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


# ------------------ REUSABLE INTERVENED CIRCUIT DRAWING ------------------


def draw_intervened_circuit(
    ax,
    layer_sizes: list[int],
    weights: list[np.ndarray],
    current_activations: list[list[float]],
    original_activations: list[list[float]] | None = None,
    intervened_nodes: set[str] | None = None,
    circuit: Circuit | None = None,
    title: str | None = None,
    node_size: int = 400,
    show_edge_labels: bool = True,
    biases: list[np.ndarray] | None = None,
):
    """
    Draw a circuit showing intervention effects with pastel colors.

    Border colors for node types:
    - Regular nodes: thin dark grey
    - Affected nodes (changed): medium purple border
    - Intervened nodes: thick magenta border

    Edge colors: warm brown for positive, cool slate for negative (subtle)

    Edge labels show weight value, and if biases are provided, also show
    (+bias) to make the bias contribution obvious when edges are patched to 0.
    """
    if intervened_nodes is None:
        intervened_nodes = set()

    if circuit is None:
        circuit = Circuit.full(layer_sizes)

    G = nx.DiGraph()
    pos = _layout_cache.get_positions(tuple(layer_sizes))

    # Build nodes
    node_data = {}

    for layer_idx, n_nodes in enumerate(layer_sizes):
        for node_idx in range(n_nodes):
            name = f"({layer_idx},{node_idx})"
            G.add_node(name)

            # Get current activation value
            current_val = None
            if layer_idx < len(current_activations):
                layer = current_activations[layer_idx]
                if isinstance(layer, (list, tuple)) and node_idx < len(layer):
                    v = layer[node_idx]
                    if isinstance(v, (int, float)):
                        current_val = float(v)

            # Get original activation value
            original_val = None
            if original_activations and layer_idx < len(original_activations):
                layer = original_activations[layer_idx]
                if isinstance(layer, (list, tuple)) and node_idx < len(layer):
                    v = layer[node_idx]
                    if isinstance(v, (int, float)):
                        original_val = float(v)

            # Compute color from current value (uses new pastel gradient)
            if current_val is not None:
                color = _activation_to_color(current_val)
                text_color = _text_color_for_background(color)
            else:
                color = (0.92, 0.92, 0.92, 1.0)
                text_color = "black"

            # Check if value changed
            is_intervened = name in intervened_nodes
            value_changed = (
                original_val is not None and
                current_val is not None and
                abs(current_val - original_val) > 0.01
            )
            is_affected = value_changed and not is_intervened

            node_data[name] = {
                "color": color,
                "current_val": current_val,
                "original_val": original_val,
                "text_color": text_color,
                "is_intervened": is_intervened,
                "is_affected": is_affected,
                "value_changed": value_changed,
            }

    # Build edges with weights and track sign
    edges = []
    edge_labels = {}
    edge_weights = {}
    edge_signs = {}  # Track positive/negative

    for layer_idx, mask in enumerate(circuit.edge_masks):
        if layer_idx >= len(weights):
            continue
        w = weights[layer_idx]
        # Get bias for this layer's output neurons (if available)
        b = biases[layer_idx] if biases is not None and layer_idx < len(biases) else None

        for out_idx, row in enumerate(mask):
            for in_idx, active in enumerate(row):
                if active:
                    e = (f"({layer_idx},{in_idx})", f"({layer_idx + 1},{out_idx})")
                    edges.append(e)
                    G.add_edge(e[0], e[1])

                    if out_idx < w.shape[0] and in_idx < w.shape[1]:
                        weight_val = float(w[out_idx, in_idx])
                        # Show weight, with bias in parentheses when available
                        # Format: "0.5 (+0.1)" - space before paren, 1 decimal for compactness
                        if b is not None and out_idx < len(b):
                            bias_val = float(b[out_idx])
                            sign = "+" if bias_val >= 0 else ""
                            edge_labels[e] = f"{weight_val:.1f} ({sign}{bias_val:.1f})"
                        else:
                            edge_labels[e] = f"{weight_val:.2f}"
                        edge_weights[e] = abs(weight_val)
                        edge_signs[e] = weight_val >= 0

    # Compute edge widths and colors
    if edge_weights:
        max_w = max(edge_weights.values()) if edge_weights.values() else 1.0
        max_w = max(max_w, 0.01)
        edge_widths = [0.5 + 2.0 * (edge_weights.get(e, 0) / max_w) for e in edges]
    else:
        edge_widths = [1.0] * len(edges)

    # Subtle edge colors: warm for positive, cool for negative
    edge_colors = [
        "#8B7355" if edge_signs.get(e, True) else "#5F6A7D"  # warm brown vs cool slate
        for e in edges
    ]

    # Categorize nodes by border type
    regular_nodes = [n for n in G.nodes()
                     if not node_data[n]["is_intervened"] and not node_data[n]["is_affected"]]
    affected_nodes = [n for n in G.nodes() if node_data[n]["is_affected"]]
    intervened_list = [n for n in G.nodes() if node_data[n]["is_intervened"]]

    # Draw regular nodes (thin dark grey border)
    if regular_nodes:
        colors = [node_data[n]["color"] for n in regular_nodes]
        nx.draw_networkx_nodes(
            G, pos, nodelist=regular_nodes, node_color=colors,
            node_size=node_size, ax=ax, edgecolors="#555555", linewidths=0.8
        )

    # Draw affected nodes (medium purple border - good contrast)
    if affected_nodes:
        colors = [node_data[n]["color"] for n in affected_nodes]
        nx.draw_networkx_nodes(
            G, pos, nodelist=affected_nodes, node_color=colors,
            node_size=node_size, ax=ax, edgecolors="#7B68EE", linewidths=2.0
        )

    # Draw intervened nodes (thick magenta border - high contrast)
    if intervened_list:
        colors = [node_data[n]["color"] for n in intervened_list]
        nx.draw_networkx_nodes(
            G, pos, nodelist=intervened_list, node_color=colors,
            node_size=node_size, ax=ax, edgecolors="#C71585", linewidths=2.5
        )

    # Draw edges with color by sign
    if edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, edge_color=edge_colors,
            width=edge_widths, arrows=True, arrowstyle="-|>",
            arrowsize=8, connectionstyle="arc3,rad=0.1", ax=ax
        )

    # Draw edge labels - small font, subtle appearance
    if show_edge_labels and edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=3.5,
            font_color="#888888", alpha=0.6, label_pos=0.3,
            bbox=dict(boxstyle="round,pad=0.05", facecolor="white",
                      alpha=0.5, edgecolor="none"), ax=ax
        )

    # Draw node labels - smaller text, tighter spacing
    for node, (x, y) in pos.items():
        data = node_data[node]
        current_val = data["current_val"]
        original_val = data["original_val"]
        text_color = data["text_color"]
        value_changed = data["value_changed"]

        if current_val is not None:
            if value_changed:
                # Current value slightly above, original below (use va for tight control)
                ax.text(
                    x, y, f"{current_val:.2f}",
                    ha="center", va="bottom",
                    fontsize=5, fontweight="bold", color=text_color
                )
                ax.text(
                    x, y, f"({original_val:.2f})",
                    ha="center", va="top",
                    fontsize=4, color=text_color, alpha=0.5
                )
            else:
                # Just current value centered
                ax.text(
                    x, y, f"{current_val:.2f}",
                    ha="center", va="center",
                    fontsize=5, fontweight="bold", color=text_color
                )

    if title:
        ax.set_title(title, fontsize=8, fontweight="bold")
    ax.axis("off")


# ------------------ CIRCUIT GRAPH ------------------


def _build_graph_fast(
    activations, circuit, weights_per_layer, vmin, vmax, cached_pos=None,
    biases_per_layer=None,
):
    """Build networkx graph with optional cached positions.

    Args:
        biases_per_layer: Optional list of bias vectors. If provided, edge labels
            show (weight + bias) to reveal bias contribution when weights are 0.
    """
    G = nx.DiGraph()
    node_colors_dict = {}  # Dict keyed by node name for proper lookup
    labels, text_colors = {}, {}

    # Get layer sizes from activations
    layer_sizes = tuple(
        act.shape[-1]
        if isinstance(act, torch.Tensor)
        else len(act[0])
        if isinstance(act[0], list)
        else act.shape[-1]
        for act in activations
    )

    # Use cached positions if available
    if cached_pos is None:
        pos = _layout_cache.get_positions(layer_sizes)
    else:
        pos = cached_pos

    # Batch add all nodes first
    all_nodes = []
    for layer_idx, layer_act in enumerate(activations):
        n = (
            layer_act.shape[-1]
            if isinstance(layer_act, torch.Tensor)
            else len(layer_act[0])
            if isinstance(layer_act[0], list)
            else layer_act.shape[-1]
        )
        for node_idx in range(n):
            name = f"({layer_idx},{node_idx})"
            all_nodes.append(name)
    G.add_nodes_from(all_nodes)

    # Build node colors and labels
    for layer_idx, layer_act in enumerate(activations):
        n = (
            layer_act.shape[-1]
            if isinstance(layer_act, torch.Tensor)
            else len(layer_act[0])
            if isinstance(layer_act[0], list)
            else layer_act.shape[-1]
        )
        for node_idx in range(n):
            name = f"({layer_idx},{node_idx})"

            if isinstance(layer_act, torch.Tensor):
                val = layer_act[0, node_idx].item()
            else:
                val = (
                    layer_act[0][node_idx]
                    if isinstance(layer_act[0], list)
                    else layer_act[0, node_idx]
                )

            # Handle case where circuit has more outputs than activations
            if layer_idx < len(circuit.node_masks) and node_idx < len(
                circuit.node_masks[layer_idx]
            ):
                active = circuit.node_masks[layer_idx][node_idx] == 1
            else:
                active = True

            if active:
                labels[name] = f"{val:.2f}"
                color = _activation_to_color(val, vmin, vmax)
                node_colors_dict[name] = color
                text_colors[name] = _text_color_for_background(color)
            else:
                # Out-of-circuit nodes: no label, gray color
                labels[name] = ""
                node_colors_dict[name] = "#d3d3d3"
                text_colors[name] = "gray"

    # Batch add edges - only add edges for nodes that exist in activations
    edge_labels = {}
    edge_weights = {}
    all_edges = []
    n_layers = len(activations)
    for layer_idx, mask in enumerate(circuit.edge_masks):
        if layer_idx + 1 >= n_layers:
            continue  # Skip edges to layers beyond activations

        w = weights_per_layer[layer_idx]
        # Get bias for this layer if available
        b = biases_per_layer[layer_idx] if biases_per_layer is not None and layer_idx < len(biases_per_layer) else None
        out_limit = (
            activations[layer_idx + 1].shape[-1]
            if isinstance(activations[layer_idx + 1], torch.Tensor)
            else len(activations[layer_idx + 1][0])
        )
        in_limit = (
            activations[layer_idx].shape[-1]
            if isinstance(activations[layer_idx], torch.Tensor)
            else len(activations[layer_idx][0])
        )

        for out_idx, row in enumerate(mask):
            if out_idx >= out_limit:
                continue  # Skip outputs beyond activation size
            for in_idx, active in enumerate(row):
                if in_idx >= in_limit:
                    continue  # Skip inputs beyond activation size
                e = (f"({layer_idx},{in_idx})", f"({layer_idx + 1},{out_idx})")
                weight_val = (
                    w[out_idx, in_idx]
                    if out_idx < w.shape[0] and in_idx < w.shape[1]
                    else 0.0
                )
                all_edges.append((e[0], e[1], {"active": active, "weight": weight_val}))
                if active == 1:
                    # Only show labels for edges with significant weight to reduce clutter
                    if abs(weight_val) >= 0.15:
                        # Show weight with bias in parentheses when available
                        if b is not None and out_idx < len(b):
                            bias_val = float(b[out_idx])
                            sign = "+" if bias_val >= 0 else ""
                            edge_labels[e] = f"{weight_val:.1f} ({sign}{bias_val:.1f})"
                        else:
                            edge_labels[e] = f"{weight_val:.1f}"
                    edge_weights[e] = abs(weight_val)

    G.add_edges_from(all_edges)

    # Convert node_colors_dict to list in G.nodes() order
    node_colors = [node_colors_dict.get(n, "#d3d3d3") for n in G.nodes()]

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

    # Draw all nodes at once
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=900,
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )

    # Draw active edges
    if active:
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

    # Draw inactive edges
    if inactive:
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

    # Draw labels
    for node, (x, y) in pos.items():
        ax.text(
            x,
            y,
            labels.get(node, ""),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=text_colors.get(node, "black"),
        )

    if edge_labels:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=7,
            font_color="#333333",
            label_pos=0.5,
            bbox=dict(
                boxstyle="round,pad=0.1", facecolor="white", alpha=0.95, edgecolor="none"
            ),
            ax=ax,
        )
    ax.axis("off")


def _draw_graph_with_output_highlight(
    ax,
    G,
    pos,
    node_colors,
    labels,
    text_colors,
    edge_labels,
    edge_weights,
    output_correct,
):
    """Draw graph with highlighted output node border (green=correct, red=incorrect)."""
    active = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 1]
    inactive = [(u, v) for u, v, d in G.edges(data=True) if d["active"] == 0]

    if edge_weights:
        max_w = max(edge_weights.values()) if edge_weights.values() else 1.0
        max_w = max(max_w, 0.01)
        active_widths = [0.5 + 3.5 * (edge_weights.get(e, 0) / max_w) for e in active]
    else:
        active_widths = [2] * len(active)

    # Find output layer
    max_layer = max(int(n.split(",")[0][1:]) for n in G.nodes())
    output_nodes = [n for n in G.nodes() if n.startswith(f"({max_layer},")]
    other_nodes = [n for n in G.nodes() if n not in output_nodes]

    # Draw non-output nodes
    if other_nodes:
        other_colors = [node_colors[list(G.nodes()).index(n)] for n in other_nodes]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=other_nodes,
            node_color=other_colors,
            node_size=900,
            edgecolors="black",
            linewidths=1.5,
            ax=ax,
        )

    # Draw output nodes with colored border
    if output_nodes:
        output_colors = [node_colors[list(G.nodes()).index(n)] for n in output_nodes]
        border_color = "green" if output_correct else "red"
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=output_nodes,
            node_color=output_colors,
            node_size=900,
            edgecolors=border_color,
            linewidths=4,
            ax=ax,
        )

    # Draw edges
    if active:
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
    if inactive:
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

    # Draw labels
    for node, (x, y) in pos.items():
        ax.text(
            x,
            y,
            labels.get(node, ""),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=text_colors.get(node, "black"),
        )
    ax.axis("off")


def _draw_circuit_from_data(ax, activations, circuit, weights, title, biases=None):
    """Draw a single circuit using pre-computed activations (no model run)."""
    vmin, vmax = _symmetric_range(activations)
    G, pos, colors, node_labels, text_colors, edge_labels, edge_weights = (
        _build_graph_fast(activations, circuit, weights, vmin, vmax, biases_per_layer=biases)
    )
    _draw_graph(ax, G, pos, colors, node_labels, text_colors, edge_labels, edge_weights)

    output = (
        activations[-1][0, 0].item()
        if isinstance(activations[-1], torch.Tensor)
        else activations[-1][0][0]
    )
    ax.set_title(f"{title} -> {output:.3f}", fontsize=10, fontweight="bold")


# ------------------ PUBLIC FUNCTIONS ------------------


def visualize_circuit_activations_from_data(
    canonical_activations: dict[str, list[torch.Tensor]],
    layer_weights: list[torch.Tensor],
    circuit: Circuit,
    output_dir: str,
    filename: str = "circuit_activations.png",
    gate_name: str = "",
    layer_biases: list[torch.Tensor] | None = None,
) -> str:
    """
    2x2 grid: circuit activations for (0,0), (0,1), (1,0), (1,1) inputs.

    Uses pre-computed activations - NO model execution.

    Args:
        layer_biases: Optional bias vectors. If provided, edge labels show
            (weight + bias) to reveal bias contribution when edges are patched.
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
    biases = [b.numpy() for b in layer_biases] if layer_biases else None

    for i, (key, label) in enumerate(labels_map.items()):
        activations = canonical_activations.get(key, [])
        if activations:
            _draw_circuit_from_data(
                axes[i], activations, circuit, weights, f"Input: {label}", biases=biases
            )
        else:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[i].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if gate_name:
        fig.suptitle(
            f"{gate_name} - Circuit Activations", fontsize=14, fontweight="bold", y=0.99
        )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_circuit_activations_mean(
    mean_activations_by_range: dict[str, list[torch.Tensor]],
    layer_weights: list[torch.Tensor],
    circuit: Circuit,
    output_dir: str,
    filename: str = "circuit_activations_mean.png",
    gate_name: str = "",
    layer_biases: list[torch.Tensor] | None = None,
) -> str:
    """
    1x4 grid: mean circuit activations for different input ranges.

    Shows how the network behaves on average for inputs from:
    - [0, 1]: Normal operating range
    - [-1, 0]: Negative inputs
    - [-2, 2]: Extended range
    - [-100, 100]: Far out-of-distribution

    Uses pre-computed mean activations - NO model execution.

    Args:
        layer_biases: Optional bias vectors. If provided, edge labels show
            (weight + bias) to reveal bias contribution when edges are patched.
    """
    labels_map = {
        "0_1": "[0, 1]",
        "-1_0": "[-1, 0]",
        "-2_2": "[-2, 2]",
        "-100_100": "[-100, 100]",
    }

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    weights = [w.numpy() for w in layer_weights]
    biases = [b.numpy() for b in layer_biases] if layer_biases else None

    for i, (key, label) in enumerate(labels_map.items()):
        activations = mean_activations_by_range.get(key, [])
        if activations:
            _draw_circuit_from_data(
                axes[i], activations, circuit, weights, f"Input Range: {label}", biases=biases
            )
        else:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[i].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if gate_name:
        fig.suptitle(
            f"{gate_name} - Mean Activations by Input Range", fontsize=14, fontweight="bold", y=0.99
        )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------ ROBUSTNESS VISUALIZATION ------------------


def _generate_robustness_circuit_figure(args):
    """Worker function for parallel robustness circuit generation.

    Uses draw_intervened_circuit for consistent visualization.
    Input nodes are marked as intervened since robustness tests vary inputs.
    Shows original (canonical) activations below current values for comparison.
    """
    (
        samples,
        circuit_dict,
        full_circuit_dict,
        weights,
        base_key,
        category,
        gt,
        output_path,
        n_samples,
        base_activations,
        biases,  # Added biases parameter
    ) = args

    # Reconstruct circuits from dicts
    circuit = Circuit.from_dict(circuit_dict)
    full_circuit = Circuit.from_dict(full_circuit_dict)

    # Get layer sizes from circuit
    layer_sizes = [len(nm) for nm in circuit.node_masks]

    # Input nodes are intervened (layer 0)
    input_size = layer_sizes[0] if layer_sizes else 2
    intervened_nodes = {f"(0,{i})" for i in range(input_size)}

    fig, axes = plt.subplots(n_samples, 2, figsize=(8, n_samples * 3))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, sample in enumerate(samples):
        if not sample["agreement_bit"]:
            for col in range(2):
                axes[i, col].set_facecolor("#FFEEEE")

        # Get activations as flat lists
        sc_acts = sample["subcircuit_activations"]
        full_acts = sample["gate_activations"]

        # Left: subcircuit (use base_activations as original for comparison)
        draw_intervened_circuit(
            axes[i, 0],
            layer_sizes=layer_sizes,
            weights=weights,
            current_activations=sc_acts,
            original_activations=base_activations,
            intervened_nodes=intervened_nodes,
            circuit=circuit,
            title=None,
            node_size=500,
            show_edge_labels=True,
            biases=biases,
        )

        # Right: full model (use base_activations as original for comparison)
        draw_intervened_circuit(
            axes[i, 1],
            layer_sizes=layer_sizes,
            weights=weights,
            current_activations=full_acts,
            original_activations=base_activations,
            intervened_nodes=intervened_nodes,
            circuit=full_circuit,
            title=None,
            node_size=500,
            show_edge_labels=True,
            biases=biases,
        )

        if not sample["agreement_bit"]:
            axes[i, 0].text(
                0.02,
                0.98,
                "DISAGREE",
                transform=axes[i, 0].transAxes,
                fontsize=8,
                color="red",
                fontweight="bold",
                verticalalignment="top",
            )

    axes[0, 0].set_title("Subcircuit", fontsize=10, fontweight="bold")
    axes[0, 1].set_title("Full", fontsize=10, fontweight="bold")

    # Parse base_key like "0_1" -> "(0, 1)"
    base_parts = base_key.split("_")
    base_str = f"({base_parts[0]}, {base_parts[1]})"

    # Category label for subtitle - concise labels
    category_labels = {
        "noise": "Noise",
        "multiply_positive": "Multiply (positive)",
        "multiply_negative": "Multiply (negative)",
        "add": "Add",
        "subtract": "Subtract",
        "bimodal": "Bimodal",
        "bimodal_inv": "Bimodal (inv)",
    }
    category_label = category_labels.get(category, category)

    # Use tight_layout first to get proper spacing, then add titles above
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Main title: "(0, 1) -> 1" format - place well above plot area
    fig.suptitle(f"{base_str} → {gt}", fontsize=14, fontweight="bold", y=0.98)
    # Subtitle: transformation type
    fig.text(0.5, 0.94, category_label, ha="center", fontsize=10, style="italic")

    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def visualize_robustness_circuit_samples(
    robustness: RobustnessMetrics,
    circuit: Circuit,
    layer_weights: list[torch.Tensor],
    output_dir: str,
    n_samples_per_grid: int = 8,
    canonical_activations: dict[str, list] | None = None,
    layer_biases: list[torch.Tensor] | None = None,
) -> dict[str, str]:
    """Visualize circuit diagrams comparing subcircuit vs full model under noise.

    Folder structure:
        circuit_viz/
        ├── noise_perturbations/
        │   └── 0_0.png, 0_1.png, 1_0.png, 1_1.png
        └── out_distribution_transformations/
            ├── multiply/
            │   └── 0_1_positive.png, 0_1_negative.png, etc.
            ├── add/
            │   └── 0_0.png, 0_1.png, etc.
            ├── subtract/
            │   └── 0_0.png, 0_1.png, etc.
            └── bimodal/
                └── 0_0.png (order-preserving), 0_0_inv.png (inverted)

    Args:
        layer_biases: Optional bias vectors. If provided, edge labels show
            (weight + bias) to reveal bias contribution when edges are patched.
    """
    circuit_viz_dir = os.path.join(output_dir, "circuit_viz")
    paths = {}

    # Create folder structure
    noise_dir = os.path.join(circuit_viz_dir, "noise_perturbations")
    ood_dir = os.path.join(circuit_viz_dir, "out_distribution_transformations")
    multiply_dir = os.path.join(ood_dir, "multiply")
    add_dir = os.path.join(ood_dir, "add")
    subtract_dir = os.path.join(ood_dir, "subtract")
    bimodal_dir = os.path.join(ood_dir, "bimodal")

    for d in [noise_dir, multiply_dir, add_dir, subtract_dir, bimodal_dir]:
        os.makedirs(d, exist_ok=True)

    base_to_key = {
        (0.0, 0.0): "0_0",
        (0.0, 1.0): "0_1",
        (1.0, 0.0): "1_0",
        (1.0, 1.0): "1_1",
    }

    weights = [w.numpy() for w in layer_weights]
    biases = [b.numpy() for b in layer_biases] if layer_biases else None
    layer_sizes = [layer_weights[0].shape[1]] + [w.shape[0] for w in layer_weights]
    full_circuit = Circuit.full(layer_sizes)

    # All sample types to handle
    sample_types = [
        "noise",
        "multiply_positive",
        "multiply_negative",
        "add",
        "subtract",
        "bimodal",
        "bimodal_inv",
    ]

    # Group samples by base input and sample_type
    samples_by_base: dict[str, dict[str, list]] = {
        k: {st: [] for st in sample_types} for k in base_to_key.values()
    }

    # Process noise samples
    for sample in robustness.noise_samples:
        base_key = base_to_key.get((sample.base_input[0], sample.base_input[1]))
        if base_key:
            samples_by_base[base_key]["noise"].append(sample)

    # Process OOD samples using sample_type field
    for sample in robustness.ood_samples:
        base_key = base_to_key.get((sample.base_input[0], sample.base_input[1]))
        if base_key:
            st = getattr(sample, "sample_type", "multiply_positive")
            if st in samples_by_base[base_key]:
                samples_by_base[base_key][st].append(sample)

    # Map sample types to directories and filename suffixes
    type_to_dir = {
        "noise": noise_dir,
        "multiply_positive": multiply_dir,
        "multiply_negative": multiply_dir,
        "add": add_dir,
        "subtract": subtract_dir,
        "bimodal": bimodal_dir,
        "bimodal_inv": bimodal_dir,
    }

    # For multiply and bimodal, we need suffixes
    # multiply: positive/negative, bimodal: none/inv
    type_to_suffix = {
        "noise": "",
        "multiply_positive": "_positive",
        "multiply_negative": "_negative",
        "add": "",
        "subtract": "",
        "bimodal": "",
        "bimodal_inv": "_inv",
    }

    # Prepare tasks for parallel execution
    tasks = []
    circuit_dict = circuit.to_dict()
    full_circuit_dict = full_circuit.to_dict()

    for base_key, by_type in samples_by_base.items():
        for sample_type in sample_types:
            all_samples = by_type[sample_type]
            if not all_samples:
                continue

            sort_key = lambda s: abs(s.noise_magnitude)
            disagree = sorted(
                [s for s in all_samples if not s.agreement_bit], key=sort_key
            )
            agree = sorted([s for s in all_samples if s.agreement_bit], key=sort_key)

            if len(disagree) >= n_samples_per_grid:
                d_idx = np.linspace(0, len(disagree) - 1, n_samples_per_grid, dtype=int)
                samples = [disagree[i] for i in d_idx]
            else:
                samples = list(disagree)
                remaining = n_samples_per_grid - len(samples)
                if remaining > 0 and agree:
                    a_idx = np.linspace(
                        0, len(agree) - 1, min(remaining, len(agree)), dtype=int
                    )
                    samples.extend([agree[i] for i in a_idx])

            samples = sorted(samples, key=sort_key)
            n_samples = len(samples)
            if n_samples == 0:
                continue

            gt = int(all_samples[0].ground_truth)

            # Build filename and path
            target_dir = type_to_dir[sample_type]
            suffix = type_to_suffix[sample_type]
            filename = f"{base_key}{suffix}.png"
            output_path = os.path.join(target_dir, filename)

            # Convert samples to dicts for pickling
            sample_dicts = [
                {
                    "subcircuit_activations": s.subcircuit_activations,
                    "gate_activations": s.gate_activations,
                    "subcircuit_correct": s.subcircuit_correct,
                    "gate_correct": s.gate_correct,
                    "agreement_bit": s.agreement_bit,
                }
                for s in samples
            ]

            # Get canonical activations for this base input (for showing original values)
            base_acts = None
            if canonical_activations and base_key in canonical_activations:
                acts = canonical_activations[base_key]
                # Convert tensors to lists for pickling
                if acts and isinstance(acts[0], torch.Tensor):
                    base_acts = [a.squeeze(0).tolist() for a in acts]
                else:
                    base_acts = acts

            tasks.append(
                (
                    sample_dicts,
                    circuit_dict,
                    full_circuit_dict,
                    weights,
                    base_key,
                    sample_type,  # Use sample_type as category
                    gt,
                    output_path,
                    n_samples,
                    base_acts,
                    biases,
                )
            )

            # Use relative path from circuit_viz_dir for paths dict
            rel_path = os.path.relpath(output_path, circuit_viz_dir)
            paths[rel_path] = output_path

    # Execute in parallel
    if tasks:
        n_workers = min(len(tasks), mp.cpu_count())
        print(
            f"[VIZ] Generating {len(tasks)} robustness circuit figures with {n_workers} workers"
        )
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            list(executor.map(_generate_robustness_circuit_figure, tasks))

    return paths


def visualize_robustness_curves(
    robustness: RobustnessMetrics,
    output_dir: str,
    gate_name: str = "",
) -> dict[str, str]:
    """Visualize robustness showing raw output values colored by correctness."""
    paths = {}
    prefix = f"{gate_name} - " if gate_name else ""

    def _compute_perturbation_effect(sample):
        """Compute perturbation effect on input difference.

        Perturbation Effect = (perturbed_left - perturbed_right) - (base_left - base_right)

        For XOR, this measures how perturbation changes the left-right relationship.
        """
        if len(sample.input_values) >= 2 and len(sample.base_input) >= 2:
            perturbed_diff = sample.input_values[0] - sample.input_values[1]
            base_diff = sample.base_input[0] - sample.base_input[1]
            return perturbed_diff - base_diff
        # Fallback for non-2D inputs
        return sum(p - b for p, b in zip(sample.input_values, sample.base_input))

    def _plot_output_values(ax, samples, x_values, gt_value, x_label_fmt=".2f"):
        """Plot raw output values colored by correctness (green=correct, red=incorrect)."""
        if not samples or not x_values:
            return

        # Sort by x_values
        sorted_pairs = sorted(zip(x_values, samples), key=lambda p: p[0])
        sorted_x = [p[0] for p in sorted_pairs]
        sorted_samples = [p[1] for p in sorted_pairs]

        # Extract outputs and correctness
        gate_outputs = [s.gate_output for s in sorted_samples]
        sc_outputs = [s.subcircuit_output for s in sorted_samples]
        gate_colors = ["#4CAF50" if s.gate_correct else "#E53935" for s in sorted_samples]
        sc_colors = ["#4CAF50" if s.subcircuit_correct else "#E53935" for s in sorted_samples]

        # Plot gate outputs (circles) and subcircuit outputs (squares)
        ax.scatter(sorted_x, gate_outputs, s=20, c=gate_colors, alpha=0.7, label="Gate", marker="o", edgecolors="none")
        ax.scatter(sorted_x, sc_outputs, s=20, c=sc_colors, alpha=0.7, label="SC", marker="s", edgecolors="none")

        # Add horizontal line at ground truth and 0.5 threshold
        ax.axhline(y=gt_value, color="#333333", linestyle="--", linewidth=1, alpha=0.5, label=f"GT={gt_value:.0f}")
        ax.axhline(y=0.5, color="#888888", linestyle=":", linewidth=1, alpha=0.5)

        # Add x-axis labels
        ax.set_xlim(min(sorted_x) - 0.1, max(sorted_x) + 0.1)

    base_to_key = {
        (0.0, 0.0): "0_0",
        (0.0, 1.0): "0_1",
        (1.0, 0.0): "1_0",
        (1.0, 1.0): "1_1",
    }
    input_keys = list(base_to_key.values())  # ["0_0", "0_1", "1_0", "1_1"]

    def _plot_single_model(ax, samples, x_values, gt_value, outputs, correct_flags):
        """Plot single model outputs colored by correctness."""
        if not samples or not x_values:
            return
        sorted_pairs = sorted(zip(x_values, outputs, correct_flags), key=lambda p: p[0])
        sorted_x = [p[0] for p in sorted_pairs]
        sorted_outputs = [p[1] for p in sorted_pairs]
        sorted_correct = [p[2] for p in sorted_pairs]
        colors = ["#4CAF50" if c else "#E53935" for c in sorted_correct]
        ax.scatter(sorted_x, sorted_outputs, s=25, c=colors, alpha=0.7, edgecolors="none")
        ax.axhline(y=gt_value, color="#333333", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(y=0.5, color="#888888", linestyle=":", linewidth=1, alpha=0.5)

    def _plot_agreement_binned(ax, samples, x_values, n_bins=6):
        """Plot binned agreement rates (overlapping: bit, best) and |Δ logit|."""
        if not samples or not x_values:
            return

        # Create bins - use logarithmic binning when x_values span orders of magnitude
        x_min, x_max = min(x_values), max(x_values)
        if x_min == x_max:
            x_min, x_max = x_min - 0.5, x_max + 0.5

        # Detect if logarithmic binning is appropriate:
        # - All values positive (required for geomspace)
        # - Span at least one order of magnitude (max/min > 10)
        use_log_bins = x_min > 0 and x_max / x_min > 10
        if use_log_bins:
            bin_edges = np.geomspace(x_min, x_max, n_bins + 1)
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean
        else:
            bin_edges = np.linspace(x_min, x_max, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute per-bin statistics
        bit_agreement_rates = []
        best_agreement_rates = []
        logit_diffs = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            bin_samples = [
                (s, x) for s, x in zip(samples, x_values)
                if lo <= x < hi or (i == n_bins - 1 and x == hi)
            ]
            if bin_samples:
                # Use pre-computed agreement fields from RobustnessSample
                bit_agrees = [s.agreement_bit for s, _ in bin_samples]
                best_agrees = [s.agreement_best for s, _ in bin_samples]
                # Logit differences
                diffs = [abs(s.gate_output - s.subcircuit_output) for s, _ in bin_samples]

                bit_agreement_rates.append(np.mean(bit_agrees))
                best_agreement_rates.append(np.mean(best_agrees))
                logit_diffs.append(np.mean(diffs))
            else:
                bit_agreement_rates.append(np.nan)
                best_agreement_rates.append(np.nan)
                logit_diffs.append(np.nan)

        # Convert to arrays
        bit_rates = np.array(bit_agreement_rates)
        best_rates = np.array(best_agreement_rates)

        # Compute bar widths - for log scale, use ratio-based width per bin
        if use_log_bins:
            ax.set_xscale('log')
            # For log scale, bar width should be proportional to bin center
            bar_widths = [(bin_edges[i+1] - bin_edges[i]) * 0.8 for i in range(n_bins)]
        else:
            bar_widths = [(x_max - x_min) / n_bins * 0.8] * n_bins

        # Pastel colors
        color_bit = "#FFB6C1"   # Pastel pink for Bit
        color_best = "#FFFACD"  # Pastel yellow for Best

        # Three regions (since Best >= Bit):
        # - Best only (Bit to Best): solid pastel yellow
        # - Bit only: doesn't exist since Best >= Bit
        # - Overlap (0 to Bit): diagonal stripes of both colors

        # 1. Solid yellow for Best-only portion (above Bit)
        diff_rates = np.maximum(0, best_rates - bit_rates)
        ax.bar(bin_centers, diff_rates, width=bar_widths, bottom=bit_rates,
               color=color_best, edgecolor="none")

        # 2. Overlap region: pink fill with yellow diagonal stripes
        ax.bar(bin_centers, bit_rates, width=bar_widths,
               color=color_bit, edgecolor=color_best, hatch="//",
               linewidth=0.5)

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Agreement", fontsize=8)
        # Legend is added at figure level, not per subplot

        # Plot logit difference (right y-axis)
        ax2 = ax.twinx()
        color_line = "#404040"  # Dark gray for the line
        ax2.plot(bin_centers, logit_diffs, "o-", color=color_line, markersize=3,
                 linewidth=1.2, alpha=0.85)
        ax2.set_ylim(0, max(0.5, np.nanmax(logit_diffs) * 1.2) if logit_diffs else 0.5)
        ax2.set_ylabel("|Δ logit|", fontsize=8, color=color_line)
        ax2.tick_params(axis="y", labelcolor=color_line)

    # Per-input breakdown for noise samples (4 rows x 3 cols: SC | Gate | Agreement)
    with profile("robust_curves.per_input"):
        samples_by_base = {k: [] for k in input_keys}
        for sample in robustness.noise_samples:
            base_key = base_to_key.get((sample.base_input[0], sample.base_input[1]))
            if base_key:
                samples_by_base[base_key].append(sample)

        # Compute global y-axis range from all noise samples
        all_noise_outputs = []
        for sample in robustness.noise_samples:
            all_noise_outputs.append(sample.gate_output)
            all_noise_outputs.append(sample.subcircuit_output)
        if all_noise_outputs:
            noise_y_min = min(all_noise_outputs)
            noise_y_max = max(all_noise_outputs)
            # Add padding
            noise_y_pad = (noise_y_max - noise_y_min) * 0.05
            noise_y_min -= noise_y_pad
            noise_y_max += noise_y_pad
        else:
            noise_y_min, noise_y_max = -0.2, 1.2

        fig, axes = plt.subplots(4, 3, figsize=(15, 14))

        for row, key in enumerate(input_keys):
            samples = samples_by_base.get(key, [])
            gt = samples[0].ground_truth if samples else 0

            # Column 0: Subcircuit
            ax = axes[row, 0]
            if samples:
                signed_noise = [_compute_perturbation_effect(s) for s in samples]
                sc_outputs = [s.subcircuit_output for s in samples]
                sc_correct = [s.subcircuit_correct for s in samples]
                _plot_single_model(ax, samples, signed_noise, gt, sc_outputs, sc_correct)
            ax.set_ylabel(f"({key.replace('_', ',')})", fontsize=10, fontweight="bold")
            ax.set_ylim(noise_y_min, noise_y_max)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title("Subcircuit", fontsize=11, fontweight="bold")
            if row == 3:
                ax.set_xlabel("Perturbation Effect", fontsize=9)

            # Column 1: Gate
            ax = axes[row, 1]
            if samples:
                signed_noise = [_compute_perturbation_effect(s) for s in samples]
                gate_outputs = [s.gate_output for s in samples]
                gate_correct = [s.gate_correct for s in samples]
                _plot_single_model(ax, samples, signed_noise, gt, gate_outputs, gate_correct)
            ax.set_ylim(noise_y_min, noise_y_max)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title("Full Gate", fontsize=11, fontweight="bold")
            if row == 3:
                ax.set_xlabel("Perturbation Effect", fontsize=9)

            # Column 2: Agreement (binned)
            ax = axes[row, 2]
            if samples:
                signed_noise = [_compute_perturbation_effect(s) for s in samples]
                _plot_agreement_binned(ax, samples, signed_noise, n_bins=8)
            ax.grid(alpha=0.3, axis="y")
            if row == 0:
                ax.set_title("Agreement", fontsize=11, fontweight="bold")
            if row == 3:
                ax.set_xlabel("Perturbation Effect", fontsize=9)

        # Use tight_layout first with proper rect, then add title
        plt.tight_layout(rect=[0, 0.08, 1, 0.94])

        fig.suptitle(
            f"{prefix}Noise Robustness",
            fontsize=14, fontweight="bold", y=0.98
        )

        # Add figure-level legend at bottom
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        color_bit = "#FFB6C1"   # Pastel pink
        color_best = "#FFFACD"  # Pastel yellow
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=8, label='Correct'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Incorrect'),
            Patch(facecolor=color_bit, edgecolor="none", label="Bit"),
            Patch(facecolor=color_best, edgecolor="none", label="Best"),
            Patch(facecolor=color_bit, edgecolor=color_best, hatch="//", label="Overlap"),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5,
                   fontsize=9, framealpha=0.9, edgecolor="none", fancybox=False,
                   bbox_to_anchor=(0.5, 0.01))

        # Add definition text below legend
        fig.text(0.5, -0.02,
                 "Perturbation Effect = (perturbed_left - perturbed_right) - (base_left - base_right)",
                 ha='center', fontsize=8, style='italic', color='#555555')
        # Save in circuit_viz/noise_perturbations/summary.png
        noise_perturb_dir = os.path.join(output_dir, "circuit_viz", "noise_perturbations")
        os.makedirs(noise_perturb_dir, exist_ok=True)
        path = os.path.join(noise_perturb_dir, "summary.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths["noise_perturbations/summary"] = path

    # Per-input breakdown for OOD samples - create per-subtype summaries
    with profile("robust_curves.ood_per_input"):
        # Each subtype gets its own summary file
        # For multiply: summary_positive.png, summary_negative.png
        # For bimodal: summary.png, summary_inv.png
        # For add/subtract: summary.png
        ood_subtypes = [
            {
                "folder": "multiply",
                "sample_type": "multiply_positive",
                "title": "Multiply (positive)",
                "xlabel": "Scale Factor",
                "filename": "summary_positive.png",
                "skip_0_0": True,
            },
            {
                "folder": "multiply",
                "sample_type": "multiply_negative",
                "title": "Multiply (negative)",
                "xlabel": "Scale Factor",
                "filename": "summary_negative.png",
                "skip_0_0": True,
            },
            {
                "folder": "add",
                "sample_type": "add",
                "title": "Add",
                "xlabel": "Added Value",
                "filename": "summary.png",
                "skip_0_0": False,
            },
            {
                "folder": "subtract",
                "sample_type": "subtract",
                "title": "Subtract",
                "xlabel": "Subtracted Value",
                "filename": "summary.png",
                "skip_0_0": False,
            },
            {
                "folder": "bimodal",
                "sample_type": "bimodal",
                "title": "Bimodal",
                "xlabel": "Input",
                "filename": "summary.png",
                "skip_0_0": False,
            },
            {
                "folder": "bimodal",
                "sample_type": "bimodal_inv",
                "title": "Bimodal (inv)",
                "xlabel": "Input",
                "filename": "summary_inv.png",
                "skip_0_0": False,
            },
        ]

        for subtype_info in ood_subtypes:
            samples_by_base = {k: [] for k in input_keys}
            for sample in robustness.ood_samples:
                st = getattr(sample, "sample_type", "multiply_positive")
                if st == subtype_info["sample_type"]:
                    base_key = base_to_key.get((sample.base_input[0], sample.base_input[1]))
                    if base_key:
                        samples_by_base[base_key].append(sample)

            # Skip if no samples for this subtype
            total_samples = sum(len(s) for s in samples_by_base.values())
            if total_samples == 0:
                continue

            # Compute global y-axis range from samples
            all_outputs = []
            for samples in samples_by_base.values():
                for sample in samples:
                    all_outputs.append(sample.gate_output)
                    all_outputs.append(sample.subcircuit_output)
            if all_outputs:
                y_min = min(all_outputs)
                y_max = max(all_outputs)
                y_pad = (y_max - y_min) * 0.05
                y_min -= y_pad
                y_max += y_pad
            else:
                y_min, y_max = -0.2, 1.2

            # Skip 0_0 for multiply (it stays 0 regardless of scale)
            if subtype_info["skip_0_0"]:
                ood_input_keys = [k for k in input_keys if k != "0_0"]
                n_rows = 3
            else:
                ood_input_keys = input_keys
                n_rows = 4

            fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows - 2))

            for row, key in enumerate(ood_input_keys):
                samples = samples_by_base.get(key, [])
                gt = samples[0].ground_truth if samples else 0

                # Column 0: Subcircuit
                ax = axes[row, 0]
                if samples:
                    x_vals = [abs(s.noise_magnitude) for s in samples]
                    sc_outputs = [s.subcircuit_output for s in samples]
                    sc_correct = [s.subcircuit_correct for s in samples]
                    _plot_single_model(ax, samples, x_vals, gt, sc_outputs, sc_correct)
                ax.set_ylabel(f"({key.replace('_', ',')})", fontsize=10, fontweight="bold")
                ax.set_ylim(y_min, y_max)
                ax.grid(alpha=0.3)
                if row == 0:
                    ax.set_title("Subcircuit", fontsize=11, fontweight="bold")
                if row == n_rows - 1:
                    ax.set_xlabel(subtype_info["xlabel"], fontsize=9)

                # Column 1: Gate
                ax = axes[row, 1]
                if samples:
                    x_vals = [abs(s.noise_magnitude) for s in samples]
                    gate_outputs = [s.gate_output for s in samples]
                    gate_correct = [s.gate_correct for s in samples]
                    _plot_single_model(ax, samples, x_vals, gt, gate_outputs, gate_correct)
                ax.set_ylim(y_min, y_max)
                ax.grid(alpha=0.3)
                if row == 0:
                    ax.set_title("Full Gate", fontsize=11, fontweight="bold")
                if row == n_rows - 1:
                    ax.set_xlabel(subtype_info["xlabel"], fontsize=9)

                # Column 2: Agreement (binned)
                ax = axes[row, 2]
                if samples:
                    x_vals = [abs(s.noise_magnitude) for s in samples]
                    _plot_agreement_binned(ax, samples, x_vals, n_bins=8)
                ax.grid(alpha=0.3, axis="y")
                if row == 0:
                    ax.set_title("Agreement", fontsize=11, fontweight="bold")
                if row == n_rows - 1:
                    ax.set_xlabel(subtype_info["xlabel"], fontsize=9)

            # Use tight_layout first, then add title with proper spacing
            plt.tight_layout(rect=[0, 0.08, 1, 0.94])
            fig.suptitle(
                f"{prefix}OOD: {subtype_info['title']}",
                fontsize=14, fontweight="bold", y=0.98
            )

            # Add figure-level legend at bottom
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            color_bit = "#FFB6C1"   # Pastel pink
            color_best = "#FFFACD"  # Pastel yellow
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                       markersize=8, label='Correct'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=8, label='Incorrect'),
                Patch(facecolor=color_bit, edgecolor="none", label="Bit"),
                Patch(facecolor=color_best, edgecolor="none", label="Best"),
                Patch(facecolor=color_bit, edgecolor=color_best, hatch="//", label="Overlap"),
            ]
            fig.legend(handles=legend_elements, loc='lower center', ncol=5,
                       fontsize=9, framealpha=0.9, edgecolor="none", fancybox=False,
                       bbox_to_anchor=(0.5, 0.01))

            # Save in circuit_viz/out_distribution_transformations/{folder}/{filename}
            ood_type_dir = os.path.join(
                output_dir, "circuit_viz", "out_distribution_transformations", subtype_info["folder"]
            )
            os.makedirs(ood_type_dir, exist_ok=True)
            path = os.path.join(ood_type_dir, subtype_info["filename"])
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            paths[f"ood_{subtype_info['folder']}/{subtype_info['filename']}"] = path

    return paths


# ------------------ FAITHFULNESS VISUALIZATION ------------------


def _patch_key_to_filename(patch_key: str) -> str:
    """Convert patch key to readable filename."""
    import re

    layer_match = re.search(r"layers=\((\d+),?\)", patch_key)
    indices_match = re.search(r"indices=\(([^)]*)\)", patch_key)

    layer = layer_match.group(1) if layer_match else "0"
    indices_str = indices_match.group(1).replace(" ", "") if indices_match else ""

    if indices_str:
        indices_clean = indices_str.rstrip(",").replace(",", "_")
        return f"L{layer}_n{indices_clean}"
    else:
        return f"L{layer}_all"


def _faithfulness_score_to_color(score: float) -> tuple:
    """Pastel color gradient for faithfulness score: red (0) -> yellow (0.5) -> green (1)."""
    score = max(0, min(1, score))

    # Pastel colors (softer, less saturated)
    pastel_red = (1.0, 0.8, 0.8)      # #FFCCCC - soft pink-red
    pastel_yellow = (1.0, 0.98, 0.8)  # #FFFACD - lemon chiffon
    pastel_green = (0.8, 1.0, 0.8)    # #CCFFCC - soft mint green

    if score < 0.5:
        # Pastel red to pastel yellow
        t = score * 2
        r = pastel_red[0] + (pastel_yellow[0] - pastel_red[0]) * t
        g = pastel_red[1] + (pastel_yellow[1] - pastel_red[1]) * t
        b = pastel_red[2] + (pastel_yellow[2] - pastel_red[2]) * t
        return (r, g, b, 1.0)
    else:
        # Pastel yellow to pastel green
        t = (score - 0.5) * 2
        r = pastel_yellow[0] + (pastel_green[0] - pastel_yellow[0]) * t
        g = pastel_yellow[1] + (pastel_green[1] - pastel_yellow[1]) * t
        b = pastel_yellow[2] + (pastel_green[2] - pastel_yellow[2]) * t
        return (r, g, b, 1.0)


def visualize_faithfulness_intervention_effects(
    faithfulness: FaithfulnessMetrics,
    output_dir: str,
    gate_name: str = "",
) -> dict[str, str]:
    """
    Comprehensive faithfulness visualization suite.

    Creates in output_dir/:
    - interventional/in_circuit/*_stats.png - per-patch scatter plots
    - interventional/out_circuit/*_stats.png - per-patch scatter plots
    - interventional/in_distribution_summary.png - overview of all patches
    - interventional/out_distribution_summary.png - OOD overview
    - counterfactual/counterfact_summary.png - 2x2 matrix visualization
    - counterfactual/denoising_per_input.png - per-input denoising circuits
    - counterfactual/noising_per_input.png - per-input noising circuits
    """
    paths = {}
    prefix = f"{gate_name} - " if gate_name else ""

    def _plot_intervention_scatter(ax, samples, title, show_xlabel=True):
        """Plot intervention values vs outputs colored by agreement."""
        if not samples:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=10, fontweight="bold")
            return

        x_vals = [np.mean(s.intervention_values) if s.intervention_values else 0 for s in samples]
        outputs = [s.subcircuit_output if "SC" in title else s.gate_output for s in samples]
        correct = [s.bit_agreement for s in samples]

        sorted_data = sorted(zip(x_vals, outputs, correct), key=lambda d: d[0])
        sorted_x = [d[0] for d in sorted_data]
        sorted_out = [d[1] for d in sorted_data]
        sorted_correct = [d[2] for d in sorted_data]

        colors = ["#4CAF50" if c else "#E53935" for c in sorted_correct]
        ax.scatter(sorted_x, sorted_out, s=25, c=colors, alpha=0.7, edgecolors="none")
        ax.axhline(y=0.5, color="#888888", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_ylim(-0.2, 1.2)
        ax.set_title(title, fontsize=10, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel("Intervention Value (mode: add)", fontsize=9)
        ax.grid(alpha=0.3)

    def _plot_agreement_binned(ax, samples, n_bins=15, show_xlabel=True):
        """Plot binned agreement rate and output difference."""
        if not samples:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Agreement", fontsize=10, fontweight="bold")
            return

        x_vals = [np.mean(s.intervention_values) if s.intervention_values else 0 for s in samples]
        x_min, x_max = min(x_vals), max(x_vals)
        if x_min == x_max:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        agreement_rates = []
        output_diffs = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            bin_samples = [s for s, x in zip(samples, x_vals) if lo <= x < hi or (i == n_bins - 1 and x == hi)]
            if bin_samples:
                agreement_rates.append(np.mean([s.bit_agreement for s in bin_samples]))
                output_diffs.append(np.mean([abs(s.gate_output - s.subcircuit_output) for s in bin_samples]))
            else:
                agreement_rates.append(np.nan)
                output_diffs.append(np.nan)

        color1 = "#2196F3"
        ax.bar(bin_centers, agreement_rates, width=(x_max - x_min) / n_bins * 0.8, color=color1, alpha=0.6)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Agreement", color=color1, fontsize=8)
        ax.tick_params(axis="y", labelcolor=color1)

        ax2 = ax.twinx()
        color2 = "#FF5722"
        ax2.plot(bin_centers, output_diffs, "o-", color=color2, markersize=3, linewidth=1.5)
        max_diff = np.nanmax(output_diffs) if output_diffs and not np.all(np.isnan(output_diffs)) else 0.5
        ax2.set_ylim(0, max(0.5, max_diff * 1.2))
        ax2.set_ylabel("|Δ Output|", color=color2, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=color2)

        ax.set_title("Agreement", fontsize=10, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel("Intervention Value", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    # === 1. Per-patch intervention scatter plots (in interventional/{in,out}_circuit/*_stats.png) ===
    interventional_base = os.path.join(output_dir, "interventional")

    def _create_patch_figures(patch_stats_dict, circuit_type, base_dir):
        """Create scatter plots for all patches of a given type."""
        patch_paths = {}
        # Save in interventional/{in,out}_circuit/ with _stats suffix
        out_dir = os.path.join(base_dir, circuit_type)
        os.makedirs(out_dir, exist_ok=True)

        for patch_key, patch_stats in patch_stats_dict.items():
            samples = patch_stats.samples
            if not samples:
                continue

            filename = _patch_key_to_filename(patch_key)

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            _plot_intervention_scatter(axes[0], samples, "Subcircuit")
            _plot_intervention_scatter(axes[1], samples, "Full Gate")
            _plot_agreement_binned(axes[2], samples)

            layer = samples[0].patch_layer if samples else "?"
            indices = samples[0].patch_indices if samples else []
            indices_str = ",".join(map(str, indices)) if indices else "all"
            n_nodes = len(indices) if indices else 0

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            fig.suptitle(
                f"{prefix}{circuit_type} | L{layer} nodes=[{indices_str}] ({n_nodes} nodes) | mode=add",
                fontsize=12, fontweight="bold", y=0.99
            )

            # Save with _stats suffix to avoid collision with circuit diagrams
            path = os.path.join(out_dir, f"{filename}_stats.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            patch_paths[f"interventional/{circuit_type}/{filename}_stats"] = path

        return patch_paths

    if faithfulness.in_circuit_stats:
        paths.update(_create_patch_figures(faithfulness.in_circuit_stats, "in_circuit", interventional_base))
    if faithfulness.out_circuit_stats:
        paths.update(_create_patch_figures(faithfulness.out_circuit_stats, "out_circuit", interventional_base))

    # === 2. Intervention Summaries (in interventional/ folder) ===
    interventional_dir = interventional_base
    os.makedirs(interventional_dir, exist_ok=True)

    def _create_intervention_summary(in_stats, out_stats, title_suffix, filename):
        """Helper to create intervention summary bar chart."""
        all_patches = []
        for pk, ps in in_stats.items():
            all_patches.append(("in", pk, ps))
        for pk, ps in out_stats.items():
            all_patches.append(("out", pk, ps))

        def sort_key(item):
            import re
            layer_match = re.search(r"layers=\((\d+)", item[1])
            idx_match = re.search(r"indices=\((\d+)", item[1])
            return (int(layer_match.group(1)) if layer_match else 0,
                    int(idx_match.group(1)) if idx_match else 0)

        all_patches.sort(key=sort_key)

        if not all_patches:
            return None

        n_patches = len(all_patches)
        labels = [_patch_key_to_filename(p[1]) for p in all_patches]

        bit_sim = [p[2].mean_bit_similarity for p in all_patches]
        logit_sim = [p[2].mean_logit_similarity for p in all_patches]
        best_sim = [p[2].mean_best_similarity for p in all_patches]
        n_samples = [p[2].n_interventions for p in all_patches]
        is_in_circuit = [p[0] == "in" for p in all_patches]

        fig, ax = plt.subplots(1, 1, figsize=(max(14, n_patches * 1.5), 7))

        x = np.arange(n_patches)
        width = 0.25

        ax.bar(x - width, bit_sim, width, label="Bit Agreement", color="#4CAF50", alpha=0.8)
        ax.bar(x, logit_sim, width, label="Logit Similarity", color="#2196F3", alpha=0.8)
        ax.bar(x + width, best_sim, width, label="Best Similarity", color="#FF9800", alpha=0.8)

        for i, is_in in enumerate(is_in_circuit):
            marker = "▲" if is_in else "▼"
            color = "#77DD77" if is_in else "#6495ED"  # Pastel green/blue
            ax.text(x[i], -0.08, marker, ha="center", va="top", fontsize=10, color=color)

        ax.set_ylabel("Score", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(-0.15, 1.15)
        ax.axhline(y=0, color="#888888", linestyle="-", alpha=0.3)
        ax.axhline(y=0.5, color="#888888", linestyle="--", alpha=0.3)
        ax.axhline(y=1.0, color="#888888", linestyle="-", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

        for i, (xi, n) in enumerate(zip(x, n_samples)):
            ax.text(xi, 1.05, f"n={n}", ha="center", va="bottom", fontsize=7, color="#666666")

        ax.set_title(f"{prefix}{title_suffix}\n(▲=in-circuit, ▼=out-circuit)",
                     fontsize=13, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(interventional_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    # In-distribution summary
    path = _create_intervention_summary(
        faithfulness.in_circuit_stats,
        faithfulness.out_circuit_stats,
        "In-Distribution Summary",
        "in_distribution_summary.png"
    )
    if path:
        paths["interventional/in_distribution_summary"] = path

    # Out-of-distribution summary (if OOD stats exist)
    if faithfulness.in_circuit_stats_ood or faithfulness.out_circuit_stats_ood:
        path = _create_intervention_summary(
            faithfulness.in_circuit_stats_ood,
            faithfulness.out_circuit_stats_ood,
            "Out-of-Distribution Summary",
            "out_distribution_summary.png"
        )
        if path:
            paths["interventional/out_distribution_summary"] = path

    # === 3. Intervention Summary (2x2 Matrix of Patching Experiments) ===
    # Use new 2x2 matrix fields if available, fall back to legacy
    suff_cf = faithfulness.sufficiency_effects or []
    comp_cf = faithfulness.completeness_effects or []
    nec_cf = faithfulness.necessity_effects or []
    ind_cf = faithfulness.independence_effects or []

    # Legacy fallback
    out_cf = faithfulness.out_counterfactual_effects
    in_cf = faithfulness.in_counterfactual_effects

    # Build per-node scores from intervention stats (bit similarity)
    # Key: (layer, node_idx) -> score
    def _build_node_scores(stats_dict):
        """Extract per-node agreement scores from patch statistics."""
        import re
        scores = {}
        for patch_key, patch_stats in stats_dict.items():
            layer_match = re.search(r"layers=\((\d+),?\)", patch_key)
            idx_match = re.search(r"indices=\((\d+),?\)", patch_key)
            if layer_match and idx_match:
                layer = int(layer_match.group(1))
                node = int(idx_match.group(1))
                scores[(layer, node)] = patch_stats.mean_bit_similarity
        return scores

    in_bit_scores = _build_node_scores(faithfulness.in_circuit_stats)
    out_bit_scores = _build_node_scores(faithfulness.out_circuit_stats)

    # Build per-node faithfulness scores from counterfactual effects
    def _build_cf_node_scores(cf_effects, circuit_node_masks, is_in_circuit=True):
        """Extract mean faithfulness score per node from counterfactual effects."""
        if not cf_effects:
            return {}
        # Compute overall mean faithfulness score
        mean_score = np.mean([e.faithfulness_score for e in cf_effects])
        scores = {}
        # Apply to all relevant nodes
        for layer_idx in range(1, len(circuit_node_masks) - 1):
            for node_idx, active in enumerate(circuit_node_masks[layer_idx]):
                if (is_in_circuit and active == 1) or (not is_in_circuit and active == 0):
                    scores[(layer_idx, node_idx)] = mean_score
        return scores

    # Get circuit node masks for determining in/out circuit nodes
    circuit_node_masks = None
    if faithfulness.in_circuit_stats or faithfulness.out_circuit_stats:
        # Infer from stats
        all_keys = list(in_bit_scores.keys()) + list(out_bit_scores.keys())
        if all_keys:
            max_layer = max(k[0] for k in all_keys)
            # Build simple mask: nodes with in_circuit stats are in-circuit
            circuit_node_masks = [[1, 1]]  # Input layer
            for l in range(1, max_layer + 1):
                in_nodes = {k[1] for k in in_bit_scores.keys() if k[0] == l}
                out_nodes = {k[1] for k in out_bit_scores.keys() if k[0] == l}
                all_nodes = in_nodes | out_nodes
                max_node = max(all_nodes) + 1 if all_nodes else 3
                layer_mask = [1 if n in in_nodes else 0 for n in range(max_node)]
                circuit_node_masks.append(layer_mask)
            circuit_node_masks.append([1])  # Output layer

    # Compute faithfulness scores per node for all 4 experiments
    suff_scores = {}  # Denoise in-circuit
    comp_scores = {}  # Denoise out-circuit
    nec_scores = {}   # Noise in-circuit
    ind_scores = {}   # Noise out-circuit
    if circuit_node_masks:
        suff_scores = _build_cf_node_scores(suff_cf, circuit_node_masks, is_in_circuit=True)
        comp_scores = _build_cf_node_scores(comp_cf, circuit_node_masks, is_in_circuit=False)
        nec_scores = _build_cf_node_scores(nec_cf, circuit_node_masks, is_in_circuit=True)
        ind_scores = _build_cf_node_scores(ind_cf, circuit_node_masks, is_in_circuit=False)

    # Infer layer sizes from stats
    all_keys = list(in_bit_scores.keys()) + list(out_bit_scores.keys())
    if all_keys:
        max_layer = max(k[0] for k in all_keys)
        layer_sizes = [2]  # Input layer
        for l in range(1, max_layer + 1):
            nodes_in_layer = [k[1] for k in all_keys if k[0] == l]
            layer_sizes.append(max(nodes_in_layer) + 1 if nodes_in_layer else 3)
        layer_sizes.append(1)  # Output layer

        def _draw_metric_circuit(ax, out_scores, in_scores, title, layer_sizes):
            """Draw circuit with nodes colored by metric score."""
            G = nx.DiGraph()
            pos = _layout_cache.get_positions(tuple(layer_sizes))

            node_colors = []
            node_labels = {}

            for layer_idx, n_nodes in enumerate(layer_sizes):
                for node_idx in range(n_nodes):
                    name = f"({layer_idx},{node_idx})"
                    G.add_node(name)

                    # Check if in-circuit or out-circuit
                    in_score = in_scores.get((layer_idx, node_idx))
                    out_score = out_scores.get((layer_idx, node_idx))

                    if in_score is not None:
                        node_colors.append(_faithfulness_score_to_color(in_score))
                        node_labels[name] = f"{in_score:.2f}"
                    elif out_score is not None:
                        node_colors.append(_faithfulness_score_to_color(out_score))
                        node_labels[name] = f"{out_score:.2f}"
                    else:
                        node_colors.append((0.85, 0.85, 0.85, 1.0))
                        node_labels[name] = ""

            for l in range(len(layer_sizes) - 1):
                for i in range(layer_sizes[l]):
                    for j in range(layer_sizes[l + 1]):
                        G.add_edge(f"({l},{i})", f"({l+1},{j})")

            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="#888888", width=0.5)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=400,
                                   edgecolors="#555555", linewidths=1)
            nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=6, font_weight="bold")

            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

        # Create 2x2 matrix summary figure (counterfact_summary.png)
        # Rows: Denoising, Noising | Cols: In-Circuit, Out-Circuit
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))

        # Compute mean scores for each experiment
        suff_mean = faithfulness.mean_sufficiency if hasattr(faithfulness, 'mean_sufficiency') else (
            np.mean([e.faithfulness_score for e in suff_cf]) if suff_cf else 0.5)
        comp_mean = faithfulness.mean_completeness if hasattr(faithfulness, 'mean_completeness') else (
            np.mean([e.faithfulness_score for e in comp_cf]) if comp_cf else 0.5)
        nec_mean = faithfulness.mean_necessity if hasattr(faithfulness, 'mean_necessity') else (
            np.mean([e.faithfulness_score for e in nec_cf]) if nec_cf else 0.5)
        ind_mean = faithfulness.mean_independence if hasattr(faithfulness, 'mean_independence') else (
            np.mean([e.faithfulness_score for e in ind_cf]) if ind_cf else 0.5)

        # Row 1: DENOISING (run corrupted, patch with clean)
        _draw_metric_circuit(axes[0, 0], {}, suff_scores,
                            f"Sufficiency: {suff_mean:.2f}", layer_sizes)
        _draw_metric_circuit(axes[0, 1], comp_scores, {},
                            f"Completeness: {comp_mean:.2f}", layer_sizes)

        # Row 2: NOISING (run clean, patch with corrupted)
        _draw_metric_circuit(axes[1, 0], {}, nec_scores,
                            f"Necessity: {nec_mean:.2f}", layer_sizes)
        _draw_metric_circuit(axes[1, 1], ind_scores, {},
                            f"Independence: {ind_mean:.2f}", layer_sizes)

        # Apply tight_layout first to get proper subplot arrangement
        plt.tight_layout(rect=[0.08, 0.03, 1.0, 0.90])

        # Simple title with just gate name - positioned within the safe area
        fig.suptitle(prefix.strip(" -") if prefix else "Counterfactual Summary",
                     fontsize=12, fontweight="bold", y=0.96)

        # Column labels at top (positioned to not get cut off)
        fig.text(0.30, 0.92, "IN-CIRCUIT", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color="#6495ED")
        fig.text(0.73, 0.92, "OUT-CIRCUIT", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color="#DA70D6")

        # Row labels on left (pastel colors)
        fig.text(0.035, 0.68, "DENOISING", ha="center", va="center", fontsize=10,
                 fontweight="bold", rotation=90, color="#77DD77")
        fig.text(0.035, 0.26, "NOISING", ha="center", va="center", fontsize=10,
                 fontweight="bold", rotation=90, color="#FFB6C1")

        # Save in counterfactual/ subdirectory
        counterfactual_dir = os.path.join(output_dir, "counterfactual")
        os.makedirs(counterfactual_dir, exist_ok=True)
        path = os.path.join(counterfactual_dir, "counterfact_summary.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths["counterfactual/counterfact_summary"] = path

    # === 4. Per-Input Visualizations (split into denoising and noising) ===
    if suff_cf or comp_cf or nec_cf or ind_cf or out_cf or in_cf:
        base_to_key = {"(0, 0)": "0_0", "(0, 1)": "0_1", "(1, 0)": "1_0", "(1, 1)": "1_1"}

        # Group counterfactuals by clean_input
        cf_by_input = {k: {"sufficiency": [], "completeness": [], "necessity": [], "independence": []}
                       for k in base_to_key.values()}

        all_effects = [
            (suff_cf, "sufficiency"),
            (comp_cf, "completeness"),
            (nec_cf, "necessity"),
            (ind_cf, "independence"),
        ]

        for effects, score_type in all_effects:
            for e in effects:
                input_str = f"({int(e.clean_input[0])}, {int(e.clean_input[1])})"
                key = base_to_key.get(input_str)
                if key:
                    cf_by_input[key][score_type].append(e.faithfulness_score)

        # Legacy fallback
        if not suff_cf and out_cf:
            for e in out_cf:
                input_str = f"({int(e.clean_input[0])}, {int(e.clean_input[1])})"
                key = base_to_key.get(input_str)
                if key:
                    cf_by_input[key]["independence"].append(e.faithfulness_score)
        if not nec_cf and in_cf:
            for e in in_cf:
                input_str = f"({int(e.clean_input[0])}, {int(e.clean_input[1])})"
                key = base_to_key.get(input_str)
                if key:
                    cf_by_input[key]["necessity"].append(e.faithfulness_score)

        def _draw_per_input_figure(score_types, title, filename):
            """Helper to create per-input circuit visualization for 2 scores."""
            fig, axes = plt.subplots(2, 2, figsize=(10, 9))
            axes = axes.flatten()

            for i, input_key in enumerate(["0_0", "0_1", "1_0", "1_1"]):
                ax = axes[i]
                scores_dict = cf_by_input[input_key]

                # Get scores for the two types (in-circuit and out-circuit)
                in_score_type, out_score_type = score_types
                in_mean = np.mean(scores_dict[in_score_type]) if scores_dict[in_score_type] else 0.5
                out_mean = np.mean(scores_dict[out_score_type]) if scores_dict[out_score_type] else 0.5

                if layer_sizes:
                    G = nx.DiGraph()
                    pos = _layout_cache.get_positions(tuple(layer_sizes))

                    in_circuit_nodes, out_circuit_nodes, other_nodes = [], [], []
                    node_colors_dict, node_labels = {}, {}

                    for layer_idx, n_nodes in enumerate(layer_sizes):
                        for node_idx in range(n_nodes):
                            name = f"({layer_idx},{node_idx})"
                            G.add_node(name)

                            if (layer_idx, node_idx) in in_bit_scores:
                                node_colors_dict[name] = _faithfulness_score_to_color(in_mean)
                                node_labels[name] = f"{in_mean:.2f}"
                                in_circuit_nodes.append(name)
                            elif (layer_idx, node_idx) in out_bit_scores:
                                node_colors_dict[name] = _faithfulness_score_to_color(out_mean)
                                node_labels[name] = f"{out_mean:.2f}"
                                out_circuit_nodes.append(name)
                            else:
                                node_colors_dict[name] = (0.85, 0.85, 0.85, 1.0)
                                node_labels[name] = ""
                                other_nodes.append(name)

                    for l in range(len(layer_sizes) - 1):
                        for ii in range(layer_sizes[l]):
                            for jj in range(layer_sizes[l + 1]):
                                G.add_edge(f"({l},{ii})", f"({l+1},{jj})")

                    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="#888888", width=0.5)

                    if other_nodes:
                        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=other_nodes,
                                               node_color=[node_colors_dict[n] for n in other_nodes],
                                               node_size=400, edgecolors="#555555", linewidths=1)
                    if out_circuit_nodes:
                        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=out_circuit_nodes,
                                               node_color=[node_colors_dict[n] for n in out_circuit_nodes],
                                               node_size=400, edgecolors="#DA70D6", linewidths=2.5)
                    if in_circuit_nodes:
                        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=in_circuit_nodes,
                                               node_color=[node_colors_dict[n] for n in in_circuit_nodes],
                                               node_size=400, edgecolors="#6495ED", linewidths=2.5)

                    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=6, font_weight="bold")

                input_label = input_key.replace("_", ", ")
                in_label = in_score_type.capitalize()[:4]
                out_label = out_score_type.capitalize()[:4]

                # Create colored subtitle using multiple text elements
                # Position at top of axes in axes coordinates
                ax.text(0.5, 1.08, f"({input_label})  |  ", transform=ax.transAxes,
                       fontsize=9, fontweight="bold", ha="right", va="bottom")
                ax.text(0.5, 1.08, f"{in_label}={in_mean:.2f}", transform=ax.transAxes,
                       fontsize=9, fontweight="bold", ha="left", va="bottom", color="#6495ED")
                ax.text(0.72, 1.08, f"  {out_label}={out_mean:.2f}", transform=ax.transAxes,
                       fontsize=9, fontweight="bold", ha="left", va="bottom", color="#DA70D6")
                ax.axis("off")

            fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, hspace=0.12, wspace=0.08)
            path = os.path.join(counterfactual_dir, filename)
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return path

        # Denoising: Sufficiency (in-circuit) + Completeness (out-circuit)
        path = _draw_per_input_figure(
            ("sufficiency", "completeness"),
            "Denoising",
            "denoising_per_input.png"
        )
        paths["counterfactual/denoising_per_input"] = path

        # Noising: Necessity (in-circuit) + Independence (out-circuit)
        path = _draw_per_input_figure(
            ("necessity", "independence"),
            "Noising",
            "noising_per_input.png"
        )
        paths["counterfactual/noising_per_input"] = path

    return paths


def _generate_faithfulness_circuit_figure(args):
    """Worker function for parallel faithfulness circuit generation.

    Uses draw_intervened_circuit for consistent visualization across all circuit_viz.
    Shows intervened network with original values (grey, small) below modified values.
    """
    (
        effect_dict,
        circuit_dict,
        weights,
        intervened_nodes,
        output_path,
        fig_type,
        index,
    ) = args

    circuit = Circuit.from_dict(circuit_dict)

    # Get activations as flat lists [layer][node]
    clean_acts = effect_dict["clean_activations"]
    corrupt_acts = effect_dict["corrupted_activations"]
    intervened_acts = effect_dict.get("intervened_activations", clean_acts)

    # Determine experiment type for proper labeling
    experiment_type = effect_dict.get("experiment_type", "noising")
    is_denoising = experiment_type == "denoising"

    # Compute layer sizes from activations
    layer_sizes = [len(layer) for layer in clean_acts]

    # Use slightly taller figure to avoid title occlusion
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # Simplified panel labels
    # Panel 1: Clean
    clean_label = "Clean (source)" if is_denoising else "Clean (base)"
    draw_intervened_circuit(
        axes[0],
        layer_sizes=layer_sizes,
        weights=weights,
        current_activations=clean_acts,
        original_activations=None,
        intervened_nodes=set(),
        circuit=circuit,
        title=f"{clean_label}: {effect_dict['expected_clean_output']:.2f}",
    )

    # Panel 2: Corrupted
    corrupt_label = "Corrupted (base)" if is_denoising else "Corrupted (source)"
    draw_intervened_circuit(
        axes[1],
        layer_sizes=layer_sizes,
        weights=weights,
        current_activations=corrupt_acts,
        original_activations=None,
        intervened_nodes=set(),
        circuit=circuit,
        title=f"{corrupt_label}: {effect_dict['expected_corrupted_output']:.2f}",
    )

    # Panel 3: Intervened
    # For denoising: run corrupted, patch with clean
    # For noising: run clean, patch with corrupted
    if is_denoising:
        original_acts = corrupt_acts  # Base was corrupted
        intervened_label = "Patched"
    else:
        original_acts = clean_acts  # Base was clean
        intervened_label = "Patched"

    draw_intervened_circuit(
        axes[2],
        layer_sizes=layer_sizes,
        weights=weights,
        current_activations=intervened_acts,
        original_activations=original_acts,
        intervened_nodes=intervened_nodes,
        circuit=circuit,
        title=f"{intervened_label}: {effect_dict['actual_output']:.2f}",
    )

    clean_str = ",".join(f"{v:.0f}" for v in effect_dict["clean_input"])
    corrupt_str = ",".join(f"{v:.0f}" for v in effect_dict["corrupted_input"])

    # Determine experiment type and score type for title
    score_type = effect_dict.get("score_type", fig_type)

    # Build descriptive experiment label
    # Sufficiency/Completeness = Denoising, Necessity/Independence = Noising
    if score_type in ("sufficiency", "completeness"):
        circuit_type = "In-Circuit" if score_type == "sufficiency" else "Out-of-Circuit"
        exp_label = f"{circuit_type} Denoising"
    else:
        circuit_type = "In-Circuit" if score_type == "necessity" else "Out-of-Circuit"
        exp_label = f"{circuit_type} Noising"

    score_display = score_type.capitalize()

    # Simplified title format:
    # "Counterfactual (1,0)→(1,1) | Out-of-Circuit Noising | Independence: 1.00"
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave room for title
    fig.suptitle(
        f"Counterfactual ({clean_str})→({corrupt_str})  |  {exp_label}  |  "
        f"{score_display}: {effect_dict['faithfulness_score']:.2f}",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def visualize_faithfulness_circuit_samples(
    faithfulness: FaithfulnessMetrics,
    circuit: Circuit,
    layer_weights: list[torch.Tensor],
    output_dir: str,
    n_samples_per_grid: int = 8,
    layer_biases: list[torch.Tensor] | None = None,
) -> dict[str, str]:
    """Visualize circuit diagrams showing interventions.

    Args:
        layer_biases: Optional bias vectors. If provided, edge labels show
            (weight + bias) to reveal bias contribution when edges are patched.
    """
    # Create organized folder structure inside output_dir (which is already .../faithfulness):
    # counterfactual/{sufficiency,completeness,necessity,independence}/
    # interventional/{in_circuit,out_circuit}/
    counterfactual_dir = os.path.join(output_dir, "counterfactual")
    interventional_dir = os.path.join(output_dir, "interventional")
    os.makedirs(counterfactual_dir, exist_ok=True)
    os.makedirs(interventional_dir, exist_ok=True)
    paths = {}

    weights = [w.numpy() for w in layer_weights]
    biases = [b.numpy() for b in layer_biases] if layer_biases else None
    # Use full circuit for counterfactual viz (we run full model with interventions)
    layer_sizes = [layer_weights[0].shape[1]] + [w.shape[0] for w in layer_weights]
    full_circuit = Circuit.full(layer_sizes)
    circuit_dict = full_circuit.to_dict()

    # Get out-circuit nodes
    out_circuit_nodes = set()
    for layer_idx in range(1, len(circuit.node_masks) - 1):
        for node_idx, active in enumerate(circuit.node_masks[layer_idx]):
            if active == 0:
                out_circuit_nodes.add(f"({layer_idx},{node_idx})")

    # Get in-circuit nodes
    in_circuit_nodes = set()
    for layer_idx in range(1, len(circuit.node_masks) - 1):
        for node_idx, active in enumerate(circuit.node_masks[layer_idx]):
            if active == 1:
                in_circuit_nodes.add(f"({layer_idx},{node_idx})")

    # Prepare parallel tasks
    tasks = []

    # Helper to add counterfactual effects to tasks
    def _add_counterfactual_tasks(effects, score_type, target_nodes):
        """Add counterfactual visualization tasks."""
        if not effects:
            return
        cf_subdir = os.path.join(counterfactual_dir, score_type)
        os.makedirs(cf_subdir, exist_ok=True)

        for i, effect in enumerate(effects[:6]):
            if not effect.clean_activations or not effect.corrupted_activations:
                continue

            effect_dict = {
                "clean_activations": effect.clean_activations,
                "corrupted_activations": effect.corrupted_activations,
                "intervened_activations": effect.intervened_activations,
                "expected_clean_output": effect.expected_clean_output,
                "expected_corrupted_output": effect.expected_corrupted_output,
                "actual_output": effect.actual_output,
                "output_changed_to_corrupted": effect.output_changed_to_corrupted,
                "faithfulness_score": effect.faithfulness_score,
                "clean_input": effect.clean_input,
                "corrupted_input": effect.corrupted_input,
                "experiment_type": getattr(effect, 'experiment_type', 'noising'),
                "score_type": getattr(effect, 'score_type', score_type),
            }
            output_path = os.path.join(cf_subdir, f"{i}.png")
            tasks.append(
                (
                    effect_dict,
                    circuit_dict,
                    weights,
                    target_nodes,
                    output_path,
                    score_type,  # Just the score type, title built in worker
                    i,
                )
            )

        paths[f"counterfactual/{score_type}"] = cf_subdir

    # ===== 2x2 Matrix Counterfactuals =====
    # Denoising experiments (run corrupted, patch with clean)
    _add_counterfactual_tasks(
        faithfulness.sufficiency_effects,
        "sufficiency",
        in_circuit_nodes,
    )
    _add_counterfactual_tasks(
        faithfulness.completeness_effects,
        "completeness",
        out_circuit_nodes,
    )

    # Noising experiments (run clean, patch with corrupted)
    _add_counterfactual_tasks(
        faithfulness.necessity_effects,
        "necessity",
        in_circuit_nodes,
    )
    _add_counterfactual_tasks(
        faithfulness.independence_effects,
        "independence",
        out_circuit_nodes,
    )

    # Legacy fallback: use old fields if new ones are empty
    if not faithfulness.sufficiency_effects and faithfulness.out_counterfactual_effects:
        _add_counterfactual_tasks(
            faithfulness.out_counterfactual_effects,
            "independence",  # Old out-circuit was mislabeled
            out_circuit_nodes,
        )

    if not faithfulness.necessity_effects and faithfulness.in_counterfactual_effects:
        _add_counterfactual_tasks(
            faithfulness.in_counterfactual_effects,
            "necessity",
            in_circuit_nodes,
        )

    # Execute in parallel
    if tasks:
        n_workers = min(len(tasks), mp.cpu_count())
        print(
            f"[VIZ] Generating {len(tasks)} faithfulness circuit figures with {n_workers} workers"
        )
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            list(executor.map(_generate_faithfulness_circuit_figure, tasks))

    # In-circuit/out-circuit patch visualizations (simpler, do sequentially)
    def visualize_patch_circuits(stats: dict, circuit_type: str, out_dir: str) -> dict:
        """Visualize patch intervention circuits using reusable draw_intervened_circuit.

        Shows the intervened circuit with marked nodes (orange border).
        No text labels - just the circuit with activation values in nodes.
        """
        os.makedirs(out_dir, exist_ok=True)
        type_paths = {}

        layer_sizes = [layer_weights[0].shape[1]] + [w.shape[0] for w in layer_weights]
        full_circuit = Circuit.full(layer_sizes)

        for patch_key, patch_stats in stats.items():
            samples = patch_stats.samples
            if not samples:
                continue

            # Format patch label
            import re

            layer_match = re.search(r"layers=\((\d+),?\)", patch_key)
            indices_match = re.search(r"indices=\(([^)]*)\)", patch_key)
            patch_layer = int(layer_match.group(1)) if layer_match else 0
            patch_indices = []
            if indices_match and indices_match.group(1).strip():
                patch_indices = [
                    int(x.strip())
                    for x in indices_match.group(1).split(",")
                    if x.strip()
                ]
            patch_label = f"L{patch_layer}_{'_'.join(map(str, patch_indices))}"

            # Build intervened nodes set
            intervened_nodes = {f"({patch_layer},{idx})" for idx in patch_indices}

            # Select samples
            disagree = [s for s in samples if not s.bit_agreement]
            agree = [s for s in samples if s.bit_agreement]
            selected = (
                disagree[:n_samples_per_grid]
                if len(disagree) >= n_samples_per_grid
                else disagree + agree[: n_samples_per_grid - len(disagree)]
            )

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

                # Get activations for this sample
                # Structure is [layer][batch][node] - extract first batch sample
                sc_acts = sample.subcircuit_activations if sample.subcircuit_activations else []
                gate_acts = sample.gate_activations if sample.gate_activations else []
                # Original (unpatched) activations for two-value display
                orig_sc_acts = getattr(sample, 'original_subcircuit_activations', None) or []
                orig_gate_acts = getattr(sample, 'original_gate_activations', None) or []

                def extract_activations(acts):
                    """Convert activations to [layer][node] format.

                    Handles both formats:
                    - [layer][node] (from mean across batch) - return as-is
                    - [layer][batch][node] - extract first batch sample
                    """
                    if not acts:
                        return []
                    result = []
                    for layer in acts:
                        if isinstance(layer, (list, tuple)) and len(layer) > 0:
                            first_elem = layer[0]
                            # Check if first element is a number (format: [layer][node])
                            # or a list/tuple (format: [layer][batch][node])
                            if isinstance(first_elem, (int, float)):
                                # Already [layer][node] format - use directly
                                result.append(list(layer))
                            elif isinstance(first_elem, (list, tuple)):
                                # [layer][batch][node] format - extract first batch
                                result.append(list(first_elem))
                            else:
                                result.append([])
                        else:
                            result.append([])
                    return result

                sc_acts_flat = extract_activations(sc_acts)
                gate_acts_flat = extract_activations(gate_acts)
                orig_sc_acts_flat = extract_activations(orig_sc_acts)
                orig_gate_acts_flat = extract_activations(orig_gate_acts)

                for col, (label, acts_flat, orig_acts_flat, circ) in enumerate(
                    [
                        ("Subcircuit", sc_acts_flat, orig_sc_acts_flat, circuit),
                        ("Full Model", gate_acts_flat, orig_gate_acts_flat, full_circuit),
                    ]
                ):
                    ax = axes[i, col]

                    # Use reusable function - pass original activations for two-value display
                    draw_intervened_circuit(
                        ax,
                        layer_sizes=layer_sizes,
                        weights=weights,
                        current_activations=acts_flat,
                        original_activations=orig_acts_flat if orig_acts_flat else None,
                        intervened_nodes=intervened_nodes,
                        circuit=circ,
                        title=None,
                        node_size=400,
                        show_edge_labels=True,
                        biases=biases,
                    )

            axes[0, 0].set_title("Subcircuit", fontsize=10, fontweight="bold")
            axes[0, 1].set_title("Full Model", fontsize=10, fontweight="bold")

            # Calculate agreement % over ALL samples, not just visualized ones
            total_samples = len(samples)
            total_agree = sum(1 for s in samples if s.bit_agreement)
            agree_pct = 100 * total_agree / total_samples if total_samples > 0 else 0

            plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave space for title
            fig.suptitle(
                f"{patch_label} | {circuit_type} | Agreement: {agree_pct:.0f}% (n={total_samples})",
                fontsize=11,
                fontweight="bold",
                y=0.99,
            )

            path = os.path.join(out_dir, f"{patch_label}.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            type_paths[patch_label] = path

        return type_paths

    # Create interventional visualizations with in_distribution/out_of_distribution subfolders
    in_circuit_base = os.path.join(interventional_dir, "in_circuit")
    out_circuit_base = os.path.join(interventional_dir, "out_circuit")

    # In-distribution interventions
    if faithfulness.in_circuit_stats:
        in_circuit_id_dir = os.path.join(in_circuit_base, "in_distribution")
        paths["interventional/in_circuit/in_distribution"] = visualize_patch_circuits(
            faithfulness.in_circuit_stats,
            "In-Circuit (ID)",
            in_circuit_id_dir,
        )

    if faithfulness.out_circuit_stats:
        out_circuit_id_dir = os.path.join(out_circuit_base, "in_distribution")
        paths["interventional/out_circuit/in_distribution"] = visualize_patch_circuits(
            faithfulness.out_circuit_stats,
            "Out-of-Circuit (ID)",
            out_circuit_id_dir,
        )

    # Out-of-distribution interventions (if OOD stats exist)
    if faithfulness.in_circuit_stats_ood:
        in_circuit_ood_dir = os.path.join(in_circuit_base, "out_of_distribution")
        paths["interventional/in_circuit/out_of_distribution"] = visualize_patch_circuits(
            faithfulness.in_circuit_stats_ood,
            "In-Circuit (OOD)",
            in_circuit_ood_dir,
        )

    if faithfulness.out_circuit_stats_ood:
        out_circuit_ood_dir = os.path.join(out_circuit_base, "out_of_distribution")
        paths["interventional/out_circuit/out_of_distribution"] = visualize_patch_circuits(
            faithfulness.out_circuit_stats_ood,
            "Out-of-Circuit (OOD)",
            out_circuit_ood_dir,
        )

    return paths


# ------------------ SPD VISUALIZATION ------------------


def visualize_spd_components(
    decomposed: DecomposedMLP,
    output_dir: str,
    filename: str = "spd_components.png",
    gate_name: str = "",
) -> str | None:
    """Visualize SPD component weights."""
    weights = _get_spd_component_weights(decomposed)
    if weights is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(weights))
    bars = ax.bar(x, weights, color="steelblue", alpha=0.8)

    # Highlight top components
    top_k = min(3, len(weights))
    top_indices = np.argsort(weights)[-top_k:]
    for idx in top_indices:
        bars[idx].set_color("coral")

    ax.set_xlabel("Component Index")
    ax.set_ylabel("Normalized Weight")
    ax.set_title(
        f"{gate_name} - SPD Component Importance"
        if gate_name
        else "SPD Component Importance",
        fontweight="bold",
    )
    ax.set_xticks(x)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------ PROFILING VISUALIZATION ------------------


def visualize_profiling_timeline(
    profiling: ProfilingData,
    output_dir: str,
    filename: str = "timeline.png",
) -> str | None:
    """Visualize profiling timeline."""
    if not profiling.events:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    events = profiling.events
    start_time = events[0].timestamp_ms if events else 0

    # Group by phase
    phases = {}
    for event in events:
        phase = event.status.split(":")[0] if ":" in event.status else event.status
        if phase not in phases:
            phases[phase] = []
        phases[phase].append((event.timestamp_ms - start_time) / 1000.0)

    colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))
    y_pos = 0
    for (phase, times), color in zip(phases.items(), colors):
        for t in times:
            ax.barh(y_pos, 0.1, left=t, color=color, alpha=0.8)
        ax.text(-0.5, y_pos, phase[:20], ha="right", va="center", fontsize=8)
        y_pos += 1

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase")
    ax.set_title("Trial Timeline", fontweight="bold")
    ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_profiling_phases(
    profiling: ProfilingData,
    output_dir: str,
    filename: str = "phases.png",
) -> str | None:
    """Visualize profiling phase durations."""
    if not profiling.events:
        return None

    # Compute phase durations
    phase_durations = {}
    events = profiling.events
    for i, event in enumerate(events):
        if event.status.startswith("STARTED_"):
            phase = event.status.replace("STARTED_", "")
            end_status = (
                f"ENDED_{phase}"
                if f"ENDED_{phase}" in [e.status for e in events]
                else f"FINISHED_{phase}"
            )
            for j in range(i + 1, len(events)):
                if events[j].status == end_status:
                    duration = (events[j].timestamp_ms - event.timestamp_ms) / 1000.0
                    phase_durations[phase] = phase_durations.get(phase, 0) + duration
                    break

    if not phase_durations:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    phases = list(phase_durations.keys())
    durations = [phase_durations[p] for p in phases]

    bars = ax.barh(phases, durations, color="steelblue", alpha=0.8)
    ax.set_xlabel("Duration (s)")
    ax.set_title("Phase Durations", fontweight="bold")

    for bar, d in zip(bars, durations):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{d:.2f}s",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_profiling_summary(
    profiling: ProfilingData,
    output_dir: str,
    filename: str = "summary.png",
) -> str | None:
    """Visualize profiling summary."""
    if not profiling.events:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Event count by phase
    ax = axes[0]
    phase_counts = {}
    for event in profiling.events:
        phase = event.status.split(":")[0] if ":" in event.status else event.status
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    phases = list(phase_counts.keys())[:15]
    counts = [phase_counts[p] for p in phases]
    ax.barh(phases, counts, color="coral", alpha=0.8)
    ax.set_xlabel("Count")
    ax.set_title("Events by Phase", fontweight="bold")

    # Right: Text summary
    ax = axes[1]
    ax.axis("off")
    total_time = (
        (profiling.events[-1].timestamp_ms - profiling.events[0].timestamp_ms) / 1000.0
        if len(profiling.events) > 1
        else 0
    )
    summary_text = f"""
    Total Events: {len(profiling.events)}
    Total Time: {total_time:.2f}s
    Unique Phases: {len(phase_counts)}
    """
    ax.text(
        0.1,
        0.5,
        summary_text.strip(),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------ METRICS COMPUTATION AND JSON SAVING ------------------


def compute_observational_metrics(robustness: "RobustnessMetrics") -> dict:
    """Compute ObservationalMetrics from RobustnessMetrics with nested structure.

    Returns nested dict structure:
    {
        "noise_perturbations": {
            "n_samples": 100,
            "gate_accuracy": 0.95,
            "subcircuit_accuracy": 0.93,
            "agreement_bit": 0.98,
            "agreement_best": 0.99,
            "mse_mean": 0.001
        },
        "out_distribution_transformations": {
            "multiply": {
                "positive": {"n_samples": 50, "agreement": 0.92},
                "negative": {"n_samples": 50, "agreement": 0.88}
            },
            "add": {"n_samples": 100, "agreement": 0.95},
            ...
        },
        "overall_score": 0.91
    }
    """
    # Noise perturbation metrics
    noise_samples = robustness.noise_samples
    noise_metrics = {
        "n_samples": len(noise_samples) if noise_samples else 0,
        "gate_accuracy": robustness.noise_gate_accuracy,
        "subcircuit_accuracy": robustness.noise_subcircuit_accuracy,
        "agreement_bit": robustness.noise_agreement_bit,
        "agreement_best": robustness.noise_agreement_best,
        "mse_mean": robustness.noise_mse_mean,
    }

    # Group OOD samples by type
    ood_by_type: dict[str, list] = {}
    for sample in robustness.ood_samples:
        st = getattr(sample, "sample_type", "multiply_positive")
        ood_by_type.setdefault(st, []).append(sample)

    # Compute per-type metrics
    def compute_ood_metrics(samples):
        n = len(samples)
        agreement = sum(s.agreement_bit for s in samples) / n if n > 0 else 0.0
        return {"n_samples": n, "agreement": agreement}

    multiply_pos = compute_ood_metrics(ood_by_type.get("multiply_positive", []))
    multiply_neg = compute_ood_metrics(ood_by_type.get("multiply_negative", []))
    add_metrics = compute_ood_metrics(ood_by_type.get("add", []))
    subtract_metrics = compute_ood_metrics(ood_by_type.get("subtract", []))
    bimodal_order = compute_ood_metrics(ood_by_type.get("bimodal", []))
    bimodal_inv = compute_ood_metrics(ood_by_type.get("bimodal_inv", []))

    ood_metrics = {
        "multiply": {
            "positive": multiply_pos,
            "negative": multiply_neg,
        },
        "add": add_metrics,
        "subtract": subtract_metrics,
        "bimodal": {
            "order_preserving": bimodal_order,
            "inverted": bimodal_inv,
        },
    }

    # Overall score: average of all agreements
    agreements = [
        noise_metrics["agreement_bit"],
        multiply_pos["agreement"],
        multiply_neg["agreement"],
        add_metrics["agreement"],
        subtract_metrics["agreement"],
        bimodal_order["agreement"],
        bimodal_inv["agreement"],
    ]
    n_valid = sum(1 for a in agreements if a > 0)
    overall = sum(agreements) / n_valid if n_valid > 0 else 0.0

    return {
        "noise_perturbations": noise_metrics,
        "out_distribution_transformations": ood_metrics,
        "overall_score": overall,
    }


def compute_interventional_metrics(faithfulness: "FaithfulnessMetrics") -> dict:
    """Compute InterventionalMetrics from FaithfulnessMetrics with nested structure.

    Returns nested dict structure:
    {
        "in_circuit": {
            "in_distribution": {"n_interventions": 10, "mean_bit_similarity": 0.95, "mean_logit_similarity": 0.92},
            "out_distribution": {"n_interventions": 10, "mean_bit_similarity": 0.88, "mean_logit_similarity": 0.85}
        },
        "out_circuit": {
            "in_distribution": {...},
            "out_distribution": {...}
        },
        "overall_score": 0.90
    }
    """
    def compute_stats(stats_dict):
        if not stats_dict:
            return {"n_interventions": 0, "mean_bit_similarity": 0.0, "mean_logit_similarity": 0.0}
        bit_sims = [ps.mean_bit_similarity for ps in stats_dict.values()]
        logit_sims = [ps.mean_logit_similarity for ps in stats_dict.values()]
        return {
            "n_interventions": len(stats_dict),
            "mean_bit_similarity": sum(bit_sims) / len(bit_sims) if bit_sims else 0.0,
            "mean_logit_similarity": sum(logit_sims) / len(logit_sims) if logit_sims else 0.0,
        }

    in_circuit_id = compute_stats(faithfulness.in_circuit_stats)
    in_circuit_ood = compute_stats(faithfulness.in_circuit_stats_ood)
    out_circuit_id = compute_stats(faithfulness.out_circuit_stats)
    out_circuit_ood = compute_stats(faithfulness.out_circuit_stats_ood)

    # Overall: average of bit similarities
    sims = [
        in_circuit_id["mean_bit_similarity"],
        in_circuit_ood["mean_bit_similarity"],
        out_circuit_id["mean_bit_similarity"],
        out_circuit_ood["mean_bit_similarity"],
    ]
    n_valid = sum(1 for s in sims if s > 0)
    overall = sum(sims) / n_valid if n_valid > 0 else 0.0

    return {
        "in_circuit": {
            "in_distribution": in_circuit_id,
            "out_distribution": in_circuit_ood,
        },
        "out_circuit": {
            "in_distribution": out_circuit_id,
            "out_distribution": out_circuit_ood,
        },
        "overall_score": overall,
    }


def compute_counterfactual_metrics(faithfulness: "FaithfulnessMetrics") -> dict:
    """Compute CounterfactualMetrics from FaithfulnessMetrics with nested structure.

    Returns nested dict structure matching 2x2 matrix:
    {
        "denoising": {
            "in_circuit": {"score": 0.95, "label": "sufficiency"},
            "out_circuit": {"score": 0.92, "label": "completeness"},
            "n_pairs": 40
        },
        "noising": {
            "in_circuit": {"score": 0.88, "label": "necessity"},
            "out_circuit": {"score": 0.90, "label": "independence"},
            "n_pairs": 40
        },
        "overall_score": 0.91
    }
    """
    suff = faithfulness.mean_sufficiency
    comp = faithfulness.mean_completeness
    nec = faithfulness.mean_necessity
    ind = faithfulness.mean_independence

    n_denoising = len(faithfulness.sufficiency_effects) + len(faithfulness.completeness_effects)
    n_noising = len(faithfulness.necessity_effects) + len(faithfulness.independence_effects)

    # Overall: average of all counterfactual scores
    scores = [suff, comp, nec, ind]
    n_valid = sum(1 for s in scores if s > 0)
    overall = sum(scores) / n_valid if n_valid > 0 else 0.0

    return {
        "denoising": {
            "in_circuit": {"score": suff, "label": "sufficiency"},
            "out_circuit": {"score": comp, "label": "completeness"},
            "n_pairs": n_denoising,
        },
        "noising": {
            "in_circuit": {"score": nec, "label": "necessity"},
            "out_circuit": {"score": ind, "label": "independence"},
            "n_pairs": n_noising,
        },
        "overall_score": overall,
    }


def save_faithfulness_json(
    observational_dir: str | None,
    interventional_dir: str | None,
    counterfactual_dir: str | None,
    faithfulness_dir: str,
    robustness: "RobustnessMetrics | None",
    faithfulness: "FaithfulnessMetrics | None",
) -> dict[str, str]:
    """Save result.json in each subfolder and summary.json in faithfulness/."""
    import json
    from identifiability_toy_study.common.schemas import FaithfulnessSummary

    paths = {}
    obs_overall = 0.0
    int_overall = 0.0
    cf_overall = 0.0

    # Observational result.json
    if observational_dir and robustness:
        obs_metrics = compute_observational_metrics(robustness)
        obs_overall = obs_metrics.get("overall_score", 0.0)
        path = os.path.join(observational_dir, "result.json")
        with open(path, "w") as f:
            json.dump(obs_metrics, f, indent=2)
        paths["observational/result.json"] = path

    # Interventional result.json
    if interventional_dir and faithfulness:
        int_metrics = compute_interventional_metrics(faithfulness)
        int_overall = int_metrics.get("overall_score", 0.0)
        path = os.path.join(interventional_dir, "result.json")
        with open(path, "w") as f:
            json.dump(int_metrics, f, indent=2)
        paths["interventional/result.json"] = path

    # Counterfactual result.json
    if counterfactual_dir and faithfulness:
        cf_metrics = compute_counterfactual_metrics(faithfulness)
        cf_overall = cf_metrics.get("overall_score", 0.0)
        path = os.path.join(counterfactual_dir, "result.json")
        with open(path, "w") as f:
            json.dump(cf_metrics, f, indent=2)
        paths["counterfactual/result.json"] = path

    # Compute epsilon from the SAME component scores used for overall score
    # Epsilon = min distance from 1.0 across components (always positive magnitude)
    from identifiability_toy_study.common.schemas import FaithfulnessCategoryScore

    # Observational: epsilon from the 7 agreement rates that make up the overall
    obs_epsilon = 0.0
    if robustness:
        obs_component_scores = [robustness.noise_agreement_bit]
        # Get per-type OOD agreements
        ood_by_type: dict[str, list] = {}
        for sample in robustness.ood_samples:
            st = getattr(sample, "sample_type", "multiply_positive")
            ood_by_type.setdefault(st, []).append(sample)
        for st in ["multiply_positive", "multiply_negative", "add", "subtract", "bimodal", "bimodal_inv"]:
            samples = ood_by_type.get(st, [])
            if samples:
                agreement = sum(s.agreement_bit for s in samples) / len(samples)
                obs_component_scores.append(agreement)
        # Filter to only valid (non-zero) scores that contributed to overall
        obs_valid_scores = [s for s in obs_component_scores if s > 0]
        if obs_valid_scores:
            obs_epsilon = abs(min(1.0 - s for s in obs_valid_scores))

    # Interventional: epsilon from the 4 mean bit similarities
    int_epsilon = 0.0
    if faithfulness:
        int_component_scores = []
        for stats_dict in [faithfulness.in_circuit_stats, faithfulness.out_circuit_stats,
                          faithfulness.in_circuit_stats_ood, faithfulness.out_circuit_stats_ood]:
            if stats_dict:
                bit_sims = [ps.mean_bit_similarity for ps in stats_dict.values()]
                if bit_sims:
                    int_component_scores.append(sum(bit_sims) / len(bit_sims))
        int_valid_scores = [s for s in int_component_scores if s > 0]
        if int_valid_scores:
            int_epsilon = abs(min(1.0 - s for s in int_valid_scores))

    # Counterfactual: epsilon from the 4 mean scores (suff, comp, nec, ind)
    cf_epsilon = 0.0
    if faithfulness:
        cf_component_scores = [
            faithfulness.mean_sufficiency,
            faithfulness.mean_completeness,
            faithfulness.mean_necessity,
            faithfulness.mean_independence,
        ]
        cf_valid_scores = [s for s in cf_component_scores if s > 0]
        if cf_valid_scores:
            cf_epsilon = abs(min(1.0 - s for s in cf_valid_scores))

    # Summary.json in faithfulness/ with nested structure
    summary = FaithfulnessSummary(
        observational=FaithfulnessCategoryScore(score=obs_overall, epsilon=obs_epsilon),
        interventional=FaithfulnessCategoryScore(score=int_overall, epsilon=int_epsilon),
        counterfactual=FaithfulnessCategoryScore(score=cf_overall, epsilon=cf_epsilon),
        overall=(obs_overall + int_overall + cf_overall) / 3.0 if (obs_overall + int_overall + cf_overall) > 0 else 0.0,
    )
    summary_path = os.path.join(faithfulness_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    paths["summary.json"] = summary_path

    # Generate explanation.md
    explanation_content = """# Faithfulness Metrics Explanation

## Overview

This folder contains faithfulness analysis results for a neural network subcircuit.
Faithfulness measures how well the identified subcircuit captures the behavior of the full network.

## Score Categories

### 1. Observational (Robustness)

Tests how well the subcircuit agrees with the full network under various input perturbations.

**Noise Perturbations:**
- Gaussian noise added to inputs
- Measures stability under small input changes

**Out-of-Distribution Transformations:**
- **Multiply**: Scale inputs by factors (positive and negative)
- **Add/Subtract**: Add or subtract values from inputs
- **Bimodal**: Map inputs to [-1, 1] range (order-preserving and inverted)

**Aggregation:** Average of agreement rates across all perturbation types.

### 2. Interventional

Tests how the subcircuit responds to internal activation interventions.

**In-Circuit Interventions:**
- Patch activations within the identified subcircuit
- High similarity = subcircuit responds consistently to internal changes

**Out-Circuit Interventions:**
- Patch activations outside the subcircuit
- High similarity = subcircuit is independent of non-circuit components

**Distribution Types:**
- In-distribution: Normal activation values
- Out-of-distribution: Extreme or unusual activation values

**Aggregation:** Average of bit similarities across all patch types and distributions.

### 3. Counterfactual (2x2 Matrix)

Tests causal faithfulness using the activation patching paradigm.

```
                | IN-Circuit Patch    | OUT-Circuit Patch
----------------|---------------------|--------------------
DENOISING       | Sufficiency         | Completeness
(corrupt→clean) | (recovery → 1)      | (disruption → 1)
----------------|---------------------|--------------------
NOISING         | Necessity           | Independence
(clean→corrupt) | (disruption → 1)    | (recovery → 1)
```

**Denoising Experiments** (run corrupted input, patch with clean activations):
- **Sufficiency**: Patching clean in-circuit → behavior recovers to clean output
- **Completeness**: Patching clean out-circuit → behavior stays corrupted (circuit is complete)

**Noising Experiments** (run clean input, patch with corrupted activations):
- **Necessity**: Patching corrupted in-circuit → behavior changes to corrupted output
- **Independence**: Patching corrupted out-circuit → behavior stays clean (circuit is self-contained)

**Aggregation:** Average of all four counterfactual scores.

## Epsilon Values

Epsilon represents the minimum margin by which any individual score falls short of 1.0:

```
epsilon = min(1.0 - individual_scores)
```

A lower epsilon means all individual scores are closer to perfect (1.0).
Epsilon = 0 would mean all tests achieved perfect faithfulness.

## File Structure

```
faithfulness/
├── summary.json          # Overall scores and epsilons
├── explanation.md        # This file
├── observational/
│   └── result.json       # Detailed robustness metrics
├── interventional/
│   └── result.json       # Detailed intervention metrics
└── counterfactual/
    └── result.json       # Detailed counterfactual metrics
```

## Interpretation

| Score Range | Interpretation |
|------------|----------------|
| 0.95 - 1.0 | Excellent faithfulness |
| 0.85 - 0.95 | Good faithfulness |
| 0.70 - 0.85 | Moderate faithfulness |
| < 0.70 | Poor faithfulness |

A faithful circuit should score high (>0.85) on all three categories.
"""

    explanation_path = os.path.join(faithfulness_dir, "explanation.md")
    with open(explanation_path, "w") as f:
        f.write(explanation_content)
    paths["explanation.md"] = explanation_path

    return paths


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

        # --- profiling/ (timing visualizations) - run in parallel ---
        if trial.profiling and trial.profiling.events:
            profiling_dir = os.path.join(trial_dir, "profiling")
            os.makedirs(profiling_dir, exist_ok=True)
            viz_paths[trial_id]["profiling"] = {}

            with ThreadPoolExecutor(max_workers=3) as executor:
                timeline_future = executor.submit(
                    visualize_profiling_timeline, trial.profiling, profiling_dir
                )
                phases_future = executor.submit(
                    visualize_profiling_phases, trial.profiling, profiling_dir
                )
                summary_future = executor.submit(
                    visualize_profiling_summary, trial.profiling, profiling_dir
                )

            if path := timeline_future.result():
                viz_paths[trial_id]["profiling"]["timeline"] = path
            if path := phases_future.result():
                viz_paths[trial_id]["profiling"]["phases"] = path
            if path := summary_future.result():
                viz_paths[trial_id]["profiling"]["summary"] = path

        # Extract pre-computed data
        canonical_activations = trial.canonical_activations or {}
        mean_activations_by_range = trial.mean_activations_by_range or {}
        layer_weights = trial.layer_weights or []
        layer_biases = trial.layer_biases or []
        gate_names = trial.setup.model_params.logic_gates

        if not canonical_activations or not layer_weights:
            continue

        # Full circuit (all edges/nodes active)
        layer_sizes = [layer_weights[0].shape[1]] + [w.shape[0] for w in layer_weights]
        full_circuit = Circuit.full(layer_sizes)

        # Pre-cache layout for this structure
        _layout_cache.get_positions(tuple(layer_sizes))

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
            layer_biases=layer_biases if layer_biases else None,
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

            act_path = visualize_circuit_activations_from_data(
                canonical_activations,
                layer_weights,
                full_circuit,
                folder,
                gate_name=gate_label,
                layer_biases=layer_biases if layer_biases else None,
            )
            viz_paths[trial_id][gname]["full"]["activations"] = act_path

            # Mean activations for different input ranges
            if mean_activations_by_range:
                mean_act_path = visualize_circuit_activations_mean(
                    mean_activations_by_range,
                    layer_weights,
                    full_circuit,
                    folder,
                    gate_name=gate_label,
                    layer_biases=layer_biases if layer_biases else None,
                )
                viz_paths[trial_id][gname]["full"]["activations_mean"] = mean_act_path

            if gname in trial.decomposed_gate_models:
                decomposed = trial.decomposed_gate_models[gname]
                if path := visualize_spd_components(
                    decomposed, folder, gate_name=gate_label
                ):
                    viz_paths[trial_id][gname]["full"]["spd"] = path

        # --- Subcircuit visualization ---
        for gate_idx, gname in enumerate(gate_names):
            best_indices = trial.metrics.per_gate_bests.get(gname, [])
            if not best_indices:
                continue

            print(
                f"[VIZ] Gate {gname}: {len(best_indices)} best subcircuits to visualize"
            )
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
                    layer_biases=layer_biases if layer_biases else None,
                )
                viz_paths[trial_id][gname][sc_idx]["activations"] = act_path

                # Mean activations for different input ranges
                if mean_activations_by_range:
                    mean_act_path = visualize_circuit_activations_mean(
                        mean_activations_by_range,
                        layer_weights,
                        circuit,
                        folder,
                        gate_name=sc_label,
                        layer_biases=layer_biases if layer_biases else None,
                    )
                    viz_paths[trial_id][gname][sc_idx]["activations_mean"] = mean_act_path

                # Robustness and Faithfulness visualization
                bests_faith = trial.metrics.per_gate_bests_faith.get(gname, [])
                has_robust = i < len(bests_robust)
                has_faith = i < len(bests_faith)

                robustness_data = bests_robust[i] if has_robust else None
                faithfulness_data = bests_faith[i] if has_faith else None

                # Create directories upfront
                # Faithfulness contains: observational/ (robustness), counterfactual/, interventional/
                faithfulness_dir = os.path.join(folder, "faithfulness")
                os.makedirs(faithfulness_dir, exist_ok=True)
                viz_paths[trial_id][gname][sc_idx]["faithfulness"] = {}

                # Observational dir (renamed from robustness) - lives inside faithfulness/
                observational_dir = (
                    os.path.join(faithfulness_dir, "observational") if has_robust else None
                )
                if observational_dir:
                    os.makedirs(observational_dir, exist_ok=True)
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"]["observational"] = {}

                # Run quick visualizations sequentially (matplotlib is not thread-safe)
                if has_robust and observational_dir:
                    with profile("robust_curves"):
                        viz_paths[trial_id][gname][sc_idx]["faithfulness"]["observational"]["stats"] = (
                            visualize_robustness_curves(
                                robustness_data, observational_dir, sc_label
                            )
                        )
                # Run expensive circuit visualizations sequentially (they use ProcessPoolExecutor internally)
                if has_robust and observational_dir:
                    with profile("robust_circuit_viz"):
                        circuit_paths = visualize_robustness_circuit_samples(
                            robustness_data, circuit, layer_weights, observational_dir,
                            canonical_activations=canonical_activations,
                            layer_biases=layer_biases if layer_biases else None,
                        )
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"]["observational"]["circuit_viz"] = (
                        circuit_paths
                    )

                if has_faith:
                    with profile("faith_circuit_viz"):
                        circuit_paths = visualize_faithfulness_circuit_samples(
                            faithfulness_data, circuit, layer_weights, faithfulness_dir,
                            layer_biases=layer_biases if layer_biases else None,
                        )
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"][
                        "circuit_viz"
                    ] = circuit_paths

                    # Add intervention effect plots (like noise_by_input for robustness)
                    with profile("faith_intervention_effects"):
                        intervention_paths = visualize_faithfulness_intervention_effects(
                            faithfulness_data, faithfulness_dir, sc_label
                        )
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"][
                        "intervention_effects"
                    ] = intervention_paths

                # Save result.json files in each subfolder and summary.json in faithfulness/
                interventional_dir = os.path.join(faithfulness_dir, "interventional") if has_faith else None
                counterfactual_dir = os.path.join(faithfulness_dir, "counterfactual") if has_faith else None
                json_paths = save_faithfulness_json(
                    observational_dir=observational_dir,
                    interventional_dir=interventional_dir,
                    counterfactual_dir=counterfactual_dir,
                    faithfulness_dir=faithfulness_dir,
                    robustness=robustness_data,
                    faithfulness=faithfulness_data,
                )
                viz_paths[trial_id][gname][sc_idx]["faithfulness"]["json"] = json_paths

                # SPD
                if gname in trial.decomposed_subcircuits:
                    if sc_idx in trial.decomposed_subcircuits[gname]:
                        decomposed = trial.decomposed_subcircuits[gname][sc_idx]
                        if path := visualize_spd_components(
                            decomposed, folder, gate_name=sc_label
                        ):
                            viz_paths[trial_id][gname][sc_idx]["spd"] = path

    return viz_paths

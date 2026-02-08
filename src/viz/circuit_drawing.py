"""Circuit graph drawing utilities.

Contains functions for drawing neural network circuits as directed graphs:
- draw_intervened_circuit: Draw circuit with intervention effects
- _build_graph_fast: Build networkx graph with cached positions
- _draw_graph: Draw graph on axis
- _draw_graph_with_output_highlight: Draw graph with highlighted output node
- _draw_circuit_from_data: Draw single circuit from pre-computed activations
- _get_spd_component_weights: Get normalized weight magnitudes per component
"""

import networkx as nx
import numpy as np
import torch

from src.circuit import Circuit
from src.model import DecomposedMLP
from .constants import (
    _activation_to_color,
    _layout_cache,
    _symmetric_range,
    _text_color_for_background,
)


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
        # Draw edge labels with varying positions to avoid overlap
        # Group edges by source node and vary positions within each group
        for i, (edge, label) in enumerate(edge_labels.items()):
            # Vary position between 0.3 and 0.7 based on edge index
            label_pos = 0.35 + (i % 3) * 0.15  # 0.35, 0.50, 0.65
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels={edge: label},
                font_size=6,
                font_color="#333333",
                label_pos=label_pos,
                bbox=dict(
                    boxstyle="round,pad=0.08", facecolor="white", alpha=0.9, edgecolor="none"
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

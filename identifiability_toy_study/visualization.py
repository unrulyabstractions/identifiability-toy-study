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

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Configure matplotlib backend BEFORE importing pyplot (critical for batch rendering)
import matplotlib
matplotlib.use('Agg')

from .profiler import profile

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

# Configure matplotlib for performance
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['text.usetex'] = False


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
    all_vals = [act[0].detach().cpu().numpy() if isinstance(act, torch.Tensor)
                else np.array(act[0] if isinstance(act[0], list) else act)
                for act in activations]
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


def _build_graph_fast(activations, circuit, weights_per_layer, vmin, vmax, cached_pos=None):
    """Build networkx graph with optional cached positions."""
    G = nx.DiGraph()
    node_colors_dict = {}  # Dict keyed by node name for proper lookup
    labels, text_colors = {}, {}

    # Get layer sizes from activations
    layer_sizes = tuple(act.shape[-1] if isinstance(act, torch.Tensor) else len(act[0]) if isinstance(act[0], list) else act.shape[-1] for act in activations)

    # Use cached positions if available
    if cached_pos is None:
        pos = _layout_cache.get_positions(layer_sizes)
    else:
        pos = cached_pos

    # Batch add all nodes first
    all_nodes = []
    for layer_idx, layer_act in enumerate(activations):
        n = layer_act.shape[-1] if isinstance(layer_act, torch.Tensor) else len(layer_act[0]) if isinstance(layer_act[0], list) else layer_act.shape[-1]
        for node_idx in range(n):
            name = f"({layer_idx},{node_idx})"
            all_nodes.append(name)
    G.add_nodes_from(all_nodes)

    # Build node colors and labels
    for layer_idx, layer_act in enumerate(activations):
        n = layer_act.shape[-1] if isinstance(layer_act, torch.Tensor) else len(layer_act[0]) if isinstance(layer_act[0], list) else layer_act.shape[-1]
        for node_idx in range(n):
            name = f"({layer_idx},{node_idx})"

            if isinstance(layer_act, torch.Tensor):
                val = layer_act[0, node_idx].item()
            else:
                val = layer_act[0][node_idx] if isinstance(layer_act[0], list) else layer_act[0, node_idx]
            labels[name] = f"{val:.2f}"

            # Handle case where circuit has more outputs than activations
            if layer_idx < len(circuit.node_masks) and node_idx < len(circuit.node_masks[layer_idx]):
                active = circuit.node_masks[layer_idx][node_idx] == 1
            else:
                active = True

            if active:
                color = _activation_to_color(val, vmin, vmax)
                node_colors_dict[name] = color
                text_colors[name] = _text_color_for_background(color)
            else:
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
        out_limit = activations[layer_idx + 1].shape[-1] if isinstance(activations[layer_idx + 1], torch.Tensor) else len(activations[layer_idx + 1][0])
        in_limit = activations[layer_idx].shape[-1] if isinstance(activations[layer_idx], torch.Tensor) else len(activations[layer_idx][0])

        for out_idx, row in enumerate(mask):
            if out_idx >= out_limit:
                continue  # Skip outputs beyond activation size
            for in_idx, active in enumerate(row):
                if in_idx >= in_limit:
                    continue  # Skip inputs beyond activation size
                e = (f"({layer_idx},{in_idx})", f"({layer_idx + 1},{out_idx})")
                weight_val = w[out_idx, in_idx] if out_idx < w.shape[0] and in_idx < w.shape[1] else 0.0
                all_edges.append((e[0], e[1], {"active": active, "weight": weight_val}))
                if active == 1:
                    edge_labels[e] = f"{weight_val:.2f}"
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
        G, pos,
        node_color=node_colors,
        node_size=900,
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )

    # Draw active edges
    if active:
        nx.draw_networkx_edges(
            G, pos,
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
            G, pos,
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
            x, y,
            labels.get(node, ""),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=text_colors.get(node, "black"),
        )

    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos,
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


def _draw_graph_with_output_highlight(
    ax, G, pos, node_colors, labels, text_colors, edge_labels, edge_weights, output_correct
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
            G, pos,
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
            G, pos,
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
            G, pos,
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
            G, pos,
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
            x, y,
            labels.get(node, ""),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=text_colors.get(node, "black"),
        )
    ax.axis("off")


def _draw_circuit_from_data(ax, activations, circuit, weights, title):
    """Draw a single circuit using pre-computed activations (no model run)."""
    vmin, vmax = _symmetric_range(activations)
    G, pos, colors, node_labels, text_colors, edge_labels, edge_weights = _build_graph_fast(
        activations, circuit, weights, vmin, vmax
    )
    _draw_graph(ax, G, pos, colors, node_labels, text_colors, edge_labels, edge_weights)

    output = activations[-1][0, 0].item() if isinstance(activations[-1], torch.Tensor) else activations[-1][0][0]
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
    """Bin samples by noise magnitude and compute stats per bin."""
    if not samples:
        return [], [], [], [], []

    sorted_samples = sorted(samples, key=lambda s: s.noise_magnitude)
    magnitudes = [s.noise_magnitude for s in sorted_samples]

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
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        gate_accs.append(sum(1 for s in bin_samples if s.gate_correct) / len(bin_samples))
        sc_accs.append(sum(1 for s in bin_samples if s.subcircuit_correct) / len(bin_samples))
        bit_agrees.append(sum(1 for s in bin_samples if s.agreement_bit) / len(bin_samples))
        mse_means.append(sum(s.mse for s in bin_samples) / len(bin_samples))

    return bin_centers, gate_accs, sc_accs, bit_agrees, mse_means


def _generate_robustness_circuit_figure(args):
    """Worker function for parallel robustness circuit generation."""
    (samples, circuit_dict, full_circuit_dict, weights, base_key, category,
     gt, output_path, n_samples) = args

    # Reconstruct circuits from dicts
    circuit = Circuit.from_dict(circuit_dict)
    full_circuit = Circuit.from_dict(full_circuit_dict)

    fig, axes = plt.subplots(n_samples, 2, figsize=(8, n_samples * 3))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, sample in enumerate(samples):
        if not sample['agreement_bit']:
            for col in range(2):
                axes[i, col].set_facecolor("#FFEEEE")

        # Convert stored list activations back to tensors
        sc_acts = [torch.tensor(a).unsqueeze(0) for a in sample['subcircuit_activations']]
        full_acts = [torch.tensor(a).unsqueeze(0) for a in sample['gate_activations']]

        # Left: subcircuit
        vmin, vmax = _symmetric_range(sc_acts)
        G, pos, colors, labels, text_colors, edge_labels, edge_w = (
            _build_graph_fast(sc_acts, circuit, weights, vmin, vmax)
        )
        _draw_graph_with_output_highlight(
            axes[i, 0], G, pos, colors, labels, text_colors,
            edge_labels, edge_w, sample['subcircuit_correct']
        )

        # Right: full model
        vmin, vmax = _symmetric_range(full_acts)
        G, pos, colors, labels, text_colors, edge_labels, edge_w = (
            _build_graph_fast(full_acts, full_circuit, weights, vmin, vmax)
        )
        _draw_graph_with_output_highlight(
            axes[i, 1], G, pos, colors, labels, text_colors,
            edge_labels, edge_w, sample['gate_correct']
        )

        if not sample['agreement_bit']:
            axes[i, 0].text(
                0.02, 0.98, "⚠ DISAGREE", transform=axes[i, 0].transAxes,
                fontsize=8, color="red", fontweight="bold",
                verticalalignment="top"
            )

    axes[0, 0].set_title("Subcircuit", fontsize=10, fontweight="bold")
    axes[0, 1].set_title("Full", fontsize=10, fontweight="bold")

    base_str = base_key.replace("_", ",")
    category_label = {
        "noise": "noise",
        "ood_positive": "ood: scale > 1",
        "ood_negative": "ood: scale < 0",
    }.get(category, category)
    fig.suptitle(f"({base_str}) → {gt}  [{category_label}]", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_path, dpi=100)  # Lower DPI for faster generation
    plt.close(fig)
    return output_path


def visualize_robustness_circuit_samples(
    robustness: RobustnessMetrics,
    circuit: Circuit,
    layer_weights: list[torch.Tensor],
    output_dir: str,
    n_samples_per_grid: int = 8,
) -> dict[str, str]:
    """Visualize circuit diagrams comparing subcircuit vs full model under noise."""
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
    layer_sizes = [layer_weights[0].shape[1]] + [w.shape[0] for w in layer_weights]
    full_circuit = Circuit.full(layer_sizes)

    # Group samples by base input
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

    # Prepare tasks for parallel execution
    tasks = []
    circuit_dict = circuit.to_dict()
    full_circuit_dict = full_circuit.to_dict()

    for base_key, by_category in samples_by_base.items():
        for category in ["noise", "ood_positive", "ood_negative"]:
            all_samples = by_category[category]
            if not all_samples:
                continue

            sort_key = (lambda s: abs(s.noise_magnitude)) if category == "ood_negative" else (lambda s: s.noise_magnitude)
            disagree = sorted([s for s in all_samples if not s.agreement_bit], key=sort_key)
            agree = sorted([s for s in all_samples if s.agreement_bit], key=sort_key)

            if len(disagree) >= n_samples_per_grid:
                d_idx = np.linspace(0, len(disagree) - 1, n_samples_per_grid, dtype=int)
                samples = [disagree[i] for i in d_idx]
            else:
                samples = list(disagree)
                remaining = n_samples_per_grid - len(samples)
                if remaining > 0 and agree:
                    a_idx = np.linspace(0, len(agree) - 1, min(remaining, len(agree)), dtype=int)
                    samples.extend([agree[i] for i in a_idx])

            samples = sorted(samples, key=sort_key)
            n_samples = len(samples)
            if n_samples == 0:
                continue

            gt = int(all_samples[0].ground_truth)
            filename = f"{base_key}_{category}.png"
            output_path = os.path.join(circuit_viz_dir, filename)

            # Convert samples to dicts for pickling
            sample_dicts = [{
                'subcircuit_activations': s.subcircuit_activations,
                'gate_activations': s.gate_activations,
                'subcircuit_correct': s.subcircuit_correct,
                'gate_correct': s.gate_correct,
                'agreement_bit': s.agreement_bit,
            } for s in samples]

            tasks.append((
                sample_dicts, circuit_dict, full_circuit_dict, weights,
                base_key, category, gt, output_path, n_samples
            ))
            paths[filename] = output_path

    # Execute in parallel
    if tasks:
        n_workers = min(len(tasks), mp.cpu_count())
        print(f"[VIZ] Generating {len(tasks)} robustness circuit figures with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            list(executor.map(_generate_robustness_circuit_figure, tasks))

    return paths


def visualize_robustness_summary(
    robustness: RobustnessMetrics,
    output_dir: str,
    gate_name: str = "",
) -> str:
    """Visualize robustness summary as a single overview chart."""
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
    ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Rate")
    ax.set_title(f"Noise Summary (MSE: {robustness.noise_mse_mean:.4f})", fontweight="bold")
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

    # Bottom-left: Overall comparison
    ax = axes[1, 0]
    x = np.arange(3)
    width = 0.35
    noise_vals = [
        robustness.noise_gate_accuracy,
        robustness.noise_subcircuit_accuracy,
        robustness.noise_agreement_bit,
    ]
    ood_vals = [
        robustness.ood_gate_accuracy,
        robustness.ood_subcircuit_accuracy,
        robustness.ood_agreement_bit,
    ]
    ax.bar(x - width/2, noise_vals, width, label="Noise", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, ood_vals, width, label="OOD", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Gate Acc", "SC Acc", "Agreement"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Rate")
    ax.set_title("Noise vs OOD Comparison", fontweight="bold")
    ax.legend()
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)

    # Bottom-right: Overall robustness score
    ax = axes[1, 1]
    score = robustness.overall_robustness
    ax.pie([score, 1 - score], colors=["green", "#e0e0e0"], startangle=90)
    ax.text(0, 0, f"{score:.1%}", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.set_title("Overall Robustness", fontweight="bold")

    fig.suptitle(f"{prefix}Robustness Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_robustness_curves(
    robustness: RobustnessMetrics,
    output_dir: str,
    gate_name: str = "",
) -> dict[str, str]:
    """Visualize accuracy/agreement curves vs noise magnitude."""
    paths = {}
    prefix = f"{gate_name} - " if gate_name else ""

    # Per-input breakdown for noise samples
    with profile("robust_curves.per_input"):
        base_to_key = {
            (0.0, 0.0): "0_0", (0.0, 1.0): "0_1",
            (1.0, 0.0): "1_0", (1.0, 1.0): "1_1",
        }

        samples_by_base = {k: [] for k in base_to_key.values()}
        for sample in robustness.noise_samples:
            base_key = base_to_key.get((sample.base_input[0], sample.base_input[1]))
            if base_key:
                samples_by_base[base_key].append(sample)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, key in enumerate(base_to_key.values()):
            ax = axes[i]
            samples = samples_by_base.get(key, [])
            if samples:
                bin_centers, gate_acc, sc_acc, bit_agree, mse = _bin_samples_by_magnitude(samples)
                if bin_centers:
                    ax.plot(bin_centers, gate_acc, "o-", label="Gate Acc", color="steelblue")
                    ax.plot(bin_centers, sc_acc, "s-", label="SC Acc", color="coral")
                    ax.plot(bin_centers, bit_agree, "^-", label="Agreement", color="purple")
            ax.set_xlabel("Noise Magnitude")
            ax.set_ylabel("Rate")
            ax.set_ylim(0, 1.1)
            ax.legend(loc="lower left")
            ax.set_title(f"Input: ({key.replace('_', ', ')})", fontweight="bold")
            ax.grid(alpha=0.3)

        fig.suptitle(f"{prefix}Noise Robustness by Input", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(output_dir, "noise_by_input.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["noise_by_input"] = path

    # Aggregate curves
    with profile("robust_curves.aggregate"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Noise aggregate
        ax = axes[0]
        bin_centers, gate_acc, sc_acc, bit_agree, mse = _bin_samples_by_magnitude(
            robustness.noise_samples
        )
        if bin_centers:
            ax.plot(bin_centers, gate_acc, "o-", label="Gate Acc", color="steelblue")
            ax.plot(bin_centers, sc_acc, "s-", label="SC Acc", color="coral")
            ax.plot(bin_centers, bit_agree, "^-", label="Agreement", color="purple")
        ax.set_xlabel("Noise Magnitude")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.set_title("Noise (All Inputs)", fontweight="bold")
        ax.grid(alpha=0.3)

        # OOD aggregate
        ax = axes[1]
        bin_centers, gate_acc, sc_acc, bit_agree, mse = _bin_samples_by_magnitude(
            robustness.ood_samples, log_scale=True
        )
        if bin_centers:
            ax.plot(bin_centers, gate_acc, "o-", label="Gate Acc", color="steelblue")
            ax.plot(bin_centers, sc_acc, "s-", label="SC Acc", color="coral")
            ax.plot(bin_centers, bit_agree, "^-", label="Agreement", color="purple")
        ax.set_xlabel("Scale Factor (log)")
        ax.set_xscale("log")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.set_title("OOD (All Inputs)", fontweight="bold")
        ax.grid(alpha=0.3)

        fig.suptitle(f"{prefix}Aggregate Robustness", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(output_dir, "aggregate.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["aggregate"] = path

    return paths


# ------------------ FAITHFULNESS VISUALIZATION ------------------


def _generate_faithfulness_circuit_figure(args):
    """Worker function for parallel faithfulness circuit generation."""
    (effect_dict, circuit_dict, weights, out_circuit_nodes, output_path, fig_type, index) = args

    circuit = Circuit.from_dict(circuit_dict)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    clean_acts = [torch.tensor(a).unsqueeze(0) for a in effect_dict['clean_activations']]
    corrupt_acts = [torch.tensor(a).unsqueeze(0) for a in effect_dict['corrupted_activations']]

    all_acts = clean_acts + corrupt_acts
    vmin, vmax = _symmetric_range(all_acts)

    def draw_cf_circuit(ax, acts, patched_nodes, title, output_correct):
        G = nx.DiGraph()
        layer_sizes = tuple(a.shape[-1] for a in acts)
        pos = _layout_cache.get_positions(layer_sizes)

        # Build graph
        node_colors = []
        labels = {}
        text_colors = {}

        all_nodes = []
        for layer_idx, layer_act in enumerate(acts):
            n = layer_act.shape[-1]
            for node_idx in range(n):
                name = f"({layer_idx},{node_idx})"
                all_nodes.append(name)
        G.add_nodes_from(all_nodes)

        for layer_idx, layer_act in enumerate(acts):
            n = layer_act.shape[-1]
            for node_idx in range(n):
                name = f"({layer_idx},{node_idx})"
                val = layer_act[0, node_idx].item()
                labels[name] = f"{val:.2f}"

                color = _activation_to_color(val, vmin, vmax)
                node_colors.append(color)
                text_colors[name] = _text_color_for_background(color)

        # Add edges
        edges = []
        for layer_idx, mask in enumerate(circuit.edge_masks):
            for out_idx, row in enumerate(mask):
                for in_idx, active in enumerate(row):
                    if active:
                        e = (f"({layer_idx},{in_idx})", f"({layer_idx + 1},{out_idx})")
                        edges.append(e)
        G.add_edges_from(edges)

        # Draw
        max_layer = max(int(n.split(",")[0][1:]) for n in G.nodes())
        output_nodes = [n for n in G.nodes() if n.startswith(f"({max_layer},")]
        patched_list = [n for n in G.nodes() if n in patched_nodes]
        regular_nodes = [n for n in G.nodes() if n not in output_nodes and n not in patched_list]

        if regular_nodes:
            reg_colors = [node_colors[list(G.nodes()).index(n)] for n in regular_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, node_color=reg_colors,
                                  node_size=500, ax=ax, edgecolors="black", linewidths=1)

        if patched_list:
            p_colors = [node_colors[list(G.nodes()).index(n)] for n in patched_list]
            nx.draw_networkx_nodes(G, pos, nodelist=patched_list, node_color=p_colors,
                                  node_size=500, ax=ax, edgecolors="#9C27B0", linewidths=4)

        if output_nodes:
            out_colors = [node_colors[list(G.nodes()).index(n)] for n in output_nodes]
            border = "green" if output_correct else "red"
            nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color=out_colors,
                                  node_size=500, ax=ax, edgecolors=border, linewidths=4)

        nx.draw_networkx_edges(G, pos, edge_color="black", width=1.0, ax=ax)

        for node, (x, y) in pos.items():
            ax.text(x, y, labels[node], ha="center", va="center", fontsize=6,
                   fontweight="bold", color=text_colors[node])

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    # Draw three circuits
    draw_cf_circuit(axes[0], clean_acts, set(),
                   f"Clean → {effect_dict['expected_clean_output']:.2f}", True)
    draw_cf_circuit(axes[1], corrupt_acts, set(),
                   f"Corrupted → {effect_dict['expected_corrupted_output']:.2f}", True)
    draw_cf_circuit(axes[2], clean_acts, out_circuit_nodes,
                   f"Clean+Patch → {effect_dict['actual_output']:.2f}",
                   not effect_dict['output_changed_to_corrupted'])

    clean_str = ",".join(f"{v:.0f}" for v in effect_dict['clean_input'])
    corrupt_str = ",".join(f"{v:.0f}" for v in effect_dict['corrupted_input'])
    fig.suptitle(f"{fig_type} Counterfactual #{index} | ({clean_str})→({corrupt_str}) | "
                f"Faith: {effect_dict['faithfulness_score']:.2f}",
                fontsize=11, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_path, dpi=100)  # Lower DPI for faster generation
    plt.close(fig)
    return output_path


def visualize_faithfulness_circuit_samples(
    faithfulness: FaithfulnessMetrics,
    circuit: Circuit,
    layer_weights: list[torch.Tensor],
    output_dir: str,
    n_samples_per_grid: int = 8,
) -> dict[str, str]:
    """Visualize circuit diagrams showing interventions."""
    circuit_viz_dir = os.path.join(output_dir, "circuit_viz")
    os.makedirs(circuit_viz_dir, exist_ok=True)
    paths = {}

    weights = [w.numpy() for w in layer_weights]
    circuit_dict = circuit.to_dict()

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

    # Out-circuit counterfactuals (sufficiency)
    if faithfulness.out_counterfactual_effects:
        cf_dir = os.path.join(circuit_viz_dir, "counterfactual_out")
        os.makedirs(cf_dir, exist_ok=True)

        for i, effect in enumerate(faithfulness.out_counterfactual_effects[:6]):
            if not effect.clean_activations or not effect.corrupted_activations:
                continue

            effect_dict = {
                'clean_activations': effect.clean_activations,
                'corrupted_activations': effect.corrupted_activations,
                'expected_clean_output': effect.expected_clean_output,
                'expected_corrupted_output': effect.expected_corrupted_output,
                'actual_output': effect.actual_output,
                'output_changed_to_corrupted': effect.output_changed_to_corrupted,
                'faithfulness_score': effect.faithfulness_score,
                'clean_input': effect.clean_input,
                'corrupted_input': effect.corrupted_input,
            }
            output_path = os.path.join(cf_dir, f"out_cf_{i}.png")
            tasks.append((effect_dict, circuit_dict, weights, out_circuit_nodes, output_path, "Out-Circuit", i))

        paths["counterfactual_out"] = cf_dir

    # In-circuit counterfactuals (necessity)
    if faithfulness.in_counterfactual_effects:
        cf_dir = os.path.join(circuit_viz_dir, "counterfactual_in")
        os.makedirs(cf_dir, exist_ok=True)

        for i, effect in enumerate(faithfulness.in_counterfactual_effects[:6]):
            if not effect.clean_activations or not effect.corrupted_activations:
                continue

            effect_dict = {
                'clean_activations': effect.clean_activations,
                'corrupted_activations': effect.corrupted_activations,
                'expected_clean_output': effect.expected_clean_output,
                'expected_corrupted_output': effect.expected_corrupted_output,
                'actual_output': effect.actual_output,
                'output_changed_to_corrupted': effect.output_changed_to_corrupted,
                'faithfulness_score': effect.faithfulness_score,
                'clean_input': effect.clean_input,
                'corrupted_input': effect.corrupted_input,
            }
            output_path = os.path.join(cf_dir, f"in_cf_{i}.png")
            tasks.append((effect_dict, circuit_dict, weights, in_circuit_nodes, output_path, "In-Circuit", i))

        paths["counterfactual_in"] = cf_dir

    # Execute in parallel
    if tasks:
        n_workers = min(len(tasks), mp.cpu_count())
        print(f"[VIZ] Generating {len(tasks)} faithfulness circuit figures with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            list(executor.map(_generate_faithfulness_circuit_figure, tasks))

    # In-circuit/out-circuit patch visualizations (simpler, do sequentially)
    def visualize_patch_circuits(stats: dict, circuit_type: str, out_dir: str) -> dict:
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
                patch_indices = [int(x.strip()) for x in indices_match.group(1).split(",") if x.strip()]
            patch_label = f"L{patch_layer}_{'_'.join(map(str, patch_indices))}"

            # Select samples
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

                # Simple drawing without full activation visualization
                for col, (label, output) in enumerate([
                    ("Subcircuit", sample.subcircuit_output),
                    ("Full", sample.gate_output)
                ]):
                    ax = axes[i, col]
                    circ = circuit if col == 0 else full_circuit

                    # Draw simple circuit structure
                    G = nx.DiGraph()
                    pos = _layout_cache.get_positions(tuple(layer_sizes))

                    for layer_idx, n_nodes in enumerate(layer_sizes):
                        for node_idx in range(n_nodes):
                            G.add_node(f"({layer_idx},{node_idx})")

                    for layer_idx, mask in enumerate(circ.edge_masks):
                        for out_idx, row in enumerate(mask):
                            for in_idx, active in enumerate(row):
                                if active:
                                    G.add_edge(f"({layer_idx},{in_idx})", f"({layer_idx + 1},{out_idx})")

                    # Color patched nodes
                    patched = {f"({patch_layer},{idx})" for idx in patch_indices}
                    node_colors = []
                    for n in G.nodes():
                        if n in patched:
                            node_colors.append("#FF6600")
                        else:
                            node_colors.append("lightblue")

                    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, ax=ax)
                    nx.draw_networkx_edges(G, pos, edge_color="black", width=1.0, ax=ax)

                    border_color = "green" if sample.bit_agreement else "red"
                    ax.text(0.5, -0.05, f"IV:{iv_val:.2f} → {output:.2f}",
                           transform=ax.transAxes, ha="center", fontsize=9,
                           fontweight="bold", color=border_color)
                    ax.axis("off")

            axes[0, 0].set_title("Subcircuit", fontsize=10, fontweight="bold")
            axes[0, 1].set_title("Full Model", fontsize=10, fontweight="bold")

            n_agree = sum(1 for s in selected if s.bit_agreement)
            fig.suptitle(f"{patch_label} | {circuit_type} | agree: {n_agree}/{n_samples}",
                        fontsize=11, fontweight="bold")
            plt.tight_layout()

            path = os.path.join(out_dir, f"{patch_label}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            type_paths[patch_label] = path

        return type_paths

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

    return paths


def visualize_faithfulness_summary(
    faithfulness: FaithfulnessMetrics,
    output_dir: str,
    gate_name: str = "",
) -> str:
    """Visualize faithfulness summary."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    prefix = f"{gate_name} - " if gate_name else ""

    # Top-left: In-circuit similarity
    ax = axes[0, 0]
    categories = ["In-Dist", "OOD"]
    values = [
        faithfulness.mean_in_circuit_similarity,
        faithfulness.mean_in_circuit_similarity_ood,
    ]
    ax.bar(categories, values, color=[COLORS["in_circuit"], "#81C784"], alpha=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Similarity")
    ax.set_title("In-Circuit Intervention Similarity", fontweight="bold")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    # Top-right: Out-circuit similarity
    ax = axes[0, 1]
    values = [
        faithfulness.mean_out_circuit_similarity,
        faithfulness.mean_out_circuit_similarity_ood,
    ]
    ax.bar(categories, values, color=[COLORS["out_circuit"], "#EF9A9A"], alpha=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Similarity")
    ax.set_title("Out-Circuit Intervention Similarity", fontweight="bold")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    # Bottom-left: Counterfactual faithfulness
    ax = axes[1, 0]
    if faithfulness.counterfactual_effects:
        scores = [e.faithfulness_score for e in faithfulness.counterfactual_effects]
        ax.hist(scores, bins=10, color=COLORS["faithfulness"], alpha=0.8, edgecolor="black")
        ax.axvline(x=faithfulness.mean_faithfulness_score, color="red", linestyle="--",
                  label=f"Mean: {faithfulness.mean_faithfulness_score:.2f}")
        ax.set_xlabel("Faithfulness Score")
        ax.set_ylabel("Count")
        ax.legend()
    ax.set_title("Counterfactual Faithfulness Distribution", fontweight="bold")

    # Bottom-right: Overall score
    ax = axes[1, 1]
    score = faithfulness.overall_faithfulness
    ax.pie([score, 1 - score], colors=["green", "#e0e0e0"], startangle=90)
    ax.text(0, 0, f"{score:.1%}", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.set_title("Overall Faithfulness", fontweight="bold")

    fig.suptitle(f"{prefix}Faithfulness Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_faithfulness_stats(
    faithfulness: FaithfulnessMetrics,
    output_dir: str,
    gate_name: str = "",
) -> dict[str, str]:
    """Visualize detailed faithfulness statistics."""
    paths = {}
    prefix = f"{gate_name} - " if gate_name else ""

    # Patch statistics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def plot_patch_stats(ax, stats: dict, title: str, color: str):
        if not stats:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title, fontweight="bold")
            return

        patch_names = list(stats.keys())[:10]
        means = [stats[p].mean_bit_similarity for p in patch_names]
        stds = [stats[p].std_bit_similarity for p in patch_names]

        x = np.arange(len(patch_names))
        ax.bar(x, means, yerr=stds, color=color, alpha=0.8, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([p.split("indices=")[1].split(")")[0] if "indices=" in p else p[:15]
                          for p in patch_names], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Bit Similarity")
        ax.set_title(title, fontweight="bold")
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)

    plot_patch_stats(axes[0, 0], faithfulness.in_circuit_stats,
                    "In-Circuit Patches (In-Dist)", COLORS["in_circuit"])
    plot_patch_stats(axes[0, 1], faithfulness.out_circuit_stats,
                    "Out-Circuit Patches (In-Dist)", COLORS["out_circuit"])
    plot_patch_stats(axes[1, 0], faithfulness.in_circuit_stats_ood,
                    "In-Circuit Patches (OOD)", "#81C784")
    plot_patch_stats(axes[1, 1], faithfulness.out_circuit_stats_ood,
                    "Out-Circuit Patches (OOD)", "#EF9A9A")

    fig.suptitle(f"{prefix}Patch Intervention Statistics", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "patch_stats.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["patch_stats"] = path

    # Counterfactual analysis
    if faithfulness.out_counterfactual_effects or faithfulness.in_counterfactual_effects:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Sufficiency (out-circuit)
        ax = axes[0]
        if faithfulness.out_counterfactual_effects:
            scores = [e.faithfulness_score for e in faithfulness.out_counterfactual_effects]
            changed = [e.output_changed_to_corrupted for e in faithfulness.out_counterfactual_effects]
            colors = ["red" if c else "green" for c in changed]
            ax.bar(range(len(scores)), scores, color=colors, alpha=0.8)
            ax.axhline(y=np.mean(scores), color="blue", linestyle="--",
                      label=f"Mean: {np.mean(scores):.2f}")
            ax.set_xlabel("Counterfactual Pair")
            ax.set_ylabel("Sufficiency Score")
            ax.legend()
        ax.set_title("Out-Circuit Counterfactuals (Sufficiency)", fontweight="bold")

        # Necessity (in-circuit)
        ax = axes[1]
        if faithfulness.in_counterfactual_effects:
            scores = [e.faithfulness_score for e in faithfulness.in_counterfactual_effects]
            changed = [e.output_changed_to_corrupted for e in faithfulness.in_counterfactual_effects]
            colors = ["red" if c else "green" for c in changed]
            ax.bar(range(len(scores)), scores, color=colors, alpha=0.8)
            ax.axhline(y=np.mean(scores), color="blue", linestyle="--",
                      label=f"Mean: {np.mean(scores):.2f}")
            ax.set_xlabel("Counterfactual Pair")
            ax.set_ylabel("Necessity Score")
            ax.legend()
        ax.set_title("In-Circuit Counterfactuals (Necessity)", fontweight="bold")

        fig.suptitle(f"{prefix}Counterfactual Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(output_dir, "counterfactual_stats.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["counterfactual_stats"] = path

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
    ax.set_title(f"{gate_name} - SPD Component Importance" if gate_name else "SPD Component Importance",
                fontweight="bold")
    ax.set_xticks(x)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
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
    plt.savefig(path, dpi=150, bbox_inches="tight")
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
            end_status = f"ENDED_{phase}" if f"ENDED_{phase}" in [e.status for e in events] else f"FINISHED_{phase}"
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
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
               f"{d:.2f}s", va="center", fontsize=9)

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
    total_time = (profiling.events[-1].timestamp_ms - profiling.events[0].timestamp_ms) / 1000.0 if len(profiling.events) > 1 else 0
    summary_text = f"""
    Total Events: {len(profiling.events)}
    Total Time: {total_time:.2f}s
    Unique Phases: {len(phase_counts)}
    """
    ax.text(
        0.1, 0.5, summary_text.strip(),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="center",
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

        # --- profiling/ (timing visualizations) - run in parallel ---
        if trial.profiling and trial.profiling.events:
            profiling_dir = os.path.join(trial_dir, "profiling")
            os.makedirs(profiling_dir, exist_ok=True)
            viz_paths[trial_id]["profiling"] = {}

            with ThreadPoolExecutor(max_workers=3) as executor:
                timeline_future = executor.submit(visualize_profiling_timeline, trial.profiling, profiling_dir)
                phases_future = executor.submit(visualize_profiling_phases, trial.profiling, profiling_dir)
                summary_future = executor.submit(visualize_profiling_summary, trial.profiling, profiling_dir)

            if path := timeline_future.result():
                viz_paths[trial_id]["profiling"]["timeline"] = path
            if path := phases_future.result():
                viz_paths[trial_id]["profiling"]["phases"] = path
            if path := summary_future.result():
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
            )
            viz_paths[trial_id][gname]["full"]["activations"] = act_path

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

            print(f"[VIZ] Gate {gname}: {len(best_indices)} best subcircuits to visualize")
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

                # Robustness and Faithfulness visualization
                bests_faith = trial.metrics.per_gate_bests_faith.get(gname, [])
                has_robust = i < len(bests_robust)
                has_faith = i < len(bests_faith)

                robustness_data = bests_robust[i] if has_robust else None
                faithfulness_data = bests_faith[i] if has_faith else None

                # Create directories upfront
                robustness_dir = os.path.join(folder, "robustness") if has_robust else None
                faithfulness_dir = os.path.join(folder, "faithfulness") if has_faith else None
                if robustness_dir:
                    os.makedirs(robustness_dir, exist_ok=True)
                    viz_paths[trial_id][gname][sc_idx]["robustness"] = {}
                if faithfulness_dir:
                    os.makedirs(faithfulness_dir, exist_ok=True)
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"] = {}

                # Run quick visualizations (summary, curves, stats) in parallel
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {}
                    if has_robust:
                        futures["robust_summary"] = executor.submit(
                            visualize_robustness_summary, robustness_data, robustness_dir, sc_label
                        )
                        futures["robust_curves"] = executor.submit(
                            visualize_robustness_curves, robustness_data, robustness_dir, sc_label
                        )
                    if has_faith:
                        futures["faith_summary"] = executor.submit(
                            visualize_faithfulness_summary, faithfulness_data, faithfulness_dir, sc_label
                        )
                        futures["faith_stats"] = executor.submit(
                            visualize_faithfulness_stats, faithfulness_data, faithfulness_dir, sc_label
                        )

                # Collect quick visualization results
                if has_robust:
                    with profile("robust_summary"):
                        viz_paths[trial_id][gname][sc_idx]["robustness"]["summary"] = futures["robust_summary"].result()
                    with profile("robust_curves"):
                        viz_paths[trial_id][gname][sc_idx]["robustness"]["stats"] = futures["robust_curves"].result()
                if has_faith:
                    with profile("faith_summary"):
                        viz_paths[trial_id][gname][sc_idx]["faithfulness"]["summary"] = futures["faith_summary"].result()
                    with profile("faith_stats"):
                        viz_paths[trial_id][gname][sc_idx]["faithfulness"]["stats"] = futures["faith_stats"].result()

                # Run expensive circuit visualizations sequentially (they use ProcessPoolExecutor internally)
                if has_robust:
                    with profile("robust_circuit_viz"):
                        circuit_paths = visualize_robustness_circuit_samples(
                            robustness_data, circuit, layer_weights, robustness_dir
                        )
                    viz_paths[trial_id][gname][sc_idx]["robustness"]["circuit_viz"] = circuit_paths

                if has_faith:
                    with profile("faith_circuit_viz"):
                        circuit_paths = visualize_faithfulness_circuit_samples(
                            faithfulness_data, circuit, layer_weights, faithfulness_dir
                        )
                    viz_paths[trial_id][gname][sc_idx]["faithfulness"]["circuit_viz"] = circuit_paths

                # SPD
                if gname in trial.decomposed_subcircuits:
                    if sc_idx in trial.decomposed_subcircuits[gname]:
                        decomposed = trial.decomposed_subcircuits[gname][sc_idx]
                        if path := visualize_spd_components(
                            decomposed, folder, gate_name=sc_label
                        ):
                            viz_paths[trial_id][gname][sc_idx]["spd"] = path

    return viz_paths

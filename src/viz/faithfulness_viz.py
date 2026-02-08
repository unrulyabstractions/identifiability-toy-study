"""Faithfulness visualization.

Contains functions for visualizing faithfulness analysis:
- _patch_key_to_filename: Convert patch key to readable filename
- _faithfulness_score_to_color: Color gradient for faithfulness scores
- visualize_faithfulness_intervention_effects: Comprehensive faithfulness visualization
- _generate_faithfulness_circuit_figure: Worker for parallel circuit generation
- visualize_faithfulness_circuit_samples: Visualize circuit diagrams with interventions
"""

import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from src.circuit import Circuit
from src.schemas import FaithfulnessMetrics
from .constants import (
    LAYOUT_RECT_DEFAULT,
    TITLE_Y,
    _layout_cache,
    finalize_figure,
)
from .circuit_drawing import draw_intervened_circuit


def _patch_key_to_filename(patch_key: str) -> str:
    """Convert patch key to readable filename."""
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
        ax2.set_ylabel("|Delta Output|", color=color2, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=color2)

        ax.set_title("Agreement", fontsize=10, fontweight="bold")
        if show_xlabel:
            ax.set_xlabel("Intervention Value", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    # === 1. Intervention Stats (in interventional/{in,out}_circuit/{in,out}_distribution_stats.png) ===
    # Format matches observational: rows per patch, columns for Subcircuit | Gate | Agreement
    interventional_base = os.path.join(output_dir, "interventional")

    def _create_circuit_distribution_summary(patch_stats_dict, circuit_type, distribution_type, base_dir):
        """Create bar chart summary for a circuit type and distribution.

        Format: Bar chart with patches on x-axis, bit/logit/best similarity bars.
        Same format as observational summaries for consistency.
        """
        if not patch_stats_dict:
            return None

        # Sort patches by layer and node index
        def sort_key(item):
            pk = item[0]
            layer_match = re.search(r"layers=\((\d+)", pk)
            idx_match = re.search(r"indices=\((\d+)", pk)
            return (int(layer_match.group(1)) if layer_match else 0,
                    int(idx_match.group(1)) if idx_match else 0)

        sorted_patches = sorted(patch_stats_dict.items(), key=sort_key)
        n_patches = len(sorted_patches)

        if n_patches == 0:
            return None

        # Create output directory
        out_dir = os.path.join(base_dir, circuit_type)
        os.makedirs(out_dir, exist_ok=True)

        # Extract metrics from patch stats
        labels = [_patch_key_to_filename(pk) for pk, _ in sorted_patches]
        bit_sim = [ps.mean_bit_similarity for _, ps in sorted_patches]
        logit_sim = [ps.mean_logit_similarity for _, ps in sorted_patches]
        best_sim = [ps.mean_best_similarity for _, ps in sorted_patches]
        n_samples = [ps.n_interventions for _, ps in sorted_patches]

        # Create bar chart (same format as observational/interventional summaries)
        fig, ax = plt.subplots(1, 1, figsize=(max(12, n_patches * 1.2), 6))

        x = np.arange(n_patches)
        width = 0.25

        # Bar colors matching observational format
        ax.bar(x - width, bit_sim, width, label="Bit Agreement", color="#4CAF50", alpha=0.8)
        ax.bar(x, logit_sim, width, label="Logit Similarity", color="#2196F3", alpha=0.8)
        ax.bar(x + width, best_sim, width, label="Best Similarity", color="#FF9800", alpha=0.8)

        ax.set_ylabel("Score", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

        # Adapt y-axis based on distribution type
        # OOD may have lower scores, so auto-scale with some padding
        if distribution_type == "out_of_distribution":
            all_scores = bit_sim + logit_sim + best_sim
            y_min = min(0, min(all_scores) - 0.1) if all_scores else -0.15
            y_max = max(1.0, max(all_scores) + 0.1) if all_scores else 1.15
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(-0.15, 1.15)

        ax.axhline(y=0, color="#888888", linestyle="-", alpha=0.3)
        ax.axhline(y=0.5, color="#888888", linestyle="--", alpha=0.3)
        ax.axhline(y=1.0, color="#888888", linestyle="-", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

        # Sample counts above bars
        for i, (xi, n) in enumerate(zip(x, n_samples)):
            y_pos = max(bit_sim[i], logit_sim[i], best_sim[i]) + 0.03
            ax.text(xi, y_pos, f"n={n}", ha="center", va="bottom", fontsize=7, color="#666666")

        # Title
        circuit_label = "In-Circuit" if circuit_type == "in_circuit" else "Out-of-Circuit"
        dist_label = "In-Distribution" if distribution_type == "in_distribution" else "Out-of-Distribution"
        finalize_figure(fig, f"{prefix}{circuit_label} - {dist_label}", fontsize=13)

        filename = f"{distribution_type}_stats.png"
        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    # In-circuit stats
    if faithfulness.interventional.in_circuit_stats if faithfulness.interventional else {}:
        path = _create_circuit_distribution_summary(
            faithfulness.interventional.in_circuit_stats if faithfulness.interventional else {}, "in_circuit", "in_distribution", interventional_base)
        if path:
            paths["interventional/in_circuit/in_distribution_stats"] = path

    if faithfulness.interventional.in_circuit_stats_ood if faithfulness.interventional else {}:
        path = _create_circuit_distribution_summary(
            faithfulness.interventional.in_circuit_stats_ood if faithfulness.interventional else {}, "in_circuit", "out_of_distribution", interventional_base)
        if path:
            paths["interventional/in_circuit/out_of_distribution_stats"] = path

    # Out-circuit stats
    if faithfulness.interventional.out_circuit_stats if faithfulness.interventional else {}:
        path = _create_circuit_distribution_summary(
            faithfulness.interventional.out_circuit_stats if faithfulness.interventional else {}, "out_circuit", "in_distribution", interventional_base)
        if path:
            paths["interventional/out_circuit/in_distribution_stats"] = path

    if faithfulness.interventional.out_circuit_stats_ood if faithfulness.interventional else {}:
        path = _create_circuit_distribution_summary(
            faithfulness.interventional.out_circuit_stats_ood if faithfulness.interventional else {}, "out_circuit", "out_of_distribution", interventional_base)
        if path:
            paths["interventional/out_circuit/out_of_distribution_stats"] = path

    # === 2. Combined Intervention Summaries (bar charts at interventional/ level) ===
    interventional_dir = interventional_base
    os.makedirs(interventional_dir, exist_ok=True)

    def _create_intervention_summary(in_stats, out_stats, title_suffix, filename):
        """Helper to create intervention summary bar chart combining in/out circuit."""
        all_patches = []
        for pk, ps in in_stats.items():
            all_patches.append(("in", pk, ps))
        for pk, ps in out_stats.items():
            all_patches.append(("out", pk, ps))

        def sort_key(item):
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
            marker = "^" if is_in else "v"
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

        ax.set_title(f"{prefix}{title_suffix}\n(^=in-circuit, v=out-circuit)",
                     fontsize=13, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(interventional_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    # In-distribution summary (combined bar chart)
    path = _create_intervention_summary(
        faithfulness.interventional.in_circuit_stats if faithfulness.interventional else {},
        faithfulness.interventional.out_circuit_stats if faithfulness.interventional else {},
        "In-Distribution Summary",
        "in_distribution_summary.png"
    )
    if path:
        paths["interventional/in_distribution_summary"] = path

    # Out-of-distribution summary (if OOD stats exist)
    if faithfulness.interventional.in_circuit_stats_ood if faithfulness.interventional else {} or faithfulness.interventional.out_circuit_stats_ood if faithfulness.interventional else {}:
        path = _create_intervention_summary(
            faithfulness.interventional.in_circuit_stats_ood if faithfulness.interventional else {},
            faithfulness.interventional.out_circuit_stats_ood if faithfulness.interventional else {},
            "Out-of-Distribution Summary",
            "out_distribution_summary.png"
        )
        if path:
            paths["interventional/out_distribution_summary"] = path

    # === 3. Intervention Summary (2x2 Matrix of Patching Experiments) ===
    suff_cf = faithfulness.counterfactual.sufficiency_effects if faithfulness.counterfactual else [] or []
    comp_cf = faithfulness.counterfactual.completeness_effects if faithfulness.counterfactual else [] or []
    nec_cf = faithfulness.counterfactual.necessity_effects if faithfulness.counterfactual else [] or []
    ind_cf = faithfulness.counterfactual.independence_effects if faithfulness.counterfactual else [] or []

    # Build per-node scores from intervention stats (bit similarity)
    # Key: (layer, node_idx) -> score
    def _build_node_scores(stats_dict):
        """Extract per-node agreement scores from patch statistics."""
        scores = {}
        for patch_key, patch_stats in stats_dict.items():
            layer_match = re.search(r"layers=\((\d+),?\)", patch_key)
            idx_match = re.search(r"indices=\((\d+),?\)", patch_key)
            if layer_match and idx_match:
                layer = int(layer_match.group(1))
                node = int(idx_match.group(1))
                scores[(layer, node)] = patch_stats.mean_bit_similarity
        return scores

    in_bit_scores = _build_node_scores(faithfulness.interventional.in_circuit_stats if faithfulness.interventional else {})
    out_bit_scores = _build_node_scores(faithfulness.interventional.out_circuit_stats if faithfulness.interventional else {})

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
    if faithfulness.interventional.in_circuit_stats if faithfulness.interventional else {} or faithfulness.interventional.out_circuit_stats if faithfulness.interventional else {}:
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
        suff_mean = faithfulness.counterfactual.mean_sufficiency if faithfulness.counterfactual else 0 if hasattr(faithfulness, 'mean_sufficiency') else (
            np.mean([e.faithfulness_score for e in suff_cf]) if suff_cf else 0.5)
        comp_mean = faithfulness.counterfactual.mean_completeness if faithfulness.counterfactual else 0 if hasattr(faithfulness, 'mean_completeness') else (
            np.mean([e.faithfulness_score for e in comp_cf]) if comp_cf else 0.5)
        nec_mean = faithfulness.counterfactual.mean_necessity if faithfulness.counterfactual else 0 if hasattr(faithfulness, 'mean_necessity') else (
            np.mean([e.faithfulness_score for e in nec_cf]) if nec_cf else 0.5)
        ind_mean = faithfulness.counterfactual.mean_independence if faithfulness.counterfactual else 0 if hasattr(faithfulness, 'mean_independence') else (
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

        # Apply tight_layout with global spacing
        plt.tight_layout(rect=[0.08, 0.03, 1.0, LAYOUT_RECT_DEFAULT[3]])

        # Simple title with just gate name
        fig.suptitle(prefix.strip(" -") if prefix else "Counterfactual Summary",
                     fontsize=12, fontweight="bold", y=TITLE_Y)

        # Column labels at top (below title)
        fig.text(0.30, 0.90, "IN-CIRCUIT", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color="#6495ED")
        fig.text(0.73, 0.90, "OUT-CIRCUIT", ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color="#DA70D6")

        # Row labels on left (pastel colors)
        fig.text(0.035, 0.65, "DENOISING", ha="center", va="center", fontsize=10,
                 fontweight="bold", rotation=90, color="#77DD77")
        fig.text(0.035, 0.25, "NOISING", ha="center", va="center", fontsize=10,
                 fontweight="bold", rotation=90, color="#FFB6C1")

        # Save in counterfactual/ subdirectory
        counterfactual_dir = os.path.join(output_dir, "counterfactual")
        os.makedirs(counterfactual_dir, exist_ok=True)
        path = os.path.join(counterfactual_dir, "counterfact_summary.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths["counterfactual/counterfact_summary"] = path

    # === 4. Per-Input Visualizations (split into denoising and noising) ===
    if suff_cf or comp_cf or nec_cf or ind_cf:
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

        # Need layer_sizes for per-input figures
        if 'layer_sizes' not in dir():
            layer_sizes = None

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

            fig.suptitle(title, fontsize=13, fontweight="bold", y=TITLE_Y)
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=LAYOUT_RECT_DEFAULT[3], hspace=0.12, wspace=0.08)
            counterfactual_dir = os.path.join(output_dir, "counterfactual")
            os.makedirs(counterfactual_dir, exist_ok=True)
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
        biases,
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
        biases=biases,
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
        biases=biases,
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
        biases=biases,
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
    # "Counterfactual (1,0)->(1,1) | Out-of-Circuit Noising | Independence: 1.00"
    finalize_figure(
        fig,
        f"Counterfactual ({clean_str})->({corrupt_str})  |  {exp_label}  |  "
        f"{score_display}: {effect_dict['faithfulness_score']:.2f}",
        fontsize=11
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
                    biases,  # Include biases for edge labels
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
        faithfulness.counterfactual.sufficiency_effects if faithfulness.counterfactual else [],
        "sufficiency",
        in_circuit_nodes,
    )
    _add_counterfactual_tasks(
        faithfulness.counterfactual.completeness_effects if faithfulness.counterfactual else [],
        "completeness",
        out_circuit_nodes,
    )

    # Noising experiments (run clean, patch with corrupted)
    _add_counterfactual_tasks(
        faithfulness.counterfactual.necessity_effects if faithfulness.counterfactual else [],
        "necessity",
        in_circuit_nodes,
    )
    _add_counterfactual_tasks(
        faithfulness.counterfactual.independence_effects if faithfulness.counterfactual else [],
        "independence",
        out_circuit_nodes,
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

            finalize_figure(
                fig,
                f"{patch_label} | {circuit_type} | Agreement: {agree_pct:.0f}% (n={total_samples})",
                fontsize=11
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
    if faithfulness.interventional.in_circuit_stats if faithfulness.interventional else {}:
        in_circuit_id_dir = os.path.join(in_circuit_base, "in_distribution")
        paths["interventional/in_circuit/in_distribution"] = visualize_patch_circuits(
            faithfulness.interventional.in_circuit_stats if faithfulness.interventional else {},
            "In-Circuit (ID)",
            in_circuit_id_dir,
        )

    if faithfulness.interventional.out_circuit_stats if faithfulness.interventional else {}:
        out_circuit_id_dir = os.path.join(out_circuit_base, "in_distribution")
        paths["interventional/out_circuit/in_distribution"] = visualize_patch_circuits(
            faithfulness.interventional.out_circuit_stats if faithfulness.interventional else {},
            "Out-of-Circuit (ID)",
            out_circuit_id_dir,
        )

    # Out-of-distribution interventions (if OOD stats exist)
    if faithfulness.interventional.in_circuit_stats_ood if faithfulness.interventional else {}:
        in_circuit_ood_dir = os.path.join(in_circuit_base, "out_of_distribution")
        paths["interventional/in_circuit/out_of_distribution"] = visualize_patch_circuits(
            faithfulness.interventional.in_circuit_stats_ood if faithfulness.interventional else {},
            "In-Circuit (OOD)",
            in_circuit_ood_dir,
        )

    if faithfulness.interventional.out_circuit_stats_ood if faithfulness.interventional else {}:
        out_circuit_ood_dir = os.path.join(out_circuit_base, "out_of_distribution")
        paths["interventional/out_circuit/out_of_distribution"] = visualize_patch_circuits(
            faithfulness.interventional.out_circuit_stats_ood if faithfulness.interventional else {},
            "Out-of-Circuit (OOD)",
            out_circuit_ood_dir,
        )

    return paths

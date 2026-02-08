"""Observational (robustness) visualization.

Contains functions for visualizing robustness tests:
- _generate_robustness_circuit_figure: Worker function for parallel generation
- visualize_robustness_circuit_samples: Visualize circuit diagrams under noise
- visualize_robustness_curves: Visualize robustness curves
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.circuit import Circuit
from src.infra import profile
from src.schemas import RobustnessMetrics
from .constants import (
    LAYOUT_RECT_DEFAULT,
    TITLE_Y,
    finalize_figure,
    set_subplot_title,
)
from .circuit_drawing import draw_intervened_circuit


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

    # Use global layout with extra space for subtitle
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.suptitle(f"{base_str} -> {gt}", fontsize=14, fontweight="bold", y=TITLE_Y)
    # Subtitle: transformation type (below main title)
    fig.text(0.5, 0.93, category_label, ha="center", fontsize=10, style="italic")

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
        |- noise_perturbations/
        |   |-- 0_0.png, 0_1.png, 1_0.png, 1_1.png
        |-- out_distribution_transformations/
            |- multiply/
            |   |-- 0_1_positive.png, 0_1_negative.png, etc.
            |- add/
            |   |-- 0_0.png, 0_1.png, etc.
            |- subtract/
            |   |-- 0_0.png, 0_1.png, etc.
            |-- bimodal/
                |-- 0_0.png (order-preserving), 0_0_inv.png (inverted)

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
        """Plot binned agreement rates (overlapping: bit, best) and |Delta logit|."""
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
        ax2.set_ylabel("|Delta logit|", fontsize=8, color=color_line)
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
                set_subplot_title(ax, "Subcircuit")
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
                set_subplot_title(ax, "Full Gate")
            if row == 3:
                ax.set_xlabel("Perturbation Effect", fontsize=9)

            # Column 2: Agreement (binned)
            ax = axes[row, 2]
            if samples:
                signed_noise = [_compute_perturbation_effect(s) for s in samples]
                _plot_agreement_binned(ax, samples, signed_noise, n_bins=8)
            ax.grid(alpha=0.3, axis="y")
            if row == 0:
                set_subplot_title(ax, "Agreement")
            if row == 3:
                ax.set_xlabel("Perturbation Effect", fontsize=9)

        # Use global layout helper
        finalize_figure(fig, f"{prefix}Noise Robustness", has_legend_below=True, fontsize=14)

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
                    set_subplot_title(ax, "Subcircuit")
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
                    set_subplot_title(ax, "Full Gate")
                if row == n_rows - 1:
                    ax.set_xlabel(subtype_info["xlabel"], fontsize=9)

                # Column 2: Agreement (binned)
                ax = axes[row, 2]
                if samples:
                    x_vals = [abs(s.noise_magnitude) for s in samples]
                    _plot_agreement_binned(ax, samples, x_vals, n_bins=8)
                ax.grid(alpha=0.3, axis="y")
                if row == 0:
                    set_subplot_title(ax, "Agreement")
                if row == n_rows - 1:
                    ax.set_xlabel(subtype_info["xlabel"], fontsize=9)

            # Use global layout helper
            finalize_figure(fig, f"{prefix}OOD: {subtype_info['title']}", has_legend_below=True, fontsize=14)

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

"""Circuit activation visualization.

Contains functions for visualizing circuit activations:
- visualize_circuit_activations_from_data: Grid for canonical inputs (adapts to n_inputs)
- visualize_circuit_activations_mean: 1x4 grid for mean activations by range
"""

import math
import os

import matplotlib.pyplot as plt
import torch

from src.circuit import Circuit
from .constants import finalize_figure
from .circuit_drawing import _draw_circuit_from_data


def _filter_activations_for_gate(
    activations: list[torch.Tensor], gate_idx: int | None
) -> list[torch.Tensor]:
    """Filter output layer activations to only show the specific gate's output."""
    if gate_idx is None or not activations:
        return activations

    # Copy all layers except the last one
    filtered = [a.clone() for a in activations[:-1]]

    # Filter the output layer to only include the specific gate
    output_layer = activations[-1]
    if output_layer.shape[-1] > 1 and gate_idx < output_layer.shape[-1]:
        # Select only the gate_idx column, keeping batch dimension
        filtered.append(output_layer[:, gate_idx : gate_idx + 1])
    else:
        filtered.append(output_layer)

    return filtered


def _build_canonical_labels(canonical_activations: dict) -> dict[str, str]:
    """Build labels map from available canonical activation keys.

    Handles both 2-input keys like "0_0" and 3-input keys like "0_0_0".
    """
    if not canonical_activations:
        return {}

    labels_map = {}
    for key in sorted(canonical_activations.keys()):
        # Convert "0_1_0" to "(0, 1, 0)"
        bits = key.split("_")
        label = "(" + ", ".join(bits) + ")"
        labels_map[key] = label

    return labels_map


def _filter_weights_biases(
    layer_weights: list[torch.Tensor],
    layer_biases: list[torch.Tensor] | None,
    gate_idx: int | None,
) -> tuple[list, list | None]:
    """Filter weights and biases for single gate output if needed."""
    if gate_idx is not None and layer_weights:
        weights = [w.numpy() for w in layer_weights[:-1]]
        last_weight = layer_weights[-1]
        if last_weight.shape[0] > 1 and gate_idx < last_weight.shape[0]:
            weights.append(last_weight[gate_idx : gate_idx + 1, :].numpy())
        else:
            weights.append(last_weight.numpy())

        if layer_biases:
            biases = [b.numpy() for b in layer_biases[:-1]]
            last_bias = layer_biases[-1]
            if last_bias.shape[0] > 1 and gate_idx < last_bias.shape[0]:
                biases.append(last_bias[gate_idx : gate_idx + 1].numpy())
            else:
                biases.append(last_bias.numpy())
        else:
            biases = None
    else:
        weights = [w.numpy() for w in layer_weights]
        biases = [b.numpy() for b in layer_biases] if layer_biases else None

    return weights, biases


def visualize_circuit_activations_from_data(
    canonical_activations: dict[str, list[torch.Tensor]],
    layer_weights: list[torch.Tensor],
    circuit: Circuit,
    output_dir: str,
    filename: str = "circuit_activations.png",
    gate_name: str = "",
    layer_biases: list[torch.Tensor] | None = None,
    gate_idx: int | None = None,
) -> str:
    """
    Grid showing circuit activations for all canonical binary inputs.

    Adapts grid size based on n_inputs:
    - 2 inputs: 2x2 grid (4 combinations)
    - 3 inputs: 2x4 grid (8 combinations)

    Uses pre-computed activations - NO model execution.

    Args:
        layer_biases: Optional bias vectors. If provided, edge labels show
            (weight + bias) to reveal bias contribution when edges are patched.
        gate_idx: If specified, only show this gate's output node (0-indexed).
            When None, shows all output nodes.
    """
    labels_map = _build_canonical_labels(canonical_activations)

    if not labels_map:
        # No data - create minimal figure with message
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "No canonical activations data", ha="center", va="center")
        ax.axis("off")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    n_combinations = len(labels_map)
    n_cols = min(4, n_combinations)
    n_rows = math.ceil(n_combinations / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))

    # Handle single subplot case
    if n_combinations == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    weights, biases = _filter_weights_biases(layer_weights, layer_biases, gate_idx)

    for i, (key, label) in enumerate(labels_map.items()):
        if i >= len(axes):
            break
        activations = canonical_activations.get(key, [])
        if activations:
            filtered_activations = _filter_activations_for_gate(activations, gate_idx)
            _draw_circuit_from_data(
                axes[i], filtered_activations, circuit, weights, f"Input: {label}", biases=biases
            )
        else:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[i].axis("off")

    # Hide unused subplots
    for i in range(len(labels_map), len(axes)):
        axes[i].axis("off")

    if gate_name:
        finalize_figure(fig, f"{gate_name} - Circuit Activations", fontsize=14)
    else:
        plt.tight_layout()

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
    gate_idx: int | None = None,
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
        gate_idx: If specified, only show this gate's output node (0-indexed).
            When None, shows all output nodes.
    """
    labels_map = {
        "0_1": "[0, 1]",
        "-1_0": "[-1, 0]",
        "-2_2": "[-2, 2]",
        "-100_100": "[-100, 100]",
    }

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    weights, biases = _filter_weights_biases(layer_weights, layer_biases, gate_idx)

    for i, (key, label) in enumerate(labels_map.items()):
        activations = mean_activations_by_range.get(key, [])
        if activations:
            filtered_activations = _filter_activations_for_gate(activations, gate_idx)
            _draw_circuit_from_data(
                axes[i], filtered_activations, circuit, weights, f"Input Range: {label}", biases=biases
            )
        else:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[i].axis("off")

    if gate_name:
        finalize_figure(fig, f"{gate_name} - Mean Activations by Input Range", fontsize=14)
    else:
        plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path

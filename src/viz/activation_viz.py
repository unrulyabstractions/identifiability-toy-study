"""Circuit activation visualization.

Contains functions for visualizing circuit activations:
- visualize_circuit_activations_from_data: 2x2 grid for canonical inputs
- visualize_circuit_activations_mean: 1x4 grid for mean activations by range
"""

import os

import matplotlib.pyplot as plt
import torch

from ..common.circuit import Circuit
from .base import finalize_figure
from .circuit_drawing import _draw_circuit_from_data


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

    if gate_name:
        finalize_figure(fig, f"{gate_name} - Mean Activations by Input Range", fontsize=14)
    else:
        plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path

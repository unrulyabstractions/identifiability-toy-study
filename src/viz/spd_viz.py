"""SPD (Spectral Decomposition) visualization.

Contains functions for visualizing SPD decompositions:
- visualize_spd_components: Visualize SPD component weights
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from src.model import DecomposedMLP
from .circuit_drawing import _get_spd_component_weights


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

"""Attribution heatmap visualizations.

Visualizes attribution scores as heatmaps for easy interpretation.
"""

from typing import TYPE_CHECKING
import os

import numpy as np
import matplotlib.pyplot as plt

from ..types import AttributionResult, InputAttribution

if TYPE_CHECKING:
    from src.model import MLP


def plot_layer_attribution_heatmap(
    result: AttributionResult,
    output_path: str,
    title: str | None = None,
    cmap: str = "RdBu_r",
) -> str:
    """Plot layer-wise attribution scores as a heatmap.

    Args:
        result: AttributionResult with layer_attributions
        output_path: Path to save the figure
        title: Optional title for the plot
        cmap: Colormap to use

    Returns:
        Path to the saved figure
    """
    if not result.layer_attributions:
        raise ValueError("No layer attributions to plot")

    # Determine figure layout
    n_layers = len(result.layer_attributions)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))

    if n_layers == 1:
        axes = [axes]

    for i, (layer_idx, scores) in enumerate(sorted(result.layer_attributions.items())):
        ax = axes[i]

        if scores.ndim == 1:
            # Neuron-level attribution: reshape to column vector
            scores = scores.reshape(-1, 1)
            im = ax.imshow(scores, cmap=cmap, aspect='auto')
            ax.set_xlabel('Attribution')
            ax.set_ylabel('Neuron')
            ax.set_xticks([])
        else:
            # Edge-level attribution: [out, in] matrix
            im = ax.imshow(scores, cmap=cmap, aspect='auto')
            ax.set_xlabel('Input neuron')
            ax.set_ylabel('Output neuron')

        ax.set_title(f'Layer {layer_idx}')
        plt.colorbar(im, ax=ax)

    if title:
        fig.suptitle(title, fontsize=14)
    else:
        fig.suptitle(f'Attribution: {result.method}', fontsize=14)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_edge_attribution_matrix(
    result: AttributionResult,
    model: "MLP",
    output_path: str,
    title: str | None = None,
    cmap: str = "RdBu_r",
) -> str:
    """Plot edge attributions as a full adjacency matrix.

    Args:
        result: AttributionResult with edge_attributions
        model: The MLP model (for structure)
        output_path: Path to save the figure
        title: Optional title for the plot
        cmap: Colormap to use

    Returns:
        Path to the saved figure
    """
    if not result.edge_attributions:
        raise ValueError("No edge attributions to plot")

    # Count total neurons
    layer_sizes = model.layer_sizes
    total_neurons = sum(layer_sizes)

    # Create adjacency matrix
    adj_matrix = np.zeros((total_neurons, total_neurons))

    # Fill in edge attributions
    cumsum = [0] + list(np.cumsum(layer_sizes))

    for (layer, in_idx, next_layer, out_idx), score in result.edge_attributions.items():
        from_idx = cumsum[layer] + in_idx
        to_idx = cumsum[next_layer] + out_idx
        adj_matrix[from_idx, to_idx] = score

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    vmax = max(abs(adj_matrix.min()), abs(adj_matrix.max()))
    if vmax == 0:
        vmax = 1
    im = ax.imshow(adj_matrix, cmap=cmap, vmin=-vmax, vmax=vmax)

    # Add layer boundaries
    for i in range(1, len(layer_sizes)):
        boundary = cumsum[i] - 0.5
        ax.axhline(y=boundary, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)

    # Labels
    layer_labels = []
    for i, size in enumerate(layer_sizes):
        layer_labels.extend([f'L{i}:{j}' for j in range(size)])

    ax.set_xticks(range(total_neurons))
    ax.set_yticks(range(total_neurons))
    ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(layer_labels, fontsize=8)

    ax.set_xlabel('To neuron')
    ax.set_ylabel('From neuron')

    plt.colorbar(im, ax=ax, label='Attribution')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Edge Attribution: {result.method}')

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_input_attribution(
    result: InputAttribution,
    output_path: str,
    title: str | None = None,
    show_samples: int = 10,
) -> str:
    """Plot input attribution scores.

    Args:
        result: InputAttribution with attribution scores
        output_path: Path to save the figure
        title: Optional title for the plot
        show_samples: Number of samples to show in detail

    Returns:
        Path to the saved figure
    """
    if result.attributions is None:
        raise ValueError("No attributions to plot")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Mean attribution per input feature
    mean_attr = result.get_mean_attribution()
    ax = axes[0]
    bars = ax.bar(range(len(mean_attr)), mean_attr, color='steelblue')
    ax.set_xlabel('Input Feature')
    ax.set_ylabel('Mean |Attribution|')
    ax.set_title('Feature Importance')
    ax.set_xticks(range(len(mean_attr)))
    ax.set_xticklabels([f'x{i}' for i in range(len(mean_attr))])

    # Right: Sample-level attributions
    ax = axes[1]
    n_show = min(show_samples, result.attributions.shape[0])
    for i in range(n_show):
        ax.plot(range(result.attributions.shape[1]), result.attributions[i],
                alpha=0.5, marker='o', label=f'Sample {i}')

    ax.set_xlabel('Input Feature')
    ax.set_ylabel('Attribution')
    ax.set_title(f'Sample Attributions (n={n_show})')
    ax.set_xticks(range(result.attributions.shape[1]))
    ax.set_xticklabels([f'x{i}' for i in range(result.attributions.shape[1])])
    ax.legend(loc='best', fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14)
    else:
        fig.suptitle(f'Input Attribution: {result.method}', fontsize=14)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_attribution_comparison(
    results: list[AttributionResult],
    output_path: str,
    layer_idx: int = 1,
    title: str | None = None,
) -> str:
    """Compare multiple attribution methods on the same layer.

    Args:
        results: List of AttributionResult to compare
        output_path: Path to save the figure
        layer_idx: Which layer to compare
        title: Optional title for the plot

    Returns:
        Path to the saved figure
    """
    # Filter to results that have the specified layer
    valid_results = [r for r in results if layer_idx in r.layer_attributions]

    if not valid_results:
        raise ValueError(f"No results have attributions for layer {layer_idx}")

    fig, ax = plt.subplots(figsize=(10, 6))

    n_neurons = max(len(r.layer_attributions[layer_idx].flatten()) for r in valid_results)
    x = np.arange(n_neurons)
    width = 0.8 / len(valid_results)

    for i, result in enumerate(valid_results):
        scores = result.layer_attributions[layer_idx].flatten()
        if len(scores) < n_neurons:
            scores = np.pad(scores, (0, n_neurons - len(scores)))
        ax.bar(x + i * width, np.abs(scores), width, label=result.method, alpha=0.8)

    ax.set_xlabel('Neuron/Edge Index')
    ax.set_ylabel('|Attribution|')
    ax.set_title(title or f'Attribution Comparison - Layer {layer_idx}')
    ax.legend()
    ax.set_xticks(x + width * len(valid_results) / 2)
    ax.set_xticklabels([str(i) for i in range(n_neurons)])

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

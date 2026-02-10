"""Probing visualization functions.

Provides visualization utilities for linear probing results:
- Accuracy across layers
- Feature importance heatmaps
- Comparison plots for multiple probes
"""

from typing import TYPE_CHECKING
import os

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from ..types import ProbeResult, ProbeAnalysis


def plot_probe_accuracy(
    analysis: "ProbeAnalysis",
    output_path: str,
    title: str | None = None,
    show: bool = False,
) -> str:
    """Plot probe accuracy across layers.

    Args:
        analysis: ProbeAnalysis to plot
        output_path: Path to save the figure
        title: Optional title for the plot
        show: Whether to display the plot

    Returns:
        Path to the saved figure
    """
    layers = [x[0] for x in analysis.layer_accuracies]
    accuracies = [x[1] for x in analysis.layer_accuracies]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(layers, accuracies, color='steelblue', alpha=0.8)

    # Highlight best layer
    if analysis.best_layer in layers:
        best_idx = layers.index(analysis.best_layer)
        bars[best_idx].set_color('red')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title(title or f"Probe Accuracy for '{analysis.target_name}'")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_probe_comparison(
    analyses: list["ProbeAnalysis"],
    output_path: str,
    title: str | None = None,
    show: bool = False,
) -> str:
    """Plot probe accuracies for multiple targets.

    Args:
        analyses: List of ProbeAnalysis to compare
        output_path: Path to save the figure
        title: Optional title for the plot
        show: Whether to display the plot

    Returns:
        Path to the saved figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_analyses = len(analyses)
    width = 0.8 / n_analyses

    # Get all layers from first analysis
    if not analyses:
        raise ValueError("No analyses provided")

    layers = [x[0] for x in analyses[0].layer_accuracies]

    for i, analysis in enumerate(analyses):
        accuracies = [x[1] for x in analysis.layer_accuracies]

        x_positions = np.array(layers) + i * width - (n_analyses - 1) * width / 2
        ax.bar(x_positions, accuracies, width, label=analysis.target_name, alpha=0.8)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title(title or "Probe Accuracy Comparison")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_feature_importance(
    result: "ProbeResult",
    output_path: str,
    title: str | None = None,
    show: bool = False,
) -> str:
    """Plot feature importance from probe weights.

    Args:
        result: ProbeResult with trained weights
        output_path: Path to save the figure
        title: Optional title for the plot
        show: Whether to display the plot

    Returns:
        Path to the saved figure
    """
    importance = result.get_feature_importance()

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(importance))
    ax.bar(x, importance, color='steelblue', alpha=0.8)

    ax.set_xlabel('Hidden Feature')
    ax.set_ylabel('Importance (|weight|)')
    ax.set_title(title or f"Feature Importance - Layer {result.layer_idx}")
    ax.set_xticks(x)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_loss_history(
    result: "ProbeResult",
    output_path: str,
    title: str | None = None,
    show: bool = False,
) -> str:
    """Plot training loss history for a probe.

    Args:
        result: ProbeResult with loss_history
        output_path: Path to save the figure
        title: Optional title for the plot
        show: Whether to display the plot

    Returns:
        Path to the saved figure
    """
    if not result.loss_history:
        raise ValueError("No loss history to plot")

    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(len(result.loss_history))
    ax.plot(epochs, result.loss_history, color='steelblue', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title or f"Training Loss - {result.target_name} (Layer {result.layer_idx})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_layer_accuracy_heatmap(
    analyses: list["ProbeAnalysis"],
    output_path: str,
    title: str | None = None,
    show: bool = False,
) -> str:
    """Plot accuracy heatmap with targets on y-axis and layers on x-axis.

    Args:
        analyses: List of ProbeAnalysis to plot
        output_path: Path to save the figure
        title: Optional title for the plot
        show: Whether to display the plot

    Returns:
        Path to the saved figure
    """
    if not analyses:
        raise ValueError("No analyses provided")

    # Build matrix
    target_names = [a.target_name for a in analyses]
    layers = [x[0] for x in analyses[0].layer_accuracies]
    n_layers = len(layers)
    n_targets = len(analyses)

    matrix = np.zeros((n_targets, n_layers))
    for i, analysis in enumerate(analyses):
        for j, (layer_idx, acc) in enumerate(analysis.layer_accuracies):
            matrix[i, j] = acc

    fig, ax = plt.subplots(figsize=(10, max(4, n_targets * 0.5)))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_yticks(range(n_targets))
    ax.set_yticklabels(target_names)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Target')
    ax.set_title(title or 'Probe Accuracy by Layer and Target')

    # Add text annotations
    for i in range(n_targets):
        for j in range(n_layers):
            val = matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label='Accuracy')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path

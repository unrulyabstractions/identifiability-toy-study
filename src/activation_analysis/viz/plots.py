"""Activation analysis visualization functions.

Provides visualization utilities for activation analysis:
- Statistics distribution plots
- Correlation heatmaps
- Clustering visualizations
"""

from typing import TYPE_CHECKING
import os

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from ..types import (
        ActivationStatistics,
        LayerStatistics,
        ActivationCorrelation,
        ClusteringResult,
    )


def plot_layer_statistics(
    statistics: "LayerStatistics",
    output_path: str,
    title: str | None = None,
    show: bool = False,
) -> str:
    """Plot statistics across all layers.

    Args:
        statistics: LayerStatistics to plot
        output_path: Path to save the figure
        title: Optional title
        show: Whether to display the plot

    Returns:
        Path to saved figure
    """
    n_layers = statistics.n_layers
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    layer_indices = sorted(statistics.per_layer.keys())

    # Plot 1: Mean activation per layer
    ax = axes[0, 0]
    for layer_idx in layer_indices:
        stats = statistics.per_layer[layer_idx]
        if stats.mean is not None:
            x = np.arange(len(stats.mean))
            ax.bar(x + layer_idx * 0.1, stats.mean, width=0.1,
                   label=f'Layer {layer_idx}', alpha=0.7)
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Mean Activation by Layer')
    ax.legend(loc='best', fontsize=8)

    # Plot 2: Standard deviation per layer
    ax = axes[0, 1]
    for layer_idx in layer_indices:
        stats = statistics.per_layer[layer_idx]
        if stats.std is not None:
            x = np.arange(len(stats.std))
            ax.bar(x + layer_idx * 0.1, stats.std, width=0.1,
                   label=f'Layer {layer_idx}', alpha=0.7)
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Std Activation')
    ax.set_title('Standard Deviation by Layer')
    ax.legend(loc='best', fontsize=8)

    # Plot 3: Sparsity per layer
    ax = axes[1, 0]
    sparsities = []
    labels = []
    for layer_idx in layer_indices:
        stats = statistics.per_layer[layer_idx]
        if stats.sparsity is not None:
            sparsities.append(stats.sparsity.mean())
            labels.append(f'L{layer_idx}')
    if sparsities:
        ax.bar(labels, sparsities, color='steelblue', alpha=0.8)
        ax.set_ylabel('Mean Sparsity')
        ax.set_title('Average Sparsity by Layer')
        ax.set_ylim(0, 1)

    # Plot 4: Range (max - min) per layer
    ax = axes[1, 1]
    for layer_idx in layer_indices:
        stats = statistics.per_layer[layer_idx]
        if stats.min_val is not None and stats.max_val is not None:
            ranges = stats.max_val - stats.min_val
            x = np.arange(len(ranges))
            ax.bar(x + layer_idx * 0.1, ranges, width=0.1,
                   label=f'Layer {layer_idx}', alpha=0.7)
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Activation Range')
    ax.set_title('Activation Range by Layer')
    ax.legend(loc='best', fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_correlation_heatmap(
    correlation: "ActivationCorrelation",
    output_path: str,
    title: str | None = None,
    show: bool = False,
) -> str:
    """Plot correlation matrix as a heatmap.

    Args:
        correlation: ActivationCorrelation to plot
        output_path: Path to save the figure
        title: Optional title
        show: Whether to display the plot

    Returns:
        Path to saved figure
    """
    if correlation.correlation_matrix is None:
        raise ValueError("No correlation matrix to plot")

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(correlation.correlation_matrix, cmap='RdBu_r',
                   vmin=-1, vmax=1, aspect='auto')

    n_neurons = correlation.correlation_matrix.shape[0]
    ax.set_xticks(range(n_neurons))
    ax.set_yticks(range(n_neurons))
    ax.set_xticklabels([str(i) for i in range(n_neurons)])
    ax.set_yticklabels([str(i) for i in range(n_neurons)])

    ax.set_xlabel('Neuron')
    ax.set_ylabel('Neuron')

    plt.colorbar(im, ax=ax, label='Correlation')

    ax.set_title(title or f'Neuron Correlation - Layer {correlation.layer_idx}')

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_correlation_summary(
    correlations: dict[int, "ActivationCorrelation"],
    output_path: str,
    title: str | None = None,
    show: bool = False,
) -> str:
    """Plot mean correlation across layers.

    Args:
        correlations: Dictionary of layer_idx to ActivationCorrelation
        output_path: Path to save the figure
        title: Optional title
        show: Whether to display the plot

    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    layers = sorted(correlations.keys())
    mean_corrs = [correlations[l].mean_abs_correlation for l in layers]

    ax.bar([f'L{l}' for l in layers], mean_corrs, color='steelblue', alpha=0.8)
    ax.set_ylabel('Mean |Correlation|')
    ax.set_xlabel('Layer')
    ax.set_title(title or 'Mean Absolute Correlation by Layer')
    ax.set_ylim(0, 1)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_clustering_result(
    result: "ClusteringResult",
    output_path: str,
    title: str | None = None,
    show: bool = False,
) -> str:
    """Plot clustering result.

    Args:
        result: ClusteringResult to plot
        output_path: Path to save the figure
        title: Optional title
        show: Whether to display the plot

    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Cluster sizes
    ax = axes[0]
    cluster_labels = [f'C{i}' for i in range(result.n_clusters)]
    colors = plt.cm.tab10(np.arange(result.n_clusters))
    ax.bar(cluster_labels, result.cluster_sizes, color=colors, alpha=0.8)
    ax.set_ylabel('Number of Neurons')
    ax.set_xlabel('Cluster')
    ax.set_title('Cluster Sizes')

    # Plot 2: Neuron assignments (as a scatter-like plot)
    ax = axes[1]
    if result.cluster_labels is not None:
        n_neurons = len(result.cluster_labels)
        for c in range(result.n_clusters):
            neuron_indices = np.where(result.cluster_labels == c)[0]
            ax.scatter(neuron_indices, [c] * len(neuron_indices),
                      c=[colors[c]], s=100, alpha=0.8, label=f'Cluster {c}')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Cluster')
        ax.set_yticks(range(result.n_clusters))
        ax.set_title(f'Neuron Assignments (silhouette={result.silhouette_score:.3f})')
        ax.legend(loc='best')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_cluster_centers(
    result: "ClusteringResult",
    output_path: str,
    title: str | None = None,
    show: bool = False,
) -> str:
    """Plot cluster centers (mean activation patterns).

    Args:
        result: ClusteringResult with cluster_centers
        output_path: Path to save the figure
        title: Optional title
        show: Whether to display the plot

    Returns:
        Path to saved figure
    """
    if result.cluster_centers is None:
        raise ValueError("No cluster centers to plot")

    fig, ax = plt.subplots(figsize=(10, 6))

    n_clusters = result.cluster_centers.shape[0]
    n_features = result.cluster_centers.shape[1]

    for c in range(n_clusters):
        ax.plot(range(n_features), result.cluster_centers[c],
                marker='o', label=f'Cluster {c}', alpha=0.7)

    ax.set_xlabel('Feature Index (Sample)')
    ax.set_ylabel('Mean Activation')
    ax.set_title(title or f'Cluster Centers - Layer {result.layer_idx}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_silhouette_scores(
    scores: list[float],
    output_path: str,
    title: str | None = None,
    show: bool = False,
) -> str:
    """Plot silhouette scores vs number of clusters.

    Args:
        scores: Silhouette scores for k=2, 3, 4, ...
        output_path: Path to save the figure
        title: Optional title
        show: Whether to display the plot

    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    k_values = list(range(2, len(scores) + 2))
    ax.plot(k_values, scores, marker='o', color='steelblue', linewidth=2)

    # Mark optimal
    best_k = np.argmax(scores) + 2
    ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7,
               label=f'Optimal k={best_k}')

    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title(title or 'Silhouette Score vs Number of Clusters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path

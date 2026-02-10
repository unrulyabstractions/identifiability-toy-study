"""Activation analysis type definitions.

Contains dataclasses for activation analysis results:
- ActivationStatistics: Summary statistics for layer activations
- ActivationCorrelation: Correlation analysis between neurons
- ClusteringResult: Results from activation clustering
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ActivationStatistics:
    """Statistics for activations at a single layer.

    Attributes:
        layer_idx: Which layer these statistics are for
        mean: Mean activation per neuron [n_neurons]
        std: Standard deviation per neuron [n_neurons]
        min_val: Minimum activation per neuron [n_neurons]
        max_val: Maximum activation per neuron [n_neurons]
        sparsity: Fraction of zero/near-zero activations per neuron [n_neurons]
        n_samples: Number of samples used to compute statistics
    """
    layer_idx: int = 0
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None
    sparsity: Optional[np.ndarray] = None
    n_samples: int = 0

    def summary(self) -> str:
        """Return a summary string of the statistics."""
        if self.mean is None:
            return f"ActivationStatistics(layer={self.layer_idx}, no data)"
        return (
            f"ActivationStatistics(layer={self.layer_idx}, "
            f"n_samples={self.n_samples}, n_neurons={len(self.mean)}, "
            f"mean_range=[{self.mean.min():.3f}, {self.mean.max():.3f}], "
            f"avg_sparsity={self.sparsity.mean():.3f})"
        )


@dataclass
class LayerStatistics:
    """Complete statistics across all layers.

    Attributes:
        per_layer: Dictionary mapping layer_idx to ActivationStatistics
        n_layers: Number of layers
        n_samples: Number of samples used
    """
    per_layer: dict[int, ActivationStatistics] = field(default_factory=dict)
    n_layers: int = 0
    n_samples: int = 0

    def summary(self) -> str:
        """Return a summary of statistics across all layers."""
        lines = [f"LayerStatistics (n_layers={self.n_layers}, n_samples={self.n_samples})"]
        for layer_idx in sorted(self.per_layer.keys()):
            stats = self.per_layer[layer_idx]
            lines.append(f"  {stats.summary()}")
        return "\n".join(lines)


@dataclass
class ActivationCorrelation:
    """Correlation analysis for a layer's activations.

    Attributes:
        layer_idx: Which layer this correlation is for
        correlation_matrix: Pearson correlation between neurons [n_neurons, n_neurons]
        top_positive_pairs: List of (neuron_i, neuron_j, correlation) for highest correlations
        top_negative_pairs: List of (neuron_i, neuron_j, correlation) for most negative correlations
        mean_abs_correlation: Average absolute correlation (measure of redundancy)
    """
    layer_idx: int = 0
    correlation_matrix: Optional[np.ndarray] = None
    top_positive_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    top_negative_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    mean_abs_correlation: float = 0.0

    def summary(self) -> str:
        """Return a summary of the correlation analysis."""
        lines = [f"ActivationCorrelation(layer={self.layer_idx})"]
        lines.append(f"  Mean |correlation|: {self.mean_abs_correlation:.3f}")
        if self.top_positive_pairs:
            lines.append(f"  Top positive: {self.top_positive_pairs[0]}")
        if self.top_negative_pairs:
            lines.append(f"  Top negative: {self.top_negative_pairs[0]}")
        return "\n".join(lines)


@dataclass
class ClusteringResult:
    """Results from clustering activations or neurons.

    Attributes:
        layer_idx: Which layer was clustered
        n_clusters: Number of clusters found
        cluster_labels: Cluster assignment for each neuron [n_neurons]
        cluster_sizes: Number of neurons in each cluster
        cluster_centers: Mean activation pattern for each cluster [n_clusters, n_features]
        inertia: Sum of squared distances to cluster centers (lower is better)
        silhouette_score: Clustering quality metric (-1 to 1, higher is better)
    """
    layer_idx: int = 0
    n_clusters: int = 0
    cluster_labels: Optional[np.ndarray] = None
    cluster_sizes: list[int] = field(default_factory=list)
    cluster_centers: Optional[np.ndarray] = None
    inertia: float = 0.0
    silhouette_score: float = 0.0

    def summary(self) -> str:
        """Return a summary of the clustering result."""
        return (
            f"ClusteringResult(layer={self.layer_idx}, n_clusters={self.n_clusters}, "
            f"sizes={self.cluster_sizes}, silhouette={self.silhouette_score:.3f})"
        )


@dataclass
class ActivationAnalysisResult:
    """Complete activation analysis result.

    Attributes:
        statistics: Per-layer activation statistics
        correlations: Per-layer correlation analysis
        clustering: Per-layer clustering results
        model_path: Path to the analyzed model
    """
    statistics: Optional[LayerStatistics] = None
    correlations: dict[int, ActivationCorrelation] = field(default_factory=dict)
    clustering: dict[int, ClusteringResult] = field(default_factory=dict)
    model_path: str = ""

    def summary(self) -> str:
        """Return a comprehensive summary."""
        lines = ["ActivationAnalysisResult"]
        lines.append("=" * 40)

        if self.statistics:
            lines.append("\n" + self.statistics.summary())

        if self.correlations:
            lines.append("\nCorrelation Analysis:")
            for layer_idx in sorted(self.correlations.keys()):
                lines.append(f"  {self.correlations[layer_idx].summary()}")

        if self.clustering:
            lines.append("\nClustering:")
            for layer_idx in sorted(self.clustering.keys()):
                lines.append(f"  {self.clustering[layer_idx].summary()}")

        return "\n".join(lines)

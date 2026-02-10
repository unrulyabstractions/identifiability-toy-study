"""Activation clustering for interpretability.

Clusters neurons or activation patterns to find functional groups:
- K-means clustering of neuron activations
- Hierarchical clustering based on correlation
- Optimal cluster number estimation
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

from .types import ClusteringResult

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

if TYPE_CHECKING:
    from src.model import MLP


def cluster_neurons_kmeans(
    model: "MLP",
    x: torch.Tensor,
    layer_idx: int,
    n_clusters: int = 2,
    device: str = "cpu",
    random_state: int = 42,
) -> ClusteringResult:
    """Cluster neurons by their activation patterns using K-means.

    Each neuron is represented by its activation vector across samples.
    Similar neurons will be grouped together.

    Args:
        model: The MLP model to analyze
        x: Input samples [n_samples, n_inputs]
        layer_idx: Which layer to cluster
        n_clusters: Number of clusters
        device: Device to run on
        random_state: Random seed for reproducibility

    Returns:
        ClusteringResult with cluster assignments
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for K-means clustering. Install with: pip install scikit-learn")

    model = model.to(device)
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)

    if layer_idx >= len(activations):
        raise ValueError(f"Layer {layer_idx} does not exist")

    acts = activations[layer_idx].cpu().numpy()  # [n_samples, n_neurons]
    n_neurons = acts.shape[1]

    if n_neurons < n_clusters:
        # Not enough neurons for requested clusters
        return ClusteringResult(
            layer_idx=layer_idx,
            n_clusters=n_neurons,
            cluster_labels=np.arange(n_neurons),
            cluster_sizes=[1] * n_neurons,
            inertia=0.0,
            silhouette_score=0.0,
        )

    # Transpose so each row is a neuron's activation pattern
    neuron_features = acts.T  # [n_neurons, n_samples]

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(neuron_features)

    # Compute cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = [int(counts[np.where(unique_labels == i)[0][0]]) if i in unique_labels else 0
                     for i in range(n_clusters)]

    # Silhouette score (if we have enough clusters and samples)
    sil_score = 0.0
    if n_clusters > 1 and n_clusters < n_neurons:
        try:
            sil_score = silhouette_score(neuron_features, labels)
        except ValueError:
            pass  # Can fail if a cluster has only one sample

    return ClusteringResult(
        layer_idx=layer_idx,
        n_clusters=n_clusters,
        cluster_labels=labels,
        cluster_sizes=cluster_sizes,
        cluster_centers=kmeans.cluster_centers_,
        inertia=kmeans.inertia_,
        silhouette_score=sil_score,
    )


def cluster_neurons_hierarchical(
    model: "MLP",
    x: torch.Tensor,
    layer_idx: int,
    n_clusters: int | None = None,
    distance_threshold: float = 0.5,
    device: str = "cpu",
) -> ClusteringResult:
    """Cluster neurons using hierarchical clustering on correlation.

    Uses correlation-based distance (1 - |correlation|) to group
    neurons that have similar activation patterns.

    Args:
        model: The MLP model to analyze
        x: Input samples
        layer_idx: Which layer to cluster
        n_clusters: Number of clusters (if None, use distance_threshold)
        distance_threshold: Maximum distance for merging clusters
        device: Device to run on

    Returns:
        ClusteringResult with cluster assignments
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for hierarchical clustering. Install with: pip install scipy")

    model = model.to(device)
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)

    if layer_idx >= len(activations):
        raise ValueError(f"Layer {layer_idx} does not exist")

    acts = activations[layer_idx].cpu().numpy()  # [n_samples, n_neurons]
    n_neurons = acts.shape[1]

    if n_neurons < 2:
        return ClusteringResult(
            layer_idx=layer_idx,
            n_clusters=1,
            cluster_labels=np.array([0]),
            cluster_sizes=[1],
            silhouette_score=0.0,
        )

    # Compute correlation matrix
    corr_matrix = np.corrcoef(acts.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Convert to distance (1 - |correlation|)
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0)

    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method="average")

    if n_clusters is not None:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    else:
        labels = fcluster(Z, t=distance_threshold, criterion="distance") - 1

    # Renumber contiguously
    unique_labels = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([remap[l] for l in labels])

    actual_n_clusters = len(unique_labels)

    # Compute cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = [int(counts[np.where(unique == i)[0][0]]) if i in unique else 0
                     for i in range(actual_n_clusters)]

    # Silhouette score
    sil_score = 0.0
    if actual_n_clusters > 1 and actual_n_clusters < n_neurons:
        try:
            if SKLEARN_AVAILABLE:
                sil_score = silhouette_score(acts.T, labels)
        except ValueError:
            pass

    return ClusteringResult(
        layer_idx=layer_idx,
        n_clusters=actual_n_clusters,
        cluster_labels=labels,
        cluster_sizes=cluster_sizes,
        silhouette_score=sil_score,
    )


def cluster_samples_by_activation(
    model: "MLP",
    x: torch.Tensor,
    layer_idx: int,
    n_clusters: int = 2,
    device: str = "cpu",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster input samples by their activation patterns.

    Groups inputs that produce similar activations at a given layer.

    Args:
        model: The MLP model to analyze
        x: Input samples [n_samples, n_inputs]
        layer_idx: Which layer to use for clustering
        n_clusters: Number of clusters
        device: Device to run on
        random_state: Random seed

    Returns:
        Tuple of (cluster_labels, cluster_centers)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for clustering")

    model = model.to(device)
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)

    acts = activations[layer_idx].cpu().numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(acts)

    return labels, kmeans.cluster_centers_


def find_optimal_clusters(
    model: "MLP",
    x: torch.Tensor,
    layer_idx: int,
    max_clusters: int = 10,
    device: str = "cpu",
) -> tuple[int, list[float]]:
    """Find optimal number of clusters using silhouette score.

    Args:
        model: The MLP model
        x: Input samples
        layer_idx: Which layer to analyze
        max_clusters: Maximum number of clusters to try
        device: Device to run on

    Returns:
        Tuple of (optimal_n_clusters, silhouette_scores)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required")

    model = model.to(device)
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)

    acts = activations[layer_idx].cpu().numpy()
    n_neurons = acts.shape[1]

    # Limit max clusters to number of neurons
    max_k = min(max_clusters, n_neurons - 1)
    if max_k < 2:
        return 1, [0.0]

    neuron_features = acts.T  # [n_neurons, n_samples]

    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(neuron_features)
        try:
            score = silhouette_score(neuron_features, labels)
        except ValueError:
            score = 0.0
        scores.append(score)

    if not scores:
        return 1, [0.0]

    optimal_k = np.argmax(scores) + 2  # +2 because we started at k=2
    return optimal_k, scores

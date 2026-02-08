"""
SPD Clustering Module - Hierarchical clustering and dead component detection.

This module provides functions for clustering SPD components based on
coactivation patterns and detecting dead/alive components.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

# Scipy imports for hierarchical clustering (optional dependency)
try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

if TYPE_CHECKING:
    from src.model import DecomposedMLP


def detect_dead_components(
    decomposed_model: "DecomposedMLP",
    threshold: float = 0.01,
) -> tuple[list[str], list[str]]:
    """
    Detect dead components (those with negligible weight norms).

    Dead components are superfluous for replicating target model behavior.
    A good decomposition will have some dead components if n_components > needed.

    Args:
        decomposed_model: Trained SPD decomposition
        threshold: Relative threshold for considering a component dead

    Returns:
        Tuple of (alive_labels, dead_labels)
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return [], []

    component_model = decomposed_model.component_model

    alive_labels = []
    dead_labels = []

    for module_name, components in component_model.components.items():
        U = components.U  # [C, d_out]
        V = components.V  # [d_in, C]

        n_components = U.shape[0]

        # Compute norm of each rank-1 component contribution
        norms = []
        for c in range(n_components):
            u_norm = torch.norm(U[c, :]).item()
            v_norm = torch.norm(V[:, c]).item()
            norms.append(u_norm * v_norm)

        max_norm = max(norms) if norms else 1.0

        for c in range(n_components):
            label = f"{module_name}:{c}"
            if norms[c] < threshold * max_norm:
                dead_labels.append(label)
            else:
                alive_labels.append(label)

    return alive_labels, dead_labels


def cluster_components_hierarchical(
    coactivation_matrix: np.ndarray,
    n_clusters: int = None,
    merge_threshold: float = 0.7,
) -> list[int]:
    """
    Cluster components based on coactivation patterns using hierarchical clustering.

    Uses scipy's fast hierarchical clustering implementation.

    Args:
        coactivation_matrix: Shape [n_components, n_components]
        n_clusters: Target number of clusters (if None, determined by threshold)
        merge_threshold: Similarity threshold for merging (higher = more merging)

    Returns:
        cluster_assignments: List where index i gives cluster ID for component i
    """
    if coactivation_matrix.size == 0:
        return []

    n_components = coactivation_matrix.shape[0]

    # Handle trivial cases
    if n_components == 1:
        return [0]

    # Compute Jaccard-like similarity (vectorized)
    diag = np.diag(coactivation_matrix)
    diag_safe = np.maximum(diag, 1e-8)

    # Broadcasting: union[i,j] = diag[i] + diag[j] - coact[i,j]
    diag_i = diag_safe[:, np.newaxis]
    diag_j = diag_safe[np.newaxis, :]
    union = diag_i + diag_j - coactivation_matrix
    union = np.maximum(union, 1e-8)  # Avoid division by zero

    similarity = coactivation_matrix / union
    np.fill_diagonal(similarity, 1.0)

    # Convert to distance (condensed form for scipy)
    distance = 1 - similarity
    np.fill_diagonal(distance, 0)

    if SCIPY_AVAILABLE:
        # Convert to condensed distance matrix
        condensed_dist = squareform(distance, checks=False)

        # Perform hierarchical clustering (average linkage)
        Z = linkage(condensed_dist, method="average")

        if n_clusters is not None:
            # Cut to get exactly n_clusters
            cluster_assignments = fcluster(Z, t=n_clusters, criterion="maxclust")
        else:
            # Cut at distance threshold (1 - merge_threshold = distance threshold)
            dist_threshold = 1 - merge_threshold
            cluster_assignments = fcluster(Z, t=dist_threshold, criterion="distance")

        # Convert to 0-indexed list
        cluster_assignments = [int(c - 1) for c in cluster_assignments]

    else:
        # Fallback: simple greedy clustering if scipy not available
        cluster_assignments = list(range(n_components))

        # Merge similar components based on similarity threshold
        merged = [False] * n_components
        current_cluster = 0

        for i in range(n_components):
            if merged[i]:
                continue

            cluster_assignments[i] = current_cluster
            merged[i] = True

            # Find all components similar to i
            for j in range(i + 1, n_components):
                if not merged[j] and similarity[i, j] >= merge_threshold:
                    cluster_assignments[j] = current_cluster
                    merged[j] = True

            current_cluster += 1

    # Renumber clusters to be 0, 1, 2, ... contiguously
    unique_clusters = sorted(set(cluster_assignments))
    remap = {old: new for new, old in enumerate(unique_clusters)}
    cluster_assignments = [remap[c] for c in cluster_assignments]

    return cluster_assignments

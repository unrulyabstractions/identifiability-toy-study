"""SPD clustering: group components by activation patterns.

After computing importance and coactivation matrices, we need to group
components that behave similarly. Components that activate together on the
same inputs likely implement the same function.

This module provides:
1. Dead component detection: Find components with negligible weight norms
2. Hierarchical clustering: Group components by coactivation similarity
3. Function mapping: Match clusters to known boolean gates

The clustering pipeline:
    coactivation_matrix -> similarity -> distance -> hierarchical clustering
                                                            |
                                                            v
    cluster_assignments <- function mapping <- activation patterns
"""

from typing import TYPE_CHECKING

import numpy as np

from src.domain import ALL_LOGIC_GATES

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

if TYPE_CHECKING:
    import torch

    from src.model import DecomposedMLP


def detect_dead_components(
    decomposed_model: "DecomposedMLP",
    threshold: float = 0.01,
) -> tuple[list[str], list[str]]:
    """Detect dead components (negligible weight norms).

    Dead components are superfluous for replicating target model behavior.
    A good decomposition will have some dead components if n_components > needed.

    The check compares each component's weight norm to the maximum norm in its
    layer. Components below threshold * max_norm are considered dead.

    Args:
        decomposed_model: Trained SPD decomposition
        threshold: Fraction of max norm below which a component is "dead"

    Returns:
        Tuple of (alive_labels, dead_labels) where each label is like "layers.0.0:3"
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return [], []

    component_model = decomposed_model.component_model
    alive_labels = []
    dead_labels = []

    for module_name, components in component_model.components.items():
        U = components.U
        V = components.V
        n_components = U.shape[0]

        norms = []
        for c in range(n_components):
            u_norm = U[c, :].norm().item()
            v_norm = V[:, c].norm().item()
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
    """Group components that activate together into clusters.

    Components implementing the same function should have similar activation
    patterns. We use hierarchical clustering to group them.

    Algorithm:
        1. Convert coactivation counts to Jaccard similarity:
           sim[i,j] = coact[i,j] / (diag[i] + diag[j] - coact[i,j])
        2. Convert similarity to distance: dist = 1 - similarity
        3. Run hierarchical clustering (average linkage)
        4. Cut tree at threshold or to get n_clusters

    Args:
        coactivation_matrix: From compute_coactivation_matrix()
        n_clusters: If set, force exactly this many clusters
        merge_threshold: Similarity threshold for merging (higher = more merging)

    Returns:
        cluster_assignments: List where cluster_assignments[i] = cluster for component i
    """
    if coactivation_matrix.size == 0:
        return []

    n_components = coactivation_matrix.shape[0]
    if n_components == 1:
        return [0]

    # Compute Jaccard-like similarity
    diag = np.diag(coactivation_matrix)
    diag_safe = np.maximum(diag, 1e-8)

    diag_i = diag_safe[:, np.newaxis]
    diag_j = diag_safe[np.newaxis, :]
    union = diag_i + diag_j - coactivation_matrix
    union = np.maximum(union, 1e-8)

    similarity = coactivation_matrix / union
    np.fill_diagonal(similarity, 1.0)

    distance = 1 - similarity
    np.fill_diagonal(distance, 0)

    if SCIPY_AVAILABLE:
        condensed_dist = squareform(distance, checks=False)
        Z = linkage(condensed_dist, method="average")

        if n_clusters is not None:
            cluster_assignments = fcluster(Z, t=n_clusters, criterion="maxclust")
        else:
            dist_threshold = 1 - merge_threshold
            cluster_assignments = fcluster(Z, t=dist_threshold, criterion="distance")

        cluster_assignments = [int(c - 1) for c in cluster_assignments]
    else:
        # Fallback: greedy clustering when scipy not available
        cluster_assignments = list(range(n_components))
        merged = [False] * n_components
        current_cluster = 0

        for i in range(n_components):
            if merged[i]:
                continue
            cluster_assignments[i] = current_cluster
            merged[i] = True

            for j in range(i + 1, n_components):
                if not merged[j] and similarity[i, j] >= merge_threshold:
                    cluster_assignments[j] = current_cluster
                    merged[j] = True

            current_cluster += 1

    # Renumber contiguously
    unique_clusters = sorted(set(cluster_assignments))
    remap = {old: new for new, old in enumerate(unique_clusters)}
    return [remap[c] for c in cluster_assignments]


def map_clusters_to_functions(
    importance_matrix: np.ndarray,
    cluster_assignments: list[int],
    n_inputs: int = 2,
    gate_names: list[str] = None,
) -> dict[int, str]:
    """Identify which boolean function each cluster implements.

    Each cluster has an activation pattern: the inputs where it's active.
    We compare this to known boolean gate truth tables using Jaccard similarity.

    Example:
        XOR truth table: outputs 1 on inputs (0,1) and (1,0)
        If a cluster activates on exactly (0,1) and (1,0), Jaccard = 1.0 (perfect match)
        If it activates on (0,1), (1,0), and (0,0), Jaccard = 2/3 = 0.67

    Algorithm:
        1. For each cluster, find inputs where mean importance > 0.5
        2. For each known gate, find inputs where gate output = 1
        3. Compute Jaccard = |intersection| / |union|
        4. Assign best match if Jaccard > 0.5, else "UNKNOWN"

    Args:
        importance_matrix: Shape [n_inputs, n_components]
        cluster_assignments: Which cluster each component belongs to
        n_inputs: Number of input bits
        gate_names: Which gates to check (default: all known gates)

    Returns:
        Dict mapping cluster_idx -> "GATE_NAME (similarity)" or "UNKNOWN"/"INACTIVE"
    """
    if importance_matrix.size == 0 or not cluster_assignments:
        return {}

    n_total_inputs = 2**n_inputs
    n_clusters = max(cluster_assignments) + 1

    # Generate all binary inputs
    all_inputs = []
    for i in range(n_total_inputs):
        inp = tuple((i >> j) & 1 for j in range(n_inputs))
        all_inputs.append(inp)

    cluster_functions = {}
    gates_to_check = gate_names if gate_names else list(ALL_LOGIC_GATES.keys())

    for cluster_idx in range(n_clusters):
        component_indices = [
            i for i, c in enumerate(cluster_assignments) if c == cluster_idx
        ]

        if not component_indices:
            cluster_functions[cluster_idx] = "EMPTY"
            continue

        # Average importance for this cluster
        cluster_importance = importance_matrix[:, component_indices].mean(axis=1)
        active_inputs = set(np.where(cluster_importance > 0.5)[0])

        if not active_inputs:
            cluster_functions[cluster_idx] = "INACTIVE"
            continue

        # Find best matching gate
        best_match = "UNKNOWN"
        best_jaccard = 0

        for gate_name in gates_to_check:
            if gate_name not in ALL_LOGIC_GATES:
                continue
            gate = ALL_LOGIC_GATES[gate_name]
            if gate.n_inputs != n_inputs:
                continue

            truth_table = gate.truth_table()
            gate_active = {idx for idx, inp in enumerate(all_inputs) if truth_table.get(inp, 0) == 1}

            intersection = len(active_inputs & gate_active)
            union = len(active_inputs | gate_active)
            jaccard = intersection / union if union > 0 else 0

            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_match = gate_name

        if best_jaccard > 0.5:
            cluster_functions[cluster_idx] = f"{best_match} ({best_jaccard:.2f})"
        else:
            cluster_functions[cluster_idx] = "UNKNOWN"

    return cluster_functions

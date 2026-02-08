"""SPD analysis: importance, clustering, and function mapping.

This module provides the core analysis logic for SPD decomposition:
- Importance matrix computation (causal importance values per input)
- Coactivation matrix (which components fire together)
- Hierarchical clustering of components
- Function mapping (matching clusters to boolean gates)
- Per-cluster robustness and faithfulness metrics
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

from src.domain import ALL_LOGIC_GATES

from .types import ClusterInfo, SPDAnalysisResult, SPDSubcircuitEstimate
from .validation import compute_validation_metrics

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

if TYPE_CHECKING:
    from src.model import DecomposedMLP


# =============================================================================
# Importance Matrix Computation
# =============================================================================


def compute_importance_matrix(
    decomposed_model: "DecomposedMLP",
    n_inputs: int = 2,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """Compute causal importance values for all binary input combinations.

    Args:
        decomposed_model: Trained SPD decomposition
        n_inputs: Number of input dimensions (default 2 for boolean gates)
        device: Compute device

    Returns:
        importance_matrix: Shape [2^n_inputs, total_components]
        component_labels: List of component labels like "layers.0.0:3"
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return np.array([]), []

    component_model = decomposed_model.component_model

    # Generate all binary input combinations
    n_total_inputs = 2**n_inputs
    all_inputs = torch.zeros(n_total_inputs, n_inputs, device=device)
    for i in range(n_total_inputs):
        for j in range(n_inputs):
            all_inputs[i, j] = (i >> j) & 1

    # Get causal importances
    with torch.inference_mode():
        output_with_cache = component_model(all_inputs, cache_type="input")
        pre_weight_acts = output_with_cache.cache

        ci_outputs = component_model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sampling="continuous",
            detach_inputs=False,
        )
        ci_values = ci_outputs.upper_leaky

    # Concatenate all layer importances
    all_importances = []
    component_labels = []

    for module_name in sorted(ci_values.keys()):
        ci_tensor = ci_values[module_name]
        ci_np = ci_tensor.detach().cpu().numpy()
        all_importances.append(ci_np)

        n_components = ci_np.shape[1]
        for c in range(n_components):
            component_labels.append(f"{module_name}:{c}")

    if all_importances:
        importance_matrix = np.concatenate(all_importances, axis=1)
    else:
        importance_matrix = np.array([])

    return importance_matrix, component_labels


def compute_coactivation_matrix(
    importance_matrix: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute coactivation matrix showing which components fire together.

    Args:
        importance_matrix: Shape [n_samples, n_components]
        threshold: Component is "active" if importance > threshold

    Returns:
        coactivation_matrix: Shape [n_components, n_components], symmetric
    """
    if importance_matrix.size == 0:
        return np.array([])

    active_mask = (importance_matrix > threshold).astype(np.float32)
    return active_mask.T @ active_mask


# =============================================================================
# Component Clustering
# =============================================================================


def detect_dead_components(
    decomposed_model: "DecomposedMLP",
    threshold: float = 0.01,
) -> tuple[list[str], list[str]]:
    """Detect dead components (negligible weight norms).

    Dead components are superfluous for replicating target model behavior.
    A good decomposition will have some dead components if n_components > needed.

    Returns:
        Tuple of (alive_labels, dead_labels)
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
    """Cluster components based on coactivation using hierarchical clustering.

    Args:
        coactivation_matrix: Shape [n_components, n_components]
        n_clusters: Target number of clusters (if None, determined by threshold)
        merge_threshold: Similarity threshold for merging

    Returns:
        cluster_assignments: List where index i gives cluster ID for component i
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
        # Fallback: greedy clustering
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


# =============================================================================
# Function Mapping
# =============================================================================


def map_clusters_to_functions(
    importance_matrix: np.ndarray,
    cluster_assignments: list[int],
    n_inputs: int = 2,
    gate_names: list[str] = None,
) -> dict[int, str]:
    """Map SPD clusters to boolean functions using activation patterns.

    Compares each cluster's activation pattern against truth tables of known gates
    using Jaccard similarity.

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


# =============================================================================
# Main Analysis Entry Point
# =============================================================================


def run_spd_analysis(
    decomposed_model: "DecomposedMLP",
    target_model=None,
    n_inputs: int = 2,
    gate_names: list[str] = None,
    n_clusters: int = None,
    device: str = "cpu",
) -> SPDAnalysisResult:
    """Run complete SPD analysis: clustering, function mapping, and metrics.

    Args:
        decomposed_model: Trained SPD decomposition
        target_model: Original MLP (for robustness/faithfulness tests)
        n_inputs: Number of input dimensions
        gate_names: Names of gates in the model
        n_clusters: Target number of clusters (None = auto)
        device: Compute device

    Returns:
        SPDAnalysisResult with all analysis data
    """
    result = SPDAnalysisResult()

    if decomposed_model is None or decomposed_model.component_model is None:
        return result

    # Validation metrics
    validation_metrics = compute_validation_metrics(decomposed_model)
    result.mmcs = validation_metrics["mmcs"]
    result.ml2r = validation_metrics["ml2r"]
    result.faithfulness_loss = validation_metrics["faithfulness_loss"]

    # Dead component detection
    alive_labels, dead_labels = detect_dead_components(decomposed_model)
    result.n_alive_components = len(alive_labels)
    result.n_dead_components = len(dead_labels)
    result.dead_component_labels = dead_labels

    # Importance matrix
    importance_matrix, component_labels = compute_importance_matrix(
        decomposed_model, n_inputs, device
    )

    if importance_matrix.size == 0:
        return result

    result.importance_matrix = importance_matrix
    result.component_labels = component_labels
    result.n_components = len(component_labels)

    # Count layers
    layer_names = set(label.split(":")[0] for label in component_labels)
    result.n_layers = len(layer_names)

    # Clustering
    result.coactivation_matrix = compute_coactivation_matrix(importance_matrix)
    result.cluster_assignments = cluster_components_hierarchical(
        result.coactivation_matrix, n_clusters=n_clusters
    )
    result.n_clusters = max(result.cluster_assignments) + 1 if result.cluster_assignments else 0

    # Function mapping
    cluster_functions = map_clusters_to_functions(
        importance_matrix, result.cluster_assignments, n_inputs, gate_names
    )

    # Build cluster info
    for cluster_idx in range(result.n_clusters):
        component_indices = [
            i for i, c in enumerate(result.cluster_assignments) if c == cluster_idx
        ]
        cluster_labels = [component_labels[i] for i in component_indices]

        if component_indices:
            mean_imp = importance_matrix[:, component_indices].mean()
        else:
            mean_imp = 0.0

        cluster_info = ClusterInfo(
            cluster_idx=cluster_idx,
            component_indices=component_indices,
            component_labels=cluster_labels,
            mean_importance=float(mean_imp),
            function_mapping=cluster_functions.get(cluster_idx, ""),
        )
        result.clusters.append(cluster_info)

    return result


def estimate_spd_subcircuits(
    decomposed_model: "DecomposedMLP",
    n_inputs: int = 2,
    gate_names: list[str] = None,
    device: str = "cpu",
) -> SPDSubcircuitEstimate | None:
    """Estimate subcircuits from SPD decomposition using component clustering.

    Returns:
        SPDSubcircuitEstimate with cluster assignments and statistics
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return None

    n_components = decomposed_model.get_n_components()
    if n_components == 0:
        return None

    importance_matrix, component_labels = compute_importance_matrix(
        decomposed_model, n_inputs, device
    )

    if importance_matrix.size == 0:
        return SPDSubcircuitEstimate(
            cluster_assignments=list(range(n_components)),
            n_clusters=n_components,
            cluster_sizes=[1] * n_components,
        )

    coactivation_matrix = compute_coactivation_matrix(importance_matrix)
    cluster_assignments = cluster_components_hierarchical(coactivation_matrix)
    n_clusters = max(cluster_assignments) + 1 if cluster_assignments else 0

    cluster_sizes = [0] * n_clusters
    for c in cluster_assignments:
        cluster_sizes[c] += 1

    cluster_functions = map_clusters_to_functions(
        importance_matrix, cluster_assignments, n_inputs, gate_names
    )

    return SPDSubcircuitEstimate(
        cluster_assignments=cluster_assignments,
        n_clusters=n_clusters,
        cluster_sizes=cluster_sizes,
        component_importance=importance_matrix.mean(axis=0),
        coactivation_matrix=coactivation_matrix,
        component_labels=component_labels,
        cluster_functions=cluster_functions,
    )


# =============================================================================
# Per-Cluster Analysis (Robustness/Faithfulness)
# =============================================================================


def analyze_cluster_robustness(
    decomposed_model: "DecomposedMLP",
    cluster_info: ClusterInfo,
    importance_matrix: np.ndarray,
    n_samples: int = 20,
    noise_levels: list[float] = None,
    device: str = "cpu",
) -> dict:
    """Analyze robustness of a single cluster to input perturbations."""
    if noise_levels is None:
        noise_levels = [0.05, 0.1, 0.2]

    if decomposed_model is None or decomposed_model.component_model is None:
        return {"mean_importance_stability": 0.0, "noise_sensitivity": 0.0}

    component_model = decomposed_model.component_model
    n_inputs = 2

    comp_indices = cluster_info.component_indices
    if not comp_indices:
        return {"mean_importance_stability": 0.0, "noise_sensitivity": 0.0}

    baseline_imp = importance_matrix[:, comp_indices].mean()
    stability_scores = []
    importance_changes = []

    total_samples = len(noise_levels) * n_samples
    all_base_inputs = torch.randint(
        0, 2, (total_samples, n_inputs), dtype=torch.float, device=device
    )
    all_noise = torch.randn_like(all_base_inputs)

    noise_scales = torch.tensor(
        [level for level in noise_levels for _ in range(n_samples)], device=device
    ).unsqueeze(1)
    all_noisy_inputs = all_base_inputs + all_noise * noise_scales

    try:
        with torch.inference_mode():
            output_with_cache = component_model(all_noisy_inputs, cache_type="input")
            pre_weight_acts = output_with_cache.cache

            ci_outputs = component_model.calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                sampling="continuous",
                detach_inputs=False,
            )

            all_imp = []
            for module_name in sorted(ci_outputs.upper_leaky.keys()):
                ci_tensor = ci_outputs.upper_leaky[module_name]
                all_imp.append(ci_tensor.detach().cpu().numpy())

            if all_imp:
                full_imp = np.concatenate(all_imp, axis=1)
                cluster_imp = full_imp[:, comp_indices].mean(axis=1)

                for i, (imp, noise_level) in enumerate(
                    zip(cluster_imp, noise_scales.squeeze().cpu().numpy())
                ):
                    stability = 1.0 - abs(imp - baseline_imp)
                    stability_scores.append(max(0, stability))
                    importance_changes.append(abs(imp - baseline_imp) / max(noise_level, 0.01))
    except Exception:
        pass

    return {
        "mean_importance_stability": float(np.mean(stability_scores)) if stability_scores else 0.0,
        "noise_sensitivity": float(np.mean(importance_changes)) if importance_changes else 0.0,
        "n_samples_tested": len(stability_scores),
    }


def analyze_cluster_faithfulness(
    decomposed_model: "DecomposedMLP",
    cluster_info: ClusterInfo,
    importance_matrix: np.ndarray,
    n_inputs: int = 2,
    device: str = "cpu",
) -> dict:
    """Analyze faithfulness of a cluster by testing ablation effects."""
    if decomposed_model is None or decomposed_model.component_model is None:
        return {"mean_ablation_effect": 0.0, "sufficiency_score": 0.0}

    comp_indices = cluster_info.component_indices
    if not comp_indices:
        return {"mean_ablation_effect": 0.0, "sufficiency_score": 0.0}

    cluster_importance = importance_matrix[:, comp_indices]
    importance_variance = float(np.var(cluster_importance))
    max_importance = float(np.max(cluster_importance))

    active_mask = cluster_importance > 0.5
    mean_when_active = float(np.mean(cluster_importance[active_mask])) if active_mask.any() else 0.0

    return {
        "mean_ablation_effect": importance_variance,
        "sufficiency_score": max_importance,
        "mean_when_active": mean_when_active,
        "selectivity": importance_variance / (cluster_importance.mean() + 1e-8),
    }


def analyze_all_clusters(
    decomposed_model: "DecomposedMLP",
    analysis_result: SPDAnalysisResult,
    device: str = "cpu",
) -> list[dict]:
    """Run robustness and faithfulness analysis on all clusters."""
    results = []

    for cluster_info in analysis_result.clusters:
        robustness = analyze_cluster_robustness(
            decomposed_model, cluster_info, analysis_result.importance_matrix, device=device
        )
        faithfulness = analyze_cluster_faithfulness(
            decomposed_model, cluster_info, analysis_result.importance_matrix, device=device
        )

        cluster_info.robustness_score = robustness.get("mean_importance_stability", 0.0)
        cluster_info.faithfulness_score = faithfulness.get("sufficiency_score", 0.0)

        results.append({
            "cluster_idx": cluster_info.cluster_idx,
            "robustness": robustness,
            "faithfulness": faithfulness,
        })

    return results

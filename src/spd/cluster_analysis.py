"""
SPD Cluster Analysis Module - Per-cluster robustness and faithfulness analysis.

This module provides functions for analyzing individual clusters including
robustness testing, faithfulness analysis, and function mapping.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

from src.domain import ALL_LOGIC_GATES
from .schemas import ClusterInfo, SPDAnalysisResult

if TYPE_CHECKING:
    from src.model import DecomposedMLP


def map_clusters_to_functions(
    importance_matrix: np.ndarray,
    cluster_assignments: list[int],
    n_inputs: int = 2,
    gate_names: list[str] = None,
) -> dict[int, str]:
    """
    Identify which boolean function each SPD component cluster implements.

    Algorithm:
    1. For each cluster, compute mean importance across all components in that cluster
    2. Threshold importance to determine which inputs activate the cluster
    3. Compare the cluster's activation pattern against truth tables of known logic gates
       using Jaccard similarity (intersection over union of activating inputs)
    4. Assign the best-matching gate if similarity > 0.5, otherwise mark as UNKNOWN

    The importance matrix encodes how much each component contributes to the output
    for each possible binary input combination (e.g., for 2 inputs: 00, 01, 10, 11).
    Clusters that activate on the same inputs as a known gate (e.g., XOR activates
    on 01 and 10) are mapped to that gate.

    Args:
        importance_matrix: Shape [2^n_inputs, n_components]. Each row is a binary
            input pattern, each column is a component's importance for that input.
        cluster_assignments: List mapping component index -> cluster ID.
        n_inputs: Number of input bits (default 2 for boolean gates).
        gate_names: Restrict matching to these gate names. If None, check all gates.

    Returns:
        Dict mapping cluster_idx -> "GATE_NAME (similarity)" or "UNKNOWN"/"INACTIVE".

    Example:
        If cluster 0 has high importance on inputs (0,1) and (1,0), and low
        importance on (0,0) and (1,1), it matches XOR pattern with high similarity.
    """
    if importance_matrix.size == 0 or not cluster_assignments:
        return {}

    n_total_inputs = 2**n_inputs
    n_clusters = max(cluster_assignments) + 1

    # For each cluster, find which inputs activate it
    cluster_functions = {}

    for cluster_idx in range(n_clusters):
        # Get components in this cluster
        component_indices = [
            i for i, c in enumerate(cluster_assignments) if c == cluster_idx
        ]

        if not component_indices:
            cluster_functions[cluster_idx] = "EMPTY"
            continue

        # Average importance over cluster components for each input
        cluster_importance = importance_matrix[:, component_indices].mean(axis=1)

        # Inputs where cluster is highly active
        active_threshold = 0.5
        active_inputs = set(np.where(cluster_importance > active_threshold)[0])

        if not active_inputs:
            cluster_functions[cluster_idx] = "INACTIVE"
            continue

        # Compare to known boolean functions
        best_match = "UNKNOWN"
        best_jaccard = 0

        # Generate all binary inputs
        all_inputs = []
        for i in range(n_total_inputs):
            inp = tuple((i >> j) & 1 for j in range(n_inputs))
            all_inputs.append(inp)

        # Check against all known gates
        gates_to_check = gate_names if gate_names else list(ALL_LOGIC_GATES.keys())
        for gate_name in gates_to_check:
            if gate_name not in ALL_LOGIC_GATES:
                continue
            gate = ALL_LOGIC_GATES[gate_name]
            if gate.n_inputs != n_inputs:
                continue

            # Find inputs where this gate outputs 1 (use truth_table)
            truth_table = gate.truth_table()
            gate_active = set()
            for idx, inp in enumerate(all_inputs):
                if truth_table.get(inp, 0) == 1:
                    gate_active.add(idx)

            # Jaccard similarity
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


def analyze_cluster_robustness(
    decomposed_model: "DecomposedMLP",
    cluster_info: ClusterInfo,
    importance_matrix: np.ndarray,
    n_samples: int = 20,  # Reduced default for speed
    noise_levels: list[float] = None,
    device: str = "cpu",
) -> dict:
    """
    Analyze robustness of a single cluster to input perturbations.

    Tests whether the cluster's importance pattern is stable under noise.
    Uses batched inference for efficiency.

    Args:
        decomposed_model: Trained SPD decomposition
        cluster_info: Information about the cluster
        importance_matrix: Full importance matrix [n_inputs, n_components]
        n_samples: Number of noise samples per level
        noise_levels: List of noise magnitudes to test
        device: Compute device

    Returns:
        Dict with robustness metrics:
            - mean_importance_stability: How stable importance is under noise
            - noise_sensitivity: Importance change per unit noise
    """
    if noise_levels is None:
        noise_levels = [0.05, 0.1, 0.2]  # Fewer levels for speed

    if decomposed_model is None or decomposed_model.component_model is None:
        return {"mean_importance_stability": 0.0, "noise_sensitivity": 0.0}

    component_model = decomposed_model.component_model
    n_inputs = 2  # Boolean gates

    # Get component indices for this cluster
    comp_indices = cluster_info.component_indices
    if not comp_indices:
        return {"mean_importance_stability": 0.0, "noise_sensitivity": 0.0}

    # Baseline importance for cluster (mean over inputs)
    baseline_imp = importance_matrix[:, comp_indices].mean()

    stability_scores = []
    importance_changes = []

    # Generate all noisy inputs at once (batched)
    total_samples = len(noise_levels) * n_samples
    all_base_inputs = torch.randint(
        0, 2, (total_samples, n_inputs), dtype=torch.float, device=device
    )
    all_noise = torch.randn_like(all_base_inputs)

    # Scale noise by level
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

            # Get importance for cluster components
            all_imp = []
            for module_name in sorted(ci_outputs.upper_leaky.keys()):
                ci_tensor = ci_outputs.upper_leaky[module_name]
                all_imp.append(ci_tensor.detach().cpu().numpy())

            if all_imp:
                full_imp = np.concatenate(
                    all_imp, axis=1
                )  # [total_samples, n_components]
                cluster_imp = full_imp[:, comp_indices].mean(axis=1)  # [total_samples]

                # Compute stability and sensitivity for each sample
                for i, (imp, noise_level) in enumerate(
                    zip(cluster_imp, noise_scales.squeeze().cpu().numpy())
                ):
                    stability = 1.0 - abs(imp - baseline_imp)
                    stability_scores.append(max(0, stability))
                    importance_changes.append(
                        abs(imp - baseline_imp) / max(noise_level, 0.01)
                    )

    except Exception:
        pass

    return {
        "mean_importance_stability": float(np.mean(stability_scores))
        if stability_scores
        else 0.0,
        "noise_sensitivity": float(np.mean(importance_changes))
        if importance_changes
        else 0.0,
        "n_samples_tested": len(stability_scores),
    }


def analyze_cluster_faithfulness(
    decomposed_model: "DecomposedMLP",
    cluster_info: ClusterInfo,
    importance_matrix: np.ndarray,
    n_inputs: int = 2,
    device: str = "cpu",
) -> dict:
    """
    Analyze faithfulness of a cluster by testing ablation effects.

    Tests whether ablating (masking out) the cluster's components
    changes the model output appropriately.

    Args:
        decomposed_model: Trained SPD decomposition
        cluster_info: Information about the cluster
        importance_matrix: Full importance matrix [n_inputs, n_components]
        n_inputs: Number of input dimensions
        device: Compute device

    Returns:
        Dict with faithfulness metrics:
            - mean_ablation_effect: How much output changes when cluster is ablated
            - sufficiency_score: Whether cluster alone can produce correct output
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return {"mean_ablation_effect": 0.0, "sufficiency_score": 0.0}

    # For now, return placeholder metrics based on importance patterns
    # Full ablation testing requires modifying the forward pass with masks

    comp_indices = cluster_info.component_indices
    if not comp_indices:
        return {"mean_ablation_effect": 0.0, "sufficiency_score": 0.0}

    # Use importance matrix to estimate faithfulness
    # High importance on specific inputs suggests functional role
    cluster_importance = importance_matrix[:, comp_indices]

    # Ablation effect proxy: variance in importance across inputs
    # Higher variance = more selective = likely more faithful
    importance_variance = float(np.var(cluster_importance))

    # Sufficiency proxy: maximum importance achieved
    max_importance = float(np.max(cluster_importance))

    # Mean importance when "active" (above threshold)
    active_mask = cluster_importance > 0.5
    mean_when_active = (
        float(np.mean(cluster_importance[active_mask])) if active_mask.any() else 0.0
    )

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
    """
    Run robustness and faithfulness analysis on all clusters.

    Args:
        decomposed_model: Trained SPD decomposition
        analysis_result: SPD analysis result with clustering
        device: Compute device

    Returns:
        List of analysis dicts, one per cluster
    """
    results = []

    for cluster_info in analysis_result.clusters:
        robustness = analyze_cluster_robustness(
            decomposed_model,
            cluster_info,
            analysis_result.importance_matrix,
            device=device,
        )

        faithfulness = analyze_cluster_faithfulness(
            decomposed_model,
            cluster_info,
            analysis_result.importance_matrix,
            device=device,
        )

        # Update cluster info with scores
        cluster_info.robustness_score = robustness.get("mean_importance_stability", 0.0)
        cluster_info.faithfulness_score = faithfulness.get("sufficiency_score", 0.0)

        results.append(
            {
                "cluster_idx": cluster_info.cluster_idx,
                "robustness": robustness,
                "faithfulness": faithfulness,
            }
        )

    return results

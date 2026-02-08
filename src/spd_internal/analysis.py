"""
SPD Analysis Module - Main entry points for SPD decomposition analysis.

This module provides the primary entry points for running SPD analysis:
1. run_spd_analysis - Run complete SPD analysis without visualization
2. analyze_and_visualize_spd - Run analysis and generate all visualizations

The output is organized under trial_id/spd/ with the following structure:
    spd/
        config.json              - SPD configuration and parameters
        decomposed_model.pt      - The trained decomposed model
        clustering/
            assignments.json     - Component -> cluster mapping
            coactivation.npy     - Coactivation matrix
            importance_matrix.npy - Full importance matrix
        visualizations/
            importance_heatmap.png
            coactivation_matrix.png
            components_as_circuits.png
            uv_matrices.png
            summary.png
        clusters/
            {cluster_idx}/
                analysis.json    - Cluster statistics
                robustness.json  - Robustness metrics
                faithfulness.json - Faithfulness metrics
                circuit.png      - Circuit diagram for this cluster
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .cluster_analysis import analyze_all_clusters, map_clusters_to_functions
from .clustering import cluster_components_hierarchical, detect_dead_components
from .importance import compute_coactivation_matrix, compute_importance_matrix
from .schemas import ClusterInfo, SPDAnalysisResult
from .validation import compute_validation_metrics
from .visualization import (
    visualize_ci_histograms,
    visualize_coactivation_matrix,
    visualize_components_as_circuits,
    visualize_importance_heatmap,
    visualize_l0_sparsity,
    visualize_mean_ci_per_component,
    visualize_summary,
    visualize_uv_matrices,
)

if TYPE_CHECKING:
    from ..common.neural_model import MLP, DecomposedMLP


def run_spd_analysis(
    decomposed_model: "DecomposedMLP",
    target_model: "MLP" = None,
    n_inputs: int = 2,
    gate_names: list[str] = None,
    n_clusters: int = None,
    device: str = "cpu",
) -> SPDAnalysisResult:
    """
    Run complete SPD analysis: clustering, function mapping, and metrics.

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

    # Compute validation metrics (MMCS, ML2R)
    validation_metrics = compute_validation_metrics(decomposed_model)
    result.mmcs = validation_metrics["mmcs"]
    result.ml2r = validation_metrics["ml2r"]
    result.faithfulness_loss = validation_metrics["faithfulness_loss"]

    # Detect dead components
    alive_labels, dead_labels = detect_dead_components(decomposed_model)
    result.n_alive_components = len(alive_labels)
    result.n_dead_components = len(dead_labels)
    result.dead_component_labels = dead_labels

    # Compute importance matrix
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

    # Compute coactivation matrix
    result.coactivation_matrix = compute_coactivation_matrix(importance_matrix)

    # Cluster components
    result.cluster_assignments = cluster_components_hierarchical(
        result.coactivation_matrix,
        n_clusters=n_clusters,
    )
    result.n_clusters = (
        max(result.cluster_assignments) + 1 if result.cluster_assignments else 0
    )

    # Map clusters to functions
    cluster_functions = map_clusters_to_functions(
        importance_matrix,
        result.cluster_assignments,
        n_inputs,
        gate_names,
    )

    # Build cluster info
    for cluster_idx in range(result.n_clusters):
        component_indices = [
            i for i, c in enumerate(result.cluster_assignments) if c == cluster_idx
        ]
        cluster_labels = [component_labels[i] for i in component_indices]

        # Mean importance over all inputs for this cluster
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


def analyze_and_visualize_spd(
    decomposed_model: "DecomposedMLP",
    target_model: "MLP",
    output_dir: str | Path,
    gate_names: list[str] = None,
    n_inputs: int = 2,
    device: str = "cpu",
) -> SPDAnalysisResult:
    """
    Run complete SPD analysis and generate all visualizations.

    Args:
        decomposed_model: Trained SPD decomposition
        target_model: Original MLP model
        output_dir: Directory to save all outputs
        gate_names: Names of gates in the model
        n_inputs: Number of input bits
        device: Compute device

    Returns:
        SPDAnalysisResult with all data and visualization paths
    """
    output_dir = Path(output_dir)

    # Create directory structure
    (output_dir / "clustering").mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (output_dir / "clusters").mkdir(parents=True, exist_ok=True)

    # Run analysis
    result = run_spd_analysis(
        decomposed_model=decomposed_model,
        target_model=target_model,
        n_inputs=n_inputs,
        gate_names=gate_names,
        device=device,
    )

    if result.n_components == 0:
        return result

    # Save validation metrics
    validation_data = {
        "mmcs": result.mmcs,
        "ml2r": result.ml2r,
        "faithfulness_loss": result.faithfulness_loss,
        "n_alive_components": result.n_alive_components,
        "n_dead_components": result.n_dead_components,
        "dead_component_labels": result.dead_component_labels,
    }
    with open(output_dir / "validation.json", "w") as f:
        json.dump(validation_data, f, indent=2)

    # Save clustering data
    if result.importance_matrix is not None:
        np.save(
            output_dir / "clustering" / "importance_matrix.npy",
            result.importance_matrix,
        )
    if result.coactivation_matrix is not None:
        np.save(
            output_dir / "clustering" / "coactivation_matrix.npy",
            result.coactivation_matrix,
        )

    # Save cluster assignments
    assignments_data = {
        "cluster_assignments": result.cluster_assignments,
        "component_labels": result.component_labels,
        "n_clusters": result.n_clusters,
        "clusters": [asdict(c) for c in result.clusters],
    }
    with open(output_dir / "clustering" / "assignments.json", "w") as f:
        json.dump(assignments_data, f, indent=2)

    # Generate visualizations
    layer_sizes = target_model.layer_sizes if target_model else [n_inputs, 3, 1]

    viz_paths = {}

    # Importance heatmap
    path = visualize_importance_heatmap(
        result.importance_matrix,
        result.component_labels,
        str(output_dir / "visualizations" / "importance_heatmap.png"),
        n_inputs,
    )
    if path:
        viz_paths["importance_heatmap"] = "visualizations/importance_heatmap.png"

    # Coactivation matrix
    path = visualize_coactivation_matrix(
        result.coactivation_matrix,
        result.cluster_assignments,
        result.component_labels,
        str(output_dir / "visualizations" / "coactivation_matrix.png"),
    )
    if path:
        viz_paths["coactivation_matrix"] = "visualizations/coactivation_matrix.png"

    # Components as circuits (split into 3 files by category)
    circuit_paths = visualize_components_as_circuits(
        result,
        layer_sizes,
        str(output_dir / "visualizations" / "circuits"),
        decomposed_model=decomposed_model,
    )
    for category, path in circuit_paths.items():
        viz_paths[f"circuits_{category}"] = f"visualizations/circuits_{category}.png"

    # UV matrices
    path = visualize_uv_matrices(
        decomposed_model,
        str(output_dir / "visualizations" / "uv_matrices.png"),
    )
    if path:
        viz_paths["uv_matrices"] = "visualizations/uv_matrices.png"

    # SPD paper-style visualizations
    # CI histograms (per layer)
    path = visualize_ci_histograms(
        result.importance_matrix,
        result.component_labels,
        str(output_dir / "visualizations" / "ci_histograms.png"),
    )
    if path:
        viz_paths["ci_histograms"] = "visualizations/ci_histograms.png"

    # Mean CI per component
    path = visualize_mean_ci_per_component(
        result.importance_matrix,
        result.component_labels,
        str(output_dir / "visualizations" / "mean_ci_per_component.png"),
    )
    if path:
        viz_paths["mean_ci_per_component"] = "visualizations/mean_ci_per_component.png"

    # L0 sparsity
    path = visualize_l0_sparsity(
        result.importance_matrix,
        result.component_labels,
        str(output_dir / "visualizations" / "l0_sparsity.png"),
    )
    if path:
        viz_paths["l0_sparsity"] = "visualizations/l0_sparsity.png"

    # Summary
    path = visualize_summary(
        result,
        str(output_dir / "visualizations" / "summary.png"),
    )
    if path:
        viz_paths["summary"] = "visualizations/summary.png"

    result.visualization_paths = viz_paths

    # Run cluster-level robustness and faithfulness analysis
    cluster_analyses = analyze_all_clusters(
        decomposed_model=decomposed_model,
        analysis_result=result,
        device=device,
    )

    # Create per-cluster directories and save analysis files
    def _to_serializable(obj):
        """Convert numpy types to Python native types for JSON."""
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    for cluster_info, cluster_analysis in zip(result.clusters, cluster_analyses):
        cluster_dir = output_dir / "clusters" / str(cluster_info.cluster_idx)
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Save cluster info
        with open(cluster_dir / "analysis.json", "w") as f:
            json.dump(_to_serializable(asdict(cluster_info)), f, indent=2)

        # Save robustness metrics
        with open(cluster_dir / "robustness.json", "w") as f:
            json.dump(_to_serializable(cluster_analysis["robustness"]), f, indent=2)

        # Save faithfulness metrics
        with open(cluster_dir / "faithfulness.json", "w") as f:
            json.dump(_to_serializable(cluster_analysis["faithfulness"]), f, indent=2)

    return result

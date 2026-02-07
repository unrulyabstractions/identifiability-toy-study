"""
SPD Internal Module - Internal SPD analysis and subcircuit estimation.

This module contains our internal SPD analysis code, separate from the
external goodfire-ai SPD submodule (src/spd/).

Re-exports all public functions for convenience.
"""

# Analysis functions
from .analysis import (
    ClusterInfo,
    SPDAnalysisResult,
    analyze_all_clusters,
    analyze_and_visualize_spd,
    analyze_cluster_faithfulness,
    analyze_cluster_robustness,
    cluster_components_hierarchical,
    compute_coactivation_matrix,
    compute_importance_matrix,
    compute_validation_metrics,
    detect_dead_components,
    map_clusters_to_functions,
    run_spd_analysis,
    visualize_ci_histograms,
    visualize_coactivation_matrix,
    visualize_components_as_circuits,
    visualize_importance_heatmap,
    visualize_l0_sparsity,
    visualize_mean_ci_per_component,
    visualize_summary,
    visualize_uv_matrices,
)

# Subcircuit estimation functions
from .subcircuits import (
    SPDSubcircuitEstimate,
    estimate_spd_subcircuits,
    load_spd_estimate,
    save_spd_estimate,
    spd_clusters_to_circuits,
)

__all__ = [
    # Analysis classes
    "ClusterInfo",
    "SPDAnalysisResult",
    # Analysis functions
    "analyze_all_clusters",
    "analyze_and_visualize_spd",
    "analyze_cluster_faithfulness",
    "analyze_cluster_robustness",
    "cluster_components_hierarchical",
    "compute_coactivation_matrix",
    "compute_importance_matrix",
    "compute_validation_metrics",
    "detect_dead_components",
    "map_clusters_to_functions",
    "run_spd_analysis",
    # Visualization functions
    "visualize_ci_histograms",
    "visualize_coactivation_matrix",
    "visualize_components_as_circuits",
    "visualize_importance_heatmap",
    "visualize_l0_sparsity",
    "visualize_mean_ci_per_component",
    "visualize_summary",
    "visualize_uv_matrices",
    # Subcircuit classes
    "SPDSubcircuitEstimate",
    # Subcircuit functions
    "estimate_spd_subcircuits",
    "load_spd_estimate",
    "save_spd_estimate",
    "spd_clusters_to_circuits",
]

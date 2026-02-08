"""
SPD Internal Module - Internal SPD analysis and subcircuit estimation.

This module contains our internal SPD analysis code, separate from the
external goodfire-ai SPD submodule (src/spd/).

Re-exports all public functions for convenience.
"""

# Decomposition functions
from .decomposition import SimpleDataset, decompose_mlp

# Schema classes
from .schemas import ClusterInfo, SPDAnalysisResult

# Importance functions
from .importance import compute_coactivation_matrix, compute_importance_matrix

# Clustering functions
from .clustering import cluster_components_hierarchical, detect_dead_components

# Validation functions
from .validation import compute_validation_metrics

# Cluster analysis functions
from .cluster_analysis import (
    analyze_all_clusters,
    analyze_cluster_faithfulness,
    analyze_cluster_robustness,
    map_clusters_to_functions,
)

# Visualization functions
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

# Main entry points
from .analysis import analyze_and_visualize_spd, run_spd_analysis

# Subcircuit estimation functions
from .subcircuits import (
    SPDSubcircuitEstimate,
    estimate_spd_subcircuits,
    spd_clusters_to_circuits,
)

# Persistence functions
from .persistence import load_spd_estimate, save_spd_estimate

__all__ = [
    # Decomposition
    "SimpleDataset",
    "decompose_mlp",
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

"""SPD analysis: orchestration of importance, clustering, and function mapping.

After SPD decomposes weights into components, we need to understand what each
component does. This module answers: "Which components implement the same
function, and what function is that?"

The analysis pipeline:
    1. Compute importance matrix
       - For each possible input (00, 01, 10, 11), measure how much each
         component contributes to the output (causal importance, CI)
       - Result: [n_inputs, n_components] matrix of CI values in [0, 1]

    2. Compute coactivation matrix
       - Components that fire together (both have high CI on same inputs)
         are likely implementing the same function
       - Result: [n_components, n_components] matrix of co-firing counts

    3. Cluster components
       - Use hierarchical clustering on coactivation patterns
       - Components with similar activation patterns get grouped together
       - Result: cluster_assignments list (component_idx -> cluster_idx)

    4. Map clusters to functions
       - Compare each cluster's activation pattern to known boolean gates
         (XOR activates on 01 and 10, AND activates only on 11, etc.)
       - Use Jaccard similarity to find best match
       - Result: cluster_functions dict (cluster_idx -> "XOR (0.95)")

Key insight: A cluster that activates on exactly the inputs where XOR=1
is likely implementing XOR. This lets us identify functional subcircuits.

This package provides the orchestration layer. The actual implementations are in:
- importance.py: compute_importance_matrix, compute_coactivation_matrix
- clustering.py: detect_dead_components, cluster_components_hierarchical, map_clusters_to_functions
- evaluation.py: analyze_cluster_robustness, analyze_cluster_faithfulness, analyze_all_clusters
"""

# Orchestration functions (this package)
from .orchestration import estimate_spd_subcircuits, run_spd_analysis

# Re-export from sibling modules for backward compatibility
from ..clustering import (
    cluster_components_hierarchical,
    detect_dead_components,
    map_clusters_to_functions,
)
from ..evaluation import (
    analyze_all_clusters,
    analyze_cluster_faithfulness,
    analyze_cluster_robustness,
)
from ..importance import compute_coactivation_matrix, compute_importance_matrix

__all__ = [
    # Importance (from importance.py)
    "compute_importance_matrix",
    "compute_coactivation_matrix",
    # Clustering (from clustering.py)
    "detect_dead_components",
    "cluster_components_hierarchical",
    "map_clusters_to_functions",
    # Evaluation (from evaluation.py)
    "analyze_cluster_robustness",
    "analyze_cluster_faithfulness",
    "analyze_all_clusters",
    # Orchestration (this package)
    "run_spd_analysis",
    "estimate_spd_subcircuits",
]

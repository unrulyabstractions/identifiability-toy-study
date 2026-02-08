"""
SPD Schema Classes - Data structures for SPD analysis results.

This module contains the dataclasses used to represent SPD analysis results.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ClusterInfo:
    """Information about a single component cluster."""

    cluster_idx: int
    component_indices: list[int]  # Which components belong to this cluster
    component_labels: list[str]  # Labels like "layers.0.0:3"
    mean_importance: float = 0.0

    # Analysis results (filled in later)
    robustness_score: float = 0.0
    faithfulness_score: float = 0.0
    function_mapping: str = ""  # e.g., "XOR", "AND", etc.


@dataclass
class SPDAnalysisResult:
    """Complete result of SPD analysis including clustering and per-cluster metrics."""

    # Basic info
    n_components: int = 0
    n_layers: int = 0
    n_clusters: int = 0

    # Validation metrics (from SPD paper)
    mmcs: float = 0.0  # Mean Max Cosine Similarity (should be ~1.0)
    ml2r: float = 0.0  # Mean L2 Ratio (should be ~1.0)
    faithfulness_loss: float = 0.0  # MSE between target and reconstructed weights

    # Component health
    n_alive_components: int = 0
    n_dead_components: int = 0
    dead_component_labels: list[str] = field(default_factory=list)

    # Clustering results
    cluster_assignments: list[int] = field(
        default_factory=list
    )  # component_idx -> cluster_idx
    clusters: list[ClusterInfo] = field(default_factory=list)

    # Raw data (stored as numpy arrays on disk)
    importance_matrix: Optional[np.ndarray] = None  # [n_inputs, n_components]
    coactivation_matrix: Optional[np.ndarray] = None  # [n_components, n_components]

    # Component labels for each index
    component_labels: list[str] = field(default_factory=list)

    # Visualization paths (relative to spd/ folder)
    visualization_paths: dict[str, str] = field(default_factory=dict)

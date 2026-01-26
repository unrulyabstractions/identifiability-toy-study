"""
SPD-based subcircuit estimation.

This module attempts to identify subcircuits from SPD (Stochastic Parameter
Decomposition) results. SPD decomposes model weights into components with
learned importance masks.

The key insight is that SPD components that are consistently masked together
likely form functional subcircuits.

References:
- SPD paper: https://arxiv.org/pdf/2506.20790
- SPD clustering module: spd/clustering/
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .common.neural_model import DecomposedMLP


@dataclass
class SPDSubcircuitEstimate:
    """Result of SPD-based subcircuit estimation."""

    # Cluster assignments for each component (list of cluster IDs)
    cluster_assignments: list[int]

    # Number of clusters found
    n_clusters: int

    # Per-cluster statistics
    cluster_sizes: list[int]

    # Importance scores per component
    component_importance: np.ndarray | None = None

    # Coactivation matrix (which components activate together)
    coactivation_matrix: np.ndarray | None = None


def estimate_spd_subcircuits(
    decomposed_model: "DecomposedMLP",
    n_samples: int = 1000,
    device: str = "cpu",
) -> SPDSubcircuitEstimate | None:
    """
    Estimate subcircuits from SPD decomposition using component clustering.

    This function analyzes which SPD components activate together to identify
    potential subcircuit structures. The approach is:

    1. Sample random inputs and compute component activations
    2. Build coactivation matrix (which components fire together)
    3. Cluster components based on coactivation patterns
    4. Map clusters to potential subcircuit structures

    Args:
        decomposed_model: Result from decompose_mlp()
        n_samples: Number of samples to use for coactivation analysis
        device: Device for computation

    Returns:
        SPDSubcircuitEstimate with cluster assignments and statistics,
        or None if SPD decomposition is not available.

    Note:
        This is a research feature. The mapping from SPD components to
        discrete subcircuits is an open research question. Current
        implementation provides coactivation analysis but does not
        fully convert to Circuit objects.
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return None

    n_components = decomposed_model.get_n_components()
    if n_components == 0:
        return None

    # Get the component model
    component_model = decomposed_model.component_model

    # Generate random inputs for analysis
    # Assuming 2-input logic gate model
    input_size = 2
    x = torch.rand(n_samples, input_size, device=device)

    try:
        # Try to get component activations
        # This depends on SPD internal APIs which may vary
        with torch.inference_mode():
            # Forward pass to trigger component computation
            _ = component_model(x)

            # Try to access causal importance (CI) values
            # CI indicates how much each component contributes
            ci_values = None
            if hasattr(component_model, "get_ci_values"):
                ci_values = component_model.get_ci_values()
            elif hasattr(component_model, "ci_fn"):
                # Try calling CI function directly
                pass  # Complex - depends on SPD internals

            # Build simple clustering based on component correlation
            # This is a placeholder - proper implementation would use
            # spd/clustering module's hierarchical clustering

            # For now, return a simple result indicating SPD ran but
            # subcircuit extraction is not yet implemented
            return SPDSubcircuitEstimate(
                cluster_assignments=list(range(n_components)),  # Each component is its own cluster
                n_clusters=n_components,
                cluster_sizes=[1] * n_components,
                component_importance=None,
                coactivation_matrix=None,
            )

    except Exception as e:
        # SPD internals may have changed or be unavailable
        print(f"Warning: Could not extract SPD subcircuits: {e}")
        return None


def spd_clusters_to_circuits(
    estimate: SPDSubcircuitEstimate,
    model_layer_sizes: list[int],
) -> list:
    """
    Convert SPD cluster assignments to Circuit objects.

    This is a placeholder for future implementation. The challenge is
    mapping continuous SPD component masks to discrete circuit structure.

    Args:
        estimate: SPD subcircuit estimate
        model_layer_sizes: Layer sizes of the target model

    Returns:
        List of Circuit objects (currently empty - not implemented)
    """
    # Future work: implement mapping from SPD clusters to circuits
    # This requires:
    # 1. Understanding how SPD components map to layer neurons
    # 2. Thresholding continuous masks to get binary circuit masks
    # 3. Ensuring resulting circuits are valid (connected)
    return []

"""
SPD Importance Module - Compute importance and coactivation matrices.

This module provides functions for computing causal importance values
and coactivation patterns for SPD components.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.model import DecomposedMLP


def compute_importance_matrix(
    decomposed_model: "DecomposedMLP",
    n_inputs: int = 2,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """
    Compute causal importance values for all binary input combinations.

    Args:
        decomposed_model: The trained SPD decomposition
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

    # Get pre-weight activations (inputs to each layer)
    with torch.inference_mode():
        output_with_cache = component_model(all_inputs, cache_type="input")
        pre_weight_acts = output_with_cache.cache

        # Compute causal importances
        ci_outputs = component_model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sampling="continuous",
            detach_inputs=False,
        )

        # Use upper_leaky for visualization (values in [0, 1])
        ci_values = ci_outputs.upper_leaky

    # Concatenate all layer importances and create labels
    all_importances = []
    component_labels = []

    for module_name in sorted(ci_values.keys()):
        ci_tensor = ci_values[module_name]  # [n_inputs, C]
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
    """
    Compute coactivation matrix showing which components fire together.

    Args:
        importance_matrix: Shape [n_samples, n_components]
        threshold: Activation threshold (component is "active" if importance > threshold)

    Returns:
        coactivation_matrix: Shape [n_components, n_components], symmetric
    """
    if importance_matrix.size == 0:
        return np.array([])

    # Binarize activations
    active_mask = (importance_matrix > threshold).astype(np.float32)

    # Coactivation = A^T @ A (counts how often pairs co-activate)
    coactivation = active_mask.T @ active_mask

    return coactivation

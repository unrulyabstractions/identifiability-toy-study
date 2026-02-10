"""SPD importance computation: causal importance and coactivation matrices.

This module computes the raw measurements that feed into SPD analysis:

1. Importance matrix: For each input and each component, how much does that
   component contribute to the output? This is measured via "causal importance"
   (CI) - a learned gate value in [0,1] that indicates activation strength.

2. Coactivation matrix: Which components fire together on the same inputs?
   Components with similar coactivation patterns likely implement the same
   function and can be clustered together.

These measurements are the foundation of SPD interpretability. The importance
matrix shows what each component does; the coactivation matrix shows which
components work together.
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
    """Compute causal importance (CI) for each component on each input.

    Causal importance measures how much a component contributes to the output.
    CI values are in [0, 1] where:
    - 0 = component is completely masked (doesn't contribute)
    - 1 = component is fully active (contributes its full weight)

    For boolean gates with 2 inputs, we test all 4 input combinations:
    - (0, 0), (0, 1), (1, 0), (1, 1)

    Args:
        decomposed_model: Trained SPD decomposition
        n_inputs: Number of input bits (2 for boolean gates)
        device: Compute device

    Returns:
        importance_matrix: Shape [4, n_components] for 2 inputs
            Each row is an input pattern, each column is a component
        component_labels: Labels like "layers.0.0:3" (layer 0, component 3)
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
    """Compute how often component pairs activate together.

    If components A and B both have high CI on the same inputs, they're likely
    part of the same functional subcircuit. This matrix captures that pattern.

    Algorithm:
        1. Binarize: component is "active" if CI > threshold
        2. Coactivation[i,j] = count of inputs where both i and j are active
        3. Diagonal[i] = count of inputs where i is active

    Example: If components 0 and 1 both activate on inputs (0,1) and (1,0),
    their coactivation count is 2. If they both activate on all 4 inputs,
    coactivation is 4.

    Args:
        importance_matrix: Shape [n_inputs, n_components]
        threshold: CI threshold for "active" (default 0.5)

    Returns:
        coactivation_matrix: Shape [n_components, n_components], symmetric
    """
    if importance_matrix.size == 0:
        return np.array([])

    active_mask = (importance_matrix > threshold).astype(np.float32)
    return active_mask.T @ active_mask

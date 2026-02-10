"""Activation statistics computation.

Computes summary statistics for layer activations:
- Mean, standard deviation, min, max
- Sparsity (fraction of near-zero activations)
- Correlation between neurons
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

from .types import (
    ActivationStatistics,
    LayerStatistics,
    ActivationCorrelation,
)

if TYPE_CHECKING:
    from src.model import MLP


def compute_activation_statistics(
    model: "MLP",
    x: torch.Tensor,
    sparsity_threshold: float = 0.01,
    device: str = "cpu",
) -> LayerStatistics:
    """Compute activation statistics for all layers.

    Args:
        model: The MLP model to analyze
        x: Input samples [n_samples, n_inputs]
        sparsity_threshold: Threshold below which activations are considered "sparse"
        device: Device to run on

    Returns:
        LayerStatistics with per-layer statistics
    """
    model = model.to(device)
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)

    per_layer = {}
    n_layers = len(activations)

    for layer_idx, acts in enumerate(activations):
        acts_np = acts.cpu().numpy()  # [n_samples, n_neurons]

        # Compute statistics
        mean = acts_np.mean(axis=0)
        std = acts_np.std(axis=0)
        min_val = acts_np.min(axis=0)
        max_val = acts_np.max(axis=0)

        # Sparsity: fraction of samples where activation is near zero
        sparsity = (np.abs(acts_np) < sparsity_threshold).mean(axis=0)

        per_layer[layer_idx] = ActivationStatistics(
            layer_idx=layer_idx,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            sparsity=sparsity,
            n_samples=len(x),
        )

    return LayerStatistics(
        per_layer=per_layer,
        n_layers=n_layers,
        n_samples=len(x),
    )


def compute_activation_correlation(
    model: "MLP",
    x: torch.Tensor,
    layer_idx: int,
    top_k: int = 5,
    device: str = "cpu",
) -> ActivationCorrelation:
    """Compute correlation between neurons in a layer.

    Args:
        model: The MLP model to analyze
        x: Input samples [n_samples, n_inputs]
        layer_idx: Which layer to analyze
        top_k: Number of top correlations to return
        device: Device to run on

    Returns:
        ActivationCorrelation with correlation matrix and top pairs
    """
    model = model.to(device)
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)

    if layer_idx >= len(activations):
        raise ValueError(f"Layer {layer_idx} does not exist (model has {len(activations)} layers)")

    acts = activations[layer_idx].cpu().numpy()  # [n_samples, n_neurons]
    n_neurons = acts.shape[1]

    if n_neurons < 2:
        return ActivationCorrelation(
            layer_idx=layer_idx,
            correlation_matrix=np.array([[1.0]]),
            mean_abs_correlation=0.0,
        )

    # Compute correlation matrix
    corr_matrix = np.corrcoef(acts.T)  # [n_neurons, n_neurons]

    # Handle NaN (can occur if a neuron has zero variance)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Find top positive and negative correlations
    pairs = []
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            pairs.append((i, j, corr_matrix[i, j]))

    # Sort by correlation
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    top_positive = pairs_sorted[:top_k]
    top_negative = pairs_sorted[-top_k:][::-1]

    # Mean absolute correlation (excluding diagonal)
    mask = ~np.eye(n_neurons, dtype=bool)
    mean_abs_corr = np.abs(corr_matrix[mask]).mean()

    return ActivationCorrelation(
        layer_idx=layer_idx,
        correlation_matrix=corr_matrix,
        top_positive_pairs=top_positive,
        top_negative_pairs=top_negative,
        mean_abs_correlation=mean_abs_corr,
    )


def compute_all_correlations(
    model: "MLP",
    x: torch.Tensor,
    top_k: int = 5,
    device: str = "cpu",
) -> dict[int, ActivationCorrelation]:
    """Compute correlations for all layers.

    Args:
        model: The MLP model to analyze
        x: Input samples
        top_k: Number of top correlations per layer
        device: Device to run on

    Returns:
        Dictionary mapping layer_idx to ActivationCorrelation
    """
    model = model.to(device)
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)

    correlations = {}
    for layer_idx in range(len(activations)):
        correlations[layer_idx] = compute_activation_correlation(
            model, x, layer_idx, top_k=top_k, device=device
        )

    return correlations


def compute_inter_layer_correlation(
    model: "MLP",
    x: torch.Tensor,
    layer_a: int,
    layer_b: int,
    device: str = "cpu",
) -> np.ndarray:
    """Compute correlation between neurons in two different layers.

    Args:
        model: The MLP model to analyze
        x: Input samples
        layer_a: First layer index
        layer_b: Second layer index
        device: Device to run on

    Returns:
        Correlation matrix [n_neurons_a, n_neurons_b]
    """
    model = model.to(device)
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)

    acts_a = activations[layer_a].cpu().numpy()  # [n_samples, n_a]
    acts_b = activations[layer_b].cpu().numpy()  # [n_samples, n_b]

    n_a = acts_a.shape[1]
    n_b = acts_b.shape[1]

    # Compute cross-correlation
    corr = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            if acts_a[:, i].std() > 0 and acts_b[:, j].std() > 0:
                corr[i, j] = np.corrcoef(acts_a[:, i], acts_b[:, j])[0, 1]

    return np.nan_to_num(corr, nan=0.0)

"""Representational Similarity Analysis (RSA).

RSA compares representations by computing the similarity between
Representational Dissimilarity Matrices (RDMs) at different layers.
"""

from typing import TYPE_CHECKING

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform

from .types import RDMResult, RSAResult

if TYPE_CHECKING:
    from src.model import MLP


def compute_rdm(
    activations: np.ndarray | torch.Tensor,
    distance_metric: str = "correlation",
) -> np.ndarray:
    """Compute Representational Dissimilarity Matrix.

    Args:
        activations: Activation matrix [n_samples, n_features]
        distance_metric: Distance metric (correlation, euclidean, cosine)

    Returns:
        RDM matrix [n_samples, n_samples]
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()

    # Compute pairwise distances
    if distance_metric == "correlation":
        # 1 - correlation gives dissimilarity
        distances = pdist(activations, metric="correlation")
    elif distance_metric == "euclidean":
        distances = pdist(activations, metric="euclidean")
        # Normalize to [0, 1] range
        if distances.max() > 0:
            distances = distances / distances.max()
    elif distance_metric == "cosine":
        distances = pdist(activations, metric="cosine")
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # Convert to square form
    rdm = squareform(distances)
    return rdm


def compute_layer_rdm(
    model: "MLP",
    x: torch.Tensor,
    layer_idx: int,
    distance_metric: str = "correlation",
    device: str = "cpu",
) -> RDMResult:
    """Compute RDM for a specific layer.

    Args:
        model: The MLP model
        x: Input tensor [n_samples, n_inputs]
        layer_idx: Which layer to analyze
        distance_metric: Distance metric for RDM
        device: Device to run on

    Returns:
        RDMResult with the computed RDM
    """
    model = model.to(device)
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)
        layer_acts = activations[layer_idx].cpu().numpy()

    rdm = compute_rdm(layer_acts, distance_metric)

    return RDMResult(
        layer_idx=layer_idx,
        rdm=rdm,
        distance_metric=distance_metric,
        n_samples=x.shape[0],
        n_features=layer_acts.shape[-1],
    )


def compute_all_rdms(
    model: "MLP",
    x: torch.Tensor,
    distance_metric: str = "correlation",
    device: str = "cpu",
) -> dict[int, RDMResult]:
    """Compute RDMs for all layers.

    Args:
        model: The MLP model
        x: Input tensor [n_samples, n_inputs]
        distance_metric: Distance metric for RDM
        device: Device to run on

    Returns:
        Dictionary mapping layer_idx to RDMResult
    """
    model = model.to(device)
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)

    results = {}
    for layer_idx, layer_acts in enumerate(activations):
        layer_acts_np = layer_acts.cpu().numpy()

        # Skip layers with constant activations
        if layer_acts_np.std() < 1e-8:
            continue

        rdm = compute_rdm(layer_acts_np, distance_metric)

        results[layer_idx] = RDMResult(
            layer_idx=layer_idx,
            rdm=rdm,
            distance_metric=distance_metric,
            n_samples=x.shape[0],
            n_features=layer_acts_np.shape[-1],
        )

    return results


def compare_rdms(
    rdm1: RDMResult,
    rdm2: RDMResult,
    method: str = "spearman",
) -> RSAResult:
    """Compare two RDMs using correlation.

    Args:
        rdm1: First RDM result
        rdm2: Second RDM result
        method: Correlation method (spearman, pearson)

    Returns:
        RSAResult with similarity score
    """
    if rdm1.rdm is None or rdm2.rdm is None:
        return RSAResult(
            rdm1_layer=rdm1.layer_idx,
            rdm2_layer=rdm2.layer_idx,
            similarity=0.0,
            method=method,
        )

    # Get upper triangular values
    ut1 = rdm1.get_upper_triangle()
    ut2 = rdm2.get_upper_triangle()

    if len(ut1) != len(ut2):
        raise ValueError("RDMs must have the same number of samples")

    # Compute correlation
    if method == "spearman":
        corr, p_value = spearmanr(ut1, ut2)
    elif method == "pearson":
        corr, p_value = pearsonr(ut1, ut2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return RSAResult(
        rdm1_layer=rdm1.layer_idx,
        rdm2_layer=rdm2.layer_idx,
        similarity=float(corr) if not np.isnan(corr) else 0.0,
        p_value=float(p_value) if not np.isnan(p_value) else None,
        method=method,
    )


def compute_rsa_matrix(
    rdm_results: dict[int, RDMResult],
    method: str = "spearman",
) -> np.ndarray:
    """Compute RSA similarity matrix for all layer pairs.

    Args:
        rdm_results: Dictionary of RDM results
        method: Correlation method

    Returns:
        RSA similarity matrix [n_layers, n_layers]
    """
    layer_indices = sorted(rdm_results.keys())
    n_layers = len(layer_indices)

    rsa_matrix = np.zeros((n_layers, n_layers))

    for i, layer1 in enumerate(layer_indices):
        for j, layer2 in enumerate(layer_indices):
            result = compare_rdms(rdm_results[layer1], rdm_results[layer2], method)
            rsa_matrix[i, j] = result.similarity

    return rsa_matrix

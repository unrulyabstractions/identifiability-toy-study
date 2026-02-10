"""Centered Kernel Alignment (CKA).

CKA is a similarity metric between neural network representations that is
invariant to orthogonal transformations and isotropic scaling.
"""

from typing import TYPE_CHECKING

import torch
import numpy as np

from .types import CKAResult

if TYPE_CHECKING:
    from src.model import MLP


def centering(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix.

    Args:
        K: Kernel matrix [n, n]

    Returns:
        Centered kernel matrix
    """
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n
    return H @ K @ H


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """Compute linear kernel (dot product).

    Args:
        X: Feature matrix [n_samples, n_features]

    Returns:
        Kernel matrix [n_samples, n_samples]
    """
    return X @ X.T


def rbf_kernel(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Compute RBF (Gaussian) kernel.

    Args:
        X: Feature matrix [n_samples, n_features]
        sigma: RBF bandwidth (auto-computed if None)

    Returns:
        Kernel matrix [n_samples, n_samples]
    """
    # Compute squared Euclidean distances
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
    sq_dists = sq_norms + sq_norms.T - 2 * X @ X.T

    if sigma is None:
        # Use median heuristic
        sigma = np.sqrt(np.median(sq_dists[sq_dists > 0]))
        if sigma == 0:
            sigma = 1.0

    return np.exp(-sq_dists / (2 * sigma ** 2))


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Compute Hilbert-Schmidt Independence Criterion.

    Args:
        K: First centered kernel matrix
        L: Second centered kernel matrix

    Returns:
        HSIC value
    """
    n = K.shape[0]
    return np.trace(K @ L) / ((n - 1) ** 2)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two activation matrices.

    Args:
        X: First activation matrix [n_samples, n_features_x]
        Y: Second activation matrix [n_samples, n_features_y]

    Returns:
        CKA similarity value in [0, 1]
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    # Compute linear kernels
    K = linear_kernel(X)
    L = linear_kernel(Y)

    # Center the kernels
    K = centering(K)
    L = centering(L)

    # Compute HSIC values
    hsic_xy = hsic(K, L)
    hsic_xx = hsic(K, K)
    hsic_yy = hsic(L, L)

    # Compute CKA
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0

    return hsic_xy / denom


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """Compute RBF kernel CKA between two activation matrices.

    Args:
        X: First activation matrix [n_samples, n_features_x]
        Y: Second activation matrix [n_samples, n_features_y]
        sigma: RBF bandwidth (auto-computed if None)

    Returns:
        CKA similarity value in [0, 1]
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    # Compute RBF kernels
    K = rbf_kernel(X, sigma)
    L = rbf_kernel(Y, sigma)

    # Center the kernels
    K = centering(K)
    L = centering(L)

    # Compute HSIC values
    hsic_xy = hsic(K, L)
    hsic_xx = hsic(K, K)
    hsic_yy = hsic(L, L)

    # Compute CKA
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0

    return hsic_xy / denom


def compare_layers_cka(
    model: "MLP",
    x: torch.Tensor,
    kernel: str = "linear",
    device: str = "cpu",
) -> np.ndarray:
    """Compute CKA similarity matrix for all layer pairs.

    Args:
        model: The MLP model
        x: Input tensor [n_samples, n_inputs]
        kernel: Kernel type (linear, rbf)
        device: Device to run on

    Returns:
        CKA similarity matrix [n_layers, n_layers]
    """
    model = model.to(device)
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)

    n_layers = len(activations)
    cka_matrix = np.zeros((n_layers, n_layers))

    # Convert to numpy
    acts_np = [a.cpu().numpy() for a in activations]

    cka_fn = linear_cka if kernel == "linear" else rbf_cka

    for i in range(n_layers):
        for j in range(n_layers):
            cka_matrix[i, j] = cka_fn(acts_np[i], acts_np[j])

    return cka_matrix


def compute_cka_result(
    model: "MLP",
    x: torch.Tensor,
    layer1_idx: int,
    layer2_idx: int,
    kernel: str = "linear",
    device: str = "cpu",
) -> CKAResult:
    """Compute CKA between two specific layers.

    Args:
        model: The MLP model
        x: Input tensor [n_samples, n_inputs]
        layer1_idx: First layer index
        layer2_idx: Second layer index
        kernel: Kernel type (linear, rbf)
        device: Device to run on

    Returns:
        CKAResult with detailed metrics
    """
    model = model.to(device)
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)

    X = activations[layer1_idx].cpu().numpy()
    Y = activations[layer2_idx].cpu().numpy()

    # Compute kernels
    if kernel == "linear":
        K = linear_kernel(X)
        L = linear_kernel(Y)
    else:
        K = rbf_kernel(X)
        L = rbf_kernel(Y)

    # Center kernels
    K = centering(K)
    L = centering(L)

    # Compute HSIC values
    hsic_xy = hsic(K, L)
    hsic_xx = hsic(K, K)
    hsic_yy = hsic(L, L)

    # Compute CKA
    denom = np.sqrt(hsic_xx * hsic_yy)
    cka_value = hsic_xy / denom if denom > 1e-10 else 0.0

    return CKAResult(
        layer1_idx=layer1_idx,
        layer2_idx=layer2_idx,
        cka=cka_value,
        hsic_xy=hsic_xy,
        hsic_xx=hsic_xx,
        hsic_yy=hsic_yy,
        kernel=kernel,
    )

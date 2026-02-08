"""Metric calculation functions for model evaluation.

These functions are in a separate module to avoid circular imports
between helpers.py and neural_model.py.
"""

import numpy as np
import torch


@torch.no_grad()
def calculate_match_rate(
    y_pred: torch.Tensor,  # [*] - any shape, will be flattened
    y_gt: torch.Tensor,  # [*] - any shape, will be flattened
) -> torch.Tensor:  # [] scalar
    """Compare two binary (0/1) tensors. Callers must threshold first."""
    y_pred = y_pred.reshape(-1)  # [total]
    y_gt = y_gt.reshape(-1)  # [total]
    return y_pred.eq(y_gt).float().mean()  # [] scalar


@torch.no_grad()
def calculate_match_rate_batched(
    y_preds: torch.Tensor,  # [batch, n_samples, n_gates]
    y_gt: torch.Tensor,  # [n_samples, n_gates]
) -> np.ndarray:  # [batch]
    """y_preds has leading batch dim, y_gt does not."""
    return y_preds.eq(y_gt.unsqueeze(0)).float().mean(dim=(1, 2)).cpu().numpy()


@torch.no_grad()
def logits_to_binary(
    y: torch.Tensor,  # [*] - any shape
) -> torch.Tensor:  # [*] - same shape as input
    """Convert raw logits to binary predictions. Decision boundary at 0."""
    return (y > 0).float()


@torch.no_grad()
def calculate_mse(
    y_target: torch.Tensor,  # [*] - any shape
    y_proxy: torch.Tensor,  # [*] - same shape as y_target
) -> torch.Tensor:  # [] scalar
    """Calculate mean squared error between two tensors."""
    return ((y_target - y_proxy) ** 2).mean()


@torch.no_grad()
def calculate_logit_similarity(
    y_target: torch.Tensor,  # [*] - any shape
    y_proxy: torch.Tensor,  # [*] - same shape as y_target
) -> torch.Tensor:  # [] scalar
    """R²-like similarity: 1.0 = perfect match, 0.0 = predicting the mean."""
    mse = calculate_mse(y_target, y_proxy)  # [] scalar
    var = y_target.var().clamp(min=1e-8)  # [] scalar
    return 1 - mse / var  # [] scalar


@torch.no_grad()
def calculate_logit_similarity_batched(
    y_target: torch.Tensor,  # [n_samples, n_gates]
    y_proxies: torch.Tensor,  # [batch, n_samples, n_gates]
) -> np.ndarray:  # [batch]
    """Batched R²: y_proxies has leading batch dim, y_target does not."""
    mse = ((y_proxies - y_target.unsqueeze(0)) ** 2).mean(dim=(1, 2))  # [batch]
    var = y_target.var().clamp(min=1e-8)  # [] scalar
    return (1 - mse / var).detach().cpu().numpy()  # [batch]


@torch.no_grad()
def calculate_best_match_rate(
    y_target: torch.Tensor,  # [*] - any shape
    y_proxy: torch.Tensor,  # [*] - same shape as y_target
) -> torch.Tensor:  # [] scalar
    """Calculate match rate from raw logits."""
    return calculate_match_rate(logits_to_binary(y_target), logits_to_binary(y_proxy))

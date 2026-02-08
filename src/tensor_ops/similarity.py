"""Metric calculation functions for model evaluation.

These functions are in a separate module to avoid circular imports
between helpers.py and neural_model.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_loss(
    model: nn.Module,
    logits: torch.Tensor,
    target: torch.Tensor,
    l1_lambda: float = 1e-3,
    mse_lambda: float = 0.1,
):
    loss = F.binary_cross_entropy_with_logits(logits, target)
    # if l1_lambda > 0:
    #     l1_loss = sum(p.abs().sum() for p in model.parameters())
    #     loss = loss + l1_lambda * l1_loss
    # if mse_lambda > 0:
    #     mse_loss = F.mse_loss(torch.sigmoid(logits), target)
    #     loss = loss + mse_lambda * mse_loss
    return loss


@torch.no_grad()
def logits_to_binary(
    y: torch.Tensor,  # [*] - any shape
) -> torch.Tensor:  # [*] - same shape as input
    """Convert raw logits to binary predictions. Decision boundary at 0."""
    return (y > 0).float()


@torch.no_grad()
def calculate_match_rate(
    y_target: torch.Tensor,  # [*] - any shape, will be flattened
    y_proxy: torch.Tensor,  # [*] - any shape, will be flattened
) -> torch.Tensor:  # [] scalar
    """Compare two binary (0/1) tensors. Callers must threshold first."""
    y_target = y_target.reshape(-1)  # [total]
    y_proxy = y_proxy.reshape(-1)  # [total]
    return y_target.eq(y_proxy).float().mean()  # [] scalar


@torch.no_grad()
def calculate_best_match_rate(
    y_target: torch.Tensor,  # [*] - any shape (logits)
    y_proxy: torch.Tensor,  # [*] - same shape as y_target (logits)
) -> torch.Tensor:  # [] scalar
    """Calculate match rate from raw logits. Uses logits_to_binary internally."""
    return calculate_match_rate(logits_to_binary(y_target), logits_to_binary(y_proxy))


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


#########################
#### BATCH VERSIONS #####
#########################


@torch.no_grad()
def calculate_match_rate_batched(
    y_target: torch.Tensor,  # [n_samples, n_gates]
    y_proxies: torch.Tensor,  # [batch, n_samples, n_gates]
) -> np.ndarray:  # [batch]
    """y_proxies has leading batch dim, y_target does not."""
    return y_proxies.eq(y_target.unsqueeze(0)).float().mean(dim=(1, 2)).cpu().numpy()


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
def calculate_best_match_rate_batched(
    y_target: torch.Tensor,  # [n_samples, n_gates] - logits
    y_proxies: torch.Tensor,  # [batch, n_samples, n_gates] - logits
) -> np.ndarray:  # [batch]
    """Batched best match rate: y_proxies has leading batch dim, y_target does not."""
    bit_target = logits_to_binary(y_target)  # [n_samples, n_gates]
    bit_proxies = logits_to_binary(y_proxies)  # [batch, n_samples, n_gates]
    return calculate_match_rate_batched(bit_target, bit_proxies)

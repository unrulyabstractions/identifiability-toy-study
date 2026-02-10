"""Input attribution methods.

Gradient-based methods for attributing model predictions to input features.
"""

from typing import TYPE_CHECKING

import torch
import numpy as np

from .types import InputAttribution

if TYPE_CHECKING:
    from src.model import MLP


def compute_gradient_attribution(
    model: "MLP",
    x: torch.Tensor,
    gate_idx: int = 0,
    device: str = "cpu",
) -> InputAttribution:
    """Compute gradient-based input attribution.

    The simplest attribution method: gradient of output w.r.t. input.

    Args:
        model: The MLP model to analyze
        x: Input tensor [batch, n_inputs]
        gate_idx: Which output gate to analyze
        device: Device to run on

    Returns:
        InputAttribution with gradient-based scores
    """
    model = model.to(device)
    x = x.to(device).requires_grad_(True)

    # Forward pass
    output = model(x)[:, gate_idx]

    # Backward pass
    output.sum().backward()

    # Get gradients
    grads = x.grad.detach().cpu().numpy()

    return InputAttribution(
        method="gradient",
        attributions=grads,
        input_samples=x.detach().cpu().numpy(),
        gate_idx=gate_idx,
    )


def compute_gradient_x_input(
    model: "MLP",
    x: torch.Tensor,
    gate_idx: int = 0,
    device: str = "cpu",
) -> InputAttribution:
    """Compute gradient * input attribution.

    Multiplies gradient by input value for more interpretable attributions.

    Args:
        model: The MLP model to analyze
        x: Input tensor [batch, n_inputs]
        gate_idx: Which output gate to analyze
        device: Device to run on

    Returns:
        InputAttribution with gradient*input scores
    """
    model = model.to(device)
    x = x.to(device).requires_grad_(True)

    # Forward pass
    output = model(x)[:, gate_idx]

    # Backward pass
    output.sum().backward()

    # Gradient * input
    grads = x.grad.detach()
    attributions = (grads * x.detach()).cpu().numpy()

    return InputAttribution(
        method="gradient_x_input",
        attributions=attributions,
        input_samples=x.detach().cpu().numpy(),
        gate_idx=gate_idx,
    )


def compute_integrated_gradients(
    model: "MLP",
    x: torch.Tensor,
    baseline: torch.Tensor | None = None,
    n_steps: int = 50,
    gate_idx: int = 0,
    device: str = "cpu",
) -> InputAttribution:
    """Compute Integrated Gradients attribution.

    Integrates gradients along the path from baseline to input,
    satisfying several desirable attribution axioms.

    Args:
        model: The MLP model to analyze
        x: Input tensor [batch, n_inputs]
        baseline: Baseline tensor (default: zeros)
        n_steps: Number of integration steps
        gate_idx: Which output gate to analyze
        device: Device to run on

    Returns:
        InputAttribution with integrated gradients scores
    """
    model = model.to(device)
    x = x.to(device)
    batch_size, n_inputs = x.shape

    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.to(device)

    # Generate all interpolated inputs
    alphas = torch.linspace(0, 1, n_steps, device=device)

    # Accumulate gradients
    integrated_grads = torch.zeros_like(x)

    for alpha in alphas:
        x_interp = baseline + alpha * (x - baseline)
        x_interp.requires_grad_(True)

        output = model(x_interp)[:, gate_idx]

        model.zero_grad()
        output.sum().backward()

        integrated_grads += x_interp.grad.detach() / n_steps

    # Scale by input - baseline
    attributions = integrated_grads * (x - baseline)

    return InputAttribution(
        method="integrated_gradients",
        attributions=attributions.detach().cpu().numpy(),
        input_samples=x.detach().cpu().numpy(),
        gate_idx=gate_idx,
        baseline=baseline.detach().cpu().numpy(),
    )


def compute_integrated_gradients_fast(
    model: "MLP",
    x: torch.Tensor,
    baseline: torch.Tensor | None = None,
    n_steps: int = 50,
    gate_idx: int = 0,
    device: str = "cpu",
) -> InputAttribution:
    """Fast Integrated Gradients using batched computation.

    Computes all interpolation steps in a single forward/backward pass.

    Args:
        model: The MLP model to analyze
        x: Input tensor [batch, n_inputs]
        baseline: Baseline tensor (default: zeros)
        n_steps: Number of integration steps
        gate_idx: Which output gate to analyze
        device: Device to run on

    Returns:
        InputAttribution with integrated gradients scores
    """
    model = model.to(device)
    x = x.to(device)
    batch_size, n_inputs = x.shape

    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.to(device)

    # Generate all interpolated inputs at once
    alphas = torch.linspace(0, 1, n_steps, device=device).view(-1, 1, 1)
    x_expanded = x.unsqueeze(0)  # [1, batch, input]
    baseline_expanded = baseline.unsqueeze(0)  # [1, batch, input]

    # [n_steps, batch, input]
    x_interp = baseline_expanded + alphas * (x_expanded - baseline_expanded)
    x_interp = x_interp.reshape(-1, n_inputs)  # [n_steps * batch, input]

    # Create leaf tensor for gradient computation
    x_interp_leaf = x_interp.detach().clone().requires_grad_(True)

    # Forward pass
    output = model(x_interp_leaf)[:, gate_idx]

    # Backward pass
    output.sum().backward()

    # Get gradients and reshape
    grads = x_interp_leaf.grad.view(n_steps, batch_size, n_inputs)

    # Average over steps
    integrated_grads = grads.mean(dim=0)

    # Scale by input - baseline
    attributions = integrated_grads * (x - baseline)

    return InputAttribution(
        method="integrated_gradients_fast",
        attributions=attributions.detach().cpu().numpy(),
        input_samples=x.detach().cpu().numpy(),
        gate_idx=gate_idx,
        baseline=baseline.detach().cpu().numpy(),
    )

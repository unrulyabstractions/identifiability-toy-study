"""Edge Attribution Patching with Integrated Gradients.

Combines EAP with Integrated Gradients for more accurate edge attribution
by integrating over the path from baseline to input.
"""

from typing import TYPE_CHECKING

import torch
import numpy as np

from .types import AttributionResult

if TYPE_CHECKING:
    from src.model import MLP


def compute_eap_ig(
    model: "MLP",
    x: torch.Tensor,
    baseline: torch.Tensor | None = None,
    n_steps: int = 50,
    gate_idx: int = 0,
    device: str = "cpu",
) -> AttributionResult:
    """Compute EAP with Integrated Gradients.

    Integrates edge attributions along the path from baseline to input,
    providing more stable and accurate attribution scores.

    Args:
        model: The MLP model to analyze
        x: Input tensor [batch, n_inputs]
        baseline: Baseline tensor (default: zeros)
        n_steps: Number of integration steps
        gate_idx: Which output gate to analyze
        device: Device to run on

    Returns:
        AttributionResult with integrated edge attribution scores
    """
    model = model.to(device)
    x = x.to(device)

    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.to(device)

    # Generate interpolation points
    alphas = torch.linspace(0, 1, n_steps, device=device)

    # Accumulate gradients along the path
    edge_attributions = {}
    layer_attributions = {}

    # Initialize accumulators
    for layer_idx in range(model.num_layers):
        linear = model.layers[layer_idx][0]
        out_features, in_features = linear.weight.shape
        layer_attributions[layer_idx] = np.zeros((out_features, in_features))

    for alpha in alphas:
        # Interpolated input
        x_interp = baseline + alpha * (x - baseline)
        x_interp.requires_grad_(True)

        # Forward pass
        activations = [x_interp]
        current = x_interp
        for layer in model.layers:
            current = layer(current)
            activations.append(current)

        # Backward pass
        output = current[:, gate_idx]
        model.zero_grad()
        if x_interp.grad is not None:
            x_interp.grad.zero_()
        output.sum().backward(retain_graph=True)

        # Accumulate edge gradients
        for layer_idx in range(model.num_layers):
            linear = model.layers[layer_idx][0]
            weight = linear.weight.detach()

            # Input activation to this layer
            if layer_idx == 0:
                act = x_interp.detach()
            else:
                act = activations[layer_idx].detach()

            out_features, in_features = weight.shape

            for out_idx in range(out_features):
                for in_idx in range(in_features):
                    # Gradient contribution at this step
                    grad_contrib = (act[:, in_idx] * weight[out_idx, in_idx]).mean().item()
                    layer_attributions[layer_idx][out_idx, in_idx] += grad_contrib / n_steps

    # Scale by input difference (IG formula)
    input_diff = (x - baseline).mean(dim=0)

    # Populate edge_attributions dict
    for layer_idx in range(model.num_layers):
        out_features, in_features = layer_attributions[layer_idx].shape
        for out_idx in range(out_features):
            for in_idx in range(in_features):
                score = layer_attributions[layer_idx][out_idx, in_idx]
                if layer_idx == 0:
                    # Scale by input difference for first layer
                    score *= input_diff[in_idx].item() if in_idx < len(input_diff) else 1.0
                edge_attributions[(layer_idx, in_idx, layer_idx + 1, out_idx)] = score

    return AttributionResult(
        method="eap_ig",
        layer_attributions=layer_attributions,
        edge_attributions=edge_attributions,
    )


def compute_eap_ig_fast(
    model: "MLP",
    x: torch.Tensor,
    baseline: torch.Tensor | None = None,
    n_steps: int = 20,
    gate_idx: int = 0,
    device: str = "cpu",
) -> AttributionResult:
    """Fast version of EAP-IG using vectorized computation.

    Uses batched interpolation for faster computation.

    Args:
        model: The MLP model to analyze
        x: Input tensor [batch, n_inputs]
        baseline: Baseline tensor (default: zeros)
        n_steps: Number of integration steps
        gate_idx: Which output gate to analyze
        device: Device to run on

    Returns:
        AttributionResult with integrated edge attribution scores
    """
    model = model.to(device)
    x = x.to(device)
    batch_size = x.shape[0]
    n_inputs = x.shape[-1]

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
    output = model(x_interp_leaf)[:, gate_idx]  # [n_steps * batch]

    # Backward pass
    output.sum().backward()

    # Get input gradients
    input_grads = x_interp_leaf.grad  # [n_steps * batch, input]
    input_grads = input_grads.view(n_steps, batch_size, -1)  # [n_steps, batch, input]

    # Average over steps and batch
    mean_grads = input_grads.mean(dim=(0, 1))  # [input]

    # Compute edge attributions from gradient flow
    edge_attributions = {}
    layer_attributions = {}

    with torch.no_grad():
        # Get activations for attribution weighting
        acts = model(x, return_activations=True)
        input_diff = (x - baseline).mean(dim=0)

        for layer_idx in range(model.num_layers):
            linear = model.layers[layer_idx][0]
            weight = linear.weight
            out_features, in_features = weight.shape

            act = acts[layer_idx]  # [batch, in]
            layer_scores = np.zeros((out_features, in_features))

            for out_idx in range(out_features):
                for in_idx in range(in_features):
                    # Attribution based on activation * weight
                    base_score = (act[:, in_idx].abs() * weight[out_idx, in_idx].abs()).mean().item()

                    # Weight by input gradient if first layer
                    if layer_idx == 0 and in_idx < len(mean_grads):
                        base_score *= mean_grads[in_idx].abs().item()

                    edge_attributions[(layer_idx, in_idx, layer_idx + 1, out_idx)] = base_score
                    layer_scores[out_idx, in_idx] = base_score

            layer_attributions[layer_idx] = layer_scores

    return AttributionResult(
        method="eap_ig_fast",
        layer_attributions=layer_attributions,
        edge_attributions=edge_attributions,
    )

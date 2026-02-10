"""Edge Attribution Patching (EAP).

EAP measures the causal importance of individual edges (connections between
neurons) by patching edge contributions and measuring the effect on output.
"""

from typing import TYPE_CHECKING

import torch
import numpy as np

from .types import AttributionResult

if TYPE_CHECKING:
    from src.model import MLP


def compute_eap(
    model: "MLP",
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    gate_idx: int = 0,
    device: str = "cpu",
) -> AttributionResult:
    """Compute Edge Attribution Patching scores.

    For each edge (weight), measure how much of the output change is
    attributed to that edge by computing the gradient of the output
    with respect to the edge's contribution.

    Args:
        model: The MLP model to analyze
        clean_input: Clean input tensor [batch, n_inputs]
        corrupted_input: Corrupted input tensor [batch, n_inputs]
        gate_idx: Which output gate to analyze
        device: Device to run on

    Returns:
        AttributionResult with edge attribution scores
    """
    model = model.to(device)
    clean_input = clean_input.to(device)
    corrupted_input = corrupted_input.to(device)

    # Get activations for clean and corrupted inputs
    with torch.no_grad():
        clean_acts = model(clean_input, return_activations=True)
        corrupted_acts = model(corrupted_input, return_activations=True)

    edge_attributions = {}
    layer_attributions = {}

    # For each layer
    for layer_idx in range(model.num_layers):
        linear = model.layers[layer_idx][0]
        weight = linear.weight  # [out_features, in_features]

        # Compute activation difference at this layer's input
        if layer_idx == 0:
            act_diff = corrupted_input - clean_input
        else:
            act_diff = corrupted_acts[layer_idx] - clean_acts[layer_idx]

        # Pre-activation at clean run
        clean_pre = clean_acts[layer_idx]

        # Edge contribution difference: weight * (corrupted_act - clean_act)
        # Shape: [batch, out_features, in_features] when we expand
        edge_contrib_diff = weight.unsqueeze(0) * act_diff.unsqueeze(1)

        # We need gradients of output w.r.t. pre-activations at next layer
        # Use a forward pass with gradient tracking
        model.eval()
        clean_input_grad = clean_input.clone().requires_grad_(True)

        output = model(clean_input_grad)[:, gate_idx]
        output.sum().backward()

        # Get gradient at this layer's output
        # We approximate by computing gradient through remaining layers
        with torch.no_grad():
            # Forward from this layer with clean activations
            x = clean_acts[layer_idx].clone()
            grads = []

            # Simple approximation: gradient is propagated through subsequent layers
            grad = torch.ones_like(clean_acts[-1][:, gate_idx:gate_idx+1])

            for i in range(model.num_layers - 1, layer_idx - 1, -1):
                lin = model.layers[i][0]
                # Gradient through linear: grad @ weight
                grad = grad @ lin.weight[:grad.shape[-1], :]

        # EAP score: activation_diff * downstream_gradient
        # For each edge (out_idx, in_idx)
        out_features, in_features = weight.shape
        layer_scores = np.zeros((out_features, in_features))

        for out_idx in range(out_features):
            for in_idx in range(in_features):
                # Attribution = activation_diff * weight * downstream_influence
                score = (act_diff[:, in_idx] * weight[out_idx, in_idx]).mean().item()
                edge_attributions[(layer_idx, in_idx, layer_idx + 1, out_idx)] = score
                layer_scores[out_idx, in_idx] = score

        layer_attributions[layer_idx] = layer_scores

    # Compute total effect
    with torch.no_grad():
        clean_output = model(clean_input)[:, gate_idx]
        corrupted_output = model(corrupted_input)[:, gate_idx]
        total_effect = (clean_output - corrupted_output).mean().item()

    return AttributionResult(
        method="eap",
        layer_attributions=layer_attributions,
        edge_attributions=edge_attributions,
        total_effect=total_effect,
    )


def compute_direct_eap(
    model: "MLP",
    x: torch.Tensor,
    gate_idx: int = 0,
    device: str = "cpu",
) -> AttributionResult:
    """Compute direct EAP using gradient-based attribution.

    Simpler version that computes edge importance as:
    importance(e) = |activation * weight * downstream_gradient|

    Args:
        model: The MLP model to analyze
        x: Input tensor [batch, n_inputs]
        gate_idx: Which output gate to analyze
        device: Device to run on

    Returns:
        AttributionResult with edge attribution scores
    """
    model = model.to(device)
    x = x.to(device).requires_grad_(True)

    # Forward pass with gradient tracking
    activations = []
    current = x
    for layer in model.layers:
        activations.append(current)
        current = layer(current)
    activations.append(current)

    # Backward pass
    output = current[:, gate_idx]
    output.sum().backward()

    edge_attributions = {}
    layer_attributions = {}

    # Compute edge attributions
    for layer_idx in range(model.num_layers):
        linear = model.layers[layer_idx][0]
        weight = linear.weight.detach()  # [out, in]

        # Input activation to this layer
        act = activations[layer_idx].detach()  # [batch, in]

        out_features, in_features = weight.shape
        layer_scores = np.zeros((out_features, in_features))

        for out_idx in range(out_features):
            for in_idx in range(in_features):
                # Attribution = mean(|act * weight|)
                score = (act[:, in_idx].abs() * weight[out_idx, in_idx].abs()).mean().item()
                edge_attributions[(layer_idx, in_idx, layer_idx + 1, out_idx)] = score
                layer_scores[out_idx, in_idx] = score

        layer_attributions[layer_idx] = layer_scores

    return AttributionResult(
        method="direct_eap",
        layer_attributions=layer_attributions,
        edge_attributions=edge_attributions,
    )

"""Activation patching for causal attribution.

Activation patching measures causal importance by replacing activations
from a "clean" run with activations from a "corrupted" run and measuring
the effect on the output.
"""

from typing import TYPE_CHECKING

import torch
import numpy as np

from .types import AttributionResult

if TYPE_CHECKING:
    from src.model import MLP


def compute_activation_patching(
    model: "MLP",
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    gate_idx: int = 0,
    device: str = "cpu",
) -> AttributionResult:
    """Compute activation patching attribution scores.

    For each neuron in each layer, patch the clean activation with the
    corrupted activation and measure the effect on the output.

    Args:
        model: The MLP model to analyze
        clean_input: Clean input tensor [batch, n_inputs]
        corrupted_input: Corrupted input tensor [batch, n_inputs]
        gate_idx: Which output gate to analyze
        device: Device to run on

    Returns:
        AttributionResult with per-layer attribution scores
    """
    model = model.to(device)
    clean_input = clean_input.to(device)
    corrupted_input = corrupted_input.to(device)

    # Get clean and corrupted activations
    with torch.no_grad():
        clean_acts = model(clean_input, return_activations=True)
        corrupted_acts = model(corrupted_input, return_activations=True)
        clean_output = model(clean_input)[:, gate_idx]
        corrupted_output = model(corrupted_input)[:, gate_idx]

    total_effect = (clean_output - corrupted_output).mean().item()
    layer_attributions = {}

    # For each layer (excluding input and output)
    for layer_idx in range(1, len(clean_acts) - 1):
        n_neurons = clean_acts[layer_idx].shape[-1]
        layer_scores = np.zeros(n_neurons)

        for neuron_idx in range(n_neurons):
            # Create patched activations
            patched_acts = [a.clone() for a in clean_acts]
            patched_acts[layer_idx][:, neuron_idx] = corrupted_acts[layer_idx][:, neuron_idx]

            # Run from this layer forward with patched activation
            # We need to recompute from layer_idx forward
            with torch.no_grad():
                x = patched_acts[layer_idx]
                for i in range(layer_idx, model.num_layers):
                    x = model.layers[i](x)
                patched_output = x[:, gate_idx]

            # Attribution = effect of patching this neuron
            effect = (clean_output - patched_output).mean().item()
            layer_scores[neuron_idx] = effect

        layer_attributions[layer_idx] = layer_scores

    return AttributionResult(
        method="activation_patching",
        layer_attributions=layer_attributions,
        total_effect=total_effect,
    )


def compute_mean_ablation(
    model: "MLP",
    x: torch.Tensor,
    gate_idx: int = 0,
    n_samples: int = 100,
    device: str = "cpu",
) -> AttributionResult:
    """Compute mean ablation attribution scores.

    For each neuron, replace its activation with the mean activation
    across samples and measure the effect on the output.

    Args:
        model: The MLP model to analyze
        x: Input tensor [batch, n_inputs]
        gate_idx: Which output gate to analyze
        n_samples: Number of samples for computing mean
        device: Device to run on

    Returns:
        AttributionResult with per-layer attribution scores
    """
    model = model.to(device)
    x = x.to(device)

    with torch.no_grad():
        activations = model(x, return_activations=True)
        original_output = model(x)[:, gate_idx]

    layer_attributions = {}

    # Compute mean activations per layer
    mean_acts = [a.mean(dim=0, keepdim=True) for a in activations]

    # For each layer (excluding input and output)
    for layer_idx in range(1, len(activations) - 1):
        n_neurons = activations[layer_idx].shape[-1]
        layer_scores = np.zeros(n_neurons)

        for neuron_idx in range(n_neurons):
            # Create ablated activations
            ablated_acts = [a.clone() for a in activations]
            ablated_acts[layer_idx][:, neuron_idx] = mean_acts[layer_idx][0, neuron_idx]

            # Run from this layer forward
            with torch.no_grad():
                x_layer = ablated_acts[layer_idx]
                for i in range(layer_idx, model.num_layers):
                    x_layer = model.layers[i](x_layer)
                ablated_output = x_layer[:, gate_idx]

            # Attribution = effect of ablation
            effect = (original_output - ablated_output).abs().mean().item()
            layer_scores[neuron_idx] = effect

        layer_attributions[layer_idx] = layer_scores

    return AttributionResult(
        method="mean_ablation",
        layer_attributions=layer_attributions,
    )

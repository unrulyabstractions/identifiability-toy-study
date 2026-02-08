"""Counterfactual analysis functions.

Handles clean/corrupted data pair creation and patch intervention creation
for counterfactual experiments (denoising and noising).
"""

from dataclasses import dataclass

import torch

from src.model import Intervention, PatchShape
from src.math import logits_to_binary


@dataclass
class CleanCorruptedPair:
    """A pair of clean and corrupted samples with their activations."""

    x_clean: torch.Tensor  # [1, input_dim]
    x_corrupted: torch.Tensor  # [1, input_dim]
    y_clean: torch.Tensor  # [1, n_gates]
    y_corrupted: torch.Tensor  # [1, n_gates]
    act_clean: list[torch.Tensor]  # each [1, hidden]
    act_corrupted: list[torch.Tensor]  # each [1, hidden]


def create_clean_corrupted_data(
    x: torch.Tensor,  # [N, input_dim]
    y: torch.Tensor,  # [N, n_gates] - model logits
    activations: list[torch.Tensor],  # each [N, hidden]
    n_pairs: int = 10,
) -> list[CleanCorruptedPair]:
    """
    Create clean vs corrupted data pairs where model outputs MUST differ.

    For counterfactual analysis, we need pairs where:
    - Clean sample produces one output
    - Corrupted sample produces a different output

    Args:
        x: Input data [N, input_dim]
        y: Model outputs (logits) [N, n_gates]
        activations: Activations from model forward pass
        n_pairs: Number of pairs to generate

    Returns:
        List of CleanCorruptedPair objects
    """
    pairs = []
    n_samples = x.shape[0]

    if n_samples < 2:
        return pairs

    # Round y to get binary outputs for comparison
    y_binary = logits_to_binary(y)  # [N, n_gates]

    # Find pairs with different outputs
    for i in range(min(n_pairs * 10, n_samples)):  # Try multiple times
        if len(pairs) >= n_pairs:
            break

        # Pick a random clean sample
        clean_idx = torch.randint(0, n_samples, (1,)).item()

        # Find a sample with different output
        for j in range(n_samples):
            if j == clean_idx:
                continue
            if not torch.equal(y_binary[clean_idx], y_binary[j]):
                # Found a pair with different outputs
                x_clean = x[clean_idx : clean_idx + 1]
                x_corrupted = x[j : j + 1]
                y_clean = y[clean_idx : clean_idx + 1]
                y_corrupted = y[j : j + 1]

                # Extract activations for these samples
                act_clean = [a[clean_idx : clean_idx + 1] for a in activations]
                act_corrupted = [a[j : j + 1] for a in activations]

                pairs.append(
                    CleanCorruptedPair(
                        x_clean=x_clean,
                        x_corrupted=x_corrupted,
                        y_clean=y_clean,
                        y_corrupted=y_corrupted,
                        act_clean=act_clean,
                        act_corrupted=act_corrupted,
                    )
                )
                break

    return pairs


def create_patch_intervention(
    patch: PatchShape, source_activations: list[torch.Tensor]
) -> Intervention:
    """
    Create an intervention that patches neurons with activation values.

    This is the core patching operation used in both denoising and noising:
    - Denoising: source_activations = clean activations (patch clean into corrupt run)
    - Noising: source_activations = corrupted activations (patch corrupt into clean run)

    Args:
        patch: PatchShape specifying which neurons to patch
        source_activations: List of activation tensors to patch FROM

    Returns:
        Intervention that sets specified neurons to source values
    """
    if patch is None:
        return Intervention(patches={})

    patches = {}
    for layer in patch.single_layers():
        if layer < len(source_activations):
            layer_acts = source_activations[layer]
            if patch.indices:
                vals = layer_acts[:, list(patch.indices)]
            else:
                vals = layer_acts
            single_ps = PatchShape(
                layers=(layer,), indices=patch.indices, axis="neuron"
            )
            patches[single_ps] = ("set", vals)

    return Intervention(patches=patches)

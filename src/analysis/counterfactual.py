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


def create_canonical_counterfactual_pairs(
    model,
    gate_idx: int,
    n_inputs: int,
    ground_truth_fn,
    device: str = "cpu",
) -> list[CleanCorruptedPair]:
    """
    Create counterfactual pairs from canonical inputs ONLY.

    For a 2-input gate, uses the 4 canonical inputs: (0,0), (0,1), (1,0), (1,1).
    Creates pairs where the gate output differs.

    Args:
        model: The full model (or gate model)
        gate_idx: Which gate output to compare
        n_inputs: Number of input bits (2 or 3)
        ground_truth_fn: Function(input_tensor) -> expected output
        device: Compute device

    Returns:
        List of CleanCorruptedPair objects for all pairs with differing outputs
    """
    pairs = []

    # Generate all canonical inputs
    canonical_inputs = []
    canonical_outputs = []

    for i in range(2**n_inputs):
        bits = [(i >> j) & 1 for j in range(n_inputs)]
        inp = torch.tensor([bits], dtype=torch.float32, device=device)
        canonical_inputs.append(inp)
        canonical_outputs.append(ground_truth_fn(inp))

    # Stack for batch processing
    x_all = torch.cat(canonical_inputs, dim=0)  # [4 or 8, n_inputs]

    # Get model outputs and activations for all canonical inputs
    with torch.inference_mode():
        y_all = model(x_all)  # [4 or 8, n_gates]
        activations_all = model(x_all, return_activations=True)  # list of [4 or 8, hidden]

    # Get the gate-specific outputs
    y_gate = y_all[..., gate_idx : gate_idx + 1]  # [4 or 8, 1]
    y_binary = logits_to_binary(y_gate)  # [4 or 8, 1]

    # Create pairs where outputs differ
    n_canonical = 2**n_inputs
    for i in range(n_canonical):
        for j in range(n_canonical):
            if i == j:
                continue
            # Check if outputs differ
            if not torch.equal(y_binary[i], y_binary[j]):
                pairs.append(
                    CleanCorruptedPair(
                        x_clean=x_all[i : i + 1],
                        x_corrupted=x_all[j : j + 1],
                        y_clean=y_gate[i : i + 1],
                        y_corrupted=y_gate[j : j + 1],
                        act_clean=[a[i : i + 1] for a in activations_all],
                        act_corrupted=[a[j : j + 1] for a in activations_all],
                    )
                )

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


def create_batched_patch_intervention(
    patches: list[PatchShape], batched_activations: list[torch.Tensor]
) -> Intervention:
    """
    Create a batched intervention for multiple samples at once.

    This enables batched counterfactual computation where each sample in the batch
    gets its own intervention values (per-sample activations).

    The MLP's _apply_neuron_patches_inplace supports per-sample values with shape [B, k],
    where B is batch size and k is the number of neurons being patched.

    Args:
        patches: List of PatchShape objects, one per layer with layer-specific indices.
                 Each PatchShape specifies which neurons to patch in that layer.
        batched_activations: List of activation tensors [N, hidden] per layer,
                            where N is batch size. Length must cover all layers in patches.

    Returns:
        Intervention with per-sample values [N, k] where k = number of neurons patched per layer
    """
    if not patches:
        return Intervention(patches={})

    patches_dict = {}
    for patch in patches:
        for layer in patch.single_layers():
            if layer < len(batched_activations):
                layer_acts = batched_activations[layer]  # [N, hidden]
                if patch.indices:
                    vals = layer_acts[:, list(patch.indices)]  # [N, k]
                else:
                    vals = layer_acts  # [N, hidden]
                single_ps = PatchShape(
                    layers=(layer,), indices=patch.indices, axis="neuron"
                )
                patches_dict[single_ps] = ("set", vals)

    return Intervention(patches=patches_dict)

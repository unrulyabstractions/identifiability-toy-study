"""Data generation functions for robustness analysis.

Generates noise and OOD samples for testing subcircuit behavior under perturbation.
"""

import torch

from ..common.schemas import SampleType


# Ground truth for 2-input logic gates (XOR by default)
GROUND_TRUTH = {
    (0, 0): 0.0,
    (0, 1): 1.0,
    (1, 0): 1.0,
    (1, 1): 0.0,
}


def _generate_noise_samples(
    base_inputs: list[torch.Tensor], n_samples_per_base: int = 100
) -> list[tuple[torch.Tensor, torch.Tensor, float, str]]:
    """Generate noisy inputs with magnitude always < 0.5.

    Returns list of (perturbed_input, base_input, magnitude, sample_type) tuples.
    """
    results = []
    for base_input in base_inputs:
        for _ in range(n_samples_per_base):
            target_mag = 0.01 + torch.rand(1).item() * 0.49
            noise = torch.randn_like(base_input)
            noise = noise / (noise.norm() + 1e-8) * target_mag
            perturbed = base_input + noise
            results.append((perturbed, base_input, target_mag, SampleType.NOISE))
    return results


def _generate_ood_multiply_samples(
    base_inputs: list[torch.Tensor], n_samples_per_base: int = 100
) -> list[tuple[torch.Tensor, torch.Tensor, float, str]]:
    """Generate OOD inputs via multiplicative scaling.

    Returns (perturbed, base, scale, sample_type) tuples.
    sample_type is "multiply_positive" or "multiply_negative".
    """
    results = []
    n_each = n_samples_per_base // 2

    for base_input in base_inputs:
        if base_input[0].item() == 0.0 and base_input[1].item() == 0.0:
            continue

        # Positive: scale > 1
        for _ in range(n_each):
            scale = 10 ** (torch.rand(1).item() * 2)
            perturbed = base_input * scale
            results.append((perturbed, base_input, scale, SampleType.MULTIPLY_POSITIVE))

        # Negative: scale < 0
        for _ in range(n_samples_per_base - n_each):
            scale = -(10 ** (torch.rand(1).item() * 2))
            perturbed = base_input * scale
            results.append((perturbed, base_input, abs(scale), SampleType.MULTIPLY_NEGATIVE))

    return results


def _generate_ood_add_samples(
    base_inputs: list[torch.Tensor], n_samples_per_base: int = 100
) -> list[tuple[torch.Tensor, torch.Tensor, float, str]]:
    """Generate OOD inputs by adding large positive values.

    Returns (perturbed, base, magnitude, sample_type) tuples.
    """
    results = []
    for base_input in base_inputs:
        for _ in range(n_samples_per_base):
            # Add value in range [2, 100] (outside [0,1] training range)
            add_val = 2 + torch.rand(1).item() * 98
            perturbed = base_input + add_val
            results.append((perturbed, base_input, add_val, SampleType.ADD))
    return results


def _generate_ood_subtract_samples(
    base_inputs: list[torch.Tensor], n_samples_per_base: int = 100
) -> list[tuple[torch.Tensor, torch.Tensor, float, str]]:
    """Generate OOD inputs by subtracting large values (adding negative).

    Returns (perturbed, base, magnitude, sample_type) tuples.
    """
    results = []
    for base_input in base_inputs:
        for _ in range(n_samples_per_base):
            # Subtract value in range [2, 100]
            sub_val = 2 + torch.rand(1).item() * 98
            perturbed = base_input - sub_val
            results.append((perturbed, base_input, sub_val, SampleType.SUBTRACT))
    return results


def _generate_ood_bimodal_samples(
    base_inputs: list[torch.Tensor], n_samples_per_base: int = 100
) -> list[tuple[torch.Tensor, torch.Tensor, float, str]]:
    """Generate OOD inputs by mapping [0,1] -> [-1,1].

    Two isomorphic maps:
    - Order-preserving: x -> 2x - 1 (0->-1, 1->1)
    - Inverted: x -> 1 - 2x (0->1, 1->-1)

    Returns (perturbed, base, scale, sample_type) tuples.
    scale is 1.0 for order-preserving, -1.0 for inverted (used in ground_truth adjustment).
    """
    results = []
    n_each = n_samples_per_base // 2

    for base_input in base_inputs:
        # Order-preserving: 0->-1, 1->1 (x -> 2x - 1)
        for _ in range(n_each):
            perturbed = 2 * base_input - 1
            results.append((perturbed, base_input, 1.0, SampleType.BIMODAL))

        # Inverted: 0->1, 1->-1 (x -> 1 - 2x)
        for _ in range(n_samples_per_base - n_each):
            perturbed = 1 - 2 * base_input
            results.append((perturbed, base_input, -1.0, SampleType.BIMODAL_INV))

    return results

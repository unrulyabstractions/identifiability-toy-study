"""Observational analysis functions.

Tests how well subcircuit matches gate_model when both receive the SAME perturbed input.
We compare:
  1. Both outputs to ground truth (accuracy)
  2. Subcircuit output to gate_model output (agreement + MSE)
"""

import torch

from ..common.neural_model import MLP
from ..common.schemas import RobustnessMetrics, RobustnessSample
from .data_generation import (
    GROUND_TRUTH,
    _generate_noise_samples,
    _generate_ood_add_samples,
    _generate_ood_bimodal_samples,
    _generate_ood_multiply_samples,
    _generate_ood_subtract_samples,
)


def _evaluate_samples(
    gate_model: MLP,
    subcircuit: MLP,
    samples: list[tuple[torch.Tensor, torch.Tensor, float, str]],
    device: str,
) -> list[RobustnessSample]:
    """Evaluate gate_model and subcircuit on the same perturbed inputs.

    Pre-computes activations for circuit visualization (no model runs during viz).
    Handles bimodal transformations by adjusting output interpretation.
    """
    results = []

    for perturbed, base_input, magnitude, sample_type in samples:
        perturbed_dev = perturbed.unsqueeze(0).to(device)

        # Get ground truth from base input
        base_key = (int(base_input[0].item()), int(base_input[1].item()))
        ground_truth = GROUND_TRUTH.get(base_key, 0.0)

        # Run BOTH models on the SAME perturbed input, get activations for viz
        with torch.inference_mode():
            gate_acts = gate_model(perturbed_dev, return_activations=True)
            gate_output = gate_acts[-1].item()

            sc_acts = subcircuit(perturbed_dev, return_activations=True)
            subcircuit_output = sc_acts[-1].item()

        # For bimodal transformations, interpret outputs differently
        # Order-preserving bimodal: output in [-1,1], threshold at 0
        # Inverted bimodal: output in [-1,1], threshold at 0, then invert
        if sample_type == "bimodal":
            # Threshold at 0: negative -> 0, positive -> 1
            gate_bit = 1 if gate_output >= 0 else 0
            sc_bit = 1 if subcircuit_output >= 0 else 0
            gate_best = 1 if gate_output >= 0 else 0
            sc_best = 1 if subcircuit_output >= 0 else 0
        elif sample_type == "bimodal_inv":
            # Inverted: threshold at 0, then invert interpretation
            # In bimodal_inv: -1 corresponds to original 1, 1 corresponds to original 0
            gate_bit = 0 if gate_output >= 0 else 1
            sc_bit = 0 if subcircuit_output >= 0 else 1
            gate_best = 0 if gate_output >= 0 else 1
            sc_best = 0 if subcircuit_output >= 0 else 1
        else:
            # Standard interpretation: threshold at 0.5
            gate_bit = 1 if gate_output >= 0.5 else 0
            sc_bit = 1 if subcircuit_output >= 0.5 else 0
            gate_best = max(0, min(1, round(gate_output)))
            sc_best = max(0, min(1, round(subcircuit_output)))

        # Accuracy to ground truth
        gate_correct = gate_bit == ground_truth
        subcircuit_correct = sc_bit == ground_truth

        # Agreement between models (both interpret same class)
        agreement_bit = gate_bit == sc_bit
        agreement_best = gate_best == sc_best
        mse = (gate_output - subcircuit_output) ** 2

        # Convert activations to lists for JSON serialization
        gate_acts_list = [a.squeeze(0).tolist() for a in gate_acts]
        sc_acts_list = [a.squeeze(0).tolist() for a in sc_acts]

        results.append(
            RobustnessSample(
                input_values=[perturbed[0].item(), perturbed[1].item()],
                base_input=[base_input[0].item(), base_input[1].item()],
                noise_magnitude=magnitude,
                ground_truth=ground_truth,
                gate_output=gate_output,
                subcircuit_output=subcircuit_output,
                gate_correct=gate_correct,
                subcircuit_correct=subcircuit_correct,
                agreement_bit=agreement_bit,
                agreement_best=agreement_best,
                mse=mse,
                sample_type=sample_type,
                gate_activations=gate_acts_list,
                subcircuit_activations=sc_acts_list,
            )
        )

    return results


def calculate_robustness_metrics(
    subcircuit: MLP,
    full_model: MLP,
    n_samples_per_base: int = 100,
    device: str = "cpu",
) -> RobustnessMetrics:
    """
    Calculate robustness metrics by perturbing BOTH gate_model and subcircuit the SAME way.

    Generates many samples with varying perturbation magnitudes:
    - Noise: Gaussian noise with std uniformly sampled from [0.01, 0.5]
    - OOD transformations:
      - Multiply: Scale by factors > 1 or < 0
      - Add: Add large positive values [2, 100]
      - Subtract: Subtract large values [2, 100]
      - Bimodal: Map [0,1] -> [-1,1] (order-preserving and inverted)

    For each sample, we record:
    - Actual noise magnitude or transformation parameter
    - Gate and subcircuit outputs
    - Accuracy to ground truth
    - Bit agreement and MSE between models

    Args:
        subcircuit: Subcircuit model to evaluate
        full_model: Full gate model (both get same perturbed inputs)
        n_samples_per_base: Samples per base input (100 = 400 total per category)
        device: Device for tensor operations

    Returns:
        RobustnessMetrics with all samples and aggregates
    """
    base_inputs = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 1.0]),
    ]

    # Generate samples (quick, do sequentially)
    noise_input_pairs = _generate_noise_samples(base_inputs, n_samples_per_base)

    # Generate all OOD sample types
    n_per_type = max(1, n_samples_per_base // 4)  # Split across 4 types
    ood_multiply_pairs = _generate_ood_multiply_samples(base_inputs, n_per_type)
    ood_add_pairs = _generate_ood_add_samples(base_inputs, n_per_type)
    ood_subtract_pairs = _generate_ood_subtract_samples(base_inputs, n_per_type)
    ood_bimodal_pairs = _generate_ood_bimodal_samples(base_inputs, n_per_type)

    # Combine all OOD samples
    ood_input_pairs = ood_multiply_pairs + ood_add_pairs + ood_subtract_pairs + ood_bimodal_pairs

    # Evaluate samples sequentially (GPU ops are not thread-safe)
    noise_samples = _evaluate_samples(full_model, subcircuit, noise_input_pairs, device)
    ood_samples = _evaluate_samples(full_model, subcircuit, ood_input_pairs, device)

    # Aggregate noise stats
    n_noise = len(noise_samples)
    noise_gate_acc = sum(1 for s in noise_samples if s.gate_correct) / n_noise
    noise_sc_acc = sum(1 for s in noise_samples if s.subcircuit_correct) / n_noise
    noise_agree_bit = sum(1 for s in noise_samples if s.agreement_bit) / n_noise
    noise_agree_best = sum(1 for s in noise_samples if s.agreement_best) / n_noise
    noise_mse = sum(s.mse for s in noise_samples) / n_noise

    # Aggregate OOD stats
    n_ood = len(ood_samples)
    ood_gate_acc = sum(1 for s in ood_samples if s.gate_correct) / n_ood
    ood_sc_acc = sum(1 for s in ood_samples if s.subcircuit_correct) / n_ood
    ood_agree_bit = sum(1 for s in ood_samples if s.agreement_bit) / n_ood
    ood_agree_best = sum(1 for s in ood_samples if s.agreement_best) / n_ood
    ood_mse = sum(s.mse for s in ood_samples) / n_ood

    # Overall robustness: focus on agreement (models matching each other)
    overall = (noise_agree_bit + ood_agree_bit) / 2.0

    return RobustnessMetrics(
        noise_samples=noise_samples,
        ood_samples=ood_samples,
        noise_gate_accuracy=float(noise_gate_acc),
        noise_subcircuit_accuracy=float(noise_sc_acc),
        noise_agreement_bit=float(noise_agree_bit),
        noise_agreement_best=float(noise_agree_best),
        noise_mse_mean=float(noise_mse),
        ood_gate_accuracy=float(ood_gate_acc),
        ood_subcircuit_accuracy=float(ood_sc_acc),
        ood_agreement_bit=float(ood_agree_bit),
        ood_agreement_best=float(ood_agree_best),
        ood_mse_mean=float(ood_mse),
        overall_robustness=float(overall),
    )

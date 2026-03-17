"""Observational analysis functions.

Tests how well subcircuit matches gate_model when both receive the SAME perturbed input.
We compare:
  1. Both outputs to ground truth (accuracy)
  2. Subcircuit output to gate_model output (agreement + MSE)
"""

import torch

from src.model import MLP
from src.schemas import (
    NoiseRobustnessMetrics,
    ObservationalMetrics,
    ObservationalSample,
    OutOfDistributionMetrics,
    SampleType,
    Similarity,
)
from src.math import calculate_mse, logits_to_binary

from .perturbations import (
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
    ground_truth_fn=None,
) -> list[ObservationalSample]:
    """Evaluate gate_model and subcircuit on the same perturbed inputs (BATCHED).

    Pre-computes activations for circuit visualization (no model runs during viz).
    Handles bimodal transformations by adjusting output interpretation.

    Uses batched forward passes for ~30x speedup over sequential execution.

    Args:
        gate_model: Full gate model
        subcircuit: Subcircuit model to evaluate
        samples: List of (perturbed, base_input, magnitude, sample_type) tuples
        device: Device for tensor operations
        ground_truth_fn: Optional function to compute ground truth for a base input.
                        If None, uses GROUND_TRUTH dict (2-input only).
    """
    if not samples:
        return []

    n_samples = len(samples)

    # Extract sample components
    perturbed_list = [s[0] for s in samples]
    base_input_list = [s[1] for s in samples]
    magnitude_list = [s[2] for s in samples]
    sample_type_list = [s[3] for s in samples]

    # Stack all perturbed inputs into a single batch: [N, n_inputs]
    perturbed_batch = torch.stack(perturbed_list).to(device)

    # Compute ground truths for all samples
    ground_truths = []
    for base_input in base_input_list:
        if ground_truth_fn is not None:
            ground_truths.append(ground_truth_fn(base_input))
        else:
            base_key = tuple(int(b.item()) for b in base_input)
            ground_truths.append(GROUND_TRUTH.get(base_key, 0.0))

    # Run BOTH models on the ENTIRE batch with single forward pass each
    with torch.inference_mode():
        gate_acts = gate_model(perturbed_batch, return_activations=True)  # list of [N, hidden]
        gate_outputs = gate_acts[-1]  # [N, 1] or [N]

        sc_acts = subcircuit(perturbed_batch, return_activations=True)  # list of [N, hidden]
        sc_outputs = sc_acts[-1]  # [N, 1] or [N]

    # Flatten outputs if needed
    gate_outputs = gate_outputs.view(-1)  # [N]
    sc_outputs = sc_outputs.view(-1)  # [N]

    # Convert to binary (batched)
    gate_bits = logits_to_binary(gate_outputs)  # [N]
    sc_bits = logits_to_binary(sc_outputs)  # [N]

    # Compute MSE per sample
    mse_per_sample = (gate_outputs - sc_outputs) ** 2  # [N]

    # Move ALL tensors to CPU in bulk (single transfer, not per-item)
    # LAZY SERIALIZATION: Store tensors directly, convert to lists in to_dict()
    gate_acts_cpu = [a.cpu() for a in gate_acts]
    sc_acts_cpu = [a.cpu() for a in sc_acts]
    perturbed_cpu = torch.stack(perturbed_list).cpu()  # [N, input_dim]
    base_input_cpu = torch.stack(base_input_list).cpu()  # [N, input_dim]
    gate_bits_cpu = gate_bits.cpu()
    sc_bits_cpu = sc_bits.cpu()
    gate_outputs_cpu = gate_outputs.cpu()
    sc_outputs_cpu = sc_outputs.cpu()
    mse_per_sample_cpu = mse_per_sample.cpu()

    # Build results - store tensors, lazy conversion in to_dict()
    results = []
    for i in range(n_samples):
        sample_type = sample_type_list[i]
        gate_bit = int(gate_bits_cpu[i].item())
        sc_bit = int(sc_bits_cpu[i].item())

        # For bimodal_inv, invert the interpretation
        if sample_type == SampleType.BIMODAL_INV:
            gate_bit = 1 - gate_bit
            sc_bit = 1 - sc_bit

        gate_best = gate_bit
        sc_best = sc_bit

        # Accuracy to ground truth
        ground_truth = ground_truths[i]
        gate_correct = gate_bit == ground_truth
        subcircuit_correct = sc_bit == ground_truth

        # Agreement between models
        agreement_bit = gate_bit == sc_bit
        agreement_best = gate_best == sc_best

        # Store tensors directly - lazy serialization handles conversion
        results.append(
            ObservationalSample(
                input_values=perturbed_cpu[i],  # tensor, converted lazily
                base_input=base_input_cpu[i],  # tensor, converted lazily
                noise_magnitude=magnitude_list[i],
                ground_truth=ground_truth,
                gate_output=gate_outputs_cpu[i].item(),
                subcircuit_output=sc_outputs_cpu[i].item(),
                gate_correct=gate_correct,
                subcircuit_correct=subcircuit_correct,
                agreement_bit=agreement_bit,
                agreement_best=agreement_best,
                mse=mse_per_sample_cpu[i].item(),
                sample_type=sample_type,
                gate_activations=[a[i] for a in gate_acts_cpu],  # tensors
                subcircuit_activations=[a[i] for a in sc_acts_cpu],  # tensors
            )
        )

    return results


def calculate_observational_metrics(
    subcircuit: MLP,
    full_model: MLP,
    n_samples_per_base: int = 100,
    device: str = "cpu",
    n_inputs: int = None,
) -> ObservationalMetrics:
    """
    Calculate observational metrics by perturbing BOTH gate_model and subcircuit the SAME way.

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
        n_inputs: Number of input dimensions (defaults to model's first layer input size)

    Returns:
        ObservationalMetrics with all samples and aggregates
    """
    # Determine input size from model if not specified
    if n_inputs is None:
        n_inputs = full_model.layers[0][0].in_features

    # Generate all binary corner inputs for n-inputs
    import itertools
    base_inputs = [
        torch.tensor([float(b) for b in bits])
        for bits in itertools.product([0, 1], repeat=n_inputs)
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
    ood_input_pairs = (
        ood_multiply_pairs + ood_add_pairs + ood_subtract_pairs + ood_bimodal_pairs
    )

    # Evaluate samples in batched forward passes (much faster than sequential)
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

    noise_metrics = NoiseRobustnessMetrics(
        samples=noise_samples,
        gate_accuracy=float(noise_gate_acc),
        subcircuit_accuracy=float(noise_sc_acc),
        similarity=Similarity(
            bit=float(noise_agree_bit),
            logit=1.0 - float(noise_mse),
            best=float(noise_agree_best),
        ),
        n_samples=n_noise,
    )

    # Compute per-type OOD metrics
    def _type_agreement(samples, sample_type):
        type_samples = [s for s in samples if s.sample_type == sample_type]
        if not type_samples:
            return 0.0, 0
        agreement = sum(1 for s in type_samples if s.agreement_bit) / len(type_samples)
        return float(agreement), len(type_samples)

    mult_pos_agree, mult_pos_n = _type_agreement(ood_samples, SampleType.MULTIPLY_POSITIVE)
    mult_neg_agree, mult_neg_n = _type_agreement(ood_samples, SampleType.MULTIPLY_NEGATIVE)
    add_agree, add_n = _type_agreement(ood_samples, SampleType.ADD)
    sub_agree, sub_n = _type_agreement(ood_samples, SampleType.SUBTRACT)
    bimodal_agree, bimodal_n = _type_agreement(ood_samples, SampleType.BIMODAL)
    bimodal_inv_agree, bimodal_inv_n = _type_agreement(ood_samples, SampleType.BIMODAL_INV)

    ood_metrics = OutOfDistributionMetrics(
        samples=ood_samples,
        gate_accuracy=float(ood_gate_acc),
        subcircuit_accuracy=float(ood_sc_acc),
        similarity=Similarity(
            bit=float(ood_agree_bit),
            logit=1.0 - float(ood_mse),
            best=float(ood_agree_best),
        ),
        multiply_positive_agreement=mult_pos_agree,
        multiply_positive_n_samples=mult_pos_n,
        multiply_negative_agreement=mult_neg_agree,
        multiply_negative_n_samples=mult_neg_n,
        add_agreement=add_agree,
        add_n_samples=add_n,
        subtract_agreement=sub_agree,
        subtract_n_samples=sub_n,
        bimodal_agreement=bimodal_agree,
        bimodal_n_samples=bimodal_n,
        bimodal_inv_agreement=bimodal_inv_agree,
        bimodal_inv_n_samples=bimodal_inv_n,
    )

    return ObservationalMetrics(
        noise=noise_metrics,
        ood=ood_metrics,
        overall_observational=float(overall),
    )

"""Main script to play with different identifiability constraints

Look at calculate_subcircuit_metrics to see high-level

"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .common.causal import Intervention, InterventionEffect, PatchShape
from .common.circuit import Circuit, CircuitStructure
from .common.helpers import calculate_best_match_rate, calculate_match_rate
from .common.neural_model import MLP
from .common.schemas import (
    CounterfactualEffect,
    FaithfulnessConfig,
    FaithfulnessMetrics,
    IdentifiabilityConstraints,
    InterventionSample,
    PatchStatistics,
    RobustnessMetrics,
    RobustnessSample,
    SubcircuitMetrics,
)

##########################################
####### Faithfulness Metrics #############
##########################################


def compute_recovery(y_intervened: float, y_clean: float, y_corrupted: float) -> float:
    """Raw recovery toward clean: R = (y_intervened - y_corrupted) / (y_clean - y_corrupted).

    Used for denoising experiments where we want output to move toward clean.
    """
    delta = y_clean - y_corrupted
    if abs(delta) < 1e-10:
        return 0.0
    return (y_intervened - y_corrupted) / delta


def compute_disruption(
    y_intervened: float, y_clean: float, y_corrupted: float
) -> float:
    """Raw disruption toward corrupt: D = (y_clean - y_intervened) / (y_clean - y_corrupted).

    Used for noising experiments where we expect output to move away from clean.
    Note: D = 1 - R (disruption and recovery are complements).
    """
    delta = y_clean - y_corrupted
    if abs(delta) < 1e-10:
        return 0.0
    return (y_clean - y_intervened) / delta


# =============================================================================
# 2x2 MATRIX OF FAITHFULNESS SCORES
# =============================================================================
#
# |                | IN-Circuit Patch     | OUT-Circuit Patch      |
# |----------------|----------------------|------------------------|
# | DENOISING      | Sufficiency          | Completeness           |
# | (run corrupt,  | (recovery → 1)       | (1 - recovery → 1)     |
# | patch clean)   |                      |                        |
# |----------------|----------------------|------------------------|
# | NOISING        | Necessity            | Independence           |
# | (run clean,    | (disruption → 1)     | (1 - disruption → 1)   |
# | patch corrupt) |                      |                        |
#
# =============================================================================


def compute_sufficiency_score(
    y_intervened: float, y_clean: float, y_corrupted: float
) -> float:
    """Sufficiency = Denoise In-Circuit → recovery.

    Run corrupted input, patch IN-circuit with clean activations.
    High score (→1) means: circuit alone can recover the behavior.
    """
    return compute_recovery(y_intervened, y_clean, y_corrupted)


def compute_completeness_score(
    y_intervened: float, y_clean: float, y_corrupted: float
) -> float:
    """Completeness = Denoise Out-Circuit → 1 - recovery = disruption.

    Run corrupted input, patch OUT-circuit with clean activations.
    High score (→1) means: out-circuit doesn't help recover (circuit is complete).

    We WANT low recovery from out-circuit, so we report disruption.
    """
    return compute_disruption(y_intervened, y_clean, y_corrupted)


def compute_necessity_score(
    y_intervened: float, y_clean: float, y_corrupted: float
) -> float:
    """Necessity = Noise In-Circuit → disruption.

    Run clean input, patch IN-circuit with corrupted activations.
    High score (→1) means: corrupting circuit breaks behavior (it's necessary).
    """
    return compute_disruption(y_intervened, y_clean, y_corrupted)


def compute_independence_score(
    y_intervened: float, y_clean: float, y_corrupted: float
) -> float:
    """Independence = Noise Out-Circuit → 1 - disruption = recovery.

    Run clean input, patch OUT-circuit with corrupted activations.
    High score (→1) means: corrupting out-circuit doesn't break behavior (circuit is independent).

    We WANT low disruption from out-circuit noise, so we report recovery.
    """
    return compute_recovery(y_intervened, y_clean, y_corrupted)

##########################################
##########################################


def _node_masks_key(circuit: Circuit) -> tuple:
    """Convert node_masks to a hashable key for grouping by activation pattern."""
    return tuple(tuple(m.tolist()) for m in circuit.node_masks)


def filter_subcircuits(
    constraints: IdentifiabilityConstraints,
    subcircuit_metrics: list[SubcircuitMetrics],
    subcircuits: list[Circuit],
    subcircuit_structures: list[CircuitStructure],
    max_subcircuits: int = 1,
) -> list[int]:
    """Filter subcircuits by epsilon thresholds, then select diverse top-k.

    Steps:
    1. Filter by bit_similarity and accuracy using epsilon threshold
    2. Sort by (accuracy DESC, bit_similarity DESC, node_sparsity DESC)
    3. Select up to max_subcircuits, diversifying by jaccard distance

    Note: Edge masks are not directly filtered here since circuits with
    the same node pattern but different edges are functionally equivalent
    for initial filtering. Edge exploration happens in a later stage.

    Args:
        constraints: Epsilon thresholds for filtering
        subcircuit_metrics: Per-subcircuit accuracy/similarity metrics
        subcircuits: Circuit objects (for jaccard calculation)
        subcircuit_structures: Circuit structure info (for sparsity)
        max_subcircuits: Maximum number of subcircuits to return

    Returns:
        List of subcircuit indices (up to max_subcircuits), diversified by overlap
    """
    metrics_by_idx = {m.idx: m for m in subcircuit_metrics}

    # First pass: filter by epsilon thresholds
    passing_indices = []
    for result in subcircuit_metrics:
        # Must pass BOTH bit_similarity AND accuracy thresholds
        if 1.0 - result.bit_similarity > constraints.epsilon:
            continue
        if 1.0 - result.accuracy > constraints.epsilon:
            continue
        passing_indices.append(result.idx)

    if not passing_indices:
        return []

    # Sort by quality: (accuracy DESC, bit_similarity DESC, node_sparsity DESC)
    # Higher is better for all metrics
    sorted_indices = sorted(
        passing_indices,
        key=lambda idx: (
            metrics_by_idx[idx].accuracy,
            metrics_by_idx[idx].bit_similarity,
            subcircuit_structures[idx].node_sparsity,
        ),
        reverse=True,
    )

    if max_subcircuits == 1:
        return [sorted_indices[0]]

    # Greedy selection: pick best, then diversify
    # For ties in quality, prefer less overlap with already-selected
    selected = [sorted_indices[0]]

    for candidate_idx in sorted_indices[1:]:
        if len(selected) >= max_subcircuits:
            break

        # Calculate max overlap with any already-selected subcircuit
        # (lower is better - we want diversity)
        max_overlap = max(
            subcircuits[candidate_idx].overlap_jaccard(subcircuits[s]) for s in selected
        )

        # Check if this candidate is a quality tie with the last selected
        candidate = metrics_by_idx[candidate_idx]
        last = metrics_by_idx[selected[-1]]
        is_tie = (
            abs(candidate.accuracy - last.accuracy) < 1e-6
            and abs(candidate.bit_similarity - last.bit_similarity) < 1e-6
        )

        if is_tie:
            # For ties, only add if sufficiently different (jaccard < 0.8)
            if max_overlap < 0.8:
                selected.append(candidate_idx)
        else:
            # Not a tie - just add (it's lower quality but still passes)
            selected.append(candidate_idx)

    return selected


# ===== Robustness Analysis =====
# Tests how well subcircuit matches gate_model when both receive the SAME perturbed input.
# We compare:
#   1. Both outputs to ground truth (accuracy)
#   2. Subcircuit output to gate_model output (agreement + MSE)


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
            results.append((perturbed, base_input, target_mag, "noise"))
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
            results.append((perturbed, base_input, scale, "multiply_positive"))

        # Negative: scale < 0
        for _ in range(n_samples_per_base - n_each):
            scale = -(10 ** (torch.rand(1).item() * 2))
            perturbed = base_input * scale
            results.append((perturbed, base_input, abs(scale), "multiply_negative"))

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
            results.append((perturbed, base_input, add_val, "add"))
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
            results.append((perturbed, base_input, sub_val, "subtract"))
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
            results.append((perturbed, base_input, 1.0, "bimodal"))

        # Inverted: 0->1, 1->-1 (x -> 1 - 2x)
        for _ in range(n_samples_per_base - n_each):
            perturbed = 1 - 2 * base_input
            results.append((perturbed, base_input, -1.0, "bimodal_inv"))

    return results


# Legacy wrapper for backward compatibility
def _generate_ood_samples(
    base_inputs: list[torch.Tensor], n_samples_per_base: int = 100
) -> list[tuple[torch.Tensor, torch.Tensor, float, str]]:
    """Generate OOD multiply samples (legacy compatibility)."""
    return _generate_ood_multiply_samples(base_inputs, n_samples_per_base)


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


def calculate_intervention_effect(
    intervention: Intervention,
    y_target: torch.Tensor,
    y_proxy: torch.Tensor,
    target_activations: list = None,
    proxy_activations: list = None,
    original_target_activations: list = None,
    original_proxy_activations: list = None,
) -> InterventionEffect:
    """Calculate the effect of an intervention by comparing target and proxy outputs."""
    bit_target = torch.round(y_target)
    bit_proxy = torch.round(y_proxy)
    logit_similarity = 1 - nn.MSELoss()(y_target, y_proxy).item()
    bit_similarity = calculate_match_rate(bit_target, bit_proxy).item()
    best_similarity = calculate_best_match_rate(y_target, y_proxy).item()

    return InterventionEffect(
        intervention=intervention,
        y_target=y_target.detach(),
        y_proxy=y_proxy.detach(),
        logit_similarity=logit_similarity,
        bit_similarity=bit_similarity,
        best_similarity=best_similarity,
        target_activations=target_activations,
        proxy_activations=proxy_activations,
        original_target_activations=original_target_activations,
        original_proxy_activations=original_proxy_activations,
    )


def _sample_from_value_range(shape, value_range, device):
    """Sample uniformly from value_range (single or union of ranges)."""
    is_union = isinstance(value_range[0], (list, tuple))
    if not is_union:
        min_val, max_val = value_range
        return torch.empty(shape, device=device).uniform_(min_val, max_val)
    else:
        # Union of ranges: sample uniformly across all ranges
        ranges = value_range
        widths = torch.tensor([r[1] - r[0] for r in ranges])
        probs = widths / widths.sum()
        # Vectorized sampling: choose range for each element
        n_elements = int(np.prod(shape))
        range_choices = torch.multinomial(probs, n_elements, replacement=True)
        result = torch.empty(n_elements, device=device)
        for i, r in enumerate(ranges):
            mask = range_choices == i
            count = mask.sum().item()
            if count > 0:
                result[mask] = torch.empty(count, device=device).uniform_(r[0], r[1])
        return result.view(shape)


def calculate_patches_causal_effect(
    patches: list[PatchShape],
    x: torch.Tensor,
    target: MLP,
    proxy: MLP,
    n_interventions_per_patch: int,
    out_circuit: bool = False,
    device: str = "cpu",
    value_range: Optional[list] = None,
) -> dict[str, list[InterventionEffect]]:
    """
    Calculate causal effects for a list of patches (BATCHED for performance).

    Instead of running n_interventions forward passes per patch, this batches
    all interventions for each patch into a single forward pass.

    Args:
        patches: List of PatchShape objects to intervene on
        x: Input data [N, input_dim]
        target: Target model (full model)
        proxy: Proxy model (subcircuit)
        n_interventions_per_patch: Number of random interventions per patch
        out_circuit: If True, proxy output is computed once without intervention
        device: Device for tensor creation

    Returns:
        Dict mapping patch string repr to list of InterventionEffect
    """
    if value_range is None:
        value_range = [-1, 1]

    N = x.shape[0]
    n_ivs = n_interventions_per_patch

    # Compute original activations (without intervention) for two-value display in viz
    with torch.inference_mode():
        original_target_acts = target(x, return_activations=True)
        original_proxy_acts = proxy(x, return_activations=True)

    if out_circuit:
        # For out_circuit, proxy output is computed once without intervention
        proxy_acts_base = original_proxy_acts
        y_proxy_base = proxy_acts_base[-1]

    intervention_results = {}

    for patch in patches:
        k = len(patch.indices) if patch.indices else 1

        # Create batched intervention values: [n_ivs, k]
        batched_values = _sample_from_value_range((n_ivs, k), value_range, device)

        # Expand x from [N, input_dim] to [n_ivs * N, input_dim]
        x_expanded = x.repeat(n_ivs, 1)  # [n_ivs * N, input_dim]

        # Expand intervention values: each intervention applies to N consecutive samples
        # Shape: [n_ivs * N, k] - repeat each intervention value N times
        values_expanded = batched_values.repeat_interleave(N, dim=0)

        # Create single batched intervention
        batched_patch = PatchShape(
            layers=patch.layers, indices=patch.indices, axis=patch.axis
        )
        batched_iv = Intervention(patches={batched_patch: ("add", values_expanded)})

        # Run single forward pass for all interventions
        with torch.inference_mode():
            target_acts_batched = target(
                x_expanded, intervention=batched_iv, return_activations=True
            )
            y_target_batched = target_acts_batched[-1]  # [n_ivs * N, output_dim]

            if not out_circuit:
                proxy_acts_batched = proxy(
                    x_expanded, intervention=batched_iv, return_activations=True
                )
                y_proxy_batched = proxy_acts_batched[-1]

        # Reshape outputs: [n_ivs * N, ...] -> [n_ivs, N, ...]
        y_target_per_iv = y_target_batched.view(n_ivs, N, -1)
        if not out_circuit:
            y_proxy_per_iv = y_proxy_batched.view(n_ivs, N, -1)

        # Reshape activations per layer
        target_acts_per_iv = [a.view(n_ivs, N, -1) for a in target_acts_batched]
        if not out_circuit:
            proxy_acts_per_iv = [a.view(n_ivs, N, -1) for a in proxy_acts_batched]

        # Build per-intervention results
        patch_results = []
        for i in range(n_ivs):
            # Extract this intervention's values
            iv_values = batched_values[i]  # [k]
            single_iv = Intervention(
                patches={batched_patch: ("add", iv_values.unsqueeze(0))}
            )

            # Extract outputs for this intervention (mean over samples)
            y_target_i = y_target_per_iv[i]  # [N, output_dim]
            if out_circuit:
                y_proxy_i = y_proxy_base
                proxy_acts_i = [a for a in proxy_acts_base]
            else:
                y_proxy_i = y_proxy_per_iv[i]
                proxy_acts_i = [a[i] for a in proxy_acts_per_iv]

            target_acts_i = [a[i] for a in target_acts_per_iv]

            patch_results.append(
                calculate_intervention_effect(
                    single_iv,
                    y_target_i,
                    y_proxy_i,
                    target_activations=target_acts_i,
                    proxy_activations=proxy_acts_i,
                    original_target_activations=original_target_acts,
                    original_proxy_activations=original_proxy_acts,
                )
            )

        intervention_results[str(patch)] = patch_results

    return intervention_results


def _create_intervention_samples(
    patch_key: str,
    effects: list[InterventionEffect],
) -> list[InterventionSample]:
    """Create InterventionSample objects from InterventionEffect list.

    IMPORTANT: Activations from InterventionEffect are converted to lists
    so visualization code NEVER needs to run models. Visualization is READ-ONLY.
    """
    samples = []

    # Parse patch info from key (e.g., "PatchShape(layers=(1,), indices=(0,), axis='neuron')")
    # Extract layer and indices using simple parsing
    import re

    layer_match = re.search(r"layers=\((\d+),?\)", patch_key)
    indices_match = re.search(r"indices=\(([^)]*)\)", patch_key)

    patch_layer = int(layer_match.group(1)) if layer_match else 0
    patch_indices = []
    if indices_match and indices_match.group(1).strip():
        patch_indices = [
            int(x.strip()) for x in indices_match.group(1).split(",") if x.strip()
        ]

    for effect in effects:
        # Extract intervention values from the intervention object
        iv_values = []
        for ps, (mode, vals) in effect.intervention.patches.items():
            iv_values.extend(vals.flatten().tolist())

        # Calculate per-sample metrics
        y_target_val = effect.y_target.mean().item()
        y_proxy_val = effect.y_proxy.mean().item()
        mse = (y_target_val - y_proxy_val) ** 2
        bit_agree = round(y_target_val) == round(y_proxy_val)

        # Convert tensor activations to lists for JSON serialization
        # Use MEAN across batch for efficiency - visualization shows representative values
        gate_acts_list = []
        subcircuit_acts_list = []
        original_gate_acts_list = []
        original_subcircuit_acts_list = []
        if effect.target_activations is not None:
            gate_acts_list = [a.mean(dim=0).tolist() for a in effect.target_activations]
        if effect.proxy_activations is not None:
            subcircuit_acts_list = [
                a.mean(dim=0).tolist() for a in effect.proxy_activations
            ]
        if effect.original_target_activations is not None:
            original_gate_acts_list = [
                a.mean(dim=0).tolist() for a in effect.original_target_activations
            ]
        if effect.original_proxy_activations is not None:
            original_subcircuit_acts_list = [
                a.mean(dim=0).tolist() for a in effect.original_proxy_activations
            ]

        samples.append(
            InterventionSample(
                patch_key=patch_key,
                patch_layer=patch_layer,
                patch_indices=patch_indices,
                intervention_values=iv_values,
                gate_output=y_target_val,
                subcircuit_output=y_proxy_val,
                logit_similarity=effect.logit_similarity,
                bit_agreement=bit_agree,
                mse=mse,
                gate_activations=gate_acts_list,
                subcircuit_activations=subcircuit_acts_list,
                original_gate_activations=original_gate_acts_list,
                original_subcircuit_activations=original_subcircuit_acts_list,
            )
        )

    return samples


def _compute_patch_statistics(
    effects_dict: dict[str, list[InterventionEffect]],
) -> tuple[dict[str, PatchStatistics], list[float]]:
    """Helper to compute patch statistics from effects dict."""
    stats = {}
    all_sims = []
    for patch_key, effects in effects_dict.items():
        if not effects:
            continue
        logit_sims = [e.logit_similarity for e in effects]
        bit_sims = [e.bit_similarity for e in effects]
        best_sims = [e.best_similarity for e in effects]
        all_sims.extend(bit_sims)

        samples = _create_intervention_samples(patch_key, effects)

        stats[patch_key] = PatchStatistics(
            mean_logit_similarity=float(np.mean(logit_sims)),
            std_logit_similarity=float(np.std(logit_sims)),
            mean_bit_similarity=float(np.mean(bit_sims)),
            std_bit_similarity=float(np.std(bit_sims)),
            mean_best_similarity=float(np.mean(best_sims)),
            std_best_similarity=float(np.std(best_sims)),
            n_interventions=len(effects),
            samples=samples,
        )
    return stats, all_sims


def calculate_statistics(
    in_circuit_effects: dict[
        str, dict[str, list[InterventionEffect]] | list[InterventionEffect]
    ],
    out_circuit_effects: dict[
        str, dict[str, list[InterventionEffect]] | list[InterventionEffect]
    ],
    counterfactual_effects: list[CounterfactualEffect],
) -> dict:
    """
    Compute statistics from intervention effects.

    Args:
        in_circuit_effects: Either dict with "in"/"ood" keys containing patch dicts,
                           or flat dict of patch_key -> effects (legacy)
        out_circuit_effects: Same format as in_circuit_effects
        counterfactual_effects: List of CounterfactualEffect

    Returns:
        Dict with keys: in_circuit_stats, out_circuit_stats, in_circuit_stats_ood, out_circuit_stats_ood,
                       mean_in_sim, mean_out_sim, mean_in_sim_ood, mean_out_sim_ood,
                       mean_faith, std_faith
    """
    # Check if new format (nested with "in"/"ood") or legacy (flat dict)
    is_nested = "in" in in_circuit_effects or "ood" in in_circuit_effects

    if is_nested:
        # New format with "in" and "ood" keys
        in_stats, in_sims = _compute_patch_statistics(in_circuit_effects.get("in", {}))
        in_stats_ood, in_sims_ood = _compute_patch_statistics(
            in_circuit_effects.get("ood", {})
        )
        out_stats, out_sims = _compute_patch_statistics(
            out_circuit_effects.get("in", {})
        )
        out_stats_ood, out_sims_ood = _compute_patch_statistics(
            out_circuit_effects.get("ood", {})
        )

        mean_in_sim = float(np.mean(in_sims)) if in_sims else 0.0
        mean_out_sim = float(np.mean(out_sims)) if out_sims else 0.0
        mean_in_sim_ood = float(np.mean(in_sims_ood)) if in_sims_ood else 0.0
        mean_out_sim_ood = float(np.mean(out_sims_ood)) if out_sims_ood else 0.0
    else:
        # Legacy flat format
        in_stats, in_sims = _compute_patch_statistics(in_circuit_effects)
        out_stats, out_sims = _compute_patch_statistics(out_circuit_effects)
        in_stats_ood = {}
        out_stats_ood = {}
        mean_in_sim = float(np.mean(in_sims)) if in_sims else 0.0
        mean_out_sim = float(np.mean(out_sims)) if out_sims else 0.0
        mean_in_sim_ood = 0.0
        mean_out_sim_ood = 0.0

    # Counterfactual statistics
    faith_scores = [c.faithfulness_score for c in counterfactual_effects]
    mean_faith = float(np.mean(faith_scores)) if faith_scores else 0.0
    std_faith = float(np.std(faith_scores)) if faith_scores else 0.0

    return {
        "in_circuit_stats": in_stats,
        "out_circuit_stats": out_stats,
        "in_circuit_stats_ood": in_stats_ood,
        "out_circuit_stats_ood": out_stats_ood,
        "mean_in_sim": mean_in_sim,
        "mean_out_sim": mean_out_sim,
        "mean_in_sim_ood": mean_in_sim_ood,
        "mean_out_sim_ood": mean_out_sim_ood,
        "mean_faith": mean_faith,
        "std_faith": std_faith,
    }


@dataclass
class CleanCorruptedPair:
    """A pair of clean and corrupted samples with their activations."""

    x_clean: torch.Tensor
    x_corrupted: torch.Tensor
    y_clean: torch.Tensor
    y_corrupted: torch.Tensor
    act_clean: list[torch.Tensor]
    act_corrupted: list[torch.Tensor]


def create_clean_corrupted_data(
    x: torch.Tensor,
    y: torch.Tensor,
    activations: list[torch.Tensor],
    n_pairs: int = 10,
) -> list[CleanCorruptedPair]:
    """
    Create clean vs corrupted data pairs where outputs y MUST differ.

    For counterfactual analysis, we need pairs where:
    - Clean sample produces one output
    - Corrupted sample produces a different output

    Args:
        x: Input data [N, input_dim]
        y: Ground truth outputs [N, output_dim]
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
    y_binary = torch.round(y)

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


def calculate_faithfulness_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    model: MLP,
    activations: list[torch.Tensor],
    subcircuit: MLP,
    structure: CircuitStructure,
    counterfactual_pairs: list[CleanCorruptedPair],
    config: FaithfulnessConfig = None,
    device: str = "cpu",
) -> FaithfulnessMetrics:
    """
    Calculate comprehensive faithfulness metrics for a subcircuit.

    Implements the full 2x2 patching matrix:

    |                | IN-Circuit Patch     | OUT-Circuit Patch      |
    |----------------|----------------------|------------------------|
    | DENOISING      | Sufficiency          | Completeness           |
    | (run corrupt,  | (recovery → 1)       | (1 - recovery → 1)     |
    | patch clean)   |                      |                        |
    |----------------|----------------------|------------------------|
    | NOISING        | Necessity            | Independence           |
    | (run clean,    | (disruption → 1)     | (1 - disruption → 1)   |
    | patch corrupt) |                      |                        |

    Args:
        x: Input data
        y: Ground truth outputs
        model: Full model
        activations: Activations from model forward pass on x
        subcircuit: Subcircuit model to evaluate
        structure: CircuitStructure with patch information
        counterfactual_pairs: Pre-computed clean/corrupted pairs for counterfactual analysis
        config: FaithfulnessConfig with n_interventions_per_patch
        device: Device for tensor operations

    Returns:
        FaithfulnessMetrics with all computed statistics
    """
    if config is None:
        config = FaithfulnessConfig()

    n_interventions_per_patch = config.n_interventions_per_patch

    # ===== Helper: Build CounterfactualEffect from intervention =====
    def _build_effect(
        pair: CleanCorruptedPair,
        y_intervened: torch.Tensor,
        intervened_acts: list[torch.Tensor],
        experiment_type: str,
        score_type: str,
    ) -> CounterfactualEffect:
        """Build a CounterfactualEffect from intervention results."""
        y_clean_val = pair.y_clean.mean().item()
        y_corrupted_val = pair.y_corrupted.mean().item()
        y_intervened_val = y_intervened.mean().item()

        # Compute the appropriate score based on experiment type
        if score_type == "sufficiency":
            # Denoise in-circuit: recovery
            faith_score = compute_sufficiency_score(
                y_intervened_val, y_clean_val, y_corrupted_val
            )
        elif score_type == "completeness":
            # Denoise out-circuit: 1 - recovery = disruption
            faith_score = compute_completeness_score(
                y_intervened_val, y_clean_val, y_corrupted_val
            )
        elif score_type == "necessity":
            # Noise in-circuit: disruption
            faith_score = compute_necessity_score(
                y_intervened_val, y_clean_val, y_corrupted_val
            )
        elif score_type == "independence":
            # Noise out-circuit: 1 - disruption = recovery
            faith_score = compute_independence_score(
                y_intervened_val, y_clean_val, y_corrupted_val
            )
        else:
            raise ValueError(f"Unknown score_type: {score_type}")

        output_changed = round(y_intervened_val) == round(y_corrupted_val)

        # Convert activations to lists for JSON serialization
        clean_acts_list = [a.squeeze(0).tolist() for a in pair.act_clean]
        corrupted_acts_list = [a.squeeze(0).tolist() for a in pair.act_corrupted]
        intervened_acts_list = [a.squeeze(0).tolist() for a in intervened_acts]

        return CounterfactualEffect(
            faithfulness_score=faith_score,
            experiment_type=experiment_type,
            score_type=score_type,
            clean_input=pair.x_clean.flatten().tolist(),
            corrupted_input=pair.x_corrupted.flatten().tolist(),
            expected_clean_output=y_clean_val,
            expected_corrupted_output=y_corrupted_val,
            actual_output=y_intervened_val,
            output_changed_to_corrupted=output_changed,
            clean_activations=clean_acts_list,
            corrupted_activations=corrupted_acts_list,
            intervened_activations=intervened_acts_list,
        )

    # ===== NOISING: Run clean input, patch with corrupted activations =====
    def compute_noising_effects(
        patches: list[PatchShape],
        pairs: list[CleanCorruptedPair],
        score_type: str,  # "necessity" (in-circuit) or "independence" (out-circuit)
    ) -> list[CounterfactualEffect]:
        """Noising: Run on CLEAN input, patch specified neurons with CORRUPTED values.

        - Necessity (in-circuit): Does corrupting the circuit break behavior?
        - Independence (out-circuit): Does corrupting outside the circuit break behavior?
        """
        effects = []
        for pair in pairs:
            # Patch with corrupted activations
            iv = create_patch_intervention(patches, pair.act_corrupted)

            # Run on CLEAN input with intervention
            with torch.inference_mode():
                intervened_acts = model(
                    pair.x_clean, intervention=iv, return_activations=True
                )
                y_intervened = intervened_acts[-1]

            effects.append(
                _build_effect(pair, y_intervened, intervened_acts, "noising", score_type)
            )
        return effects

    # ===== DENOISING: Run corrupted input, patch with clean activations =====
    def compute_denoising_effects(
        patches: list[PatchShape],
        pairs: list[CleanCorruptedPair],
        score_type: str,  # "sufficiency" (in-circuit) or "completeness" (out-circuit)
    ) -> list[CounterfactualEffect]:
        """Denoising: Run on CORRUPTED input, patch specified neurons with CLEAN values.

        - Sufficiency (in-circuit): Can the circuit alone recover the behavior?
        - Completeness (out-circuit): Does patching outside the circuit help recover?
        """
        effects = []
        for pair in pairs:
            # Patch with clean activations
            iv = create_patch_intervention(patches, pair.act_clean)

            # Run on CORRUPTED input with intervention
            with torch.inference_mode():
                intervened_acts = model(
                    pair.x_corrupted, intervention=iv, return_activations=True
                )
                y_intervened = intervened_acts[-1]

            effects.append(
                _build_effect(
                    pair, y_intervened, intervened_acts, "denoising", score_type
                )
            )
        return effects

    # ===== Interventional Analysis (random value patching) =====
    in_distribution_value_range = [-1, 1]
    out_distribution_value_range = [[-1000, -2], [2, 1000]]

    in_circuit_effects = {}
    out_circuit_effects = {}

    # Interventional analysis (sequential for GPU safety)
    if structure.in_patches:
        in_circuit_effects["in"] = calculate_patches_causal_effect(
            structure.in_patches,
            x,
            model,
            subcircuit,
            n_interventions_per_patch,
            False,
            device,
            in_distribution_value_range,
        )
        in_circuit_effects["ood"] = calculate_patches_causal_effect(
            structure.in_patches,
            x,
            model,
            subcircuit,
            n_interventions_per_patch,
            False,
            device,
            out_distribution_value_range,
        )

    if structure.out_patches:
        out_circuit_effects["in"] = calculate_patches_causal_effect(
            structure.out_patches,
            x,
            model,
            subcircuit,
            n_interventions_per_patch,
            True,
            device,
            in_distribution_value_range,
        )
        out_circuit_effects["ood"] = calculate_patches_causal_effect(
            structure.out_patches,
            x,
            model,
            subcircuit,
            n_interventions_per_patch,
            True,
            device,
            in_distribution_value_range,
        )

    # ===== 2x2 Counterfactual Analysis =====
    # Denoising experiments (run corrupted, patch with clean)
    sufficiency_effects = compute_denoising_effects(
        structure.in_circuit, counterfactual_pairs, "sufficiency"
    )
    completeness_effects = compute_denoising_effects(
        structure.out_circuit, counterfactual_pairs, "completeness"
    )

    # Noising experiments (run clean, patch with corrupted)
    necessity_effects = compute_noising_effects(
        structure.in_circuit, counterfactual_pairs, "necessity"
    )
    independence_effects = compute_noising_effects(
        structure.out_circuit, counterfactual_pairs, "independence"
    )

    # ===== Calculate Statistics =====
    # Legacy: map to old field names for backwards compatibility
    out_counterfactual_effects = independence_effects  # Was mislabeled as "sufficiency"
    in_counterfactual_effects = necessity_effects
    all_counterfactual_effects = (
        sufficiency_effects
        + completeness_effects
        + necessity_effects
        + independence_effects
    )

    stats = calculate_statistics(
        in_circuit_effects, out_circuit_effects, all_counterfactual_effects
    )

    # Compute mean scores for each experiment type
    def _mean_score(effects: list[CounterfactualEffect]) -> float:
        if not effects:
            return 0.0
        return float(np.mean([e.faithfulness_score for e in effects]))

    mean_sufficiency = _mean_score(sufficiency_effects)
    mean_completeness = _mean_score(completeness_effects)
    mean_necessity = _mean_score(necessity_effects)
    mean_independence = _mean_score(independence_effects)

    # Overall faithfulness: average of all 4 scores
    overall_faithfulness = (
        mean_sufficiency + mean_completeness + mean_necessity + mean_independence
    ) / 4.0

    return FaithfulnessMetrics(
        # Interventional stats
        in_circuit_stats=stats["in_circuit_stats"],
        out_circuit_stats=stats["out_circuit_stats"],
        in_circuit_stats_ood=stats["in_circuit_stats_ood"],
        out_circuit_stats_ood=stats["out_circuit_stats_ood"],
        mean_in_circuit_similarity=stats["mean_in_sim"],
        mean_out_circuit_similarity=stats["mean_out_sim"],
        mean_in_circuit_similarity_ood=stats["mean_in_sim_ood"],
        mean_out_circuit_similarity_ood=stats["mean_out_sim_ood"],
        # 2x2 Matrix effects
        sufficiency_effects=sufficiency_effects,
        completeness_effects=completeness_effects,
        necessity_effects=necessity_effects,
        independence_effects=independence_effects,
        # 2x2 Matrix aggregate scores
        mean_sufficiency=mean_sufficiency,
        mean_completeness=mean_completeness,
        mean_necessity=mean_necessity,
        mean_independence=mean_independence,
        # Legacy fields for backwards compatibility
        out_counterfactual_effects=out_counterfactual_effects,
        in_counterfactual_effects=in_counterfactual_effects,
        counterfactual_effects=all_counterfactual_effects,
        mean_faithfulness_score=stats["mean_faith"],
        std_faithfulness_score=stats["std_faith"],
        overall_faithfulness=overall_faithfulness,
    )

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
    """Raw recovery toward clean. Used for denoising experiments."""
    delta = y_clean - y_corrupted
    if abs(delta) < 1e-10:
        return 0.0
    return (y_intervened - y_corrupted) / delta


def compute_disruption(
    y_intervened: float, y_clean: float, y_corrupted: float
) -> float:
    """Raw disruption toward corrupt. Used for noising experiments."""
    delta = y_clean - y_corrupted
    if abs(delta) < 1e-10:
        return 0.0
    return (y_clean - y_intervened) / delta


# Sufficiency  = recovery of in-circuit
compute_sufficiency_score = compute_recovery

# Necessity = disruption of in-circuit
compute_necessity_score = compute_disruption

# Completeness = disruption of out-circuit  (low out-circuit recovery → complete)
compute_completeness_score = compute_disruption

# Independence = recovery of out-circuit  (low out-circuit disruption → independent)
compute_independence_score = compute_recovery

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
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Generate noisy inputs with magnitude always < 0.5.

    Samples target magnitude uniformly from [0.01, 0.5], then generates
    noise scaled to exactly that magnitude.
    Returns list of (perturbed_input, base_input, magnitude) tuples.
    """
    results = []
    for base_input in base_inputs:
        for _ in range(n_samples_per_base):
            # Sample target magnitude uniformly from [0.01, 0.5]
            target_mag = 0.01 + torch.rand(1).item() * 0.49
            # Generate random direction and scale to exact magnitude
            noise = torch.randn_like(base_input)
            noise = noise / (noise.norm() + 1e-8) * target_mag
            perturbed = base_input + noise
            results.append((perturbed, base_input, target_mag))
    return results


def _generate_ood_samples(
    base_inputs: list[torch.Tensor], n_samples_per_base: int = 100
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Generate OOD inputs via multiplicative scaling (preserves input relationships).

    Two types of OOD (split evenly):
    - Positive: scale > 1 pushes values above 1 (e.g., [1,0] * 2 = [2,0])
    - Negative: scale < 0 pushes values below 0 (e.g., [1,0] * -2 = [-2,0])

    Both preserve the relationship between input values (multiplicative, not additive).
    Skips (0,0) base input since scaling it doesn't create OOD.

    Returns list of (perturbed_input, base_input, scale) tuples.
    Scale is positive (>1) for positive OOD, negative (<0) for negative OOD.
    """
    results = []
    n_each = n_samples_per_base // 2

    for base_input in base_inputs:
        # Skip (0,0) - scaling doesn't make it OOD
        if base_input[0].item() == 0.0 and base_input[1].item() == 0.0:
            continue

        # Positive OOD: scale > 1 (values above 1)
        for _ in range(n_each):
            # Scale sampled logarithmically: 10^[0, 2] = [1, 100]
            scale = 10 ** (torch.rand(1).item() * 2)
            perturbed = base_input * scale
            results.append((perturbed, base_input, scale))

        # Negative OOD: scale < 0 (values below 0, preserves relationship)
        for _ in range(n_samples_per_base - n_each):
            # Scale sampled logarithmically: -10^[0, 2] = [-1, -100]
            scale = -(10 ** (torch.rand(1).item() * 2))
            perturbed = base_input * scale
            results.append((perturbed, base_input, scale))

    return results


def _evaluate_samples(
    gate_model: MLP,
    subcircuit: MLP,
    samples: list[tuple[torch.Tensor, torch.Tensor, float]],
    device: str,
) -> list[RobustnessSample]:
    """Evaluate gate_model and subcircuit on the same perturbed inputs.

    Pre-computes activations for circuit visualization (no model runs during viz).
    """
    results = []

    for perturbed, base_input, magnitude in samples:
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

        # Interpret outputs as 0 or 1 (based on which is closest)
        gate_bit = 1 if gate_output >= 0.5 else 0
        sc_bit = 1 if subcircuit_output >= 0.5 else 0

        # Clamp to binary: round then clamp to [0, 1] (handles out-of-range outputs)
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
                noise_magnitude=magnitude,  # Pre-computed magnitude or scale
                ground_truth=ground_truth,
                gate_output=gate_output,
                subcircuit_output=subcircuit_output,
                gate_correct=gate_correct,
                subcircuit_correct=subcircuit_correct,
                agreement_bit=agreement_bit,
                agreement_best=agreement_best,
                mse=mse,
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
    - Noise: Gaussian noise with std uniformly sampled from [0.01, 1.0]
    - OOD: Offsets uniformly sampled from [-5, 5] per dimension

    For each sample, we record:
    - Actual noise magnitude (L2 norm of perturbation)
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
    ood_input_pairs = _generate_ood_samples(base_inputs, n_samples_per_base)

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
    patch: PatchShape, corrupted_activations: list[torch.Tensor]
) -> Intervention:
    """
    Create an intervention that patches with corrupted activation values.

    Args:
        patch: PatchShape specifying which neurons to patch
        corrupted_activations: List of activation tensors from corrupted input

    Returns:
        Intervention that sets specified neurons to corrupted values
    """
    if patch is None:
        return Intervention(patches={})

    patches = {}
    for layer in patch.single_layers():
        if layer < len(corrupted_activations):
            layer_acts = corrupted_activations[layer]
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

    # ===== Define helper for counterfactual effects =====
    def compute_counterfactual_effects(
        patches: list[PatchShape], pairs: list[CleanCorruptedPair], score_type: str
    ) -> list[CounterfactualEffect]:
        """Compute counterfactual effects for a set of patches.

        Runs the FULL MODEL with interventions that patch specific neurons
        to corrupted values. Captures activations from the intervention run
        for visualization.
        """
        effects = []
        for pair in pairs:
            iv = create_patch_intervention(patches, pair.act_corrupted)

            # Run FULL MODEL with intervention and capture activations
            with torch.inference_mode():
                intervened_acts = model(
                    pair.x_clean, intervention=iv, return_activations=True
                )
                y_intervened = intervened_acts[-1]  # Last activation is output

            denominator = pair.y_clean - pair.y_corrupted
            assert torch.abs(denominator).mean() > 1e-6

            faith_score = None
            if score_type == "necessity":
                faith_score = (
                    ((pair.y_clean - y_intervened) / denominator).mean().item()
                )
            if score_type == "sufficiency":
                faith_score = (
                    ((y_intervened - pair.y_corrupted) / denominator).mean().item()
                )

            assert faith_score is not None, f"Unknown score_type: {score_type}"

            y_clean_val = pair.y_clean.mean().item()
            y_corrupted_val = pair.y_corrupted.mean().item()
            y_intervened_val = y_intervened.mean().item()
            output_changed = round(y_intervened_val) == round(y_corrupted_val)

            # Reference activations from clean/corrupted (no intervention)
            clean_acts_list = [a.squeeze(0).tolist() for a in pair.act_clean]
            corrupted_acts_list = [a.squeeze(0).tolist() for a in pair.act_corrupted]

            # Activations from the actual intervention run (FULL MODEL with patches)
            intervened_acts_list = [a.squeeze(0).tolist() for a in intervened_acts]

            effects.append(
                CounterfactualEffect(
                    faithfulness_score=faith_score,
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
            )
        return effects

    # ===== Interventional + Counterfactual Analysis =====
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

    # Counterfactual analysis
    out_counterfactual_effects = compute_counterfactual_effects(
        structure.out_circuit, counterfactual_pairs, "sufficiency"
    )
    in_counterfactual_effects = compute_counterfactual_effects(
        structure.in_circuit, counterfactual_pairs, "necessity"
    )

    # ===== Calculate Statistics =====
    # Combine counterfactual effects (out-circuit counterfactuals are more meaningful
    # since patching out-circuit neurons tests if the subcircuit is sufficient)
    all_counterfactual_effects = out_counterfactual_effects + in_counterfactual_effects

    stats = calculate_statistics(
        in_circuit_effects, out_circuit_effects, all_counterfactual_effects
    )

    # Overall faithfulness: combine in-circuit similarity and counterfactual faithfulness
    # Higher in-circuit similarity is good, higher faithfulness score is good
    mean_in_sim = stats["mean_in_sim"]
    mean_faith = stats["mean_faith"]
    overall_faithfulness = (
        (mean_in_sim + mean_faith) / 2.0 if mean_faith > 0 else mean_in_sim
    )

    return FaithfulnessMetrics(
        in_circuit_stats=stats["in_circuit_stats"],
        out_circuit_stats=stats["out_circuit_stats"],
        in_circuit_stats_ood=stats["in_circuit_stats_ood"],
        out_circuit_stats_ood=stats["out_circuit_stats_ood"],
        mean_in_circuit_similarity=stats["mean_in_sim"],
        mean_out_circuit_similarity=stats["mean_out_sim"],
        mean_in_circuit_similarity_ood=stats["mean_in_sim_ood"],
        mean_out_circuit_similarity_ood=stats["mean_out_sim_ood"],
        out_counterfactual_effects=out_counterfactual_effects,
        in_counterfactual_effects=in_counterfactual_effects,
        counterfactual_effects=all_counterfactual_effects,  # Legacy combined
        mean_faithfulness_score=stats["mean_faith"],
        std_faithfulness_score=stats["std_faith"],
        overall_faithfulness=overall_faithfulness,
    )

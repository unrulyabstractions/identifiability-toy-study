"""Main script to play with different identifiability constraints

Look at calculate_subcircuit_metrics to see high-level

"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .common.causal import Intervention, InterventionEffect, PatchShape
from .common.circuit import CircuitStructure
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


def filter_subcircuits(
    constraints: IdentifiabilityConstraints, subcircuit_metrics: list[SubcircuitMetrics]
) -> list[int]:
    subcircuit_metrics = sorted(
        subcircuit_metrics,
        key=lambda x: (x.bit_similarity, x.accuracy, x.logit_similarity),
        reverse=True,
    )

    subcircuit_indices = []
    for result in subcircuit_metrics:
        if 1.0 - result.bit_similarity > constraints.epsilon:
            break
        subcircuit_indices.append(result.idx)
    return subcircuit_indices


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
    """Evaluate gate_model and subcircuit on the same perturbed inputs."""
    results = []

    for perturbed, base_input, magnitude in samples:
        perturbed_dev = perturbed.unsqueeze(0).to(device)

        # Get ground truth from base input
        base_key = (int(base_input[0].item()), int(base_input[1].item()))
        ground_truth = GROUND_TRUTH.get(base_key, 0.0)

        # Run BOTH models on the SAME perturbed input
        with torch.no_grad():
            gate_output = gate_model(perturbed_dev).item()
            subcircuit_output = subcircuit(perturbed_dev).item()

        # Interpret outputs as 0 or 1 (based on which is closest)
        gate_bit = 1 if gate_output >= 0.5 else 0
        sc_bit = 1 if subcircuit_output >= 0.5 else 0

        # Accuracy to ground truth
        gate_correct = gate_bit == ground_truth
        subcircuit_correct = sc_bit == ground_truth

        # Agreement between models (both interpret same class)
        agreement_bit = gate_bit == sc_bit
        mse = (gate_output - subcircuit_output) ** 2

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
                mse=mse,
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

    # Generate and evaluate noise samples
    noise_input_pairs = _generate_noise_samples(base_inputs, n_samples_per_base)
    noise_samples = _evaluate_samples(full_model, subcircuit, noise_input_pairs, device)

    # Generate and evaluate OOD samples
    ood_input_pairs = _generate_ood_samples(base_inputs, n_samples_per_base)
    ood_samples = _evaluate_samples(full_model, subcircuit, ood_input_pairs, device)

    # Aggregate noise stats
    n_noise = len(noise_samples)
    noise_gate_acc = sum(1 for s in noise_samples if s.gate_correct) / n_noise
    noise_sc_acc = sum(1 for s in noise_samples if s.subcircuit_correct) / n_noise
    noise_agree_bit = sum(1 for s in noise_samples if s.agreement_bit) / n_noise
    noise_mse = sum(s.mse for s in noise_samples) / n_noise

    # Aggregate OOD stats
    n_ood = len(ood_samples)
    ood_gate_acc = sum(1 for s in ood_samples if s.gate_correct) / n_ood
    ood_sc_acc = sum(1 for s in ood_samples if s.subcircuit_correct) / n_ood
    ood_agree_bit = sum(1 for s in ood_samples if s.agreement_bit) / n_ood
    ood_mse = sum(s.mse for s in ood_samples) / n_ood

    # Overall robustness: focus on agreement (models matching each other)
    overall = (noise_agree_bit + ood_agree_bit) / 2.0

    return RobustnessMetrics(
        noise_samples=noise_samples,
        ood_samples=ood_samples,
        noise_gate_accuracy=float(noise_gate_acc),
        noise_subcircuit_accuracy=float(noise_sc_acc),
        noise_agreement_bit=float(noise_agree_bit),
        noise_mse_mean=float(noise_mse),
        ood_gate_accuracy=float(ood_gate_acc),
        ood_subcircuit_accuracy=float(ood_sc_acc),
        ood_agreement_bit=float(ood_agree_bit),
        ood_mse_mean=float(ood_mse),
        overall_robustness=float(overall),
    )


def calculate_intervention_effect(
    intervention: Intervention,
    y_target: torch.Tensor,
    y_proxy: torch.Tensor,
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
    )


def calculate_patches_causal_effect(
    patches: list[PatchShape],
    x: torch.Tensor,
    target: MLP,
    proxy: MLP,
    n_interventions_per_patch: int,
    out_circuit: bool = False,
    device: str = "cpu",
) -> dict[str, list[InterventionEffect]]:
    """
    Calculate causal effects for a list of patches.

    Args:
        patches: List of PatchShape objects to intervene on
        x: Input data
        target: Target model (full model)
        proxy: Proxy model (subcircuit)
        n_interventions_per_patch: Number of random interventions per patch
        out_circuit: If True, proxy output is computed once without intervention
        device: Device for tensor creation

    Returns:
        Dict mapping patch string repr to list of InterventionEffect
    """
    if out_circuit:
        y_proxy = proxy(x)

    intervention_results = {}
    for patch in patches:
        interventions = Intervention.create_random_interventions(
            patch, n_interventions=n_interventions_per_patch, device=device
        )
        patch_results = []
        for iv in interventions:
            y_target = target(x, intervention=iv)
            if not out_circuit:
                y_proxy = proxy(x, intervention=iv)
            patch_results.append(calculate_intervention_effect(iv, y_target, y_proxy))
        intervention_results[str(patch)] = patch_results
    return intervention_results


def _create_intervention_samples(
    patch_key: str,
    effects: list[InterventionEffect],
) -> list[InterventionSample]:
    """Create InterventionSample objects from InterventionEffect list."""
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
            )
        )

    return samples


def calculate_statistics(
    in_circuit_effects: dict[str, list[InterventionEffect]],
    out_circuit_effects: dict[str, list[InterventionEffect]],
    counterfactual_effects: list[CounterfactualEffect],
) -> tuple[
    dict[str, PatchStatistics], dict[str, PatchStatistics], float, float, float, float
]:
    """
    Compute statistics from intervention effects.

    Returns:
        Tuple of (in_circuit_stats, out_circuit_stats,
                  mean_in_sim, mean_out_sim, mean_faith, std_faith)
    """
    # Per-patch statistics for in-circuit
    in_circuit_stats = {}
    all_in_sims = []
    for patch_key, effects in in_circuit_effects.items():
        if not effects:
            continue
        logit_sims = [e.logit_similarity for e in effects]
        bit_sims = [e.bit_similarity for e in effects]
        best_sims = [e.best_similarity for e in effects]
        all_in_sims.extend(bit_sims)

        # Create individual samples for visualization
        samples = _create_intervention_samples(patch_key, effects)

        in_circuit_stats[patch_key] = PatchStatistics(
            mean_logit_similarity=float(np.mean(logit_sims)),
            std_logit_similarity=float(np.std(logit_sims)),
            mean_bit_similarity=float(np.mean(bit_sims)),
            std_bit_similarity=float(np.std(bit_sims)),
            mean_best_similarity=float(np.mean(best_sims)),
            std_best_similarity=float(np.std(best_sims)),
            n_interventions=len(effects),
            samples=samples,
        )

    # Per-patch statistics for out-circuit
    out_circuit_stats = {}
    all_out_sims = []
    for patch_key, effects in out_circuit_effects.items():
        if not effects:
            continue
        logit_sims = [e.logit_similarity for e in effects]
        bit_sims = [e.bit_similarity for e in effects]
        best_sims = [e.best_similarity for e in effects]
        all_out_sims.extend(bit_sims)

        # Create individual samples for visualization
        samples = _create_intervention_samples(patch_key, effects)

        out_circuit_stats[patch_key] = PatchStatistics(
            mean_logit_similarity=float(np.mean(logit_sims)),
            std_logit_similarity=float(np.std(logit_sims)),
            mean_bit_similarity=float(np.mean(bit_sims)),
            std_bit_similarity=float(np.std(bit_sims)),
            mean_best_similarity=float(np.mean(best_sims)),
            std_best_similarity=float(np.std(best_sims)),
            n_interventions=len(effects),
            samples=samples,
        )

    # Aggregate statistics
    mean_in_sim = float(np.mean(all_in_sims)) if all_in_sims else 0.0
    mean_out_sim = float(np.mean(all_out_sims)) if all_out_sims else 0.0

    # Counterfactual statistics
    faith_scores = [c.faithfulness_score for c in counterfactual_effects]
    mean_faith = float(np.mean(faith_scores)) if faith_scores else 0.0
    std_faith = float(np.std(faith_scores)) if faith_scores else 0.0

    return (
        in_circuit_stats,
        out_circuit_stats,
        mean_in_sim,
        mean_out_sim,
        mean_faith,
        std_faith,
    )


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

    # ===== Interventional Analysis =====
    in_circuit_effects = {}
    out_circuit_effects = {}

    if structure.in_patches:
        in_circuit_effects = calculate_patches_causal_effect(
            structure.in_patches,
            x,
            model,
            subcircuit,
            n_interventions_per_patch,
            out_circuit=False,
            device=device,
        )

    if structure.out_patches:
        out_circuit_effects = calculate_patches_causal_effect(
            structure.out_patches,
            x,
            model,
            subcircuit,
            n_interventions_per_patch,
            out_circuit=True,
            device=device,
        )

    # ===== Counterfactual Analysis =====
    def compute_counterfactual_effects(
        patches: list[PatchShape],
        pairs: list[CleanCorruptedPair],
    ) -> list[CounterfactualEffect]:
        """Compute counterfactual effects for a set of patches."""
        effects = []
        for pair in pairs:
            iv = create_patch_intervention(patches, pair.act_corrupted)
            y_sc = model(pair.x_clean, intervention=iv)

            # faithfulness = (y_sc - y_corrupted) / (y_clean - y_corrupted)
            denominator = pair.y_clean - pair.y_corrupted
            assert torch.abs(denominator).mean() > 1e-6
            faith_score = ((y_sc - pair.y_corrupted) / denominator).mean().item()

            y_clean_val = pair.y_clean.mean().item()
            y_corrupted_val = pair.y_corrupted.mean().item()
            y_sc_val = y_sc.mean().item()
            output_changed = round(y_sc_val) == round(y_corrupted_val)

            effects.append(
                CounterfactualEffect(
                    faithfulness_score=faith_score,
                    clean_input=pair.x_clean.flatten().tolist(),
                    corrupted_input=pair.x_corrupted.flatten().tolist(),
                    expected_clean_output=y_clean_val,
                    expected_corrupted_output=y_corrupted_val,
                    actual_output=y_sc_val,
                    output_changed_to_corrupted=output_changed,
                )
            )
        return effects

    out_counterfactual_effects = compute_counterfactual_effects(
        structure.out_circuit, counterfactual_pairs
    )
    in_counterfactual_effects = compute_counterfactual_effects(
        structure.in_circuit, counterfactual_pairs
    )

    # ===== Calculate Statistics =====
    # Combine counterfactual effects (out-circuit counterfactuals are more meaningful
    # since patching out-circuit neurons tests if the subcircuit is sufficient)
    all_counterfactual_effects = out_counterfactual_effects + in_counterfactual_effects

    (
        in_circuit_stats,
        out_circuit_stats,
        mean_in_sim,
        mean_out_sim,
        mean_faith,
        std_faith,
    ) = calculate_statistics(
        in_circuit_effects, out_circuit_effects, all_counterfactual_effects
    )

    # Overall faithfulness: combine in-circuit similarity and counterfactual faithfulness
    # Higher in-circuit similarity is good, higher faithfulness score is good
    overall_faithfulness = (
        (mean_in_sim + mean_faith) / 2.0 if mean_faith > 0 else mean_in_sim
    )

    return FaithfulnessMetrics(
        in_circuit_stats=in_circuit_stats,
        out_circuit_stats=out_circuit_stats,
        mean_in_circuit_similarity=mean_in_sim,
        mean_out_circuit_similarity=mean_out_sim,
        out_counterfactual_effects=out_counterfactual_effects,
        in_counterfactual_effects=in_counterfactual_effects,
        counterfactual_effects=all_counterfactual_effects,  # Legacy combined
        mean_faithfulness_score=mean_faith,
        std_faithfulness_score=std_faith,
        overall_faithfulness=overall_faithfulness,
    )

"""Interventional analysis functions.

Handles causal effect calculation for intervention experiments where
we patch activations with random values and measure the effect.
"""

import re
from typing import Optional

import numpy as np
import torch

from src.model import Intervention, InterventionEffect, MLP, PatchShape
from src.schemas import InterventionalSample, PatchStatistics
from src.tensor_ops import (
    calculate_best_match_rate,
    calculate_logit_similarity,
    calculate_match_rate,
    calculate_mse,
    logits_to_binary,
)


def calculate_intervention_effect(
    intervention: Intervention,
    y_target: torch.Tensor,  # [n_samples, n_gates]
    y_proxy: torch.Tensor,  # [n_samples, n_gates]
    target_activations: list = None,
    proxy_activations: list = None,
    original_target_activations: list = None,
    original_proxy_activations: list = None,
) -> InterventionEffect:
    """Calculate the effect of an intervention by comparing target and proxy outputs."""
    bit_target = logits_to_binary(y_target)  # [n_samples, n_gates]
    bit_proxy = logits_to_binary(y_proxy)  # [n_samples, n_gates]
    logit_similarity = calculate_logit_similarity(y_target, y_proxy).item()  # [] scalar
    bit_similarity = calculate_match_rate(bit_target, bit_proxy).item()  # [] scalar
    best_similarity = calculate_best_match_rate(y_target, y_proxy).item()  # [] scalar

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
) -> list[InterventionalSample]:
    """Create InterventionalSample objects from InterventionEffect list.

    IMPORTANT: Activations from InterventionEffect are converted to lists
    so visualization code NEVER needs to run models. Visualization is READ-ONLY.
    """
    samples = []

    # Parse patch info from key (e.g., "PatchShape(layers=(1,), indices=(0,), axis='neuron')")
    # Extract layer and indices using simple parsing
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

        # Calculate per-sample metrics using helper functions
        logit_sim = calculate_logit_similarity(effect.y_target, effect.y_proxy).item()  # [] scalar
        mse = calculate_mse(effect.y_target, effect.y_proxy).item()  # [] scalar
        y_target_val = effect.y_target.mean().item()  # [] scalar -> float
        y_proxy_val = effect.y_proxy.mean().item()  # [] scalar -> float
        bit_target = logits_to_binary(effect.y_target.mean()).item()  # [] scalar -> int
        bit_proxy = logits_to_binary(effect.y_proxy.mean()).item()  # [] scalar -> int
        bit_agree = bit_target == bit_proxy

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
            InterventionalSample(
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

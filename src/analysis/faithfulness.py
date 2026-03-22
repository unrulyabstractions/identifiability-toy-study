"""Metrics calculation functions for faithfulness analysis.

Combines observational, interventional, and counterfactual analysis to compute
comprehensive faithfulness metrics for subcircuits.

OPTIMIZATIONS:
- Lazy serialization: Tensors stored directly, converted to lists only during JSON serialization
- Parallelization: Independent analyses run concurrently using ThreadPoolExecutor
"""

import numpy as np
import torch

from src.circuit import CircuitStructure
from src.infra.profiler import trace, traced
from src.model import InterventionEffect, MLP, PatchShape
from src.experiment_config import FaithfulnessConfig
from src.schemas import (
    CircuitInterventionEffects,
    CounterfactualEffect,
    CounterfactualMetrics,
    FaithfulnessMetrics,
    InterventionalMetrics,
    InterventionStatistics,
    ObservationalMetrics,
)
from src.math import logits_to_binary

from .counterfactual import CleanCorruptedPair, create_batched_patch_intervention
from .interventional import _compute_patch_statistics, calculate_patches_causal_effect
from .observational import calculate_observational_metrics
from .scoring import (
    compute_completeness_score,
    compute_independence_score,
    compute_necessity_score,
    compute_sufficiency_score,
)


def calculate_statistics(
    in_circuit_effects: CircuitInterventionEffects,
    out_circuit_effects: CircuitInterventionEffects,
    counterfactual_effects: list[CounterfactualEffect],
    x: torch.Tensor = None,
) -> InterventionStatistics:
    """
    Compute statistics from intervention effects.

    Args:
        in_circuit_effects: Effects from in-circuit interventions
        out_circuit_effects: Effects from out-circuit interventions
        counterfactual_effects: List of CounterfactualEffect
        x: Input tensor [N, input_dim] - used to track base_input per sample

    Returns:
        InterventionStatistics with all computed statistics including sample counts
    """
    in_stats, in_sims = _compute_patch_statistics(in_circuit_effects.in_distribution, x=x)
    in_stats_ood, in_sims_ood = _compute_patch_statistics(
        in_circuit_effects.out_distribution, x=x
    )
    out_stats, out_sims = _compute_patch_statistics(out_circuit_effects.in_distribution, x=x)
    out_stats_ood, out_sims_ood = _compute_patch_statistics(
        out_circuit_effects.out_distribution, x=x
    )

    mean_in_sim = float(np.mean(in_sims)) if in_sims else 0.0
    mean_out_sim = float(np.mean(out_sims)) if out_sims else 0.0
    mean_in_sim_ood = float(np.mean(in_sims_ood)) if in_sims_ood else 0.0
    mean_out_sim_ood = float(np.mean(out_sims_ood)) if out_sims_ood else 0.0

    # Counterfactual statistics
    faith_scores = [c.faithfulness_score for c in counterfactual_effects]
    mean_faith = float(np.mean(faith_scores)) if faith_scores else 0.0
    std_faith = float(np.std(faith_scores)) if faith_scores else 0.0

    return InterventionStatistics(
        in_circuit_stats=in_stats,
        out_circuit_stats=out_stats,
        in_circuit_stats_ood=in_stats_ood,
        out_circuit_stats_ood=out_stats_ood,
        mean_in_sim=mean_in_sim,
        mean_out_sim=mean_out_sim,
        mean_in_sim_ood=mean_in_sim_ood,
        mean_out_sim_ood=mean_out_sim_ood,
        mean_faith=mean_faith,
        std_faith=std_faith,
        # Sample counts for availability tracking
        n_in_sim=len(in_sims),
        n_out_sim=len(out_sims),
        n_in_sim_ood=len(in_sims_ood),
        n_out_sim_ood=len(out_sims_ood),
    )


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

    Three categories of faithfulness:
    1. OBSERVATIONAL: How well subcircuit matches model under input perturbations
    2. INTERVENTIONAL: How well subcircuit matches under activation patching
    3. COUNTERFACTUAL: 2x2 patching matrix (sufficiency, completeness, necessity, independence)

    The counterfactual 2x2 matrix:

    |                | IN-Circuit Patch     | OUT-Circuit Patch      |
    |----------------|----------------------|------------------------|
    | DENOISING      | Sufficiency          | Completeness           |
    | (run corrupt,  | (recovery -> 1)      | (1 - recovery -> 1)    |
    | patch clean)   |                      |                        |
    |----------------|----------------------|------------------------|
    | NOISING        | Necessity            | Independence           |
    | (run clean,    | (disruption -> 1)    | (1 - disruption -> 1)  |
    | patch corrupt) |                      |                        |

    Args:
        x: Input data
        y: Ground truth outputs
        model: Full model (gate model)
        activations: Activations from model forward pass on x
        subcircuit: Subcircuit model to evaluate
        structure: CircuitStructure with patch information
        counterfactual_pairs: Pre-computed clean/corrupted pairs for counterfactual analysis
        config: FaithfulnessConfig with n_interventions_per_patch
        device: Device for tensor operations

    Returns:
        FaithfulnessMetrics with observational, interventional, and counterfactual metrics
    """
    if config is None:
        config = FaithfulnessConfig()

    n_interventions_per_patch = config.n_interventions_per_patch

    # ===== OBSERVATIONAL: Input perturbation analysis =====
    observational = None
    if not config.skip_observational:
        with traced("observational_metrics"):
            observational = calculate_observational_metrics(
                subcircuit=subcircuit,
                full_model=model,
                n_samples_per_base=200,
                device=device,
            )
    else:
        print("[SKIP] Observational analysis")

    # ===== Helper: Build CounterfactualEffect from intervention =====
    def _build_effect(
        pair: CleanCorruptedPair,
        y_intervened: torch.Tensor,
        intervened_acts: list[torch.Tensor],
        experiment_type: str,
        score_type: str,
    ) -> CounterfactualEffect:
        """Build a CounterfactualEffect from intervention results."""
        # Convert logits to bits for scoring (user requested bit comparison, not logit)
        y_clean_bit = logits_to_binary(pair.y_clean).mean().item()
        y_corrupted_bit = logits_to_binary(pair.y_corrupted).mean().item()
        y_intervened_bit = logits_to_binary(y_intervened).mean().item()

        # Compute the appropriate score based on experiment type (using bits)
        if score_type == "sufficiency":
            # Denoise in-circuit: recovery
            faith_score = compute_sufficiency_score(
                y_intervened_bit, y_clean_bit, y_corrupted_bit
            )
        elif score_type == "completeness":
            # Denoise out-circuit: 1 - recovery = disruption
            faith_score = compute_completeness_score(
                y_intervened_bit, y_clean_bit, y_corrupted_bit
            )
        elif score_type == "necessity":
            # Noise in-circuit: disruption
            faith_score = compute_necessity_score(
                y_intervened_bit, y_clean_bit, y_corrupted_bit
            )
        elif score_type == "independence":
            # Noise out-circuit: 1 - disruption = recovery
            faith_score = compute_independence_score(
                y_intervened_bit, y_clean_bit, y_corrupted_bit
            )
        else:
            raise ValueError(f"Unknown score_type: {score_type}")

        # Check if output bit matches corrupted bit
        output_changed = y_intervened_bit == y_corrupted_bit

        # LAZY SERIALIZATION: Store tensors directly, convert to lists in to_dict()
        # Move to CPU but don't call .tolist() - that happens lazily during serialization
        clean_acts = [a.squeeze(0).cpu() for a in pair.act_clean]  # tensors
        corrupted_acts = [a.squeeze(0).cpu() for a in pair.act_corrupted]  # tensors
        intervened_acts_cpu = [a.squeeze(0).cpu() for a in intervened_acts]  # tensors

        return CounterfactualEffect(
            faithfulness_score=faith_score,
            experiment_type=experiment_type,
            score_type=score_type,
            clean_input=pair.x_clean.cpu().flatten(),  # tensor, converted lazily
            corrupted_input=pair.x_corrupted.cpu().flatten(),  # tensor, converted lazily
            expected_clean_output=y_clean_bit,
            expected_corrupted_output=y_corrupted_bit,
            actual_output=y_intervened_bit,
            output_changed_to_corrupted=output_changed,
            clean_activations=clean_acts,  # tensors, converted lazily
            corrupted_activations=corrupted_acts,  # tensors, converted lazily
            intervened_activations=intervened_acts_cpu,  # tensors, converted lazily
        )

    # ===== BATCHED NOISING: Run clean inputs, patch with corrupted activations =====
    def compute_noising_effects_batched(
        patches: list[PatchShape],
        pairs: list[CleanCorruptedPair],
        score_type: str,  # "necessity" (in-circuit) or "independence" (out-circuit)
    ) -> list[CounterfactualEffect]:
        """Noising: Run on CLEAN inputs, patch with CORRUPTED values (BATCHED).

        - Necessity (in-circuit): Does corrupting the circuit break behavior?
        - Independence (out-circuit): Does corrupting outside the circuit break behavior?

        Uses batched forward pass for ~10x speedup over sequential execution.
        """
        if not pairs or not patches:
            return []

        n_pairs = len(pairs)

        # Stack all clean inputs: [N, input_dim]
        x_batch = torch.cat([pair.x_clean for pair in pairs], dim=0)

        # Stack corrupted activations for batched intervention: [N, hidden] per layer
        n_layers = len(pairs[0].act_corrupted)
        batched_corrupted_acts = [
            torch.cat([pair.act_corrupted[layer] for pair in pairs], dim=0)
            for layer in range(n_layers)
        ]

        # Create batched intervention with per-sample values
        iv = create_batched_patch_intervention(patches, batched_corrupted_acts)

        # Single batched forward pass
        with torch.inference_mode():
            intervened_acts_batch = model(x_batch, intervention=iv, return_activations=True)
            y_intervened_batch = intervened_acts_batch[-1]  # [N, 1] or [N]

        # Move to CPU before per-pair iteration (avoids MPS->CPU transfer per item)
        intervened_acts_cpu = [a.cpu() for a in intervened_acts_batch]
        y_intervened_cpu = y_intervened_batch.cpu()

        # Build per-pair effects from batched results
        effects = []
        for i, pair in enumerate(pairs):
            y_intervened = y_intervened_cpu[i:i+1]  # [1, ...]
            intervened_acts = [a[i:i+1] for a in intervened_acts_cpu]

            effects.append(
                _build_effect(pair, y_intervened, intervened_acts, "noising", score_type)
            )

        return effects

    # ===== BATCHED DENOISING: Run corrupted inputs, patch with clean activations =====
    def compute_denoising_effects_batched(
        patches: list[PatchShape],
        pairs: list[CleanCorruptedPair],
        score_type: str,  # "sufficiency" (in-circuit) or "completeness" (out-circuit)
    ) -> list[CounterfactualEffect]:
        """Denoising: Run on CORRUPTED inputs, patch with CLEAN values (BATCHED).

        - Sufficiency (in-circuit): Can the circuit alone recover the behavior?
        - Completeness (out-circuit): Does patching outside the circuit help recover?

        Uses batched forward pass for ~10x speedup over sequential execution.
        """
        if not pairs or not patches:
            return []

        n_pairs = len(pairs)

        # Stack all corrupted inputs: [N, input_dim]
        x_batch = torch.cat([pair.x_corrupted for pair in pairs], dim=0)

        # Stack clean activations for batched intervention: [N, hidden] per layer
        n_layers = len(pairs[0].act_clean)
        batched_clean_acts = [
            torch.cat([pair.act_clean[layer] for pair in pairs], dim=0)
            for layer in range(n_layers)
        ]

        # Create batched intervention with per-sample values
        iv = create_batched_patch_intervention(patches, batched_clean_acts)

        # Single batched forward pass
        with torch.inference_mode():
            intervened_acts_batch = model(x_batch, intervention=iv, return_activations=True)
            y_intervened_batch = intervened_acts_batch[-1]  # [N, 1] or [N]

        # Move to CPU before per-pair iteration (avoids MPS->CPU transfer per item)
        intervened_acts_cpu = [a.cpu() for a in intervened_acts_batch]
        y_intervened_cpu = y_intervened_batch.cpu()

        # Build per-pair effects from batched results
        effects = []
        for i, pair in enumerate(pairs):
            y_intervened = y_intervened_cpu[i:i+1]  # [1, ...]
            intervened_acts = [a[i:i+1] for a in intervened_acts_cpu]

            effects.append(
                _build_effect(pair, y_intervened, intervened_acts, "denoising", score_type)
            )

        return effects

    # ===== Interventional Analysis (random value patching) =====
    in_circuit_effects = CircuitInterventionEffects()
    out_circuit_effects = CircuitInterventionEffects()

    if not config.skip_interventional:
        # In-distribution: small perturbations within training data range
        in_distribution_value_range = [-0.25, 0.25]
        # Out-of-distribution: large values far outside training range
        out_distribution_value_range = [[-1000, -2], [2, 1000]]

        # Interventional analysis (sequential for GPU safety)
        with traced("interventional_in_circuit", n_patches=len(structure.in_patches) if structure.in_patches else 0):
            if structure.in_patches:
                in_circuit_effects.in_distribution = calculate_patches_causal_effect(
                    structure.in_patches,
                    x,
                    model,
                    subcircuit,
                    n_interventions_per_patch,
                    False,
                    device,
                    in_distribution_value_range,
                )
                in_circuit_effects.out_distribution = calculate_patches_causal_effect(
                    structure.in_patches,
                    x,
                    model,
                    subcircuit,
                    n_interventions_per_patch,
                    False,
                    device,
                    out_distribution_value_range,
                )

        with traced("interventional_out_circuit", n_patches=len(structure.out_patches) if structure.out_patches else 0):
            if structure.out_patches:
                out_circuit_effects.in_distribution = calculate_patches_causal_effect(
                    structure.out_patches,
                    x,
                    model,
                    subcircuit,
                    n_interventions_per_patch,
                    True,
                    device,
                    in_distribution_value_range,
                )
                out_circuit_effects.out_distribution = calculate_patches_causal_effect(
                    structure.out_patches,
                    x,
                    model,
                    subcircuit,
                    n_interventions_per_patch,
                    True,
                    device,
                    out_distribution_value_range,
                )
    else:
        print("[SKIP] Interventional analysis")

    # ===== 2x2 Counterfactual Analysis (BATCHED) =====
    sufficiency_effects = []
    completeness_effects = []
    necessity_effects = []
    independence_effects = []

    if not config.skip_counterfactual:
        with traced("counterfactual_2x2_batched", n_pairs=len(counterfactual_pairs)):
            # Denoising experiments (run corrupted, patch with clean)
            sufficiency_effects = compute_denoising_effects_batched(
                structure.in_circuit, counterfactual_pairs, "sufficiency"
            )
            completeness_effects = compute_denoising_effects_batched(
                structure.out_circuit, counterfactual_pairs, "completeness"
            )

            # Noising experiments (run clean, patch with corrupted)
            necessity_effects = compute_noising_effects_batched(
                structure.in_circuit, counterfactual_pairs, "necessity"
            )
            independence_effects = compute_noising_effects_batched(
                structure.out_circuit, counterfactual_pairs, "independence"
            )
    else:
        print("[SKIP] Counterfactual analysis")

    # ===== Calculate Statistics =====
    all_counterfactual_effects = (
        sufficiency_effects
        + completeness_effects
        + necessity_effects
        + independence_effects
    )

    stats = calculate_statistics(
        in_circuit_effects, out_circuit_effects, all_counterfactual_effects, x=x
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

    # Overall counterfactual: average of all 4 scores
    overall_counterfactual = (
        mean_sufficiency + mean_completeness + mean_necessity + mean_independence
    ) / 4.0

    # Build nested metrics with availability-aware averaging
    # Only average components that have data (n > 0)
    component_scores = {
        "in_circuit_id": stats.mean_in_sim if stats.n_in_sim > 0 else None,
        "out_circuit_id": stats.mean_out_sim if stats.n_out_sim > 0 else None,
        "in_circuit_ood": stats.mean_in_sim_ood if stats.n_in_sim_ood > 0 else None,
        "out_circuit_ood": stats.mean_out_sim_ood if stats.n_out_sim_ood > 0 else None,
    }
    component_n = {
        "in_circuit_id": stats.n_in_sim,
        "out_circuit_id": stats.n_out_sim,
        "in_circuit_ood": stats.n_in_sim_ood,
        "out_circuit_ood": stats.n_out_sim_ood,
    }

    # Calculate overall from available components only
    available_scores = [s for s in component_scores.values() if s is not None]
    n_components_averaged = len(available_scores)
    overall_interventional = (
        float(np.mean(available_scores)) if available_scores else 0.0
    )

    interventional = InterventionalMetrics(
        in_circuit_stats=stats.in_circuit_stats,
        out_circuit_stats=stats.out_circuit_stats,
        in_circuit_stats_ood=stats.in_circuit_stats_ood,
        out_circuit_stats_ood=stats.out_circuit_stats_ood,
        mean_in_circuit_similarity=stats.mean_in_sim,
        mean_out_circuit_similarity=stats.mean_out_sim,
        mean_in_circuit_similarity_ood=stats.mean_in_sim_ood,
        mean_out_circuit_similarity_ood=stats.mean_out_sim_ood,
        overall_interventional=overall_interventional,
        component_scores=component_scores,
        component_n=component_n,
        n_components_averaged=n_components_averaged,
    )

    counterfactual = CounterfactualMetrics(
        sufficiency_effects=sufficiency_effects,
        completeness_effects=completeness_effects,
        necessity_effects=necessity_effects,
        independence_effects=independence_effects,
        mean_sufficiency=mean_sufficiency,
        mean_completeness=mean_completeness,
        mean_necessity=mean_necessity,
        mean_independence=mean_independence,
        overall_counterfactual=overall_counterfactual,
    )

    # Overall faithfulness: average of non-skipped pillars only
    pillar_scores = []
    if not config.skip_observational and observational is not None:
        pillar_scores.append(observational.overall_observational)
    if not config.skip_interventional:
        pillar_scores.append(interventional.overall_interventional)
    if not config.skip_counterfactual:
        pillar_scores.append(overall_counterfactual)

    overall_faithfulness = float(np.mean(pillar_scores)) if pillar_scores else 0.0

    # Create default observational if skipped (for consistent return structure)
    if observational is None:
        observational = ObservationalMetrics()

    return FaithfulnessMetrics(
        observational=observational,
        interventional=interventional,
        counterfactual=counterfactual,
        overall_faithfulness=overall_faithfulness,
    )

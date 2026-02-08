"""Metrics calculation functions for faithfulness analysis.

Combines observational, interventional, and counterfactual analysis to compute
comprehensive faithfulness metrics for subcircuits.
"""

import numpy as np
import torch

from src.circuit import CircuitStructure
from src.model import InterventionEffect, MLP, PatchShape
from src.schemas import (
    CounterfactualEffect,
    CounterfactualMetrics,
    FaithfulnessConfig,
    FaithfulnessMetrics,
    InterventionalMetrics,
)
from src.tensor_ops import logits_to_binary

from .counterfactual import CleanCorruptedPair, create_patch_intervention
from .interventional import _compute_patch_statistics, calculate_patches_causal_effect
from .observational import calculate_observational_metrics
from .scoring import (
    compute_completeness_score,
    compute_independence_score,
    compute_necessity_score,
    compute_sufficiency_score,
)


def calculate_statistics(
    in_circuit_effects: dict[str, dict[str, list[InterventionEffect]]],
    out_circuit_effects: dict[str, dict[str, list[InterventionEffect]]],
    counterfactual_effects: list[CounterfactualEffect],
) -> dict:
    """
    Compute statistics from intervention effects.

    Args:
        in_circuit_effects: Dict with "in"/"ood" keys containing patch dicts
        out_circuit_effects: Same format as in_circuit_effects
        counterfactual_effects: List of CounterfactualEffect

    Returns:
        Dict with keys: in_circuit_stats, out_circuit_stats, in_circuit_stats_ood, out_circuit_stats_ood,
                       mean_in_sim, mean_out_sim, mean_in_sim_ood, mean_out_sim_ood,
                       mean_faith, std_faith
    """
    in_stats, in_sims = _compute_patch_statistics(in_circuit_effects.get("in", {}))
    in_stats_ood, in_sims_ood = _compute_patch_statistics(
        in_circuit_effects.get("ood", {})
    )
    out_stats, out_sims = _compute_patch_statistics(out_circuit_effects.get("in", {}))
    out_stats_ood, out_sims_ood = _compute_patch_statistics(
        out_circuit_effects.get("ood", {})
    )

    mean_in_sim = float(np.mean(in_sims)) if in_sims else 0.0
    mean_out_sim = float(np.mean(out_sims)) if out_sims else 0.0
    mean_in_sim_ood = float(np.mean(in_sims_ood)) if in_sims_ood else 0.0
    mean_out_sim_ood = float(np.mean(out_sims_ood)) if out_sims_ood else 0.0

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
    observational = calculate_observational_metrics(
        subcircuit=subcircuit,
        full_model=model,
        n_samples_per_base=200,
        device=device,
    )

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

        output_changed = (
            logits_to_binary(torch.tensor(y_intervened_val)).item()
            == logits_to_binary(torch.tensor(y_corrupted_val)).item()
        )

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
                _build_effect(
                    pair, y_intervened, intervened_acts, "noising", score_type
                )
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
            out_distribution_value_range,
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

    # Overall counterfactual: average of all 4 scores
    overall_counterfactual = (
        mean_sufficiency + mean_completeness + mean_necessity + mean_independence
    ) / 4.0

    # Build nested metrics
    interventional = InterventionalMetrics(
        in_circuit_stats=stats["in_circuit_stats"],
        out_circuit_stats=stats["out_circuit_stats"],
        in_circuit_stats_ood=stats["in_circuit_stats_ood"],
        out_circuit_stats_ood=stats["out_circuit_stats_ood"],
        mean_in_circuit_similarity=stats["mean_in_sim"],
        mean_out_circuit_similarity=stats["mean_out_sim"],
        mean_in_circuit_similarity_ood=stats["mean_in_sim_ood"],
        mean_out_circuit_similarity_ood=stats["mean_out_sim_ood"],
        overall_interventional=(stats["mean_in_sim"] + stats["mean_out_sim"]) / 2.0,
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

    # Overall faithfulness: average of observational, interventional, and counterfactual
    overall_faithfulness = (
        observational.overall_observational
        + interventional.overall_interventional
        + overall_counterfactual
    ) / 3.0

    return FaithfulnessMetrics(
        observational=observational,
        interventional=interventional,
        counterfactual=counterfactual,
        overall_faithfulness=overall_faithfulness,
    )

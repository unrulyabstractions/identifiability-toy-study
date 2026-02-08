"""Faithfulness-related schema classes.

Contains all faithfulness analysis dataclasses:
- PatchStatistics: Statistics for a single patch's intervention effects
- CounterfactualEffect: Result of a single counterfactual test (2x2 matrix)
- ObservationalMetrics: Detailed observational robustness data + summary
- InterventionalMetrics: Detailed interventional patching data
- CounterfactualMetrics: Detailed counterfactual effects data
- FaithfulnessMetrics: Comprehensive faithfulness metrics (combines all three)
- ObservationalSummary: Aggregated observational summary for result.json
- InterventionalSummary: Aggregated interventional summary for result.json
- CounterfactualSummary: Aggregated counterfactual summary for result.json
- FaithfulnessCategoryScore: Score and epsilon for a single category
- FaithfulnessSummary: Summary of all faithfulness metrics
"""

from dataclasses import dataclass, field

from .samples import CounterfactualSample, InterventionSample, RobustnessSample
from .schema_class import SchemaClass

# Alias for naming consistency with InterventionSample, ObservationalSample
CounterfactualEffect = CounterfactualSample

# =============================================================================
# Basic Building Blocks
# =============================================================================


@dataclass
class PatchStatistics(SchemaClass):
    """Statistics for a single patch's intervention effects."""

    mean_logit_similarity: float = 0.0
    std_logit_similarity: float = 0.0
    mean_bit_similarity: float = 0.0
    std_bit_similarity: float = 0.0
    mean_best_similarity: float = 0.0
    std_best_similarity: float = 0.0
    n_interventions: int = 0

    # Individual samples for visualization (optional, may be large)
    samples: list[InterventionSample] = field(default_factory=list)


# =============================================================================
# Detailed Metrics Classes (contain samples and full data)
# =============================================================================


@dataclass
class ObservationalMetrics(SchemaClass):
    """Observational robustness metrics for a subcircuit.

    This is part of observational faithfulness analysis. Measures how well
    subcircuit predictions match full model predictions under perturbations.
    """

    # All samples (for scatter plots by actual noise magnitude)
    noise_samples: list[RobustnessSample] = field(default_factory=list)
    ood_samples: list[RobustnessSample] = field(default_factory=list)

    # Noise perturbation metrics
    noise_gate_accuracy: float = 0.0
    noise_subcircuit_accuracy: float = 0.0
    noise_agreement_bit: float = 0.0
    noise_agreement_best: float = 0.0
    noise_mse_mean: float = 0.0
    noise_n_samples: int = 0

    # Aggregate OOD metrics
    ood_gate_accuracy: float = 0.0
    ood_subcircuit_accuracy: float = 0.0
    ood_agreement_bit: float = 0.0
    ood_agreement_best: float = 0.0
    ood_mse_mean: float = 0.0

    # Per-type OOD metrics
    multiply_positive_agreement: float = 0.0
    multiply_positive_n_samples: int = 0
    multiply_negative_agreement: float = 0.0
    multiply_negative_n_samples: int = 0

    add_agreement: float = 0.0
    add_n_samples: int = 0

    subtract_agreement: float = 0.0
    subtract_n_samples: int = 0

    bimodal_agreement: float = 0.0
    bimodal_n_samples: int = 0
    bimodal_inv_agreement: float = 0.0
    bimodal_inv_n_samples: int = 0

    # Overall
    overall_observational: float = 0.0


@dataclass
class InterventionalMetrics(SchemaClass):
    """Detailed interventional metrics for a subcircuit.

    Contains per-patch statistics for in-circuit and out-circuit interventions.
    """

    # Per-patch statistics for in-circuit and out-circuit interventions
    in_circuit_stats: dict[str, PatchStatistics] = field(default_factory=dict)
    out_circuit_stats: dict[str, PatchStatistics] = field(default_factory=dict)

    # OOD (out-of-distribution) intervention statistics
    in_circuit_stats_ood: dict[str, PatchStatistics] = field(default_factory=dict)
    out_circuit_stats_ood: dict[str, PatchStatistics] = field(default_factory=dict)

    # Aggregate statistics (in-distribution)
    mean_in_circuit_similarity: float = 0.0
    mean_out_circuit_similarity: float = 0.0

    # Aggregate statistics (out-of-distribution)
    mean_in_circuit_similarity_ood: float = 0.0
    mean_out_circuit_similarity_ood: float = 0.0

    # Overall
    overall_interventional: float = 0.0


@dataclass
class CounterfactualMetrics(SchemaClass):
    """Detailed counterfactual metrics for a subcircuit.

    Contains the 2x2 patching matrix effects (sufficiency, completeness, necessity, independence).
    """

    # Denoising experiments (run corrupted, patch with clean)
    sufficiency_effects: list[CounterfactualEffect] = field(
        default_factory=list
    )  # Denoise in-circuit
    completeness_effects: list[CounterfactualEffect] = field(
        default_factory=list
    )  # Denoise out-circuit

    # Noising experiments (run clean, patch with corrupted)
    necessity_effects: list[CounterfactualEffect] = field(
        default_factory=list
    )  # Noise in-circuit
    independence_effects: list[CounterfactualEffect] = field(
        default_factory=list
    )  # Noise out-circuit

    # Aggregate Scores (2x2 Matrix)
    mean_sufficiency: float = 0.0  # Denoise in-circuit: recovery
    mean_completeness: float = 0.0  # Denoise out-circuit: 1 - recovery
    mean_necessity: float = 0.0  # Noise in-circuit: disruption
    mean_independence: float = 0.0  # Noise out-circuit: 1 - disruption

    # Overall
    overall_counterfactual: float = 0.0


@dataclass
class FaithfulnessMetrics(SchemaClass):
    """Comprehensive faithfulness metrics for a subcircuit.

    Contains all three categories of faithfulness:
    - Observational: robustness under input perturbations
    - Interventional: activation patching effects
    - Counterfactual: 2x2 patching matrix (sufficiency, completeness, necessity, independence)

    A faithful circuit should score high on all tests.
    """

    observational: ObservationalMetrics | None = None
    interventional: InterventionalMetrics | None = None
    counterfactual: CounterfactualMetrics | None = None

    # Overall faithfulness score (higher is better)
    overall_faithfulness: float = 0.0


# =============================================================================
# Summary Classes (for result.json, aggregated stats only)
# =============================================================================


@dataclass
class ObservationalSummary(SchemaClass):
    """Aggregated observational summary for result.json."""

    # Noise perturbation metrics
    noise_gate_accuracy: float = 0.0
    noise_subcircuit_accuracy: float = 0.0
    noise_agreement_bit: float = 0.0
    noise_agreement_best: float = 0.0
    noise_mse_mean: float = 0.0
    noise_n_samples: int = 0

    # Per-type OOD metrics
    multiply_positive_agreement: float = 0.0
    multiply_positive_n_samples: int = 0
    multiply_negative_agreement: float = 0.0
    multiply_negative_n_samples: int = 0

    add_agreement: float = 0.0
    add_n_samples: int = 0

    subtract_agreement: float = 0.0
    subtract_n_samples: int = 0

    bimodal_agreement: float = 0.0
    bimodal_n_samples: int = 0
    bimodal_inv_agreement: float = 0.0
    bimodal_inv_n_samples: int = 0

    # Overall
    overall_observational: float = 0.0


@dataclass
class InterventionalSummary(SchemaClass):
    """Aggregated interventional summary for result.json."""

    # In-circuit (in-distribution)
    in_circuit_mean_bit_similarity: float = 0.0
    in_circuit_mean_logit_similarity: float = 0.0
    in_circuit_n_interventions: int = 0

    # In-circuit (out-of-distribution)
    in_circuit_ood_mean_bit_similarity: float = 0.0
    in_circuit_ood_mean_logit_similarity: float = 0.0
    in_circuit_ood_n_interventions: int = 0

    # Out-circuit (in-distribution)
    out_circuit_mean_bit_similarity: float = 0.0
    out_circuit_mean_logit_similarity: float = 0.0
    out_circuit_n_interventions: int = 0

    # Out-circuit (out-of-distribution)
    out_circuit_ood_mean_bit_similarity: float = 0.0
    out_circuit_ood_mean_logit_similarity: float = 0.0
    out_circuit_ood_n_interventions: int = 0

    # Overall
    overall_interventional: float = 0.0


@dataclass
class CounterfactualSummary(SchemaClass):
    """Aggregated counterfactual summary for result.json (2x2 matrix)."""

    # Denoising experiments
    mean_sufficiency: float = 0.0  # Denoise in-circuit
    mean_completeness: float = 0.0  # Denoise out-circuit
    n_denoising_pairs: int = 0

    # Noising experiments
    mean_necessity: float = 0.0  # Noise in-circuit
    mean_independence: float = 0.0  # Noise out-circuit
    n_noising_pairs: int = 0

    # Overall
    overall_counterfactual: float = 0.0


@dataclass
class FaithfulnessCategoryScore(SchemaClass):
    """Score and epsilon for a single faithfulness category."""

    score: float = 0.0
    epsilon: float = 0.0  # Similarity threshold used in identifiability constraints


@dataclass
class FaithfulnessSummary(SchemaClass):
    """Summary of all faithfulness metrics for summary.json.

    Each category has:
    - score: Average of component scores
    - epsilon: Minimum margin from 1.0 across component scores (always positive)
    """

    observational: FaithfulnessCategoryScore = field(
        default_factory=FaithfulnessCategoryScore
    )
    interventional: FaithfulnessCategoryScore = field(
        default_factory=FaithfulnessCategoryScore
    )
    counterfactual: FaithfulnessCategoryScore = field(
        default_factory=FaithfulnessCategoryScore
    )
    overall: float = 0.0  # Combined overall score

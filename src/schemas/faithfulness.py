"""Faithfulness-related schema classes."""

from dataclasses import dataclass, field

from .samples import CounterfactualSample, InterventionalSample, ObservationalSample
from src.schema_class import SchemaClass

# Aliases for backwards compatibility
CounterfactualEffect = CounterfactualSample
InterventionalEffect = InterventionalSample
ObservationalEffect = ObservationalSample

# =============================================================================
# Basic Building Blocks
# =============================================================================


@dataclass
class Similarity(SchemaClass):
    """Similarity metrics between two outputs.

    Three ways to measure similarity:
    - bit: Binary agreement (round(a) == round(b))
    - logit: 1 - MSE between raw logits
    - best: Agreement after clamping to [0,1]
    """

    bit: float = 0.0
    logit: float = 0.0
    best: float = 0.0


@dataclass
class PatchStatistics(SchemaClass):
    """Statistics for a single patch's intervention effects."""

    mean: Similarity = field(default_factory=Similarity)
    std: Similarity = field(default_factory=Similarity)
    n_samples: int = 0

    # Individual samples for visualization (optional, may be large)
    samples: list[InterventionalSample] = field(default_factory=list)


# =============================================================================
# Observational Metrics Components
# =============================================================================


@dataclass
class NoiseRobustnessMetrics(SchemaClass):
    """Metrics from Gaussian noise perturbation tests."""

    samples: list[ObservationalSample] = field(default_factory=list)
    gate_accuracy: float = 0.0
    subcircuit_accuracy: float = 0.0
    similarity: Similarity = field(default_factory=Similarity)
    n_samples: int = 0


@dataclass
class OutOfDistributionMetrics(SchemaClass):
    """Metrics from out-of-distribution transformation tests."""

    samples: list[ObservationalSample] = field(default_factory=list)

    # Aggregate OOD metrics
    gate_accuracy: float = 0.0
    subcircuit_accuracy: float = 0.0
    similarity: Similarity = field(default_factory=Similarity)

    # Per-type metrics (each has similarity.bit as agreement)
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


# =============================================================================
# Detailed Metrics Classes (contain samples and full data)
# =============================================================================


@dataclass
class ObservationalMetrics(SchemaClass):
    """Observational robustness metrics for a subcircuit.

    Contains noise perturbation and OOD transformation results.
    """

    noise: NoiseRobustnessMetrics | None = None
    ood: OutOfDistributionMetrics | None = None
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
    sufficiency_effects: list[CounterfactualSample] = field(
        default_factory=list
    )  # Denoise in-circuit
    completeness_effects: list[CounterfactualSample] = field(
        default_factory=list
    )  # Denoise out-circuit

    # Noising experiments (run clean, patch with corrupted)
    necessity_effects: list[CounterfactualSample] = field(
        default_factory=list
    )  # Noise in-circuit
    independence_effects: list[CounterfactualSample] = field(
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
class NoiseRobustnessSummary(SchemaClass):
    """Summary of noise robustness metrics for result.json."""

    gate_accuracy: float = 0.0
    subcircuit_accuracy: float = 0.0
    similarity: Similarity = field(default_factory=Similarity)
    n_samples: int = 0


@dataclass
class OutOfDistributionSummary(SchemaClass):
    """Summary of OOD metrics for result.json."""

    # Per-type metrics
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


@dataclass
class ObservationalSummary(SchemaClass):
    """Aggregated observational summary for result.json."""

    noise: NoiseRobustnessSummary = field(default_factory=NoiseRobustnessSummary)
    ood: OutOfDistributionSummary = field(default_factory=OutOfDistributionSummary)
    overall_observational: float = 0.0


@dataclass
class InterventionalSummary(SchemaClass):
    """Aggregated interventional summary for result.json."""

    # In-circuit (in-distribution)
    in_circuit_mean: Similarity = field(default_factory=Similarity)
    in_circuit_n: int = 0

    # In-circuit (out-of-distribution)
    in_circuit_ood_mean: Similarity = field(default_factory=Similarity)
    in_circuit_ood_n: int = 0

    # Out-circuit (in-distribution)
    out_circuit_mean: Similarity = field(default_factory=Similarity)
    out_circuit_n: int = 0

    # Out-circuit (out-of-distribution)
    out_circuit_ood_mean: Similarity = field(default_factory=Similarity)
    out_circuit_ood_n: int = 0

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


# =============================================================================
# Intermediate Data Structures (used during computation)
# =============================================================================


@dataclass
class CircuitInterventionEffects(SchemaClass):
    """Intervention effects for a circuit (in or out of circuit).

    Groups effects by distribution type (in vs out of distribution).
    Each dict maps patch keys to lists of intervention effects.
    """

    in_distribution: dict[str, list[InterventionalSample]] = field(default_factory=dict)
    out_distribution: dict[str, list[InterventionalSample]] = field(default_factory=dict)


@dataclass
class InterventionStatistics(SchemaClass):
    """Statistics computed from intervention effects.

    This replaces the dict return type of calculate_statistics().
    """

    in_circuit_stats: dict[str, PatchStatistics] = field(default_factory=dict)
    out_circuit_stats: dict[str, PatchStatistics] = field(default_factory=dict)
    in_circuit_stats_ood: dict[str, PatchStatistics] = field(default_factory=dict)
    out_circuit_stats_ood: dict[str, PatchStatistics] = field(default_factory=dict)
    mean_in_sim: float = 0.0
    mean_out_sim: float = 0.0
    mean_in_sim_ood: float = 0.0
    mean_out_sim_ood: float = 0.0
    mean_faith: float = 0.0
    std_faith: float = 0.0


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

"""Faithfulness-related schema classes.

Contains all faithfulness analysis dataclasses:
- PatchStatistics: Statistics for a single patch's intervention effects
- CounterfactualEffect: Result of a single counterfactual test (2x2 matrix)
- FaithfulnessMetrics: Comprehensive faithfulness metrics for a subcircuit
- RobustnessMetrics: Observational robustness metrics (input perturbations)
- ObservationalMetrics: Aggregated observational metrics for result.json
- InterventionalMetrics: Aggregated interventional metrics for result.json
- CounterfactualMetrics: Aggregated counterfactual metrics for result.json
- FaithfulnessCategoryScore: Score and epsilon for a single category
- FaithfulnessSummary: Summary of all faithfulness metrics
"""

from dataclasses import dataclass, field

from .schema_class import SchemaClass
from .samples import InterventionSample, RobustnessSample


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


@dataclass
class CounterfactualEffect(SchemaClass):
    """Result of a single counterfactual test from the 2x2 patching matrix.

    The 2x2 matrix tests circuit faithfulness:

    |                | IN-Circuit Patch     | OUT-Circuit Patch      |
    |----------------|----------------------|------------------------|
    | DENOISING      | Sufficiency          | Completeness           |
    | (run corrupt,  | (recovery)           | (1 - recovery)         |
    | patch clean)   |                      |                        |
    |----------------|----------------------|------------------------|
    | NOISING        | Necessity            | Independence           |
    | (run clean,    | (disruption)         | (1 - disruption)       |
    | patch corrupt) |                      |                        |

    NOTE: Activations are pre-computed here so visualization code
    NEVER needs to run models. Visualization is READ-ONLY.
    """

    faithfulness_score: float  # Score depends on score_type

    # Experiment type: which direction are we patching?
    # - "denoising": Run corrupted input, patch with clean activations (Src: clean, Dest: corrupt)
    # - "noising": Run clean input, patch with corrupted activations (Src: corrupt, Dest: clean)
    experiment_type: str = "noising"

    # Score type: which of the 4 experiments?
    # - "sufficiency": Denoise in-circuit -> tests if circuit can produce behavior
    # - "completeness": Denoise out-circuit -> tests if anything is missing from circuit
    # - "necessity": Noise in-circuit -> tests if circuit is required
    # - "independence": Noise out-circuit -> tests if circuit is self-contained
    score_type: str = "necessity"

    # Clean/corrupted input info
    clean_input: list[float] = field(default_factory=list)  # e.g., [0, 1]
    corrupted_input: list[float] = field(default_factory=list)  # e.g., [1, 0]

    # Expected outputs (from original clean/corrupted runs, no intervention)
    expected_clean_output: float = 0.0  # y_clean (full model on clean input)
    expected_corrupted_output: float = (
        0.0  # y_corrupted (full model on corrupted input)
    )

    # Actual output from FULL MODEL with intervention (patched activations)
    actual_output: float = 0.0  # model(x_base, intervention=patch)

    # Did patching change output? (interpretation depends on experiment type)
    output_changed_to_corrupted: bool = False  # round(actual) == round(corrupted)

    # Pre-computed activations for visualization (NO model runs during viz!)
    # These are from the ORIGINAL clean/corrupted runs (reference)
    clean_activations: list[list[float]] = field(default_factory=list)
    corrupted_activations: list[list[float]] = field(default_factory=list)

    # Activations from the actual intervention run (FULL MODEL with patches)
    # This is what the visualization should show for the counterfactual
    intervened_activations: list[list[float]] = field(default_factory=list)


@dataclass
class FaithfulnessMetrics(SchemaClass):
    """Comprehensive faithfulness metrics for a subcircuit.

    Implements the 2x2 patching matrix:

    |                | IN-Circuit Patch     | OUT-Circuit Patch      |
    |----------------|----------------------|------------------------|
    | DENOISING      | Sufficiency          | Completeness           |
    | (run corrupt,  | (recovery -> 1)      | (disruption -> 1)      |
    | patch clean)   |                      |                        |
    |----------------|----------------------|------------------------|
    | NOISING        | Necessity            | Independence           |
    | (run clean,    | (disruption -> 1)    | (recovery -> 1)        |
    | patch corrupt) |                      |                        |

    A faithful circuit should score high on all 4 tests.
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

    # ===== 2x2 Matrix Counterfactual Effects =====
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

    # ===== Aggregate Scores (2x2 Matrix) =====
    mean_sufficiency: float = 0.0  # Denoise in-circuit: recovery
    mean_completeness: float = 0.0  # Denoise out-circuit: 1 - recovery
    mean_necessity: float = 0.0  # Noise in-circuit: disruption
    mean_independence: float = 0.0  # Noise out-circuit: 1 - disruption

    # Overall faithfulness score (higher is better)
    overall_faithfulness: float = 0.0


@dataclass
class RobustnessMetrics(SchemaClass):
    """Observational robustness metrics for a subcircuit.

    This is part of observational faithfulness analysis (output stored in
    faithfulness/observational/). Measures how well subcircuit predictions
    match full model predictions under perturbations.

    See also: FaithfulnessMetrics for the complete faithfulness evaluation.
    """

    # All samples (for scatter plots by actual noise magnitude)
    noise_samples: list[RobustnessSample] = field(default_factory=list)
    ood_samples: list[RobustnessSample] = field(default_factory=list)

    # Aggregate statistics
    noise_gate_accuracy: float = 0.0
    noise_subcircuit_accuracy: float = 0.0
    noise_agreement_bit: float = 0.0
    noise_agreement_best: float = 0.0
    noise_mse_mean: float = 0.0

    ood_gate_accuracy: float = 0.0
    ood_subcircuit_accuracy: float = 0.0
    ood_agreement_bit: float = 0.0
    ood_agreement_best: float = 0.0
    ood_mse_mean: float = 0.0

    overall_robustness: float = 0.0  # Combined score


@dataclass
class ObservationalMetrics(SchemaClass):
    """Aggregated observational (robustness) metrics for result.json."""

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
class InterventionalMetrics(SchemaClass):
    """Aggregated interventional metrics for result.json."""

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
class CounterfactualMetrics(SchemaClass):
    """Aggregated counterfactual metrics for result.json (2x2 matrix)."""

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


# Alias for naming consistency with InterventionSample, ObservationalSample
CounterfactualSample = CounterfactualEffect

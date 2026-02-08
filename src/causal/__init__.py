"""Causal analysis module for subcircuit evaluation.

This module provides functions for:
- Scoring: Compute faithfulness scores (sufficiency, completeness, necessity, independence)
- Filtering: Filter and select subcircuits based on metrics
- Data Generation: Generate noise and OOD samples for robustness testing
- Observational Analysis: Evaluate model behavior under perturbations
- Interventional Analysis: Measure causal effects of activation patching
- Counterfactual Analysis: Compare clean vs corrupted sample behavior
- Metrics: Comprehensive faithfulness metrics computation

All public functions are re-exported here for convenient imports.
"""

# Scoring functions
from .scoring import (
    compute_recovery,
    compute_disruption,
    compute_sufficiency_score,
    compute_completeness_score,
    compute_necessity_score,
    compute_independence_score,
)

# Filtering functions
from .filtering import (
    _node_masks_key,
    filter_subcircuits,
)

# Data generation functions
from .data_generation import (
    GROUND_TRUTH,
    _generate_noise_samples,
    _generate_ood_multiply_samples,
    _generate_ood_add_samples,
    _generate_ood_subtract_samples,
    _generate_ood_bimodal_samples,
)

# Observational analysis functions
from .observational import (
    _evaluate_samples,
    calculate_observational_metrics,
)

# Interventional analysis functions
from .interventional import (
    calculate_intervention_effect,
    _sample_from_value_range,
    calculate_patches_causal_effect,
    _create_intervention_samples,
    _compute_patch_statistics,
)

# Counterfactual analysis functions
from .counterfactual import (
    CleanCorruptedPair,
    create_clean_corrupted_data,
    create_patch_intervention,
)

# Metrics functions
from .metrics import (
    calculate_statistics,
    calculate_faithfulness_metrics,
)

__all__ = [
    # Scoring
    "compute_recovery",
    "compute_disruption",
    "compute_sufficiency_score",
    "compute_completeness_score",
    "compute_necessity_score",
    "compute_independence_score",
    # Filtering
    "_node_masks_key",
    "filter_subcircuits",
    # Data generation
    "GROUND_TRUTH",
    "_generate_noise_samples",
    "_generate_ood_multiply_samples",
    "_generate_ood_add_samples",
    "_generate_ood_subtract_samples",
    "_generate_ood_bimodal_samples",
    # Observational
    "_evaluate_samples",
    "calculate_observational_metrics",
    # Interventional
    "calculate_intervention_effect",
    "_sample_from_value_range",
    "calculate_patches_causal_effect",
    "_create_intervention_samples",
    "_compute_patch_statistics",
    # Counterfactual
    "CleanCorruptedPair",
    "create_clean_corrupted_data",
    "create_patch_intervention",
    # Metrics
    "calculate_statistics",
    "calculate_faithfulness_metrics",
]

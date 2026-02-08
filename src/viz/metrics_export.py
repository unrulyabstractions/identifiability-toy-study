"""Metrics computation and JSON export.

Contains functions for computing and exporting metrics:
- compute_observational_metrics: Compute observational metrics from robustness data
- compute_interventional_metrics: Compute interventional metrics from faithfulness data
- compute_counterfactual_metrics: Compute counterfactual metrics from faithfulness data
- save_faithfulness_json: Save result.json files and summary.json
- save_gate_summary: Save summary.json for a specific gate
"""

import json
import os
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.schemas import FaithfulnessMetrics, Metrics, RobustnessMetrics

from ..common.schemas import FaithfulnessCategoryScore, FaithfulnessSummary


def compute_observational_metrics(robustness: "RobustnessMetrics") -> dict:
    """Compute observational metrics dict from RobustnessMetrics with nested structure.

    Returns nested dict structure:
    {
        "noise_perturbations": {
            "n_samples": 100,
            "gate_accuracy": 0.95,
            "subcircuit_accuracy": 0.93,
            "agreement_bit": 0.98,
            "agreement_best": 0.99,
            "mse_mean": 0.001
        },
        "out_distribution_transformations": {
            "multiply": {
                "positive": {"n_samples": 50, "agreement": 0.92},
                "negative": {"n_samples": 50, "agreement": 0.88}
            },
            "add": {"n_samples": 100, "agreement": 0.95},
            ...
        },
        "overall_score": 0.91
    }
    """
    # Noise perturbation metrics
    noise_samples = robustness.noise_samples
    noise_metrics = {
        "n_samples": len(noise_samples) if noise_samples else 0,
        "gate_accuracy": robustness.noise_gate_accuracy,
        "subcircuit_accuracy": robustness.noise_subcircuit_accuracy,
        "agreement_bit": robustness.noise_agreement_bit,
        "agreement_best": robustness.noise_agreement_best,
        "mse_mean": robustness.noise_mse_mean,
    }

    # Group OOD samples by type
    ood_by_type: dict[str, list] = {}
    for sample in robustness.ood_samples:
        st = getattr(sample, "sample_type", "multiply_positive")
        ood_by_type.setdefault(st, []).append(sample)

    # Compute per-type metrics
    def compute_ood_metrics(samples):
        n = len(samples)
        agreement = sum(s.agreement_bit for s in samples) / n if n > 0 else 0.0
        return {"n_samples": n, "agreement": agreement}

    multiply_pos = compute_ood_metrics(ood_by_type.get("multiply_positive", []))
    multiply_neg = compute_ood_metrics(ood_by_type.get("multiply_negative", []))
    add_metrics = compute_ood_metrics(ood_by_type.get("add", []))
    subtract_metrics = compute_ood_metrics(ood_by_type.get("subtract", []))
    bimodal_order = compute_ood_metrics(ood_by_type.get("bimodal", []))
    bimodal_inv = compute_ood_metrics(ood_by_type.get("bimodal_inv", []))

    ood_metrics = {
        "multiply": {
            "positive": multiply_pos,
            "negative": multiply_neg,
        },
        "add": add_metrics,
        "subtract": subtract_metrics,
        "bimodal": {
            "order_preserving": bimodal_order,
            "inverted": bimodal_inv,
        },
    }

    # Overall score: average of all agreements
    agreements = [
        noise_metrics["agreement_bit"],
        multiply_pos["agreement"],
        multiply_neg["agreement"],
        add_metrics["agreement"],
        subtract_metrics["agreement"],
        bimodal_order["agreement"],
        bimodal_inv["agreement"],
    ]
    n_valid = sum(1 for a in agreements if a > 0)
    overall = sum(agreements) / n_valid if n_valid > 0 else 0.0

    return {
        "noise_perturbations": noise_metrics,
        "out_distribution_transformations": ood_metrics,
        "overall_score": overall,
    }


def compute_interventional_metrics(faithfulness: "FaithfulnessMetrics") -> dict:
    """Compute InterventionalMetrics from FaithfulnessMetrics with nested structure.

    Returns nested dict structure:
    {
        "in_circuit": {
            "in_distribution": {"n_interventions": 10, "mean_bit_similarity": 0.95, "mean_logit_similarity": 0.92},
            "out_distribution": {"n_interventions": 10, "mean_bit_similarity": 0.88, "mean_logit_similarity": 0.85}
        },
        "out_circuit": {
            "in_distribution": {...},
            "out_distribution": {...}
        },
        "overall_score": 0.90
    }
    """
    def compute_stats(stats_dict):
        if not stats_dict:
            return {"n_interventions": 0, "mean_bit_similarity": 0.0, "mean_logit_similarity": 0.0}
        bit_sims = [ps.mean_bit_similarity for ps in stats_dict.values()]
        logit_sims = [ps.mean_logit_similarity for ps in stats_dict.values()]
        return {
            "n_interventions": len(stats_dict),
            "mean_bit_similarity": sum(bit_sims) / len(bit_sims) if bit_sims else 0.0,
            "mean_logit_similarity": sum(logit_sims) / len(logit_sims) if logit_sims else 0.0,
        }

    in_circuit_id = compute_stats(faithfulness.in_circuit_stats)
    in_circuit_ood = compute_stats(faithfulness.in_circuit_stats_ood)
    out_circuit_id = compute_stats(faithfulness.out_circuit_stats)
    out_circuit_ood = compute_stats(faithfulness.out_circuit_stats_ood)

    # Overall: average of bit similarities
    sims = [
        in_circuit_id["mean_bit_similarity"],
        in_circuit_ood["mean_bit_similarity"],
        out_circuit_id["mean_bit_similarity"],
        out_circuit_ood["mean_bit_similarity"],
    ]
    n_valid = sum(1 for s in sims if s > 0)
    overall = sum(sims) / n_valid if n_valid > 0 else 0.0

    return {
        "in_circuit": {
            "in_distribution": in_circuit_id,
            "out_distribution": in_circuit_ood,
        },
        "out_circuit": {
            "in_distribution": out_circuit_id,
            "out_distribution": out_circuit_ood,
        },
        "overall_score": overall,
    }


def compute_counterfactual_metrics(faithfulness: "FaithfulnessMetrics") -> dict:
    """Compute CounterfactualMetrics from FaithfulnessMetrics with nested structure.

    Returns nested dict structure matching 2x2 matrix:
    {
        "denoising": {
            "in_circuit": {"score": 0.95, "label": "sufficiency"},
            "out_circuit": {"score": 0.92, "label": "completeness"},
            "n_pairs": 40
        },
        "noising": {
            "in_circuit": {"score": 0.88, "label": "necessity"},
            "out_circuit": {"score": 0.90, "label": "independence"},
            "n_pairs": 40
        },
        "overall_score": 0.91
    }
    """
    suff = faithfulness.mean_sufficiency
    comp = faithfulness.mean_completeness
    nec = faithfulness.mean_necessity
    ind = faithfulness.mean_independence

    n_denoising = len(faithfulness.sufficiency_effects) + len(faithfulness.completeness_effects)
    n_noising = len(faithfulness.necessity_effects) + len(faithfulness.independence_effects)

    # Overall: average of all counterfactual scores
    scores = [suff, comp, nec, ind]
    n_valid = sum(1 for s in scores if s > 0)
    overall = sum(scores) / n_valid if n_valid > 0 else 0.0

    return {
        "denoising": {
            "in_circuit": {"score": suff, "label": "sufficiency"},
            "out_circuit": {"score": comp, "label": "completeness"},
            "n_pairs": n_denoising,
        },
        "noising": {
            "in_circuit": {"score": nec, "label": "necessity"},
            "out_circuit": {"score": ind, "label": "independence"},
            "n_pairs": n_noising,
        },
        "overall_score": overall,
    }


def save_faithfulness_json(
    observational_dir: str | None,
    interventional_dir: str | None,
    counterfactual_dir: str | None,
    faithfulness_dir: str,
    robustness: "RobustnessMetrics | None",
    faithfulness: "FaithfulnessMetrics | None",
) -> dict[str, str]:
    """Save result.json in each subfolder and summary.json in faithfulness/."""
    paths = {}
    obs_overall = 0.0
    int_overall = 0.0
    cf_overall = 0.0

    # Observational result.json
    if observational_dir and robustness:
        obs_metrics = compute_observational_metrics(robustness)
        obs_overall = obs_metrics.get("overall_score", 0.0)
        path = os.path.join(observational_dir, "result.json")
        with open(path, "w") as f:
            json.dump(obs_metrics, f, indent=2)
        paths["observational/result.json"] = path

    # Interventional result.json
    if interventional_dir and faithfulness:
        int_metrics = compute_interventional_metrics(faithfulness)
        int_overall = int_metrics.get("overall_score", 0.0)
        path = os.path.join(interventional_dir, "result.json")
        with open(path, "w") as f:
            json.dump(int_metrics, f, indent=2)
        paths["interventional/result.json"] = path

    # Counterfactual result.json
    if counterfactual_dir and faithfulness:
        cf_metrics = compute_counterfactual_metrics(faithfulness)
        cf_overall = cf_metrics.get("overall_score", 0.0)
        path = os.path.join(counterfactual_dir, "result.json")
        with open(path, "w") as f:
            json.dump(cf_metrics, f, indent=2)
        paths["counterfactual/result.json"] = path

    # Compute epsilon from the SAME component scores used for overall score
    # Epsilon = min distance from 1.0 across components (always positive magnitude)

    # Observational: epsilon from the 7 agreement rates that make up the overall
    obs_epsilon = 0.0
    if robustness:
        obs_component_scores = [robustness.noise_agreement_bit]
        # Get per-type OOD agreements
        ood_by_type: dict[str, list] = {}
        for sample in robustness.ood_samples:
            st = getattr(sample, "sample_type", "multiply_positive")
            ood_by_type.setdefault(st, []).append(sample)
        for st in ["multiply_positive", "multiply_negative", "add", "subtract", "bimodal", "bimodal_inv"]:
            samples = ood_by_type.get(st, [])
            if samples:
                agreement = sum(s.agreement_bit for s in samples) / len(samples)
                obs_component_scores.append(agreement)
        # Filter to only valid (non-zero) scores that contributed to overall
        obs_valid_scores = [s for s in obs_component_scores if s > 0]
        if obs_valid_scores:
            obs_epsilon = abs(min(1.0 - s for s in obs_valid_scores))

    # Interventional: epsilon from the 4 mean bit similarities
    int_epsilon = 0.0
    if faithfulness:
        int_component_scores = []
        for stats_dict in [faithfulness.in_circuit_stats, faithfulness.out_circuit_stats,
                          faithfulness.in_circuit_stats_ood, faithfulness.out_circuit_stats_ood]:
            if stats_dict:
                bit_sims = [ps.mean_bit_similarity for ps in stats_dict.values()]
                if bit_sims:
                    int_component_scores.append(sum(bit_sims) / len(bit_sims))
        int_valid_scores = [s for s in int_component_scores if s > 0]
        if int_valid_scores:
            int_epsilon = abs(min(1.0 - s for s in int_valid_scores))

    # Counterfactual: epsilon from the 4 mean scores (suff, comp, nec, ind)
    cf_epsilon = 0.0
    if faithfulness:
        cf_component_scores = [
            faithfulness.mean_sufficiency,
            faithfulness.mean_completeness,
            faithfulness.mean_necessity,
            faithfulness.mean_independence,
        ]
        cf_valid_scores = [s for s in cf_component_scores if s > 0]
        if cf_valid_scores:
            cf_epsilon = abs(min(1.0 - s for s in cf_valid_scores))

    # Summary.json in faithfulness/ with nested structure
    summary = FaithfulnessSummary(
        observational=FaithfulnessCategoryScore(score=obs_overall, epsilon=obs_epsilon),
        interventional=FaithfulnessCategoryScore(score=int_overall, epsilon=int_epsilon),
        counterfactual=FaithfulnessCategoryScore(score=cf_overall, epsilon=cf_epsilon),
        overall=(obs_overall + int_overall + cf_overall) / 3.0 if (obs_overall + int_overall + cf_overall) > 0 else 0.0,
    )
    summary_path = os.path.join(faithfulness_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    paths["summary.json"] = summary_path

    # Generate explanation.md
    explanation_content = """# Faithfulness Metrics Explanation

## Overview

This folder contains faithfulness analysis results for a neural network subcircuit.
Faithfulness measures how well the identified subcircuit captures the behavior of the full network.

## Score Categories

### 1. Observational (Robustness)

Tests how well the subcircuit agrees with the full network under various input perturbations.

**Noise Perturbations:**
- Gaussian noise added to inputs
- Measures stability under small input changes

**Out-of-Distribution Transformations:**
- **Multiply**: Scale inputs by factors (positive and negative)
- **Add/Subtract**: Add or subtract values from inputs
- **Bimodal**: Map inputs to [-1, 1] range (order-preserving and inverted)

**Aggregation:** Average of agreement rates across all perturbation types.

### 2. Interventional

Tests how the subcircuit responds to internal activation interventions.

**In-Circuit Interventions:**
- Patch activations within the identified subcircuit
- High similarity = subcircuit responds consistently to internal changes

**Out-Circuit Interventions:**
- Patch activations outside the subcircuit
- High similarity = subcircuit is independent of non-circuit components

**Distribution Types:**
- In-distribution: Normal activation values
- Out-of-distribution: Extreme or unusual activation values

**Aggregation:** Average of bit similarities across all patch types and distributions.

### 3. Counterfactual (2x2 Matrix)

Tests causal faithfulness using the activation patching paradigm.

```
                | IN-Circuit Patch    | OUT-Circuit Patch
----------------|---------------------|--------------------
DENOISING       | Sufficiency         | Completeness
(corrupt->clean) | (recovery -> 1)      | (disruption -> 1)
----------------|---------------------|--------------------
NOISING         | Necessity           | Independence
(clean->corrupt) | (disruption -> 1)    | (recovery -> 1)
```

**Denoising Experiments** (run corrupted input, patch with clean activations):
- **Sufficiency**: Patching clean in-circuit -> behavior recovers to clean output
- **Completeness**: Patching clean out-circuit -> behavior stays corrupted (circuit is complete)

**Noising Experiments** (run clean input, patch with corrupted activations):
- **Necessity**: Patching corrupted in-circuit -> behavior changes to corrupted output
- **Independence**: Patching corrupted out-circuit -> behavior stays clean (circuit is self-contained)

**Aggregation:** Average of all four counterfactual scores.

## Epsilon Values

Epsilon represents the minimum margin by which any individual score falls short of 1.0:

```
epsilon = min(1.0 - individual_scores)
```

A lower epsilon means all individual scores are closer to perfect (1.0).
Epsilon = 0 would mean all tests achieved perfect faithfulness.

## File Structure

```
faithfulness/
|- summary.json          # Overall scores and epsilons
|- explanation.md        # This file
|- observational/
|   |-- result.json       # Detailed robustness metrics
|- interventional/
|   |-- result.json       # Detailed intervention metrics
|-- counterfactual/
    |-- result.json       # Detailed counterfactual metrics
```

## Interpretation

| Score Range | Interpretation |
|------------|----------------|
| 0.95 - 1.0 | Excellent faithfulness |
| 0.85 - 0.95 | Good faithfulness |
| 0.70 - 0.85 | Moderate faithfulness |
| < 0.70 | Poor faithfulness |

A faithful circuit should score high (>0.85) on all three categories.
"""

    explanation_path = os.path.join(faithfulness_dir, "explanation.md")
    with open(explanation_path, "w") as f:
        f.write(explanation_content)
    paths["explanation.md"] = explanation_path

    return paths


def _compute_subcircuit_scores(
    robustness: "RobustnessMetrics | None",
    faithfulness: "FaithfulnessMetrics | None",
) -> dict:
    """Compute observational, interventional, counterfactual scores and epsilons for a subcircuit."""
    # Observational score and epsilon
    obs_score = 0.0
    obs_epsilon = 0.0
    if robustness:
        obs_component_scores = [robustness.noise_agreement_bit]
        ood_by_type: dict[str, list] = {}
        for sample in robustness.ood_samples:
            st = getattr(sample, "sample_type", "multiply_positive")
            ood_by_type.setdefault(st, []).append(sample)
        for st in ["multiply_positive", "multiply_negative", "add", "subtract", "bimodal", "bimodal_inv"]:
            samples = ood_by_type.get(st, [])
            if samples:
                agreement = sum(s.agreement_bit for s in samples) / len(samples)
                obs_component_scores.append(agreement)
        obs_valid_scores = [s for s in obs_component_scores if s > 0]
        if obs_valid_scores:
            obs_score = sum(obs_valid_scores) / len(obs_valid_scores)
            obs_epsilon = abs(min(1.0 - s for s in obs_valid_scores))

    # Interventional score and epsilon
    int_score = 0.0
    int_epsilon = 0.0
    if faithfulness:
        int_component_scores = []
        for stats_dict in [faithfulness.in_circuit_stats, faithfulness.out_circuit_stats,
                          faithfulness.in_circuit_stats_ood, faithfulness.out_circuit_stats_ood]:
            if stats_dict:
                bit_sims = [ps.mean_bit_similarity for ps in stats_dict.values()]
                if bit_sims:
                    int_component_scores.append(sum(bit_sims) / len(bit_sims))
        int_valid_scores = [s for s in int_component_scores if s > 0]
        if int_valid_scores:
            int_score = sum(int_valid_scores) / len(int_valid_scores)
            int_epsilon = abs(min(1.0 - s for s in int_valid_scores))

    # Counterfactual score and epsilon
    cf_score = 0.0
    cf_epsilon = 0.0
    if faithfulness:
        cf_component_scores = [
            faithfulness.mean_sufficiency,
            faithfulness.mean_completeness,
            faithfulness.mean_necessity,
            faithfulness.mean_independence,
        ]
        cf_valid_scores = [s for s in cf_component_scores if s > 0]
        if cf_valid_scores:
            cf_score = sum(cf_valid_scores) / len(cf_valid_scores)
            cf_epsilon = abs(min(1.0 - s for s in cf_valid_scores))

    # Overall score
    all_scores = [obs_score, int_score, cf_score]
    valid_scores = [s for s in all_scores if s > 0]
    overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    return {
        "observational": {"score": obs_score, "epsilon": obs_epsilon},
        "interventional": {"score": int_score, "epsilon": int_epsilon},
        "counterfactual": {"score": cf_score, "epsilon": cf_epsilon},
        "overall": overall,
    }


def save_gate_summary(
    gate_name: str,
    gate_dir: str,
    metrics: "Metrics",
) -> str:
    """Save summary.json for a specific gate.

    Creates a summary file with:
    - Gate name
    - Number of subcircuits
    - List of subcircuit indices (ranked by overall score)
    - Per-subcircuit scores (observational, interventional, counterfactual, overall)
    """
    best_indices = metrics.per_gate_bests.get(gate_name, [])
    bests_robust = metrics.per_gate_bests_robust.get(gate_name, [])
    bests_faith = metrics.per_gate_bests_faith.get(gate_name, [])

    # Build subcircuit summaries with scores
    subcircuit_summaries = []
    for i, sc_idx in enumerate(best_indices):
        robust = bests_robust[i] if i < len(bests_robust) else None
        faith = bests_faith[i] if i < len(bests_faith) else None

        scores = _compute_subcircuit_scores(robust, faith)
        sc_summary = {"index": sc_idx, **scores}
        subcircuit_summaries.append(sc_summary)

    # Sort by overall score (descending)
    subcircuit_summaries.sort(key=lambda x: x["overall"], reverse=True)

    # Extract sorted indices
    sorted_indices = [sc["index"] for sc in subcircuit_summaries]

    summary = {
        "gate": gate_name,
        "n_subcircuits": len(best_indices),
        "subcircuit_indices": sorted_indices,
        "subcircuits": subcircuit_summaries,
    }

    path = os.path.join(gate_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    return path

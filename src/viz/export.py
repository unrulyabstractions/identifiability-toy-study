"""Metrics computation and JSON export.

Contains functions for computing and exporting metrics:
- compute_observational_metrics: Compute observational metrics from observational data
- compute_interventional_metrics: Compute interventional metrics from faithfulness data
- compute_counterfactual_metrics: Compute counterfactual metrics from faithfulness data
- rank_subcircuits: Rank subcircuits by faithfulness scores (single source of truth)
- save_faithfulness_json: Save result.json files and summary.json
- save_gate_summary: Save summary.json for a specific gate (best per node pattern)
- save_node_pattern_summary: Save summary.json ranking edge variations within a node pattern
- save_full_results: Save full_results.json in leaf folders
"""

import json
import os
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas import FaithfulnessMetrics, Metrics, ObservationalMetrics

from src.schemas import FaithfulnessCategoryScore, FaithfulnessSummary


def rank_subcircuits(
    subcircuit_scores: list[dict],
    key: str = "overall",
    reverse: bool = True,
) -> list[dict]:
    """Rank subcircuits by a score field.

    This is the single source of truth for ranking logic.
    Change this function to change ranking everywhere.

    Args:
        subcircuit_scores: List of dicts with score fields
        key: Which score field to sort by (default: "overall")
        reverse: If True, higher scores rank first (default: True)

    Returns:
        Sorted list with "rank" field added (1-indexed)
    """
    sorted_list = sorted(subcircuit_scores, key=lambda x: x.get(key, 0), reverse=reverse)
    for i, item in enumerate(sorted_list):
        item["rank"] = i + 1
    return sorted_list


def compute_observational_metrics(observational: "ObservationalMetrics") -> dict:
    """Compute observational metrics dict from ObservationalMetrics with nested structure.

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
    noise = observational.noise
    noise_samples = noise.samples if noise else []
    noise_metrics = {
        "n_samples": len(noise_samples) if noise_samples else 0,
        "gate_accuracy": noise.gate_accuracy if noise else 0,
        "subcircuit_accuracy": noise.subcircuit_accuracy if noise else 0,
        "agreement_bit": noise.similarity.bit if noise else 0,
        "agreement_best": noise.similarity.best if noise else 0,
        "mse_mean": 1.0 - noise.similarity.logit if noise else 0,
    }

    # Group OOD samples by type
    ood = observational.ood
    ood_samples = ood.samples if ood else []
    ood_by_type: dict[str, list] = {}
    for sample in ood_samples:
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
        bit_sims = [ps.mean.bit for ps in stats_dict.values()]
        logit_sims = [ps.mean.logit for ps in stats_dict.values()]
        return {
            "n_interventions": len(stats_dict),
            "mean_bit_similarity": sum(bit_sims) / len(bit_sims) if bit_sims else 0.0,
            "mean_logit_similarity": sum(logit_sims) / len(logit_sims) if logit_sims else 0.0,
        }

    interventional = faithfulness.interventional
    in_circuit_id = compute_stats(interventional.in_circuit_stats if interventional else {})
    in_circuit_ood = compute_stats(interventional.in_circuit_stats_ood if interventional else {})
    out_circuit_id = compute_stats(interventional.out_circuit_stats if interventional else {})
    out_circuit_ood = compute_stats(interventional.out_circuit_stats_ood if interventional else {})

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
    cf = faithfulness.counterfactual
    suff = cf.mean_sufficiency if cf else 0.0
    comp = cf.mean_completeness if cf else 0.0
    nec = cf.mean_necessity if cf else 0.0
    ind = cf.mean_independence if cf else 0.0

    n_denoising = (len(cf.sufficiency_effects) + len(cf.completeness_effects)) if cf else 0
    n_noising = (len(cf.necessity_effects) + len(cf.independence_effects)) if cf else 0

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
    faithfulness: "FaithfulnessMetrics | None",
) -> dict[str, str]:
    """Save result.json in each subfolder and summary.json in faithfulness/."""
    paths = {}
    obs_overall = 0.0
    int_overall = 0.0
    cf_overall = 0.0

    # Observational result.json (from faithfulness.observational)
    observational = faithfulness.observational if faithfulness else None
    if observational_dir and observational:
        obs_metrics = compute_observational_metrics(observational)
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
    if observational:
        obs_component_scores = []
        if observational.noise:
            obs_component_scores.append(observational.noise.similarity.bit)
        # Get per-type OOD agreements
        if observational.ood:
            ood_by_type: dict[str, list] = {}
            for sample in observational.ood.samples:
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
    if faithfulness and faithfulness.interventional:
        int_component_scores = []
        for stats_dict in [faithfulness.interventional.in_circuit_stats, faithfulness.interventional.out_circuit_stats,
                          faithfulness.interventional.in_circuit_stats_ood, faithfulness.interventional.out_circuit_stats_ood]:
            if stats_dict:
                bit_sims = [ps.mean.bit for ps in stats_dict.values()]
                if bit_sims:
                    int_component_scores.append(sum(bit_sims) / len(bit_sims))
        int_valid_scores = [s for s in int_component_scores if s > 0]
        if int_valid_scores:
            int_epsilon = abs(min(1.0 - s for s in int_valid_scores))

    # Counterfactual: epsilon from the 4 mean scores (suff, comp, nec, ind)
    cf_epsilon = 0.0
    if faithfulness and faithfulness.counterfactual:
        cf_component_scores = [
            faithfulness.counterfactual.mean_sufficiency,
            faithfulness.counterfactual.mean_completeness,
            faithfulness.counterfactual.mean_necessity,
            faithfulness.counterfactual.mean_independence,
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
|   |-- result.json       # Detailed observational metrics
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
    faithfulness: "FaithfulnessMetrics | None",
) -> dict:
    """Compute observational, interventional, counterfactual scores and epsilons for a subcircuit."""
    # Observational score and epsilon (from faithfulness.observational)
    obs_score = 0.0
    obs_epsilon = 0.0
    observational = faithfulness.observational if faithfulness else None
    if observational:
        obs_component_scores = []
        # Noise agreement
        if observational.noise:
            obs_component_scores.append(observational.noise.similarity.bit)
        # OOD agreement by type
        if observational.ood:
            ood_by_type: dict[str, list] = {}
            for sample in observational.ood.samples:
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

    # Interventional score and epsilon (from faithfulness.interventional)
    int_score = 0.0
    int_epsilon = 0.0
    interventional = faithfulness.interventional if faithfulness else None
    if interventional:
        int_component_scores = []
        for stats_dict in [interventional.in_circuit_stats, interventional.out_circuit_stats,
                          interventional.in_circuit_stats_ood, interventional.out_circuit_stats_ood]:
            if stats_dict:
                bit_sims = [ps.mean.bit for ps in stats_dict.values()]
                if bit_sims:
                    int_component_scores.append(sum(bit_sims) / len(bit_sims))
        int_valid_scores = [s for s in int_component_scores if s > 0]
        if int_valid_scores:
            int_score = sum(int_valid_scores) / len(int_valid_scores)
            int_epsilon = abs(min(1.0 - s for s in int_valid_scores))

    # Counterfactual score and epsilon (from faithfulness.counterfactual)
    cf_score = 0.0
    cf_epsilon = 0.0
    counterfactual = faithfulness.counterfactual if faithfulness else None
    if counterfactual:
        cf_component_scores = [
            counterfactual.mean_sufficiency,
            counterfactual.mean_completeness,
            counterfactual.mean_necessity,
            counterfactual.mean_independence,
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

    Shows ONLY the best edge variation per node pattern for readability.
    For full edge variation details, see each node pattern's summary.json.

    Creates a clean, readable summary with:
    - Gate name and accuracy
    - Number of node patterns analyzed
    - Best subcircuit per node pattern (ranked by overall score)
    """
    best_keys = metrics.per_gate_bests.get(gate_name, [])
    bests_faith = metrics.per_gate_bests_faith.get(gate_name, [])
    gate_metrics = metrics.per_gate_metrics.get(gate_name)

    # Group by node pattern and find best edge variation per pattern
    node_pattern_best: dict[int, tuple[int, dict, "FaithfulnessMetrics | None"]] = {}

    for i, key in enumerate(best_keys):
        faith = bests_faith[i] if i < len(bests_faith) else None
        scores = _compute_subcircuit_scores(faith)

        # Get node_idx (handle both tuple and int keys)
        if isinstance(key, tuple):
            node_idx, edge_var_idx = key
        else:
            node_idx = key
            edge_var_idx = 0

        # Keep only the best edge variation per node pattern
        if node_idx not in node_pattern_best:
            node_pattern_best[node_idx] = (edge_var_idx, scores, faith)
        else:
            _, existing_scores, _ = node_pattern_best[node_idx]
            if scores["overall"] > existing_scores["overall"]:
                node_pattern_best[node_idx] = (edge_var_idx, scores, faith)

    # Build summaries for best per node pattern
    node_summaries = []
    for node_idx, (edge_var_idx, scores, _) in node_pattern_best.items():
        node_summaries.append({
            "node_pattern": node_idx,
            "best_edge_variation": edge_var_idx,
            "observational": round(scores["observational"]["score"], 3),
            "interventional": round(scores["interventional"]["score"], 3),
            "counterfactual": round(scores["counterfactual"]["score"], 3),
            "overall": round(scores["overall"], 3),
        })

    # Rank using the standard ranking function
    ranked_summaries = rank_subcircuits(node_summaries, key="overall")

    summary = {
        "gate": gate_name,
        "accuracy": round(gate_metrics.test_acc, 4) if gate_metrics else None,
        "n_node_patterns": len(node_pattern_best),
        "subcircuits": ranked_summaries,
    }

    path = os.path.join(gate_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    return path


def save_node_pattern_summary(
    node_idx: int,
    node_dir: str,
    edge_variations: list[tuple[int, "FaithfulnessMetrics | None"]],
) -> str:
    """Save summary.json for a node pattern, ranking its edge variations.

    Args:
        node_idx: The node pattern index
        node_dir: Directory for this node pattern (e.g., XOR/46/)
        edge_variations: List of (edge_var_idx, faithfulness_metrics) tuples

    Returns:
        Path to the saved summary.json
    """
    # Build edge variation summaries
    edge_summaries = []
    for edge_var_idx, faith in edge_variations:
        scores = _compute_subcircuit_scores(faith)
        edge_summaries.append({
            "edge_variation": edge_var_idx,
            "observational": round(scores["observational"]["score"], 3),
            "interventional": round(scores["interventional"]["score"], 3),
            "counterfactual": round(scores["counterfactual"]["score"], 3),
            "overall": round(scores["overall"], 3),
        })

    # Rank using the standard ranking function
    ranked_edges = rank_subcircuits(edge_summaries, key="overall")

    summary = {
        "node_pattern": node_idx,
        "n_edge_variations": len(edge_variations),
        "edge_variations": ranked_edges,
    }

    os.makedirs(node_dir, exist_ok=True)
    path = os.path.join(node_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    return path


def save_full_results(
    output_dir: str,
    faithfulness: "FaithfulnessMetrics | None",
    subcircuit_key: int | tuple[int, int],
) -> str:
    """Save full_results.json with complete faithfulness data.

    This file contains all detailed metrics and is saved in leaf folders only.

    Args:
        output_dir: Directory to save to (e.g., XOR/46/0/)
        faithfulness: Full faithfulness metrics
        subcircuit_key: The subcircuit key (int or tuple)

    Returns:
        Path to the saved full_results.json
    """
    if faithfulness is None:
        return ""

    # Format the key for JSON
    if isinstance(subcircuit_key, tuple):
        key_info = {"node_pattern": subcircuit_key[0], "edge_variation": subcircuit_key[1]}
    else:
        key_info = {"index": subcircuit_key}

    # Compute all scores
    scores = _compute_subcircuit_scores(faithfulness)

    # Build full results with detailed breakdowns
    full_results = {
        "subcircuit": key_info,
        "scores": scores,
        "observational": compute_observational_metrics(faithfulness.observational) if faithfulness.observational else None,
        "interventional": compute_interventional_metrics(faithfulness) if faithfulness.interventional else None,
        "counterfactual": compute_counterfactual_metrics(faithfulness) if faithfulness.counterfactual else None,
    }

    path = os.path.join(output_dir, "full_results.json")
    with open(path, "w") as f:
        json.dump(full_results, f, indent=2)

    return path

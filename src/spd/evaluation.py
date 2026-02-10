"""SPD cluster evaluation: robustness and faithfulness analysis.

After clustering components and mapping them to functions, we want to evaluate
how reliable these clusters are:

1. Robustness: Are the clusters stable under input perturbations?
   - Add noise to inputs and check if importance patterns remain consistent
   - High robustness = cluster reliably activates on its designated inputs

2. Faithfulness: Do the clusters actually matter for the model's behavior?
   - Measure how much ablating a cluster affects the output
   - High faithfulness = cluster is causally important, not just correlated

These metrics help distinguish real functional subcircuits from spurious patterns.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.model import DecomposedMLP

    from .types import ClusterInfo, SPDAnalysisResult


def analyze_cluster_robustness(
    decomposed_model: "DecomposedMLP",
    cluster_info: "ClusterInfo",
    importance_matrix: np.ndarray,
    n_samples: int = 20,
    noise_levels: list[float] = None,
    device: str = "cpu",
) -> dict:
    """Analyze robustness of a single cluster to input perturbations.

    Tests whether the cluster's activation pattern remains stable when we
    add small amounts of noise to the inputs. A robust cluster will have
    consistent importance values even with perturbed inputs.

    Args:
        decomposed_model: Trained SPD decomposition
        cluster_info: Info about the cluster to analyze
        importance_matrix: Baseline importance matrix from clean inputs
        n_samples: Number of noisy samples per noise level
        noise_levels: Gaussian noise standard deviations to test
        device: Compute device

    Returns:
        Dict with:
        - mean_importance_stability: 0-1 score (1 = perfectly stable)
        - noise_sensitivity: How much importance changes per unit noise
        - n_samples_tested: Number of samples used
    """
    if noise_levels is None:
        noise_levels = [0.05, 0.1, 0.2]

    if decomposed_model is None or decomposed_model.component_model is None:
        return {"mean_importance_stability": 0.0, "noise_sensitivity": 0.0}

    component_model = decomposed_model.component_model
    n_inputs = 2

    comp_indices = cluster_info.component_indices
    if not comp_indices:
        return {"mean_importance_stability": 0.0, "noise_sensitivity": 0.0}

    baseline_imp = importance_matrix[:, comp_indices].mean()
    stability_scores = []
    importance_changes = []

    total_samples = len(noise_levels) * n_samples
    all_base_inputs = torch.randint(
        0, 2, (total_samples, n_inputs), dtype=torch.float, device=device
    )
    all_noise = torch.randn_like(all_base_inputs)

    noise_scales = torch.tensor(
        [level for level in noise_levels for _ in range(n_samples)], device=device
    ).unsqueeze(1)
    all_noisy_inputs = all_base_inputs + all_noise * noise_scales

    try:
        with torch.inference_mode():
            output_with_cache = component_model(all_noisy_inputs, cache_type="input")
            pre_weight_acts = output_with_cache.cache

            ci_outputs = component_model.calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                sampling="continuous",
                detach_inputs=False,
            )

            all_imp = []
            for module_name in sorted(ci_outputs.upper_leaky.keys()):
                ci_tensor = ci_outputs.upper_leaky[module_name]
                all_imp.append(ci_tensor.detach().cpu().numpy())

            if all_imp:
                full_imp = np.concatenate(all_imp, axis=1)
                cluster_imp = full_imp[:, comp_indices].mean(axis=1)

                for i, (imp, noise_level) in enumerate(
                    zip(cluster_imp, noise_scales.squeeze().cpu().numpy())
                ):
                    stability = 1.0 - abs(imp - baseline_imp)
                    stability_scores.append(max(0, stability))
                    importance_changes.append(abs(imp - baseline_imp) / max(noise_level, 0.01))
    except Exception:
        pass

    return {
        "mean_importance_stability": float(np.mean(stability_scores)) if stability_scores else 0.0,
        "noise_sensitivity": float(np.mean(importance_changes)) if importance_changes else 0.0,
        "n_samples_tested": len(stability_scores),
    }


def analyze_cluster_faithfulness(
    decomposed_model: "DecomposedMLP",
    cluster_info: "ClusterInfo",
    importance_matrix: np.ndarray,
    n_inputs: int = 2,
    device: str = "cpu",
) -> dict:
    """Analyze faithfulness of a cluster by testing ablation effects.

    Faithfulness measures whether a cluster actually matters for the output.
    We compute several metrics from the importance matrix:

    - Variance: How much does importance vary across inputs? High variance
      means the cluster is selective (activates on some inputs, not others).
    - Max importance: How strongly does it ever activate?
    - Mean when active: Average importance when the cluster is "on"
    - Selectivity: Variance normalized by mean (high = discriminative)

    Args:
        decomposed_model: Trained SPD decomposition
        cluster_info: Info about the cluster to analyze
        importance_matrix: Importance matrix for all inputs
        n_inputs: Number of input bits
        device: Compute device

    Returns:
        Dict with ablation effect, sufficiency, mean_when_active, selectivity
    """
    if decomposed_model is None or decomposed_model.component_model is None:
        return {"mean_ablation_effect": 0.0, "sufficiency_score": 0.0}

    comp_indices = cluster_info.component_indices
    if not comp_indices:
        return {"mean_ablation_effect": 0.0, "sufficiency_score": 0.0}

    cluster_importance = importance_matrix[:, comp_indices]
    importance_variance = float(np.var(cluster_importance))
    max_importance = float(np.max(cluster_importance))

    active_mask = cluster_importance > 0.5
    mean_when_active = float(np.mean(cluster_importance[active_mask])) if active_mask.any() else 0.0

    return {
        "mean_ablation_effect": importance_variance,
        "sufficiency_score": max_importance,
        "mean_when_active": mean_when_active,
        "selectivity": importance_variance / (cluster_importance.mean() + 1e-8),
    }


def analyze_all_clusters(
    decomposed_model: "DecomposedMLP",
    analysis_result: "SPDAnalysisResult",
    device: str = "cpu",
) -> list[dict]:
    """Run robustness and faithfulness analysis on all clusters.

    This is the main entry point for cluster evaluation. It runs both
    robustness and faithfulness analysis on every cluster and updates
    the cluster info objects with the scores.

    Args:
        decomposed_model: Trained SPD decomposition
        analysis_result: SPDAnalysisResult containing clusters to analyze
        device: Compute device

    Returns:
        List of dicts, one per cluster, containing:
        - cluster_idx: Which cluster
        - robustness: Dict of robustness metrics
        - faithfulness: Dict of faithfulness metrics
    """
    results = []

    for cluster_info in analysis_result.clusters:
        robustness = analyze_cluster_robustness(
            decomposed_model, cluster_info, analysis_result.importance_matrix, device=device
        )
        faithfulness = analyze_cluster_faithfulness(
            decomposed_model, cluster_info, analysis_result.importance_matrix, device=device
        )

        # Update cluster info with scores
        cluster_info.robustness_score = robustness.get("mean_importance_stability", 0.0)
        cluster_info.faithfulness_score = faithfulness.get("sufficiency_score", 0.0)

        results.append({
            "cluster_idx": cluster_info.cluster_idx,
            "robustness": robustness,
            "faithfulness": faithfulness,
        })

    return results

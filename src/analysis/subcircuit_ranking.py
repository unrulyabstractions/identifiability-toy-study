"""Subcircuit metric extraction and ranking utilities.

This module defines:
- ALL_SUBCIRCUIT_METRICS: Complete list of all possible metrics
- SUBCIRCUIT_METRICS_RANKING: Ordered list used for ranking (tuple ordering)
- Functions for extracting and ranking subcircuit metrics
"""

# =============================================================================
# All possible metrics (exhaustive list)
# =============================================================================

ALL_SUBCIRCUIT_METRICS = [
    # Basic metrics (from SubcircuitMetrics)
    "accuracy",
    "bit_similarity",
    "logit_similarity",
    "best_similarity",
    # Structure metrics (from circuit structure)
    "edge_sparsity",
    "node_sparsity",
    "connectivity_density",
    # Observational metrics
    "overall_observational",
    "noise_perturbations_agreement_bit",
    "noise_perturbations_agreement_logit",
    "out_distribution_transformations_agreement",
    "out_distribution_multiply_agreement",
    "out_distribution_bimodal_agreement",
    # Interventional metrics
    "overall_interventional",
    "mean_in_circuit_effect",
    "mean_out_circuit_effect",
    # Counterfactual metrics
    "overall_counterfactual",
    "sufficiency",
    "completeness",
    "necessity",
    "independence",
    # Overall faithfulness
    "overall_faithfulness",
]

# =============================================================================
# Ranking configuration
# =============================================================================

# Metrics used for ranking subcircuits (in order of priority for tuple comparison)
# First metric is primary sort key, second is tiebreaker, etc.
# Higher values are better for all metrics.
SUBCIRCUIT_METRICS_RANKING = ALL_SUBCIRCUIT_METRICS.copy()

# Alias for common use
DEFAULT_SUBCIRCUIT_METRICS = ALL_SUBCIRCUIT_METRICS


# =============================================================================
# Metric extraction
# =============================================================================


def extract_all_metrics(
    subcircuit_metrics=None,
    faithfulness_data=None,
    structure_data=None,
) -> dict[str, float | None]:
    """Extract all available metrics into a flat dictionary.

    Args:
        subcircuit_metrics: SubcircuitMetrics object with basic metrics
        faithfulness_data: Optional FaithfulnessMetrics with detailed scores
        structure_data: Optional dict with sparsity metrics

    Returns:
        Dict mapping metric name to value (or None if not available)
    """
    all_metrics = {}

    # From SubcircuitMetrics
    if subcircuit_metrics:
        all_metrics["accuracy"] = getattr(subcircuit_metrics, "accuracy", None)
        all_metrics["bit_similarity"] = getattr(
            subcircuit_metrics, "bit_similarity", None
        )
        all_metrics["logit_similarity"] = getattr(
            subcircuit_metrics, "logit_similarity", None
        )
        all_metrics["best_similarity"] = getattr(
            subcircuit_metrics, "best_similarity", None
        )

    # From structure data (sparsity metrics)
    if structure_data:
        all_metrics["edge_sparsity"] = structure_data.get("edge_sparsity")
        all_metrics["node_sparsity"] = structure_data.get("node_sparsity")
        all_metrics["connectivity_density"] = structure_data.get("connectivity_density")

    # From FaithfulnessMetrics
    if faithfulness_data:
        # Observational
        if (
            hasattr(faithfulness_data, "observational")
            and faithfulness_data.observational
        ):
            obs = faithfulness_data.observational
            all_metrics["overall_observational"] = getattr(
                obs, "overall_observational", None
            )
            if hasattr(obs, "noise") and obs.noise:
                all_metrics["noise_perturbations_agreement_bit"] = getattr(
                    obs.noise, "agreement_bit", None
                )
                all_metrics["noise_perturbations_agreement_logit"] = getattr(
                    obs.noise, "agreement_logit", None
                )
            if hasattr(obs, "ood") and obs.ood:
                all_metrics["out_distribution_transformations_agreement"] = getattr(
                    obs.ood, "overall_agreement", None
                )
                all_metrics["out_distribution_multiply_agreement"] = getattr(
                    obs.ood, "multiply_agreement", None
                )
                all_metrics["out_distribution_bimodal_agreement"] = getattr(
                    obs.ood, "bimodal_agreement", None
                )

        # Interventional
        if (
            hasattr(faithfulness_data, "interventional")
            and faithfulness_data.interventional
        ):
            intv = faithfulness_data.interventional
            all_metrics["overall_interventional"] = getattr(
                intv, "overall_interventional", None
            )
            all_metrics["mean_in_circuit_effect"] = getattr(
                intv, "mean_in_circuit_effect", None
            )
            all_metrics["mean_out_circuit_effect"] = getattr(
                intv, "mean_out_circuit_effect", None
            )

        # Counterfactual
        if (
            hasattr(faithfulness_data, "counterfactual")
            and faithfulness_data.counterfactual
        ):
            cf = faithfulness_data.counterfactual
            all_metrics["overall_counterfactual"] = getattr(
                cf, "overall_counterfactual", None
            )
            all_metrics["sufficiency"] = getattr(cf, "mean_sufficiency", None)
            all_metrics["completeness"] = getattr(cf, "mean_completeness", None)
            all_metrics["necessity"] = getattr(cf, "mean_necessity", None)
            all_metrics["independence"] = getattr(cf, "mean_independence", None)

        # Overall faithfulness
        all_metrics["overall_faithfulness"] = getattr(
            faithfulness_data, "overall_faithfulness", None
        )

    return all_metrics


def get_ranking_tuple(
    metrics: dict[str, float | None],
    ranking_metrics: list[str] | None = None,
) -> tuple[float, ...]:
    """Get a tuple of metric values for ranking comparison.

    Args:
        metrics: Dict of metric name -> value
        ranking_metrics: Ordered list of metrics to use (defaults to SUBCIRCUIT_METRICS_RANKING)

    Returns:
        Tuple of values (None/NaN replaced with -inf for sorting)
    """
    import math

    if ranking_metrics is None:
        ranking_metrics = SUBCIRCUIT_METRICS_RANKING

    values = []
    for name in ranking_metrics:
        val = metrics.get(name)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            val = float("-inf")
        values.append(val)

    return tuple(values)


def get_ranking_dict(
    metrics: dict[str, float | None],
    ranking_metrics: list[str] | None = None,
) -> dict[str, float | None]:
    """Get a dict of ranking metrics with their values.

    Args:
        metrics: Dict of metric name -> value
        ranking_metrics: Ordered list of metrics to include (defaults to SUBCIRCUIT_METRICS_RANKING)

    Returns:
        Dict with only the ranking metrics (preserving order)
    """
    if ranking_metrics is None:
        ranking_metrics = SUBCIRCUIT_METRICS_RANKING

    return {name: metrics.get(name) for name in ranking_metrics if name in metrics}


def filter_and_rank_subcircuit_metrics(
    subcircuit_metrics,
    faithfulness_data=None,
    structure_data=None,
    metric_names: list[str] | None = None,
) -> dict:
    """Extract and filter subcircuit metrics into ranked_metrics format.

    Args:
        subcircuit_metrics: SubcircuitMetrics object with basic metrics
        faithfulness_data: Optional FaithfulnessMetrics with detailed scores
        structure_data: Optional dict with sparsity metrics from mask_idx_map
        metric_names: List of metric names to include (defaults to ALL_SUBCIRCUIT_METRICS)

    Returns:
        dict with:
            - metric_name: list of metric names (filtered)
            - metric_value: list of corresponding values
    """
    if metric_names is None:
        metric_names = ALL_SUBCIRCUIT_METRICS

    all_metrics = extract_all_metrics(subcircuit_metrics, faithfulness_data, structure_data)

    # Filter and order by requested metrics
    filtered_names = []
    filtered_values = []
    for name in metric_names:
        if name in all_metrics and all_metrics[name] is not None:
            filtered_names.append(name)
            filtered_values.append(all_metrics[name])

    return {
        "metric_name": filtered_names,
        "metric_value": filtered_values,
    }

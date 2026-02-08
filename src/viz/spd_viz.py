"""SPD (Stochastic Parameter Decomposition) visualization.

Contains functions for visualizing SPD decompositions:
- visualize_spd_components: Visualize SPD component weights for a single model
- visualize_spd_experiment: Visualize complete SPD results for an experiment
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from src.model import DecomposedMLP
from src.infra import profile_fn

from .circuit_drawing import _get_spd_component_weights

if TYPE_CHECKING:
    from src.spd import SpdResults


def visualize_spd_components(
    decomposed: DecomposedMLP,
    output_dir: str,
    filename: str = "spd_components.png",
    gate_name: str = "",
) -> str | None:
    """Visualize SPD component weights."""
    weights = _get_spd_component_weights(decomposed)
    if weights is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(weights))
    bars = ax.bar(x, weights, color="steelblue", alpha=0.8)

    # Highlight top components
    top_k = min(3, len(weights))
    top_indices = np.argsort(weights)[-top_k:]
    for idx in top_indices:
        bars[idx].set_color("coral")

    ax.set_xlabel("Component Index")
    ax.set_ylabel("Normalized Weight")
    ax.set_title(
        f"{gate_name} - SPD Component Importance"
        if gate_name
        else "SPD Component Importance",
        fontweight="bold",
    )
    ax.set_xticks(x)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


@profile_fn("SPD Visualization")
def visualize_spd_experiment(
    spd_results: "SpdResults",
    run_dir: str | Path,
) -> dict:
    """
    Visualize complete SPD results for an experiment.

    Creates visualizations in run_dir/{trial_id}/spd/visualizations/

    Args:
        spd_results: SpdResults from run_spd()
        run_dir: Run directory where results are saved

    Returns:
        Dict of visualization paths
    """
    run_dir = Path(run_dir)
    viz_paths = {}
    config_id = spd_results.config.get_config_id()

    for trial_id, trial_result in spd_results.per_trial.items():
        spd_dir = run_dir / trial_id / "spd"
        viz_dir = spd_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        viz_paths[trial_id] = {}

        # Component importance bar chart
        if trial_result.decomposed_model is not None:
            comp_path = visualize_spd_components(
                trial_result.decomposed_model,
                str(viz_dir),
                filename="component_importance.png",
                gate_name=f"Trial {trial_id[:8]} - {config_id}",
            )
            if comp_path:
                viz_paths[trial_id]["component_importance"] = comp_path

        # Cluster visualization if estimate available
        if trial_result.spd_subcircuit_estimate and trial_result.spd_subcircuit_estimate.n_clusters > 0:
            cluster_path = _visualize_cluster_summary(
                trial_result.spd_subcircuit_estimate,
                str(viz_dir),
                config_id,
            )
            if cluster_path:
                viz_paths[trial_id]["cluster_summary"] = cluster_path

    return viz_paths


def _visualize_cluster_summary(
    estimate,
    output_dir: str,
    config_id: str,
) -> str | None:
    """Visualize cluster assignment summary."""
    if not estimate.cluster_assignments:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Cluster sizes
    ax = axes[0]
    cluster_sizes = estimate.cluster_sizes
    x = np.arange(len(cluster_sizes))
    ax.bar(x, cluster_sizes, color="steelblue", alpha=0.8)
    ax.set_xlabel("Cluster Index")
    ax.set_ylabel("Number of Components")
    ax.set_title("Components per Cluster")
    ax.set_xticks(x)

    # Right: Cluster function mapping
    ax = axes[1]
    if estimate.cluster_functions:
        functions = [estimate.cluster_functions.get(i, "?") for i in range(len(cluster_sizes))]
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
        bars = ax.bar(x, cluster_sizes, color=colors, alpha=0.8)
        ax.set_xlabel("Cluster Index")
        ax.set_ylabel("Size")
        ax.set_title("Cluster Function Mapping")
        ax.set_xticks(x)
        # Add function labels
        for i, (bar, func) in enumerate(zip(bars, functions)):
            if func:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    func[:10],
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )
    else:
        ax.text(0.5, 0.5, "No function mapping", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Cluster Function Mapping")

    plt.suptitle(f"SPD Clustering Summary - {config_id}", fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "cluster_summary.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path

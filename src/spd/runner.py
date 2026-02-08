"""SPD experiment and trial runner.

This module is the main entry point for running SPD analysis. The typical flow:

    1. Run experiment to train models:
       experiment_result = run_experiment(config)

    2. Run SPD on all trials:
       spd_results = run_spd(experiment_result, run_dir)

    3. Save and visualize:
       save_spd_results(spd_results, run_dir)
       visualize_spd_experiment(spd_results, run_dir)

What SPD does:
    For each trial, SPD decomposes the trained MLP weights into interpretable
    components. Each component is a rank-1 matrix that contributes to the
    original weight. By analyzing which components activate together, we can
    discover functional subcircuits in the network.

Key functions:
    run_spd()           - Run SPD on all trials in an experiment
    run_spd_trial()     - Run SPD on a single trial
    analyze_and_visualize_spd() - Run analysis and generate visualizations
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.infra import profile, profile_fn

from .analysis import analyze_all_clusters, estimate_spd_subcircuits, run_spd_analysis
from .decomposition import decompose_mlp
from .types import SPDConfig, SpdResults, SpdTrialResult
from .visualization import (
    visualize_ci_histograms,
    visualize_coactivation_matrix,
    visualize_components_as_circuits,
    visualize_importance_heatmap,
    visualize_l0_sparsity,
    visualize_mean_ci_per_component,
    visualize_summary,
    visualize_uv_matrices,
)

if TYPE_CHECKING:
    from src.model import DecomposedMLP, MLP
    from src.schemas import ExperimentResult, TrialResult


# =============================================================================
# Main Entry Points
# =============================================================================


@profile_fn("SPD Experiment")
def run_spd(
    experiment_result: "ExperimentResult",
    run_dir: str | Path,
    device: str = "cpu",
    spd_config: SPDConfig | None = None,
) -> SpdResults:
    """Run SPD analysis on all trials in an experiment.

    This is the main entry point for SPD, called AFTER run_experiment().

    What it does:
        1. For each trial, loads the trained model
        2. Runs SPD decomposition (trains U, V matrices with sparsity)
        3. Estimates subcircuits by clustering components
        4. Returns results for all trials

    Args:
        experiment_result: The result from run_experiment() containing trained models
        run_dir: Directory where experiment results are saved
        device: Device for computation. CPU is recommended for small models
                because SPD's many small ops don't benefit from GPU parallelism.
        spd_config: Configuration for SPD. Uses sensible defaults if None.

    Returns:
        SpdResults containing per-trial decompositions and subcircuit estimates

    Example:
        experiment_result = run_experiment(config)
        spd_results = run_spd(experiment_result, run_dir, device="cpu")
        for trial_id, trial in spd_results.per_trial.items():
            print(f"Trial {trial_id}: {trial.spd_subcircuit_estimate.n_clusters} clusters")
    """
    run_dir = Path(run_dir)

    if spd_config is None:
        spd_config = SPDConfig()

    spd_results = SpdResults(config=spd_config)

    n_trials = len(experiment_result.trials)
    for trial_idx, (trial_id, trial_result) in enumerate(experiment_result.trials.items()):
        print(f"[SPD] Trial {trial_idx + 1}/{n_trials}: {trial_id[:8]}...")

        spd_trial_result = run_spd_trial(
            trial_result=trial_result,
            spd_config=spd_config,
            device=device,
        )

        spd_results.per_trial[trial_id] = spd_trial_result

    return spd_results


def run_spd_trial(
    trial_result: "TrialResult",
    spd_config: SPDConfig,
    device: str = "cpu",
) -> SpdTrialResult:
    """Run SPD analysis on a single trial.

    What it does:
        1. Decomposition: Trains SPD to factorize weights into components
           W ≈ Σ_c g_c * (U_c ⊗ V_c) with sparsity pressure on g values

        2. Subcircuit estimation: Clusters components based on co-activation
           patterns, then matches clusters to boolean functions

    Args:
        trial_result: Completed trial with trained model and test data
        spd_config: SPD hyperparameters (n_components, steps, etc.)
        device: Compute device

    Returns:
        SpdTrialResult with:
        - decomposed_model: The trained U, V matrices
        - spd_subcircuit_estimate: Cluster assignments and function mappings
    """
    model = trial_result.model
    x = trial_result.test_x
    y_pred = trial_result.test_y_pred
    gate_names = trial_result.setup.model_params.logic_gates
    input_size = x.shape[1] if x is not None else 2

    spd_trial_result = SpdTrialResult(trial_id=trial_result.trial_id)

    # Skip if model or data is missing
    if model is None or x is None or y_pred is None:
        return spd_trial_result

    config_id = spd_config.get_config_id()
    print(f"    SPD config: {config_id}")

    # Step 1: Decompose the model weights into components
    # This trains U, V matrices to approximate W with sparsity
    with profile(f"spd_mlp_{config_id}"):
        decomposed = decompose_mlp(x, y_pred, model, device, spd_config)

    spd_trial_result.decomposed_model = decomposed

    # Step 2: Estimate subcircuits by clustering components
    # Components that fire together get grouped into clusters
    print(f"    SPD subcircuit: {config_id}")
    with profile(f"spd_mlp_sc_{config_id}"):
        estimate = estimate_spd_subcircuits(
            decomposed_model=decomposed,
            n_inputs=input_size,
            gate_names=gate_names,
            device=device,
        )

    spd_trial_result.spd_subcircuit_estimate = estimate

    return spd_trial_result


# =============================================================================
# Analysis with Visualization
# =============================================================================


def analyze_and_visualize_spd(
    decomposed_model: "DecomposedMLP",
    target_model: "MLP",
    output_dir: str | Path,
    gate_names: list[str] = None,
    n_inputs: int = 2,
    device: str = "cpu",
):
    """Run complete SPD analysis and generate all visualizations.

    This is called automatically by save_spd_results() to create analysis
    artifacts for each trial.

    What it creates:
        output_dir/
        ├── validation.json         # MMCS, ML2R, faithfulness metrics
        ├── clustering/
        │   ├── assignments.json    # Which component → which cluster
        │   ├── importance_matrix.npy   # [n_inputs, n_components] CI values
        │   └── coactivation_matrix.npy # [n_comp, n_comp] co-firing counts
        ├── visualizations/
        │   ├── importance_heatmap.png  # CI values as heatmap
        │   ├── coactivation_matrix.png # Cluster structure visualization
        │   ├── circuits_matched.png    # Clusters matching known gates
        │   ├── circuits_unknown.png    # Unidentified clusters
        │   ├── uv_matrices.png         # The learned U, V matrices
        │   ├── ci_histograms.png       # Distribution of CI values
        │   ├── mean_ci_per_component.png  # Sorted component importance
        │   ├── l0_sparsity.png         # Active component counts
        │   └── summary.png             # Overview statistics
        └── clusters/
            └── {cluster_idx}/
                ├── analysis.json       # Cluster info
                ├── robustness.json     # Stability under noise
                └── faithfulness.json   # Ablation effects

    Args:
        decomposed_model: Trained SPD decomposition
        target_model: Original MLP model
        output_dir: Where to save everything
        gate_names: Names of gates to match against (e.g., ["XOR", "AND"])
        n_inputs: Number of input bits (2 for boolean gates)
        device: Compute device

    Returns:
        SPDAnalysisResult with all computed metrics and viz paths
    """
    output_dir = Path(output_dir)

    # Create directory structure
    (output_dir / "clustering").mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (output_dir / "clusters").mkdir(parents=True, exist_ok=True)

    # Run the core analysis (importance matrix, clustering, function mapping)
    result = run_spd_analysis(
        decomposed_model=decomposed_model,
        target_model=target_model,
        n_inputs=n_inputs,
        gate_names=gate_names,
        device=device,
    )

    if result.n_components == 0:
        return result

    # Save validation metrics (MMCS, ML2R from SPD paper)
    validation_data = {
        "mmcs": result.mmcs,
        "ml2r": result.ml2r,
        "faithfulness_loss": result.faithfulness_loss,
        "n_alive_components": result.n_alive_components,
        "n_dead_components": result.n_dead_components,
        "dead_component_labels": result.dead_component_labels,
    }
    with open(output_dir / "validation.json", "w") as f:
        json.dump(validation_data, f, indent=2)

    # Save clustering data (numpy arrays for large matrices)
    if result.importance_matrix is not None:
        np.save(output_dir / "clustering" / "importance_matrix.npy", result.importance_matrix)
    if result.coactivation_matrix is not None:
        np.save(output_dir / "clustering" / "coactivation_matrix.npy", result.coactivation_matrix)

    # Save cluster assignments as JSON for easy inspection
    assignments_data = {
        "cluster_assignments": result.cluster_assignments,
        "component_labels": result.component_labels,
        "n_clusters": result.n_clusters,
        "clusters": [asdict(c) for c in result.clusters],
    }
    with open(output_dir / "clustering" / "assignments.json", "w") as f:
        json.dump(assignments_data, f, indent=2)

    # Generate visualizations
    layer_sizes = target_model.layer_sizes if target_model else [n_inputs, 3, 1]
    viz_paths = {}

    # Importance heatmap: rows = inputs, cols = components, color = CI value
    path = visualize_importance_heatmap(
        result.importance_matrix,
        result.component_labels,
        str(output_dir / "visualizations" / "importance_heatmap.png"),
        n_inputs,
    )
    if path:
        viz_paths["importance_heatmap"] = "visualizations/importance_heatmap.png"

    # Coactivation matrix: shows which components fire together
    path = visualize_coactivation_matrix(
        result.coactivation_matrix,
        result.cluster_assignments,
        result.component_labels,
        str(output_dir / "visualizations" / "coactivation_matrix.png"),
    )
    if path:
        viz_paths["coactivation_matrix"] = "visualizations/coactivation_matrix.png"

    # Circuit diagrams: shows clusters as network graphs
    circuit_paths = visualize_components_as_circuits(
        result,
        layer_sizes,
        str(output_dir / "visualizations" / "circuits"),
        decomposed_model=decomposed_model,
    )
    for category, path in circuit_paths.items():
        viz_paths[f"circuits_{category}"] = f"visualizations/circuits_{category}.png"

    # U/V matrices: the actual learned decomposition
    path = visualize_uv_matrices(
        decomposed_model,
        str(output_dir / "visualizations" / "uv_matrices.png"),
    )
    if path:
        viz_paths["uv_matrices"] = "visualizations/uv_matrices.png"

    # CI histograms: distribution of importance values (should be bimodal)
    path = visualize_ci_histograms(
        result.importance_matrix,
        result.component_labels,
        str(output_dir / "visualizations" / "ci_histograms.png"),
    )
    if path:
        viz_paths["ci_histograms"] = "visualizations/ci_histograms.png"

    # Mean CI per component: sorted bar chart of component importance
    path = visualize_mean_ci_per_component(
        result.importance_matrix,
        result.component_labels,
        str(output_dir / "visualizations" / "mean_ci_per_component.png"),
    )
    if path:
        viz_paths["mean_ci_per_component"] = "visualizations/mean_ci_per_component.png"

    # L0 sparsity: how many components are active per input
    path = visualize_l0_sparsity(
        result.importance_matrix,
        result.component_labels,
        str(output_dir / "visualizations" / "l0_sparsity.png"),
    )
    if path:
        viz_paths["l0_sparsity"] = "visualizations/l0_sparsity.png"

    # Summary: overall statistics and cluster info table
    path = visualize_summary(
        result,
        str(output_dir / "visualizations" / "summary.png"),
    )
    if path:
        viz_paths["summary"] = "visualizations/summary.png"

    result.visualization_paths = viz_paths

    # Run per-cluster robustness/faithfulness analysis
    cluster_analyses = analyze_all_clusters(
        decomposed_model=decomposed_model,
        analysis_result=result,
        device=device,
    )

    # Helper to convert numpy types to JSON-serializable types
    def _to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    # Save per-cluster analysis files
    for cluster_info, cluster_analysis in zip(result.clusters, cluster_analyses):
        cluster_dir = output_dir / "clusters" / str(cluster_info.cluster_idx)
        cluster_dir.mkdir(parents=True, exist_ok=True)

        with open(cluster_dir / "analysis.json", "w") as f:
            json.dump(_to_serializable(asdict(cluster_info)), f, indent=2)

        with open(cluster_dir / "robustness.json", "w") as f:
            json.dump(_to_serializable(cluster_analysis["robustness"]), f, indent=2)

        with open(cluster_dir / "faithfulness.json", "w") as f:
            json.dump(_to_serializable(cluster_analysis["faithfulness"]), f, indent=2)

    return result

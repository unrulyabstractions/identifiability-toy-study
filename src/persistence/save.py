"""
Save functions for experiment results.

Changes to this module should be reflected in README.md.

Structure:
    runs/run_{timestamp}/
        summary.json          - Ranked results across all trials and gates
        explanation.md        - How to read this folder
        config.json           - ExperimentConfig only
        circuits.json         - Subcircuit masks and structure analysis (run-level)
        profiling/
            profiling.json    - Timing data (events, phase durations)
        trials/
            {trial_id}/
                summary.json      - Trial-level ranked results
                explanation.md    - How to read this folder
                setup.json        - TrialSetup
                metrics.json      - Metrics (training, per-gate, robustness, faithfulness)
                tensors.pt        - Test data, activations, weights
                all_gates/
                    model.pt
                gates/
                    {gate_name}/
                        summary.json      - Gate-level ranked subcircuits
                        explanation.md    - How to read this folder
                        full/
                            decomposed_model.pt
                        {subcircuit_idx}/
                            decomposed_model.pt
                spd/                    - SPD decomposition outputs
                    config.json         - SPD configuration used
                    decomposed_model.pt - Trained decomposed model
                    estimate.json       - Subcircuit clustering results
                    component_importance.npy - Per-component importance scores
                    coactivation_matrix.npy  - Component coactivation matrix
                    clustering/
                        assignments.json    - Full clustering data
                        importance_matrix.npy
                        coactivation_matrix.npy
                    visualizations/
                        importance_heatmap.png
                        coactivation_matrix.png
                        components_as_circuits.png
                        uv_matrices.png
                        summary.png
                    clusters/
                        {cluster_idx}/
                            analysis.json
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from src.schemas import ExperimentResult
from src.serialization import filter_non_serializable


# =============================================================================
# Configurable Subcircuit Metrics
# =============================================================================

DEFAULT_SUBCIRCUIT_METRICS = [
    "accuracy",
    "bit_similarity",
    "logit_similarity",
    "edge_sparsity",
    "node_sparsity",
    "sufficiency",
    "completeness",
    "necessity",
    "independence",
    "overall_counterfactual",
    "overall_interventional",
    "overall_observational",
    "noise_perturbations_agreement_bit",
    "out_distribution_transformations_agreement",
]


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
        metric_names: List of metric names to include (defaults to DEFAULT_SUBCIRCUIT_METRICS)

    Returns:
        dict with:
            - metric_name: list of metric names (filtered)
            - metric_value: list of corresponding values
    """
    if metric_names is None:
        metric_names = DEFAULT_SUBCIRCUIT_METRICS

    # Collect all available metrics
    all_metrics = {}

    # From SubcircuitMetrics
    if subcircuit_metrics:
        all_metrics["accuracy"] = getattr(subcircuit_metrics, "accuracy", None)
        all_metrics["bit_similarity"] = getattr(subcircuit_metrics, "bit_similarity", None)
        all_metrics["logit_similarity"] = getattr(subcircuit_metrics, "logit_similarity", None)
        all_metrics["best_similarity"] = getattr(subcircuit_metrics, "best_similarity", None)

    # From structure data (sparsity metrics)
    if structure_data:
        all_metrics["edge_sparsity"] = structure_data.get("edge_sparsity")
        all_metrics["node_sparsity"] = structure_data.get("node_sparsity")
        all_metrics["connectivity_density"] = structure_data.get("connectivity_density")

    # From FaithfulnessMetrics
    if faithfulness_data:
        # Observational
        if hasattr(faithfulness_data, "observational") and faithfulness_data.observational:
            obs = faithfulness_data.observational
            all_metrics["overall_observational"] = getattr(obs, "overall_observational", None)
            if hasattr(obs, "noise") and obs.noise:
                all_metrics["noise_perturbations_agreement_bit"] = getattr(obs.noise, "agreement_bit", None)
                all_metrics["noise_perturbations_agreement_logit"] = getattr(obs.noise, "agreement_logit", None)
            if hasattr(obs, "ood") and obs.ood:
                all_metrics["out_distribution_transformations_agreement"] = getattr(obs.ood, "overall_agreement", None)
                all_metrics["out_distribution_multiply_agreement"] = getattr(obs.ood, "multiply_agreement", None)
                all_metrics["out_distribution_bimodal_agreement"] = getattr(obs.ood, "bimodal_agreement", None)

        # Interventional
        if hasattr(faithfulness_data, "interventional") and faithfulness_data.interventional:
            intv = faithfulness_data.interventional
            all_metrics["overall_interventional"] = getattr(intv, "overall_interventional", None)
            all_metrics["mean_in_circuit_effect"] = getattr(intv, "mean_in_circuit_effect", None)
            all_metrics["mean_out_circuit_effect"] = getattr(intv, "mean_out_circuit_effect", None)

        # Counterfactual
        if hasattr(faithfulness_data, "counterfactual") and faithfulness_data.counterfactual:
            cf = faithfulness_data.counterfactual
            all_metrics["overall_counterfactual"] = getattr(cf, "overall_counterfactual", None)
            all_metrics["sufficiency"] = getattr(cf, "mean_sufficiency", None)
            all_metrics["completeness"] = getattr(cf, "mean_completeness", None)
            all_metrics["necessity"] = getattr(cf, "mean_necessity", None)
            all_metrics["independence"] = getattr(cf, "mean_independence", None)

        # Overall faithfulness
        all_metrics["overall_faithfulness"] = getattr(faithfulness_data, "overall_faithfulness", None)

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


def _save_json(data: dict, path: Path):
    """Save dict as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def save_results(result: ExperimentResult, run_dir: str | Path, logger=None):
    """
    Save experiment results to disk with clean folder structure.

    Creates:
        config.json           - Experiment configuration
        circuits.json         - Circuit masks and structures (run-level)
        profiling/profiling.json - Profiling data (run-level)
        trials/
            {trial_id}/
                setup.json        - Trial setup parameters
                metrics.json      - All metrics and analysis results
                tensors.pt        - Tensor data (test samples, activations, weights)
                all_gates/model.pt
                gates/{gate}/full/decomposed_model.pt
                gates/{gate}/{sc_idx}/decomposed_model.pt
    """
    os.makedirs(run_dir, exist_ok=True)
    run_dir = Path(run_dir)

    # Save experiment config
    config_data = filter_non_serializable(asdict(result.config))
    _save_json(config_data, run_dir / "config.json")

    # Save run-level summary and explanation
    run_summary = _generate_run_summary(result)
    _save_json(run_summary, run_dir / "summary.json")
    _save_explanation(run_dir / "explanation.md", RUN_EXPLANATION)

    # Save circuits at run level (from first trial - all trials share same circuit enumeration)
    _save_run_level_circuits(result, run_dir)

    # Save profiling at run level (aggregate from all trials)
    _save_run_level_profiling(result, run_dir)

    # Save subcircuits folder with rankings
    _save_subcircuits_folder(result, run_dir)

    # Create trials directory
    trials_dir = run_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Save each trial
    for trial_id, trial in result.trials.items():
        trial_dir = trials_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        # 1. Trial-level summary and explanation
        trial_summary = _generate_trial_summary(trial)
        _save_json(trial_summary, trial_dir / "summary.json")
        _save_explanation(trial_dir / "explanation.md", TRIAL_EXPLANATION)

        # 2. Setup JSON - trial parameters
        setup_data = filter_non_serializable(asdict(trial.setup))
        _save_json(setup_data, trial_dir / "setup.json")

        # 3. Metrics JSON - training and analysis results
        metrics_data = filter_non_serializable(asdict(trial.metrics))
        metrics_data["status"] = trial.status
        metrics_data["trial_id"] = trial.trial_id
        _save_json(metrics_data, trial_dir / "metrics.json")

        # 4. Tensors PT - all tensor data
        _save_tensors(trial, trial_dir, logger)

        # 5. Models
        _save_models(trial, trial_dir, logger)

        # 6. Per-gate summaries
        _save_gate_summaries(trial, trial_dir)

    logger and logger.info(f"Saved results to {run_dir}")


def _simplify_circuit_masks(subcircuit: dict) -> dict:
    """Simplify circuit masks for storage per T1.d.2.

    Per spec:
    - node_masks: Only include middle layers (input is never masked,
      output is always 1 for single gate)
    - edge_masks: Organize as if we have 1 output (since we always
      do separate_into_k_mlps)

    Args:
        subcircuit: Dict with 'node_masks' and 'edge_masks' keys

    Returns:
        Simplified subcircuit dict with only middle layer masks
    """
    node_masks = subcircuit.get("node_masks", [])
    edge_masks = subcircuit.get("edge_masks", [])

    # node_masks: exclude input (index 0) and output (last index)
    # Middle layers are indices 1 to n-2 inclusive
    simplified_node_masks = node_masks[1:-1] if len(node_masks) > 2 else []

    # edge_masks: keep as-is since each gate will use adapted version
    # but we could simplify to single output if needed
    # For now, keep edge_masks but note they apply to full architecture
    simplified_edge_masks = edge_masks

    return {
        "idx": subcircuit.get("idx"),
        "node_masks": simplified_node_masks,
        "edge_masks": simplified_edge_masks,
    }


def _save_run_level_circuits(result: ExperimentResult, run_dir: Path):
    """Save circuits.json at run level from first trial.

    Saves simplified circuit masks per T1.d.2:
    - node_masks: Only middle layers (not input/output)
    - edge_masks: Full connections (used with adapt_masks_for_gate)
    - subcircuit_idx, node_mask_idx, edge_mask_idx: Index mappings
    """
    # Get circuits from first trial (all trials share the same circuit enumeration)
    first_trial = next(iter(result.trials.values()), None)
    if first_trial is None:
        return

    # Track unique node patterns to assign node_mask_idx
    seen_node_patterns = {}  # tuple(hidden_masks) -> node_mask_idx
    node_mask_counter = 0
    edge_mask_counts = {}  # node_mask_idx -> count (for edge_mask_idx)

    simplified_subcircuits = []
    for sc in first_trial.subcircuits:
        subcircuit_idx = sc.get("idx", len(simplified_subcircuits))
        node_masks = sc.get("node_masks", [])
        edge_masks = sc.get("edge_masks", [])

        # Simplify node_masks: exclude input (index 0) and output (last index)
        simplified_node_masks = node_masks[1:-1] if len(node_masks) > 2 else []

        # Extract hidden layer pattern for node_mask_idx assignment
        hidden_pattern = tuple(tuple(m) for m in simplified_node_masks)

        # Assign node_mask_idx
        if hidden_pattern not in seen_node_patterns:
            seen_node_patterns[hidden_pattern] = node_mask_counter
            edge_mask_counts[node_mask_counter] = 0
            node_mask_counter += 1
        node_mask_idx = seen_node_patterns[hidden_pattern]

        # Assign edge_mask_idx (increments for each subcircuit with same node pattern)
        edge_mask_idx = edge_mask_counts[node_mask_idx]
        edge_mask_counts[node_mask_idx] += 1

        simplified_subcircuits.append({
            "subcircuit_idx": subcircuit_idx,
            "node_mask_idx": node_mask_idx,
            "edge_mask_idx": edge_mask_idx,
            "node_masks": simplified_node_masks,
            "edge_masks": edge_masks,
        })

    circuits_data = {
        "subcircuits": simplified_subcircuits,
        "subcircuit_structure_analysis": first_trial.subcircuit_structure_analysis,
    }
    _save_json(circuits_data, run_dir / "circuits.json")


def _save_run_level_profiling(result: ExperimentResult, run_dir: Path):
    """Save profiling data at run level (aggregated from all trials)."""
    profiling_dir = run_dir / "profiling"
    profiling_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate profiling from all trials
    all_profiling = {}
    for trial_id, trial in result.trials.items():
        profiling_data = filter_non_serializable(asdict(trial.profiling))
        all_profiling[trial_id] = profiling_data

    _save_json(all_profiling, profiling_dir / "profiling.json")


def _save_tensors(trial, trial_dir: Path, logger=None):
    """Save trial tensors to tensors.pt."""
    tensors_path = trial_dir / "tensors.pt"
    data = {}

    if trial.test_x is not None:
        data["test_x"] = trial.test_x.cpu()
    if trial.test_y is not None:
        data["test_y"] = trial.test_y.cpu()
    if trial.test_y_pred is not None:
        data["test_y_pred"] = trial.test_y_pred.cpu()
    if trial.activations is not None:
        data["activations"] = [a.cpu() for a in trial.activations]
    if trial.canonical_activations is not None:
        data["canonical_activations"] = {
            label: [a.cpu() for a in acts]
            for label, acts in trial.canonical_activations.items()
        }
    if trial.mean_activations_by_range is not None:
        data["mean_activations_by_range"] = {
            label: [a.cpu() for a in acts]
            for label, acts in trial.mean_activations_by_range.items()
        }
    if trial.layer_weights is not None:
        data["layer_weights"] = [w.cpu() for w in trial.layer_weights]
    if trial.layer_biases is not None:
        data["layer_biases"] = [b.cpu() for b in trial.layer_biases]

    if data:
        torch.save(data, tensors_path)
        logger and logger.info(f"Saved tensors to {tensors_path}")


def _save_models(trial, trial_dir: Path, logger=None):
    """Save model files."""
    # All gates model
    all_gates_dir = trial_dir / "all_gates"
    all_gates_dir.mkdir(parents=True, exist_ok=True)

    if trial.model is not None:
        model_path = all_gates_dir / "model.pt"
        trial.model.save_to_file(str(model_path))


def _generate_run_summary(result: ExperimentResult) -> dict:
    """Generate run-level summary with ranked results across all trials and gates."""
    # Get epsilon from config (default 0.2)
    epsilon = 0.2
    if hasattr(result.config, "base_trial") and hasattr(result.config.base_trial, "constraints"):
        epsilon = getattr(result.config.base_trial.constraints, "epsilon", 0.2)

    summary = {
        "experiment_id": result.experiment_id,
        "n_trials": len(result.trials),
        "gates": list(result.config.target_logic_gates),
        "architecture": {
            "widths": result.config.widths,
            "depths": result.config.depths,
        },
        "epsilon": epsilon,
        "best_subcircuits_by_gate": {},
        "trial_results": [],
    }

    # Aggregate results by gate across all trials
    gate_results = {}  # gate -> list of subcircuit result dicts
    for trial_id, trial in result.trials.items():
        trial_summary = {
            "trial_id": trial_id,
            "test_acc": trial.metrics.test_acc,
            "val_acc": trial.metrics.val_acc,
            "gates": {},
        }

        # Get faithfulness data for this trial
        bests_keys = trial.metrics.per_gate_bests or {}
        bests_faith = trial.metrics.per_gate_bests_faith or {}

        for gate, gate_metrics in trial.metrics.per_gate_metrics.items():
            if gate not in gate_results:
                gate_results[gate] = []

            # Build idx -> faithfulness mapping
            gate_best_keys = bests_keys.get(gate, [])
            gate_faith_list = bests_faith.get(gate, [])
            idx_to_faith = {}
            for k, faith in zip(gate_best_keys, gate_faith_list):
                idx = k[0] if isinstance(k, (tuple, list)) else k
                if idx not in idx_to_faith:
                    idx_to_faith[idx] = faith

            # Get subcircuit metrics and find best ones
            sc_metrics = getattr(gate_metrics, "subcircuit_metrics", [])
            sorted_by_acc = sorted(sc_metrics, key=lambda x: getattr(x, "accuracy", 0) or 0, reverse=True)

            for sm in sorted_by_acc[:5]:  # Top 5 per gate
                acc = sm.accuracy or 0
                faith_data = idx_to_faith.get(sm.idx)
                ranked = filter_and_rank_subcircuit_metrics(sm, faithfulness_data=faith_data)
                gate_results[gate].append({
                    "trial_id": trial_id,
                    "subcircuit_idx": sm.idx,
                    "passes_epsilon": acc >= (1 - epsilon),
                    "ranked_metrics": ranked,
                })

            trial_summary["gates"][gate] = {
                "best_idx": sorted_by_acc[0].idx if sorted_by_acc else None,
                "best_acc": sorted_by_acc[0].accuracy if sorted_by_acc else None,
            }

        summary["trial_results"].append(trial_summary)

    # Rank and store best subcircuits per gate (by first metric value, typically accuracy)
    for gate, results in gate_results.items():
        def get_first_metric(r):
            vals = r.get("ranked_metrics", {}).get("metric_value", [])
            return vals[0] if vals else 0
        sorted_results = sorted(results, key=get_first_metric, reverse=True)
        summary["best_subcircuits_by_gate"][gate] = sorted_results[:10]

    return summary


def _generate_trial_summary(trial) -> dict:
    """Generate trial-level summary with ranked results per gate."""
    # Get epsilon from trial setup
    epsilon = 0.2
    if hasattr(trial, "setup") and hasattr(trial.setup, "constraints"):
        epsilon = getattr(trial.setup.constraints, "epsilon", 0.2)

    gates_summary = {}
    for gate, gate_metrics in trial.metrics.per_gate_metrics.items():
        sc_metrics = getattr(gate_metrics, "subcircuit_metrics", [])
        sorted_by_acc = sorted(sc_metrics, key=lambda x: getattr(x, "accuracy", 0) or 0, reverse=True)
        gates_summary[gate] = {
            "gate_accuracy": getattr(gate_metrics, "test_acc", None),
            "best_subcircuits": [
                {
                    "subcircuit_idx": sm.idx,
                    "accuracy": sm.accuracy,
                    "bit_similarity": sm.bit_similarity,
                    "passes_epsilon": (sm.accuracy or 0) >= (1 - epsilon),
                }
                for sm in sorted_by_acc[:5]
            ]
        }

    return {
        "trial_id": trial.trial_id,
        "status": trial.status,
        "test_acc": trial.metrics.test_acc,
        "val_acc": trial.metrics.val_acc,
        "avg_loss": trial.metrics.avg_loss,
        "epsilon": epsilon,
        "gates": gates_summary,
    }


def _generate_gate_summary(gate_name: str, gate_metrics, epsilon: float = 0.2) -> dict:
    """Generate gate-level summary with ranked subcircuits."""
    subcircuit_ranking = []
    for sm in getattr(gate_metrics, "subcircuit_metrics", []):
        acc = sm.accuracy or 0
        subcircuit_ranking.append({
            "subcircuit_idx": sm.idx,
            "accuracy": acc,
            "bit_similarity": sm.bit_similarity,
            "passes_epsilon": acc >= (1 - epsilon),
        })
    subcircuit_ranking.sort(key=lambda x: x.get("accuracy", 0) or 0, reverse=True)

    return {
        "gate_name": gate_name,
        "gate_accuracy": gate_metrics.test_acc if hasattr(gate_metrics, "test_acc") else None,
        "epsilon": epsilon,
        "n_subcircuits_passing": sum(1 for s in subcircuit_ranking if s.get("passes_epsilon")),
        "n_subcircuits_total": len(subcircuit_ranking),
        "ranked_subcircuits": subcircuit_ranking[:20],
    }


RUN_EXPLANATION = """# Run Summary

## Structure

- `summary.json`: Ranked results across all trials and gates
- `config.json`: Experiment configuration
- `circuits.json`: All enumerated subcircuits (shared across trials)
- `profiling/`: Timing information
- `trials/`: Per-trial results

## Key Metrics

Let M = trained MLP, G_k = gate k's subnetwork, S_i = subcircuit i.

- **accuracy**: Pr[S_i(x) = G_k(x)] over test set
- **bit_similarity**: mean(1 - |S_i(x) - G_k(x)|) for binary outputs
- **faithfulness**: Combined metric from observational, interventional, counterfactual tests

## Reading summary.json

`best_subcircuits_by_gate[gate]` lists top subcircuits ranked by accuracy.
Each entry has: subcircuit_idx, accuracy, bit_similarity, faithfulness.
"""


TRIAL_EXPLANATION = """# Trial Summary

## Structure

- `summary.json`: Trial-level ranked results
- `setup.json`: Trial hyperparameters
- `metrics.json`: Full metrics data
- `tensors.pt`: Test data, activations, weights
- `all_gates/model.pt`: Trained MLP
- `gates/{name}/`: Per-gate subcircuit analysis

## Key Metrics

- **test_acc**: Accuracy on test set
- **val_acc**: Accuracy on validation set
- **gates.{name}.best_subcircuits**: Top subcircuits for each gate
"""


GATE_EXPLANATION = """# Gate Analysis

## Structure

- `summary.json`: Ranked subcircuits for this gate
- Subcircuit folders contain decomposed models

## Metrics

Let G = this gate's subnetwork, S = subcircuit.

- **accuracy**: Pr[S(x) = G(x)]
- **passes_epsilon**: accuracy >= 1 - ε (identifiability threshold)
- **ranked_subcircuits**: Sorted by accuracy descending
"""


def _save_explanation(path: Path, content: str):
    """Save explanation markdown file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _save_subcircuits_folder(result: ExperimentResult, run_dir: Path):
    """Save subcircuits/ folder with score rankings and mappings.

    Creates:
        subcircuits/
            subcircuit_score_ranking.json      - Rankings by gate with all faithfulness scores
            subcircuit_score_ranking_per_trial.json - Per-trial rankings
            mask_idx_map.json                  - Mapping (node_mask_idx, edge_mask_idx) -> subcircuit_idx
            explanation.md                     - How to read these files
    """
    subcircuits_dir = run_dir / "subcircuits"
    subcircuits_dir.mkdir(parents=True, exist_ok=True)

    # Get epsilon and architecture from config
    epsilon = 0.2
    width = 3
    depth = 2
    if hasattr(result.config, "base_trial"):
        if hasattr(result.config.base_trial, "constraints"):
            epsilon = getattr(result.config.base_trial.constraints, "epsilon", 0.2)
        if hasattr(result.config.base_trial, "model_params"):
            width = getattr(result.config.base_trial.model_params, "width", 3)
            depth = getattr(result.config.base_trial.model_params, "depth", 2)

    # Aggregate scores by (gate, subcircuit_idx) across all trials
    # Now including faithfulness scores (observational, interventional, counterfactual)
    gate_subcircuit_data = {}  # gate -> subcircuit_idx -> list of score dicts

    per_trial_rankings = {}  # trial_id -> gate -> rankings

    for trial_id, trial in result.trials.items():
        per_trial_rankings[trial_id] = {}
        bests_keys = trial.metrics.per_gate_bests or {}
        bests_faith = trial.metrics.per_gate_bests_faith or {}

        for gate, gate_metrics in trial.metrics.per_gate_metrics.items():
            if gate not in gate_subcircuit_data:
                gate_subcircuit_data[gate] = {}

            # Build node_idx -> first faithfulness mapping for this gate
            # Keys are (node_idx, edge_var_idx) tuples, we want to map by node_idx
            gate_best_keys = bests_keys.get(gate, [])
            gate_faith_list = bests_faith.get(gate, [])
            node_idx_to_faith = {}  # node_idx -> first FaithfulnessMetrics
            for k, faith in zip(gate_best_keys, gate_faith_list):
                if isinstance(k, (tuple, list)):
                    node_idx = k[0]
                else:
                    node_idx = k
                # Only keep first faithfulness per node_idx (best edge variant)
                if node_idx not in node_idx_to_faith:
                    node_idx_to_faith[node_idx] = faith

            sc_metrics = getattr(gate_metrics, "subcircuit_metrics", [])
            trial_ranking = []

            for sm in sc_metrics:
                idx = sm.idx
                acc = sm.accuracy or 0

                # Get faithfulness data for this subcircuit by node_idx
                faith_data = node_idx_to_faith.get(idx)

                # Extract faithfulness scores
                obs_score = None
                int_score = None
                cf_score = None
                if faith_data:
                    if hasattr(faith_data, 'observational') and faith_data.observational:
                        obs = faith_data.observational
                        obs_score = getattr(obs, 'overall_observational', None)
                    if hasattr(faith_data, 'interventional') and faith_data.interventional:
                        int_data = faith_data.interventional
                        int_score = getattr(int_data, 'overall_interventional', None)
                    if hasattr(faith_data, 'counterfactual') and faith_data.counterfactual:
                        cf_data = faith_data.counterfactual
                        cf_score = getattr(cf_data, 'overall_counterfactual', None)

                if idx not in gate_subcircuit_data[gate]:
                    gate_subcircuit_data[gate][idx] = []

                gate_subcircuit_data[gate][idx].append({
                    "accuracy": acc,
                    "bit_similarity": sm.bit_similarity,
                    "observational": obs_score,
                    "interventional": int_score,
                    "counterfactual": cf_score,
                })

                trial_ranking.append({
                    "subcircuit_idx": idx,
                    "accuracy": acc,
                    "bit_similarity": sm.bit_similarity,
                    "observational": obs_score,
                    "interventional": int_score,
                    "counterfactual": cf_score,
                })

            trial_ranking.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
            per_trial_rankings[trial_id][gate] = trial_ranking

    # Compute averaged rankings per gate with all scores
    averaged_rankings = {}
    for gate, subcircuit_data in gate_subcircuit_data.items():
        gate_rankings = []
        for idx, data_list in subcircuit_data.items():
            n = len(data_list)
            accs = [d["accuracy"] for d in data_list]
            avg_acc = sum(accs) / n if n > 0 else 0

            # Average faithfulness scores (ignoring None)
            obs_scores = [d["observational"] for d in data_list if d["observational"] is not None]
            int_scores = [d["interventional"] for d in data_list if d["interventional"] is not None]
            cf_scores = [d["counterfactual"] for d in data_list if d["counterfactual"] is not None]

            gate_rankings.append({
                "subcircuit_idx": idx,
                "avg_accuracy": avg_acc,
                "n_trials": n,
                "min_accuracy": min(accs) if accs else 0,
                "max_accuracy": max(accs) if accs else 0,
                "passes_epsilon": avg_acc >= (1 - epsilon),
                "observational": {
                    "avg": sum(obs_scores) / len(obs_scores) if obs_scores else None,
                    "n": len(obs_scores),
                },
                "interventional": {
                    "avg": sum(int_scores) / len(int_scores) if int_scores else None,
                    "n": len(int_scores),
                },
                "counterfactual": {
                    "avg": sum(cf_scores) / len(cf_scores) if cf_scores else None,
                    "n": len(cf_scores),
                },
            })
        gate_rankings.sort(key=lambda x: x.get("avg_accuracy", 0), reverse=True)
        averaged_rankings[gate] = gate_rankings

    # Save averaged rankings
    _save_json({
        "epsilon": epsilon,
        "n_trials": len(result.trials),
        "rankings_by_gate": averaged_rankings,
    }, subcircuits_dir / "subcircuit_score_ranking.json")

    # Save per-trial rankings
    _save_json({
        "epsilon": epsilon,
        "per_trial": per_trial_rankings,
    }, subcircuits_dir / "subcircuit_score_ranking_per_trial.json")

    # Create mask_idx_map with node_mask_idx, edge_mask_idx -> subcircuit_idx mapping
    first_trial = next(iter(result.trials.values()), None)
    if first_trial and first_trial.subcircuits:
        # Track unique node patterns to assign node_mask_idx
        seen_node_patterns = {}  # tuple(hidden_masks) -> node_mask_idx
        node_mask_counter = 0

        mask_idx_entries = []
        idx_mapping = {}  # (node_mask_idx, edge_mask_idx) -> subcircuit_idx

        for sc in first_trial.subcircuits:
            subcircuit_idx = sc.get("idx", len(mask_idx_entries))
            node_masks = sc.get("node_masks", [])
            edge_masks = sc.get("edge_masks", [])

            # Extract hidden layer masks only (exclude input/output)
            hidden_masks = tuple(tuple(m) for m in node_masks[1:-1]) if len(node_masks) > 2 else ()

            # Assign node_mask_idx
            if hidden_masks not in seen_node_patterns:
                seen_node_patterns[hidden_masks] = node_mask_counter
                node_mask_counter += 1
            node_mask_idx = seen_node_patterns[hidden_masks]

            # For full_edges_only=True, edge_mask_idx is always 0 for each node pattern
            # Count how many subcircuits share this node pattern to assign edge_mask_idx
            edge_mask_idx = sum(1 for e in mask_idx_entries if e["node_mask_idx"] == node_mask_idx)

            # Compute structural metrics
            hidden_nodes = node_masks[1:-1] if len(node_masks) > 2 else node_masks
            total_hidden = sum(len(m) for m in hidden_nodes)
            active_hidden = sum(sum(m) for m in hidden_nodes)
            node_sparsity = active_hidden / total_hidden if total_hidden > 0 else 0

            total_edges = sum(len(e) * len(e[0]) for e in edge_masks if e and e[0])
            active_edges = sum(sum(sum(row) for row in e) for e in edge_masks if e)
            edge_sparsity = active_edges / total_edges if total_edges > 0 else 0

            # Compute additional topological metrics
            n_active_nodes_per_layer = [sum(m) for m in hidden_nodes]
            connectivity_density = edge_sparsity  # Same as edge_sparsity for now

            mask_idx_entries.append({
                "subcircuit_idx": subcircuit_idx,
                "node_mask_idx": node_mask_idx,
                "edge_mask_idx": edge_mask_idx,
                "node_sparsity": round(node_sparsity, 4),
                "edge_sparsity": round(edge_sparsity, 4),
                "n_hidden_layers": len(hidden_nodes),
                "active_nodes_per_layer": n_active_nodes_per_layer,
                "connectivity_density": round(connectivity_density, 4),
            })

            idx_mapping[(node_mask_idx, edge_mask_idx)] = subcircuit_idx

        # Create readable mapping table
        mapping_table = [
            {"node_mask_idx": k[0], "edge_mask_idx": k[1], "subcircuit_idx": v}
            for k, v in sorted(idx_mapping.items())
        ]

        _save_json({
            "description": "Mapping (node_mask_idx, edge_mask_idx) -> subcircuit_idx with structural metrics",
            "architecture": {"width": width, "depth": depth},
            "n_unique_node_patterns": node_mask_counter,
            "mapping": mapping_table,
            "subcircuits": mask_idx_entries,
        }, subcircuits_dir / "mask_idx_map.json")

    # Save summary.json with key statistics
    n_subcircuits = len(first_trial.subcircuits) if first_trial and first_trial.subcircuits else 0
    summary = {
        "n_subcircuits": n_subcircuits,
        "n_trials": len(result.trials),
        "epsilon": epsilon,
        "gates": list(gate_subcircuit_data.keys()),
        "best_by_gate": {
            gate: rankings[0] if rankings else None
            for gate, rankings in averaged_rankings.items()
        },
        "n_passing_by_gate": {
            gate: sum(1 for r in rankings if r.get("passes_epsilon", False))
            for gate, rankings in averaged_rankings.items()
        },
    }
    _save_json(summary, subcircuits_dir / "summary.json")

    # Generate circuit diagrams (T1.h)
    _generate_circuit_diagrams(first_trial, subcircuits_dir)

    # Save explanation
    _save_explanation(subcircuits_dir / "explanation.md", SUBCIRCUITS_EXPLANATION)


def _generate_circuit_diagrams(trial, subcircuits_dir: Path):
    """Generate circuit diagrams for all subcircuits.

    Creates:
        circuit_diagrams/
            node_masks/{node_mask_idx}.png
            edge_masks/{edge_mask_idx}.png
            subcircuit_masks/{subcircuit_idx}.png
    """
    from src.circuit import Circuit

    if not trial or not trial.subcircuits:
        return

    diagrams_dir = subcircuits_dir / "circuit_diagrams"
    node_masks_dir = diagrams_dir / "node_masks"
    edge_masks_dir = diagrams_dir / "edge_masks"
    subcircuit_masks_dir = diagrams_dir / "subcircuit_masks"

    node_masks_dir.mkdir(parents=True, exist_ok=True)
    edge_masks_dir.mkdir(parents=True, exist_ok=True)
    subcircuit_masks_dir.mkdir(parents=True, exist_ok=True)

    # Track unique node patterns
    seen_node_patterns = {}
    node_mask_idx_counter = 0

    max_diagrams = min(48, len(trial.subcircuits))

    for sc_data in trial.subcircuits[:max_diagrams]:
        subcircuit_idx = sc_data.get("idx", 0)
        try:
            circuit = Circuit.from_dict(sc_data)

            # subcircuit_masks/{subcircuit_idx}.png
            sc_path = subcircuit_masks_dir / f"{subcircuit_idx}.png"
            circuit.visualize(file_path=str(sc_path), node_size="small")

            # Track unique node patterns for node_masks/
            node_masks = sc_data.get("node_masks", [])
            hidden_masks = tuple(tuple(m) for m in node_masks[1:-1]) if len(node_masks) > 2 else ()

            if hidden_masks not in seen_node_patterns:
                seen_node_patterns[hidden_masks] = node_mask_idx_counter
                nm_path = node_masks_dir / f"{node_mask_idx_counter}.png"
                circuit.visualize(file_path=str(nm_path), node_size="small")
                node_mask_idx_counter += 1
        except Exception:
            pass  # Silently skip diagram generation failures

    # edge_masks are equivalent to node_masks with full_edges_only=True
    import shutil
    for node_mask_idx in range(min(node_mask_idx_counter, 20)):
        src = node_masks_dir / f"{node_mask_idx}.png"
        dst = edge_masks_dir / f"{node_mask_idx}.png"
        if src.exists():
            shutil.copy(src, dst)


SUBCIRCUITS_EXPLANATION = """# Subcircuits Analysis

## Structure

- `subcircuit_score_ranking.json`: Rankings by gate with faithfulness scores
- `subcircuit_score_ranking_per_trial.json`: Per-trial granular rankings
- `mask_idx_map.json`: (node_mask_idx, edge_mask_idx) → subcircuit_idx mapping
- `circuit_diagrams/`: Visual diagrams
  - `node_masks/{idx}.png`: Diagrams for unique node patterns
  - `edge_masks/{idx}.png`: Diagrams for edge configurations
  - `subcircuit_masks/{idx}.png`: Diagrams for each subcircuit

## Index Mapping

subcircuit_idx = f(node_mask_idx, edge_mask_idx)

- **node_mask_idx**: Identifies which hidden nodes are active
- **edge_mask_idx**: Identifies edge configuration for that node pattern
- With full_edges_only=True, typically edge_mask_idx=0

## Key Metrics

For subcircuit S_i and gate G:

- **accuracy**: Pr[S_i(x) = G(x)] — behavioral match
- **observational**: Robustness under input perturbations
- **interventional**: Faithfulness under activation patching
- **counterfactual**: Necessity/sufficiency scores
- **node_sparsity**: |active hidden nodes| / |total hidden nodes|
- **edge_sparsity**: |active edges| / |total possible edges|
- **passes_epsilon**: accuracy ≥ 1 - ε

## Reading Files

`rankings_by_gate[gate]` in `subcircuit_score_ranking.json` lists
subcircuits sorted by avg_accuracy. Each entry includes all faithfulness
scores averaged across trials.
"""


def _save_gate_summaries(trial, trial_dir: Path):
    """Save per-gate summary.json, explanation.md, and samples files."""
    # Get epsilon from trial setup
    epsilon = 0.2
    if hasattr(trial, "setup") and hasattr(trial.setup, "constraints"):
        epsilon = getattr(trial.setup.constraints, "epsilon", 0.2)

    gates_dir = trial_dir / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)

    # Get faithfulness results keyed by gate
    bests_keys = trial.metrics.per_gate_bests or {}
    bests_faith = trial.metrics.per_gate_bests_faith or {}

    for gate_name, gate_metrics in trial.metrics.per_gate_metrics.items():
        gate_dir = gates_dir / gate_name
        gate_dir.mkdir(parents=True, exist_ok=True)

        gate_summary = _generate_gate_summary(gate_name, gate_metrics, epsilon)
        _save_json(gate_summary, gate_dir / "summary.json")
        _save_explanation(gate_dir / "explanation.md", GATE_EXPLANATION)

        # Save samples for each subcircuit (T1.i)
        gate_keys = bests_keys.get(gate_name, [])
        gate_faith = bests_faith.get(gate_name, [])
        sc_metrics = getattr(gate_metrics, "subcircuit_metrics", [])
        _save_subcircuit_samples(gate_dir, gate_keys, gate_faith, sc_metrics)


def _save_subcircuit_samples(gate_dir: Path, keys: list, faith_results: list, sc_metrics_list: list = None):
    """Save samples in folder structure per T1.i.

    Args:
        gate_dir: Path to gate directory
        keys: List of (node_mask_idx, edge_mask_idx) keys
        faith_results: List of FaithfulnessMetrics objects
        sc_metrics_list: List of SubcircuitMetrics objects for ranking

    Creates:
        {node_mask_idx}/{edge_mask_idx}/
            observational/
                noise_perturbations/samples.json
                out_distribution_transformations/samples.json
            interventional/
                in_circuit/in_distribution/samples.json
                in_circuit/out_distribution/samples.json
                out_circuit/in_distribution/samples.json
                out_circuit/out_distribution/samples.json
            counterfactual/
                sufficiency/samples.json
                completeness/samples.json
                necessity/samples.json
                independence/samples.json
    """
    from dataclasses import asdict

    # Build index from sc_metrics_list by idx
    sc_metrics_by_idx = {}
    if sc_metrics_list:
        for sm in sc_metrics_list:
            sc_metrics_by_idx[sm.idx] = sm

    for key, faith in zip(keys, faith_results):
        if faith is None:
            continue

        # Extract node_mask_idx and edge_mask_idx from key
        if isinstance(key, (tuple, list)) and len(key) >= 2:
            node_mask_idx, edge_mask_idx = key[0], key[1]
        else:
            node_mask_idx, edge_mask_idx = key, 0

        # Get corresponding SubcircuitMetrics
        sc_metrics = sc_metrics_by_idx.get(node_mask_idx)

        # Get ranked metrics for this subcircuit
        ranked = filter_and_rank_subcircuit_metrics(sc_metrics, faithfulness_data=faith)

        sc_dir = gate_dir / str(node_mask_idx) / str(edge_mask_idx)

        # Save subcircuit-level summary with ranked metrics
        sc_summary = {
            "node_mask_idx": node_mask_idx,
            "edge_mask_idx": edge_mask_idx,
            "ranked_metrics": ranked,
            "subfolders": ["observational", "interventional", "counterfactual"],
        }
        sc_dir.mkdir(parents=True, exist_ok=True)
        _save_json(sc_summary, sc_dir / "summary.json")

        # Observational samples
        if hasattr(faith, 'observational') and faith.observational:
            obs = faith.observational
            obs_dir = sc_dir / "observational"
            obs_dir.mkdir(parents=True, exist_ok=True)

            # Observational summary
            obs_summary = {
                "overall_observational": getattr(obs, "overall_observational", None),
                "ranked_metrics": ranked,  # Include ranked metrics
                "subfolders": ["noise_perturbations", "out_distribution_transformations"],
            }

            # Noise perturbations
            if hasattr(obs, 'noise') and obs.noise:
                noise_dir = obs_dir / "noise_perturbations"
                noise_dir.mkdir(parents=True, exist_ok=True)
                obs_summary["noise_agreement_bit"] = getattr(obs.noise, "agreement_bit", None)
                obs_summary["noise_agreement_logit"] = getattr(obs.noise, "agreement_logit", None)
                if hasattr(obs.noise, 'samples') and obs.noise.samples:
                    samples = [asdict(s) for s in obs.noise.samples]
                    _save_json({"samples": samples, "n": len(samples)}, noise_dir / "samples.json")

            # OOD transformations
            if hasattr(obs, 'ood') and obs.ood:
                ood_dir = obs_dir / "out_distribution_transformations"
                ood_dir.mkdir(parents=True, exist_ok=True)
                obs_summary["ood_overall"] = getattr(obs.ood, "overall_agreement", None)
                if hasattr(obs.ood, 'samples') and obs.ood.samples:
                    samples = [asdict(s) for s in obs.ood.samples]
                    _save_json({"samples": samples, "n": len(samples)}, ood_dir / "samples.json")

            _save_json(obs_summary, obs_dir / "summary.json")

        # Interventional samples
        if hasattr(faith, 'interventional') and faith.interventional:
            intv = faith.interventional
            intv_dir = sc_dir / "interventional"
            intv_dir.mkdir(parents=True, exist_ok=True)

            # Interventional summary
            intv_summary = {
                "overall_interventional": getattr(intv, "overall_interventional", None),
                "mean_in_circuit_effect": getattr(intv, "mean_in_circuit_effect", None),
                "mean_out_circuit_effect": getattr(intv, "mean_out_circuit_effect", None),
                "ranked_metrics": ranked,  # Include ranked metrics
                "subfolders": ["in_circuit", "out_circuit"],
            }

            # In-circuit folder
            in_circuit_dir = intv_dir / "in_circuit"
            in_circuit_dir.mkdir(parents=True, exist_ok=True)
            in_circuit_summary = {"ranked_metrics": ranked, "subfolders": ["in_distribution", "out_distribution"]}

            if hasattr(intv, 'in_circuit_stats') and intv.in_circuit_stats:
                in_in_dir = in_circuit_dir / "in_distribution"
                in_in_dir.mkdir(parents=True, exist_ok=True)
                _save_json({"stats": intv.in_circuit_stats}, in_in_dir / "samples.json")
                in_circuit_summary["in_distribution_n_patches"] = len(intv.in_circuit_stats)

            if hasattr(intv, 'in_circuit_stats_ood') and intv.in_circuit_stats_ood:
                in_out_dir = in_circuit_dir / "out_distribution"
                in_out_dir.mkdir(parents=True, exist_ok=True)
                _save_json({"stats": intv.in_circuit_stats_ood}, in_out_dir / "samples.json")
                in_circuit_summary["out_distribution_n_patches"] = len(intv.in_circuit_stats_ood)

            _save_json(in_circuit_summary, in_circuit_dir / "summary.json")

            # Out-circuit folder
            out_circuit_dir = intv_dir / "out_circuit"
            out_circuit_dir.mkdir(parents=True, exist_ok=True)
            out_circuit_summary = {"ranked_metrics": ranked, "subfolders": ["in_distribution", "out_distribution"]}

            if hasattr(intv, 'out_circuit_stats') and intv.out_circuit_stats:
                out_in_dir = out_circuit_dir / "in_distribution"
                out_in_dir.mkdir(parents=True, exist_ok=True)
                _save_json({"stats": intv.out_circuit_stats}, out_in_dir / "samples.json")
                out_circuit_summary["in_distribution_n_patches"] = len(intv.out_circuit_stats)

            if hasattr(intv, 'out_circuit_stats_ood') and intv.out_circuit_stats_ood:
                out_out_dir = out_circuit_dir / "out_distribution"
                out_out_dir.mkdir(parents=True, exist_ok=True)
                _save_json({"stats": intv.out_circuit_stats_ood}, out_out_dir / "samples.json")
                out_circuit_summary["out_distribution_n_patches"] = len(intv.out_circuit_stats_ood)

            _save_json(out_circuit_summary, out_circuit_dir / "summary.json")
            _save_json(intv_summary, intv_dir / "summary.json")

        # Counterfactual samples
        if hasattr(faith, 'counterfactual') and faith.counterfactual:
            cf = faith.counterfactual
            cf_dir = sc_dir / "counterfactual"
            cf_dir.mkdir(parents=True, exist_ok=True)

            # Counterfactual summary (2x2 matrix)
            cf_summary = {
                "overall_counterfactual": getattr(cf, "overall_counterfactual", None),
                "mean_sufficiency": getattr(cf, "mean_sufficiency", None),
                "mean_completeness": getattr(cf, "mean_completeness", None),
                "mean_necessity": getattr(cf, "mean_necessity", None),
                "mean_independence": getattr(cf, "mean_independence", None),
                "ranked_metrics": ranked,  # Include ranked metrics
                "subfolders": ["sufficiency", "completeness", "necessity", "independence"],
            }

            # Sufficiency
            if hasattr(cf, 'sufficiency_effects') and cf.sufficiency_effects:
                suff_dir = cf_dir / "sufficiency"
                suff_dir.mkdir(parents=True, exist_ok=True)
                samples = [asdict(e) for e in cf.sufficiency_effects]
                _save_json({"samples": samples, "n": len(samples)}, suff_dir / "samples.json")
                cf_summary["sufficiency_n_samples"] = len(samples)

            # Completeness
            if hasattr(cf, 'completeness_effects') and cf.completeness_effects:
                comp_dir = cf_dir / "completeness"
                comp_dir.mkdir(parents=True, exist_ok=True)
                samples = [asdict(e) for e in cf.completeness_effects]
                _save_json({"samples": samples, "n": len(samples)}, comp_dir / "samples.json")
                cf_summary["completeness_n_samples"] = len(samples)

            # Necessity
            if hasattr(cf, 'necessity_effects') and cf.necessity_effects:
                nec_dir = cf_dir / "necessity"
                nec_dir.mkdir(parents=True, exist_ok=True)
                samples = [asdict(e) for e in cf.necessity_effects]
                _save_json({"samples": samples, "n": len(samples)}, nec_dir / "samples.json")
                cf_summary["necessity_n_samples"] = len(samples)

            # Independence
            if hasattr(cf, 'independence_effects') and cf.independence_effects:
                ind_dir = cf_dir / "independence"
                ind_dir.mkdir(parents=True, exist_ok=True)
                samples = [asdict(e) for e in cf.independence_effects]
                _save_json({"samples": samples, "n": len(samples)}, ind_dir / "samples.json")
                cf_summary["independence_n_samples"] = len(samples)

            _save_json(cf_summary, cf_dir / "summary.json")

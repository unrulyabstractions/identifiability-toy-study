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


def _save_run_level_circuits(result: ExperimentResult, run_dir: Path):
    """Save circuits.json at run level from first trial."""
    # Get circuits from first trial (all trials share the same circuit enumeration)
    first_trial = next(iter(result.trials.values()), None)
    if first_trial is None:
        return

    circuits_data = {
        "subcircuits": first_trial.subcircuits,
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
    gate_results = {}
    for trial_id, trial in result.trials.items():
        trial_summary = {
            "trial_id": trial_id,
            "test_acc": trial.metrics.test_acc,
            "val_acc": trial.metrics.val_acc,
            "gates": {},
        }

        for gate, gate_metrics in trial.metrics.per_gate_metrics.items():
            if gate not in gate_results:
                gate_results[gate] = []

            # Get subcircuit metrics and find best ones
            sc_metrics = getattr(gate_metrics, "subcircuit_metrics", [])
            sorted_by_acc = sorted(sc_metrics, key=lambda x: getattr(x, "accuracy", 0) or 0, reverse=True)

            for sm in sorted_by_acc[:5]:  # Top 5 per gate
                acc = sm.accuracy or 0
                gate_results[gate].append({
                    "trial_id": trial_id,
                    "subcircuit_idx": sm.idx,
                    "accuracy": acc,
                    "bit_similarity": sm.bit_similarity,
                    "passes_epsilon": acc >= (1 - epsilon),
                })

            trial_summary["gates"][gate] = {
                "best_idx": sorted_by_acc[0].idx if sorted_by_acc else None,
                "best_acc": sorted_by_acc[0].accuracy if sorted_by_acc else None,
            }

        summary["trial_results"].append(trial_summary)

    # Rank and store best subcircuits per gate
    for gate, results in gate_results.items():
        sorted_results = sorted(results, key=lambda x: x.get("accuracy", 0) or 0, reverse=True)
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


def _save_gate_summaries(trial, trial_dir: Path):
    """Save per-gate summary.json and explanation.md files."""
    # Get epsilon from trial setup
    epsilon = 0.2
    if hasattr(trial, "setup") and hasattr(trial.setup, "constraints"):
        epsilon = getattr(trial.setup.constraints, "epsilon", 0.2)

    gates_dir = trial_dir / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)

    for gate_name, gate_metrics in trial.metrics.per_gate_metrics.items():
        gate_dir = gates_dir / gate_name
        gate_dir.mkdir(parents=True, exist_ok=True)

        gate_summary = _generate_gate_summary(gate_name, gate_metrics, epsilon)
        _save_json(gate_summary, gate_dir / "summary.json")
        _save_explanation(gate_dir / "explanation.md", GATE_EXPLANATION)

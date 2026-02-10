"""
Save functions for experiment results.

Changes to this module should be reflected in README.md.

Structure:
    runs/run_{timestamp}/
        config.json           - ExperimentConfig only
        circuits.json         - Subcircuit masks and structure analysis (run-level)
        profiling/
            profiling.json    - Timing data (events, phase durations)
        trials/
            {trial_id}/
                setup.json        - TrialSetup
                metrics.json      - Metrics (training, per-gate, robustness, faithfulness)
                tensors.pt        - Test data, activations, weights
                all_gates/
                    model.pt
                gates/
                    {gate_name}/
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

        # 1. Setup JSON - trial parameters
        setup_data = filter_non_serializable(asdict(trial.setup))
        _save_json(setup_data, trial_dir / "setup.json")

        # 2. Metrics JSON - training and analysis results
        metrics_data = filter_non_serializable(asdict(trial.metrics))
        metrics_data["status"] = trial.status
        metrics_data["trial_id"] = trial.trial_id
        _save_json(metrics_data, trial_dir / "metrics.json")

        # 3. Tensors PT - all tensor data
        _save_tensors(trial, trial_dir, logger)

        # 4. Models
        _save_models(trial, trial_dir, logger)

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

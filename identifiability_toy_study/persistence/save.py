"""
Save functions for experiment results.

Changes to this module should be reflected in README.md.

Structure:
    runs/run_{timestamp}/
        config.json           - ExperimentConfig only
        {trial_id}/
            setup.json        - TrialSetup
            metrics.json      - Metrics (training, per-gate, robustness, faithfulness)
            circuits.json     - Subcircuit masks and structure analysis
            tensors.pt        - Test data, activations, weights
            profiling/
                profiling.json - Timing data (events, phase durations)
            all_gates/
                model.pt
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
            {gate_name}/
                full/
                    decomposed_model.pt
                {subcircuit_idx}/
                    decomposed_model.pt
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from ..common.schemas import ExperimentResult
from ..common.utils import filter_non_serializable


def _save_json(data: dict, path: Path):
    """Save dict as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def save_results(result: ExperimentResult, run_dir: str | Path, logger=None):
    """
    Save experiment results to disk with clean folder structure.

    Creates:
        config.json           - Experiment configuration
        {trial_id}/
            setup.json        - Trial setup parameters
            metrics.json      - All metrics and analysis results
            circuits.json     - Circuit masks and structures
            tensors.pt        - Tensor data (test samples, activations, weights)
            all_gates/model.pt, decomposed_model.pt
            {gate}/full/decomposed_model.pt
            {gate}/{sc_idx}/decomposed_model.pt
    """
    os.makedirs(run_dir, exist_ok=True)
    run_dir = Path(run_dir)

    # Save experiment config
    config_data = filter_non_serializable(asdict(result.config))
    _save_json(config_data, run_dir / "config.json")

    # Save each trial
    for trial_id, trial in result.trials.items():
        trial_dir = run_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        # 1. Setup JSON - trial parameters
        setup_data = filter_non_serializable(asdict(trial.setup))
        _save_json(setup_data, trial_dir / "setup.json")

        # 2. Metrics JSON - training and analysis results
        metrics_data = filter_non_serializable(asdict(trial.metrics))
        metrics_data["status"] = trial.status
        metrics_data["trial_id"] = trial.trial_id
        _save_json(metrics_data, trial_dir / "metrics.json")

        # 3. Circuits JSON - subcircuit masks and structure
        circuits_data = {
            "subcircuits": trial.subcircuits,  # Already list of dicts
            "subcircuit_structure_analysis": trial.subcircuit_structure_analysis,
            "decomposed_subcircuit_indices": dict(trial.decomposed_subcircuit_indices),
        }
        _save_json(circuits_data, trial_dir / "circuits.json")

        # 4. Profiling JSON - timing data (in profiling/ folder)
        profiling_dir = trial_dir / "profiling"
        profiling_dir.mkdir(parents=True, exist_ok=True)
        profiling_data = filter_non_serializable(asdict(trial.profiling))
        _save_json(profiling_data, profiling_dir / "profiling.json")

        # 5. Tensors PT - all tensor data
        _save_tensors(trial, trial_dir, logger)

        # 6. Models
        _save_models(trial, trial_dir, logger)

    logger and logger.info(f"Saved results to {run_dir}")


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

    # Save SPD to new spd/ folder structure
    _save_spd(trial, trial_dir, logger)

    # Per-gate decomposed models
    for gate_name, decomposed in trial.decomposed_gate_models.items():
        full_gate_dir = trial_dir / gate_name / "full"
        full_gate_dir.mkdir(parents=True, exist_ok=True)
        path = full_gate_dir / "decomposed_model.pt"
        decomposed.save(str(path))

    # Per-subcircuit decomposed models
    for gate_name, decomposed_dict in trial.decomposed_subcircuits.items():
        for subcircuit_idx, decomposed in decomposed_dict.items():
            subcircuit_dir = trial_dir / gate_name / str(subcircuit_idx)
            subcircuit_dir.mkdir(parents=True, exist_ok=True)
            path = subcircuit_dir / "decomposed_model.pt"
            decomposed.save(str(path))


def _save_spd(trial, trial_dir: Path, logger=None):
    """Save SPD decomposition to trial_dir/spd/{config_id}/ folders.

    Each SPD config gets its own subfolder with:
        config.json         - SPD configuration used
        decomposed_model.pt - Trained decomposed model
        estimate.json       - Subcircuit clustering results
        visualizations/     - Analysis plots
    """
    from ..spd_subcircuits import save_spd_estimate

    # Must have sweep results
    if not trial.decomposed_models_sweep:
        return

    spd_base_dir = trial_dir / "spd"
    spd_base_dir.mkdir(parents=True, exist_ok=True)

    gate_names = trial.setup.model_params.logic_gates
    n_inputs = 2  # Boolean gates have 2 inputs

    # Save each config's results to its subfolder
    for config_id, decomposed in trial.decomposed_models_sweep.items():
        config_dir = spd_base_dir / config_id
        config_dir.mkdir(parents=True, exist_ok=True)

        # Find matching config
        all_configs = [trial.setup.spd_config] + (trial.setup.spd_sweep_configs or [])
        spd_config = next(c for c in all_configs if c.get_config_id() == config_id)

        # Save SPD config
        config_data = filter_non_serializable(asdict(spd_config))
        _save_json(config_data, config_dir / "config.json")

        # Save decomposed model
        decomposed_path = config_dir / "decomposed_model.pt"
        decomposed.save(str(decomposed_path))
        logger and logger.info(f"Saved decomposed model to {decomposed_path}")

        # Save SPD subcircuit estimate
        estimate = trial.spd_subcircuit_estimates_sweep.get(config_id)
        if estimate is not None:
            save_spd_estimate(estimate, config_dir)

        # Run and save SPD analysis with visualizations
        if trial.model is not None:
            try:
                from ..spd_analysis import analyze_and_visualize_spd

                analyze_and_visualize_spd(
                    decomposed_model=decomposed,
                    target_model=trial.model,
                    output_dir=config_dir,
                    gate_names=gate_names,
                    n_inputs=n_inputs,
                    device="cpu",
                )
                logger and logger.info(f"Saved SPD analysis to {config_dir}")
            except Exception as e:
                logger and logger.warning(f"SPD visualization failed for {config_id}: {e}")

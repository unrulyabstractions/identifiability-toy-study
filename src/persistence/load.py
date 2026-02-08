"""
Load functions for experiment results.

Changes to this module should be reflected in README.md.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..common.circuit import Circuit
from ..common.neural_model import MLP, DecomposedMLP
from ..common.schemas import (
    CounterfactualEffect,
    ExperimentConfig,
    ExperimentResult,
    FaithfulnessMetrics,
    GateMetrics,
    InterventionSample,
    PatchStatistics,
    RobustnessMetrics,
    RobustnessSample,
    SubcircuitMetrics,
    TrialResult,
    TrialSetup,
)
from ..spd_internal.persistence import load_spd_estimate as _load_spd_estimate


def get_all_runs(output_dir: str | Path) -> list[Path]:
    """Get all run directories in output_dir, sorted by name (newest first)."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    runs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    return sorted(runs, reverse=True)


def get_trial_dirs(run_dir: str | Path) -> list[Path]:
    """Get all trial directories in a run directory."""
    run_dir = Path(run_dir)
    return [
        d
        for d in run_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]


def load_config(run_dir: str | Path) -> dict:
    """Load config.json from run directory."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_trial_setup(trial_dir: str | Path) -> dict:
    """Load setup.json from trial directory."""
    trial_dir = Path(trial_dir)
    setup_path = trial_dir / "setup.json"
    if setup_path.exists():
        with open(setup_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_trial_metrics(trial_dir: str | Path) -> dict:
    """Load metrics.json from trial directory."""
    trial_dir = Path(trial_dir)
    metrics_path = trial_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_trial_circuits(trial_dir: str | Path) -> dict:
    """Load circuits.json from trial directory."""
    trial_dir = Path(trial_dir)
    circuits_path = trial_dir / "circuits.json"
    if circuits_path.exists():
        with open(circuits_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_trial_profiling(trial_dir: str | Path) -> dict:
    """Load profiling.json from trial's profiling/ folder."""
    trial_dir = Path(trial_dir)
    profiling_path = trial_dir / "profiling" / "profiling.json"
    if profiling_path.exists():
        with open(profiling_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_model(trial_dir: str | Path, device: str = "cpu") -> Optional[MLP]:
    """Load trained MLP model from trial directory."""
    trial_dir = Path(trial_dir)
    model_path = trial_dir / "all_gates" / "model.pt"
    if model_path.exists():
        return MLP.load_from_file(str(model_path), device=device)
    return None


def load_decomposed_model(
    trial_dir: str | Path,
    target_model: Optional[MLP] = None,
    device: str = "cpu",
) -> Optional[DecomposedMLP]:
    """Load decomposed model from trial directory.

    Tries new spd/ location first, then falls back to old all_gates/ location.
    """
    trial_dir = Path(trial_dir)

    # Try new spd/ location first
    spd_path = trial_dir / "spd" / "decomposed_model.pt"
    if spd_path.exists():
        return DecomposedMLP.load(str(spd_path), target_model=target_model, device=device)

    # Fall back to old location
    decomposed_path = trial_dir / "all_gates" / "decomposed_model.pt"
    if decomposed_path.exists():
        return DecomposedMLP.load(str(decomposed_path), target_model=target_model, device=device)

    return None


def load_spd_estimate(trial_dir: str | Path):
    """Load SPD subcircuit estimate from trial directory."""
    trial_dir = Path(trial_dir)
    spd_dir = trial_dir / "spd"
    if spd_dir.exists():
        return _load_spd_estimate(spd_dir)
    return None


def load_spd_analysis(trial_dir: str | Path) -> dict:
    """Load SPD analysis data from trial directory.

    Returns dict with:
        - clustering: Cluster assignments and labels
        - importance_matrix: Per-input component importances (numpy array)
        - coactivation_matrix: Component coactivation matrix (numpy array)
        - visualization_paths: Paths to visualization files
    """
    trial_dir = Path(trial_dir)
    spd_dir = trial_dir / "spd"
    result = {}

    if not spd_dir.exists():
        return result

    # Load clustering data
    clustering_path = spd_dir / "clustering" / "assignments.json"
    if clustering_path.exists():
        with open(clustering_path, encoding="utf-8") as f:
            result["clustering"] = json.load(f)

    # Load importance matrix
    imp_path = spd_dir / "clustering" / "importance_matrix.npy"
    if imp_path.exists():
        result["importance_matrix"] = np.load(imp_path)

    # Load coactivation matrix
    coact_path = spd_dir / "clustering" / "coactivation_matrix.npy"
    if coact_path.exists():
        result["coactivation_matrix"] = np.load(coact_path)

    # Find visualization paths
    viz_dir = spd_dir / "visualizations"
    if viz_dir.exists():
        result["visualization_paths"] = {
            f.stem: str(f.relative_to(trial_dir))
            for f in viz_dir.glob("*.png")
        }

    return result


def load_tensors(trial_dir: str | Path, device: str = "cpu") -> dict:
    """
    Load trial tensors from tensors.pt.

    Returns dict with keys:
        - test_x, test_y, test_y_pred: Test data
        - activations: List of per-layer activation tensors
        - canonical_activations: Dict of activations for binary inputs
        - layer_weights: Weight matrices per layer
    """
    trial_dir = Path(trial_dir)
    tensors_path = trial_dir / "tensors.pt"

    if not tensors_path.exists():
        return {}

    data = torch.load(tensors_path, map_location=device, weights_only=True)

    result = {}
    for key in ["test_x", "test_y", "test_y_pred"]:
        if key in data:
            result[key] = data[key].to(device)

    if "activations" in data:
        result["activations"] = [a.to(device) for a in data["activations"]]

    if "canonical_activations" in data:
        result["canonical_activations"] = {
            label: [a.to(device) for a in acts]
            for label, acts in data["canonical_activations"].items()
        }

    if "mean_activations_by_range" in data:
        result["mean_activations_by_range"] = {
            label: [a.to(device) for a in acts]
            for label, acts in data["mean_activations_by_range"].items()
        }

    if "layer_weights" in data:
        result["layer_weights"] = [w.to(device) for w in data["layer_weights"]]

    if "layer_biases" in data:
        result["layer_biases"] = [b.to(device) for b in data["layer_biases"]]

    return result


def load_results(run_dir: str | Path, device: str = "cpu"):
    """Load ExperimentResult from run directory for re-visualization."""
    run_dir = Path(run_dir)
    config_data = load_config(run_dir)

    # Create minimal config
    config = ExperimentConfig(
        device=device,
        spd_device=device,
    )

    result = ExperimentResult(config=config)

    for trial_dir in get_trial_dirs(run_dir):
        trial_id = trial_dir.name
        setup_data = load_trial_setup(trial_dir)
        metrics_data = load_trial_metrics(trial_dir)
        circuits_data = load_trial_circuits(trial_dir)
        tensors = load_tensors(trial_dir, device=device)

        # Create TrialSetup from saved data
        from ..common.schemas import ModelParams, FaithfulnessConfig

        model_params_data = setup_data.get("model_params", {})
        model_params = ModelParams(
            logic_gates=model_params_data.get("logic_gates", ["XOR"]),
            width=model_params_data.get("width", 3),
            depth=model_params_data.get("depth", 2),
        )

        faith_config_data = setup_data.get("faithfulness_config", {})
        faith_config = FaithfulnessConfig(
            n_interventions_per_patch=faith_config_data.get("n_interventions_per_patch", 100),
            n_counterfactual_pairs=faith_config_data.get("n_counterfactual_pairs", 10),
        )

        setup = TrialSetup(
            seed=setup_data.get("seed", 0),
            model_params=model_params,
            faithfulness_config=faith_config,
        )

        # Create TrialResult
        trial = TrialResult(setup=setup)
        trial.trial_id = trial_id
        trial.status = metrics_data.get("status", "LOADED")

        # Load model
        trial.model = load_model(trial_dir, device=device)

        # Load tensors
        trial.test_x = tensors.get("test_x")
        trial.test_y = tensors.get("test_y")
        trial.test_y_pred = tensors.get("test_y_pred")
        trial.activations = tensors.get("activations")
        trial.canonical_activations = tensors.get("canonical_activations")
        trial.mean_activations_by_range = tensors.get("mean_activations_by_range")
        trial.layer_weights = tensors.get("layer_weights")
        trial.layer_biases = tensors.get("layer_biases")

        # Load circuits
        trial.subcircuits = circuits_data.get("subcircuits", [])
        trial.subcircuit_structure_analysis = circuits_data.get("subcircuit_structure_analysis", [])

        # Load decomposed_subcircuit_indices from circuits.json
        decomposed_indices = circuits_data.get("decomposed_subcircuit_indices", {})
        for gate_name, indices in decomposed_indices.items():
            trial.decomposed_subcircuit_indices[gate_name] = indices

        # Load metrics (reconstruct nested structures)
        trial.metrics.avg_loss = metrics_data.get("avg_loss", 0)
        trial.metrics.val_acc = metrics_data.get("val_acc", 0)
        trial.metrics.test_acc = metrics_data.get("test_acc", 0)

        # Load per-gate metrics
        per_gate = metrics_data.get("per_gate_metrics", {})
        for gate_name, gate_data in per_gate.items():
            sc_metrics = [
                SubcircuitMetrics(**sm) for sm in gate_data.get("subcircuit_metrics", [])
            ]
            trial.metrics.per_gate_metrics[gate_name] = GateMetrics(
                test_acc=gate_data.get("test_acc", 0),
                subcircuit_metrics=sc_metrics,
            )

        # Load bests
        trial.metrics.per_gate_bests = metrics_data.get("per_gate_bests", {})

        # Load robustness (simplified - just need samples for viz)
        bests_robust = metrics_data.get("per_gate_bests_robust", {})
        for gate_name, robust_list in bests_robust.items():
            trial.metrics.per_gate_bests_robust[gate_name] = []
            for r in robust_list:
                noise = [RobustnessSample(**s) for s in r.get("noise_samples", [])]
                ood = [RobustnessSample(**s) for s in r.get("ood_samples", [])]
                trial.metrics.per_gate_bests_robust[gate_name].append(
                    RobustnessMetrics(noise_samples=noise, ood_samples=ood)
                )

        # Load faithfulness
        bests_faith = metrics_data.get("per_gate_bests_faith", {})
        for gate_name, faith_list in bests_faith.items():
            trial.metrics.per_gate_bests_faith[gate_name] = []
            for f in faith_list:
                # Reconstruct PatchStatistics with samples
                in_stats = {}
                for pk, ps in f.get("in_circuit_stats", {}).items():
                    samples = [InterventionSample(**s) for s in ps.get("samples", [])]
                    in_stats[pk] = PatchStatistics(
                        mean_bit_similarity=ps.get("mean_bit_similarity", 0),
                        samples=samples,
                    )
                out_stats = {}
                for pk, ps in f.get("out_circuit_stats", {}).items():
                    samples = [InterventionSample(**s) for s in ps.get("samples", [])]
                    out_stats[pk] = PatchStatistics(
                        mean_bit_similarity=ps.get("mean_bit_similarity", 0),
                        samples=samples,
                    )
                # OOD stats
                in_stats_ood = {}
                for pk, ps in f.get("in_circuit_stats_ood", {}).items():
                    samples = [InterventionSample(**s) for s in ps.get("samples", [])]
                    in_stats_ood[pk] = PatchStatistics(
                        mean_bit_similarity=ps.get("mean_bit_similarity", 0),
                        samples=samples,
                    )
                out_stats_ood = {}
                for pk, ps in f.get("out_circuit_stats_ood", {}).items():
                    samples = [InterventionSample(**s) for s in ps.get("samples", [])]
                    out_stats_ood[pk] = PatchStatistics(
                        mean_bit_similarity=ps.get("mean_bit_similarity", 0),
                        samples=samples,
                    )
                trial.metrics.per_gate_bests_faith[gate_name].append(
                    FaithfulnessMetrics(
                        in_circuit_stats=in_stats,
                        out_circuit_stats=out_stats,
                        in_circuit_stats_ood=in_stats_ood,
                        out_circuit_stats_ood=out_stats_ood,
                        mean_in_circuit_similarity=f.get("mean_in_circuit_similarity", 0),
                        mean_out_circuit_similarity=f.get("mean_out_circuit_similarity", 0),
                        mean_in_circuit_similarity_ood=f.get("mean_in_circuit_similarity_ood", 0),
                        mean_out_circuit_similarity_ood=f.get("mean_out_circuit_similarity_ood", 0),
                        overall_faithfulness=f.get("overall_faithfulness", 0),
                    )
                )

        result.trials[trial_id] = trial

    return result


def load_experiment(run_dir: str | Path, device: str = "cpu") -> dict:
    """
    Load complete experiment data.

    Returns dict with:
        - config: Experiment configuration
        - trials: Dict of trial_id -> trial data including:
            - setup: Trial setup parameters
            - metrics: Training and analysis metrics
            - circuits: Subcircuit data
            - profiling: Timing data
            - tensors: Tensor data
            - model: Loaded MLP (if available)
    """
    run_dir = Path(run_dir)

    config = load_config(run_dir)
    trials = {}

    for trial_dir in get_trial_dirs(run_dir):
        trial_id = trial_dir.name
        model = load_model(trial_dir, device=device)

        trials[trial_id] = {
            "setup": load_trial_setup(trial_dir),
            "metrics": load_trial_metrics(trial_dir),
            "circuits": load_trial_circuits(trial_dir),
            "profiling": load_trial_profiling(trial_dir),
            "tensors": load_tensors(trial_dir, device=device),
            "model": model,
            "decomposed_model": load_decomposed_model(trial_dir, model, device=device),
        }

    return {"config": config, "trials": trials}

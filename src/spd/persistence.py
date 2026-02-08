"""
SPD persistence functions for saving and loading SPD estimates and results.

Contains:
- save_spd_estimate / load_spd_estimate: Per-config estimate persistence
- save_spd_results / load_spd_results: Full SpdResults persistence
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.schemas.serialization import filter_non_serializable

if TYPE_CHECKING:
    from .spd_types import SpdResults, SpdTrialResult
    from .subcircuits import SPDSubcircuitEstimate


def save_spd_estimate(estimate: "SPDSubcircuitEstimate", output_dir: str | Path) -> None:
    """Save SPD subcircuit estimate to disk."""
    from .subcircuits import SPDSubcircuitEstimate  # Avoid circular import

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON-serializable data
    estimate_data = {
        "cluster_assignments": estimate.cluster_assignments,
        "n_clusters": estimate.n_clusters,
        "cluster_sizes": estimate.cluster_sizes,
        "component_labels": estimate.component_labels,
        "cluster_functions": {str(k): v for k, v in estimate.cluster_functions.items()},
    }
    with open(output_dir / "estimate.json", "w", encoding="utf-8") as f:
        json.dump(estimate_data, f, indent=2)

    # Save numpy arrays
    if estimate.component_importance is not None:
        np.save(output_dir / "component_importance.npy", estimate.component_importance)

    if estimate.coactivation_matrix is not None:
        np.save(output_dir / "coactivation_matrix.npy", estimate.coactivation_matrix)


def load_spd_estimate(input_dir: str | Path) -> "SPDSubcircuitEstimate | None":
    """Load SPD subcircuit estimate from disk."""
    from .subcircuits import SPDSubcircuitEstimate  # Avoid circular import

    input_dir = Path(input_dir)

    estimate_path = input_dir / "estimate.json"
    if not estimate_path.exists():
        return None

    with open(estimate_path, encoding="utf-8") as f:
        data = json.load(f)

    estimate = SPDSubcircuitEstimate(
        cluster_assignments=data["cluster_assignments"],
        n_clusters=data["n_clusters"],
        cluster_sizes=data["cluster_sizes"],
        component_labels=data["component_labels"],
        cluster_functions={int(k): v for k, v in data["cluster_functions"].items()},
    )

    # Load numpy arrays if they exist
    importance_path = input_dir / "component_importance.npy"
    if importance_path.exists():
        estimate.component_importance = np.load(importance_path)

    coact_path = input_dir / "coactivation_matrix.npy"
    if coact_path.exists():
        estimate.coactivation_matrix = np.load(coact_path)

    return estimate


def save_spd_results(results: "SpdResults", run_dir: str | Path) -> None:
    """
    Save complete SPD results to run_dir/spd/ folder.

    Creates:
        spd/
            config.json           - SPD configuration
            {trial_id}/
                {config_id}/
                    decomposed_model.pt
                    estimate.json
                    component_importance.npy
                    coactivation_matrix.npy
    """
    from .analysis import analyze_and_visualize_spd
    from .spd_types import SPDConfig, SpdResults, SpdTrialResult

    run_dir = Path(run_dir)
    spd_dir = run_dir / "spd"
    spd_dir.mkdir(parents=True, exist_ok=True)

    # Save main config
    config_data = filter_non_serializable(asdict(results.config))
    with open(spd_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # Save sweep configs if any
    if results.sweep_configs:
        sweep_data = [filter_non_serializable(asdict(c)) for c in results.sweep_configs]
        with open(spd_dir / "sweep_configs.json", "w", encoding="utf-8") as f:
            json.dump(sweep_data, f, indent=2)

    # Save per-trial results
    for trial_id, trial_result in results.per_trial.items():
        trial_spd_dir = spd_dir / trial_id
        trial_spd_dir.mkdir(parents=True, exist_ok=True)

        # Save each config's decomposition
        for config_id, decomposed in trial_result.decomposed_models_sweep.items():
            config_dir = trial_spd_dir / config_id
            config_dir.mkdir(parents=True, exist_ok=True)

            # Save decomposed model
            decomposed_path = config_dir / "decomposed_model.pt"
            decomposed.save(str(decomposed_path))

            # Save estimate if available
            estimate = trial_result.spd_subcircuit_estimates_sweep.get(config_id)
            if estimate is not None:
                save_spd_estimate(estimate, config_dir)

            # Run and save analysis with visualizations
            if decomposed.target_model is not None:
                try:
                    analyze_and_visualize_spd(
                        decomposed_model=decomposed,
                        target_model=decomposed.target_model,
                        output_dir=config_dir,
                        gate_names=None,  # Will be inferred
                        n_inputs=2,
                        device="cpu",
                    )
                except Exception as e:
                    print(f"SPD visualization failed for {config_id}: {e}")


def load_spd_results(run_dir: str | Path, device: str = "cpu") -> "SpdResults | None":
    """
    Load SPD results from run_dir/spd/ folder.

    Args:
        run_dir: Run directory containing spd/ subfolder
        device: Device to load models to

    Returns:
        SpdResults or None if no SPD results found
    """
    from src.model import DecomposedMLP, MLP
    from src.persistence import load_model

    from .spd_types import SPDConfig, SpdResults, SpdTrialResult

    run_dir = Path(run_dir)
    spd_dir = run_dir / "spd"

    if not spd_dir.exists():
        return None

    # Load main config
    config_path = spd_dir / "config.json"
    if not config_path.exists():
        return None

    with open(config_path, encoding="utf-8") as f:
        config_data = json.load(f)
    config = SPDConfig(**{k: v for k, v in config_data.items() if k != "_hash"})

    # Load sweep configs if any
    sweep_configs = []
    sweep_path = spd_dir / "sweep_configs.json"
    if sweep_path.exists():
        with open(sweep_path, encoding="utf-8") as f:
            sweep_data = json.load(f)
        sweep_configs = [
            SPDConfig(**{k: v for k, v in c.items() if k != "_hash"})
            for c in sweep_data
        ]

    results = SpdResults(config=config, sweep_configs=sweep_configs)

    # Load per-trial results
    for trial_dir in spd_dir.iterdir():
        if not trial_dir.is_dir() or trial_dir.name.startswith("."):
            continue

        trial_id = trial_dir.name
        trial_result = SpdTrialResult(trial_id=trial_id)

        # Load target model for this trial
        trial_model_dir = run_dir / trial_id
        target_model = load_model(trial_model_dir, device=device)

        # Load each config's decomposition
        for config_dir in trial_dir.iterdir():
            if not config_dir.is_dir():
                continue

            config_id = config_dir.name
            decomposed_path = config_dir / "decomposed_model.pt"

            if decomposed_path.exists() and target_model is not None:
                try:
                    decomposed = DecomposedMLP.load(
                        str(decomposed_path),
                        target_model=target_model,
                        device=device,
                    )
                    trial_result.decomposed_models_sweep[config_id] = decomposed

                    # Set primary decomposition
                    if trial_result.decomposed_model is None:
                        trial_result.decomposed_model = decomposed
                except Exception as e:
                    print(f"Failed to load decomposed model {config_id}: {e}")

            # Load estimate
            estimate = load_spd_estimate(config_dir)
            if estimate is not None:
                trial_result.spd_subcircuit_estimates_sweep[config_id] = estimate
                if trial_result.spd_subcircuit_estimate is None:
                    trial_result.spd_subcircuit_estimate = estimate

        results.per_trial[trial_id] = trial_result

    return results

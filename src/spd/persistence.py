"""SPD persistence: save and load SPD results.

Functions for persisting SPD analysis to disk:
- save_spd_estimate / load_spd_estimate: Subcircuit estimate persistence
- save_spd_results / load_spd_results: Full SpdResults persistence
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.serialization import filter_non_serializable

if TYPE_CHECKING:
    from .types import SPDSubcircuitEstimate, SpdResults, SpdTrialResult


def save_spd_estimate(estimate: "SPDSubcircuitEstimate", output_dir: str | Path) -> None:
    """Save SPD subcircuit estimate to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    estimate_data = {
        "cluster_assignments": estimate.cluster_assignments,
        "n_clusters": estimate.n_clusters,
        "cluster_sizes": estimate.cluster_sizes,
        "component_labels": estimate.component_labels,
        "cluster_functions": {str(k): v for k, v in estimate.cluster_functions.items()},
    }
    with open(output_dir / "estimate.json", "w", encoding="utf-8") as f:
        json.dump(estimate_data, f, indent=2)

    if estimate.component_importance is not None:
        np.save(output_dir / "component_importance.npy", estimate.component_importance)

    if estimate.coactivation_matrix is not None:
        np.save(output_dir / "coactivation_matrix.npy", estimate.coactivation_matrix)


def load_spd_estimate(input_dir: str | Path) -> "SPDSubcircuitEstimate | None":
    """Load SPD subcircuit estimate from disk."""
    from .types import SPDSubcircuitEstimate

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

    importance_path = input_dir / "component_importance.npy"
    if importance_path.exists():
        estimate.component_importance = np.load(importance_path)

    coact_path = input_dir / "coactivation_matrix.npy"
    if coact_path.exists():
        estimate.coactivation_matrix = np.load(coact_path)

    return estimate


def save_spd_results(results: "SpdResults", run_dir: str | Path) -> None:
    """Save complete SPD results to run_dir/{trial_id}/spd/ folders.

    Creates:
        {trial_id}/
            spd/
                config.json
                decomposed_model.pt
                estimate.json
                component_importance.npy
                coactivation_matrix.npy
    """
    from .spd_executor import analyze_and_visualize_spd

    run_dir = Path(run_dir)

    for trial_id, trial_result in results.per_trial.items():
        trial_spd_dir = run_dir / trial_id / "spd"
        trial_spd_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_data = filter_non_serializable(asdict(results.config))
        with open(trial_spd_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        # Save decomposed model
        if trial_result.decomposed_model is not None:
            decomposed_path = trial_spd_dir / "decomposed_model.pt"
            trial_result.decomposed_model.save(str(decomposed_path))

        # Save estimate
        if trial_result.spd_subcircuit_estimate is not None:
            save_spd_estimate(trial_result.spd_subcircuit_estimate, trial_spd_dir)

        # Run analysis with visualizations
        if trial_result.decomposed_model is not None and trial_result.decomposed_model.target_model is not None:
            try:
                analyze_and_visualize_spd(
                    decomposed_model=trial_result.decomposed_model,
                    target_model=trial_result.decomposed_model.target_model,
                    output_dir=trial_spd_dir,
                    gate_names=None,
                    n_inputs=2,
                    device="cpu",
                )
            except Exception as e:
                print(f"SPD visualization failed for {trial_id}: {e}")


def load_spd_results(run_dir: str | Path, device: str = "cpu") -> "SpdResults | None":
    """Load SPD results from run_dir/{trial_id}/spd/ folders.

    Args:
        run_dir: Run directory containing trial folders with spd/ subfolders
        device: Device to load models to

    Returns:
        SpdResults or None if no SPD results found
    """
    from src.model import DecomposedMLP
    from src.persistence import load_model

    from .types import SPDConfig, SpdResults, SpdTrialResult

    run_dir = Path(run_dir)

    # Find trial directories with spd/ subfolders
    trial_dirs_with_spd = []
    for trial_dir in run_dir.iterdir():
        if trial_dir.is_dir() and (trial_dir / "spd").exists():
            trial_dirs_with_spd.append(trial_dir)

    if not trial_dirs_with_spd:
        return None

    # Load config from first trial
    first_spd_dir = trial_dirs_with_spd[0] / "spd"
    config_path = first_spd_dir / "config.json"
    if not config_path.exists():
        return None

    with open(config_path, encoding="utf-8") as f:
        config_data = json.load(f)
    config = SPDConfig(**{k: v for k, v in config_data.items() if k != "_hash"})

    results = SpdResults(config=config)

    for trial_dir in trial_dirs_with_spd:
        trial_id = trial_dir.name
        spd_dir = trial_dir / "spd"
        trial_result = SpdTrialResult(trial_id=trial_id)

        # Load target model
        target_model = load_model(trial_dir, device=device)

        # Load decomposed model
        decomposed_path = spd_dir / "decomposed_model.pt"
        if decomposed_path.exists() and target_model is not None:
            try:
                trial_result.decomposed_model = DecomposedMLP.load(
                    str(decomposed_path),
                    target_model=target_model,
                    device=device,
                )
            except Exception as e:
                print(f"Failed to load decomposed model for {trial_id}: {e}")

        # Load estimate
        trial_result.spd_subcircuit_estimate = load_spd_estimate(spd_dir)

        results.per_trial[trial_id] = trial_result

    return results

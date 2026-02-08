"""
SPD persistence functions for saving and loading SPD estimates.

Separated from subcircuits.py to follow the pattern of keeping
save/load functions in dedicated persistence modules.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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

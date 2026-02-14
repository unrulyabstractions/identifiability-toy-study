"""Result schema classes.

Contains result-related dataclasses:
- Dataset: Training/validation/test data container
- TrialData: Complete trial data (train, val, test datasets)
- TrialResult: Complete result from a single trial
- ExperimentResult: Complete result from an experiment
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from src.experiment_config import ExperimentConfig, TrialSetup
from src.infra import parse_subcircuit_key
from src.schema_class import SchemaClass
from .evaluation import Metrics, ProfilingData

# Avoid circular import - these are only needed for type hints
if TYPE_CHECKING:
    from src.model import MLP


@dataclass
class Dataset:
    x: torch.tensor
    y: torch.tensor


@dataclass
class TrialData:
    train: Dataset
    val: Dataset
    test: Dataset

    def select_gates(self, gate_indices: list[int], n_inputs: int) -> "TrialData":
        """Create a new TrialData with only the specified gate columns and input size.

        Args:
            gate_indices: Which columns to select from y (e.g., [0, 2] for gates 0 and 2)
            n_inputs: Number of input columns to use (slices x to [:, :n_inputs])

        Returns:
            New TrialData with x sliced to n_inputs columns and y sliced to gate columns
        """
        return TrialData(
            train=Dataset(
                x=self.train.x[:, :n_inputs],
                y=self.train.y[:, gate_indices],
            ),
            val=Dataset(
                x=self.val.x[:, :n_inputs],
                y=self.val.y[:, gate_indices],
            ),
            test=Dataset(
                x=self.test.x[:, :n_inputs],
                y=self.test.y[:, gate_indices],
            ),
        )


@dataclass
class TrialResult(SchemaClass):
    # Basic info
    setup: TrialSetup
    status: str = "UNKNOWN"
    metrics: Metrics = field(default_factory=Metrics)
    profiling: ProfilingData = field(default_factory=ProfilingData)

    # Test data (saved as tensors.pt, not in JSON)
    test_x: Optional[torch.Tensor] = None
    test_y: Optional[torch.Tensor] = None
    test_y_pred: Optional[torch.Tensor] = None

    # Activations per layer from forward pass on test data (saved as tensors.pt)
    activations: Optional[list[torch.Tensor]] = None

    # Pre-computed activations for canonical binary inputs (0,0), (0,1), (1,0), (1,1)
    # Dict mapping "input_label" -> list of layer activations
    # Used for visualization without running models
    canonical_activations: Optional[dict[str, list[torch.Tensor]]] = None

    # Weight matrices per layer (extracted from model, for visualization)
    layer_weights: Optional[list[torch.Tensor]] = None

    # Bias vectors per layer (extracted from model, for visualization)
    # Shows (weight + bias) on edge labels to reveal bias contribution when edges are patched
    layer_biases: Optional[list[torch.Tensor]] = None

    # Mean activations for inputs from different ranges (for visualization)
    # Dict mapping range_label (e.g., "0_1", "-1_0") -> list of mean activations per layer
    mean_activations_by_range: Optional[dict[str, list[torch.Tensor]]] = None

    # Decision boundary data for full model (per gate)
    # Dict mapping gate_name -> data dict from generate_grid_data/generate_monte_carlo_data
    decision_boundary_data: Optional[dict[str, dict]] = None

    # Decision boundary data for subcircuits
    # Dict mapping gate_name -> {subcircuit_idx: data dict}
    # subcircuit_idx is from make_subcircuit_idx(node_mask_idx, edge_variant_rank)
    subcircuit_decision_boundary_data: Optional[dict[str, dict]] = None

    # Models stored at runtime (saved as model.pt, not in JSON)
    model: Optional["MLP"] = None
    subcircuits: list = field(default_factory=list)
    subcircuit_structure_analysis: list = field(default_factory=list)

    # Each TrialSetup defines an deterministic id
    trial_id: str = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.trial_id = self.setup.get_id()


@dataclass
class ExperimentResult(SchemaClass):
    config: ExperimentConfig
    trials: dict[str, TrialResult] = field(default_factory=dict)

    # Each ExperimentConfig defines an deterministic id
    experiment_id: str = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.experiment_id = self.config.get_id()

    def print_summary(self, viz_paths: dict = None) -> str:
        """Print summary showing best subcircuits per gate."""
        viz_paths = viz_paths or {}
        summary = {"experiment_id": self.experiment_id, "trials": {}}

        for trial_id, trial in self.trials.items():
            # Get architecture params for subcircuit index parsing
            width = trial.setup.model_params.width
            depth = trial.setup.model_params.depth

            gates = {}
            for gate, gm in trial.metrics.per_gate_metrics.items():
                by_idx = {sm.idx: sm for sm in gm.subcircuit_metrics}
                bests = trial.metrics.per_gate_bests.get(gate, [])[:5]
                viz = viz_paths.get(trial_id, {}).get(gate, {})
                best_list = []
                for key in bests:
                    node_mask_idx, edge_variant_rank = parse_subcircuit_key(key, width, depth)
                    if node_mask_idx not in by_idx:
                        continue
                    sm = by_idx[node_mask_idx]

                    entry = {
                        "node_pattern": node_mask_idx,
                        "edge_variant_rank": edge_variant_rank,
                        "acc": sm.accuracy,
                        "sim": sm.bit_similarity,
                    }

                    if key in viz:
                        entry["viz"] = viz[key]
                    best_list.append(entry)
                gates[gate] = {
                    "test_acc": gm.test_acc,
                    "best": best_list,
                }
            summary["trials"][trial_id] = {
                "status": trial.status,
                "test_acc": trial.metrics.test_acc,
                "gates": gates,
            }

        return json.dumps(summary, indent=2)

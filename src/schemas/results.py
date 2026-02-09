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
            gates = {}
            for gate, gm in trial.metrics.per_gate_metrics.items():
                by_idx = {sm.idx: sm for sm in gm.subcircuit_metrics}
                bests = trial.metrics.per_gate_bests.get(gate, [])[:5]
                viz = viz_paths.get(trial_id, {}).get(gate, {})
                best_list = []
                for key in bests:
                    node_idx, edge_var_idx = parse_subcircuit_key(key)
                    if node_idx not in by_idx:
                        continue
                    sm = by_idx[node_idx]

                    # Build entry with appropriate key format
                    if isinstance(key, (tuple, list)):
                        entry = {
                            "node_idx": node_idx,
                            "edge_var_idx": edge_var_idx,
                            "acc": sm.accuracy,
                            "sim": sm.bit_similarity,
                        }
                    else:
                        entry = {
                            "idx": key,
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

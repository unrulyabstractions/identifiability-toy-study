"""Result schema classes.

Contains result-related dataclasses:
- Dataset: Training/validation/test data container
- TrialData: Complete trial data (train, val, test datasets)
- TrialResult: Complete result from a single trial
- ExperimentResult: Complete result from an experiment
"""

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from .schema_class import SchemaClass
from .config import ExperimentConfig, TrialSetup
from .evaluation import Metrics, ProfilingData

# Avoid circular import - these are only needed for type hints
if TYPE_CHECKING:
    from src.model import DecomposedMLP, MLP


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
    decomposed_model: Optional["DecomposedMLP"] = (
        None  # Full multi-gate model (primary config)
    )
    spd_subcircuit_estimate: Optional[Any] = (
        None  # SPD-based subcircuit clustering (primary config)
    )
    # Multi-config SPD sweep results: maps config_id -> DecomposedMLP/estimate
    decomposed_models_sweep: dict[str, "DecomposedMLP"] = field(default_factory=dict)
    spd_subcircuit_estimates_sweep: dict[str, Any] = field(default_factory=dict)
    subcircuits: list = field(default_factory=list)
    subcircuit_structure_analysis: list = field(default_factory=list)
    # Maps gate_name -> DecomposedMLP for full single-gate model (saved separately)
    decomposed_gate_models: dict[str, "DecomposedMLP"] = field(default_factory=dict)
    # Maps gate_name -> subcircuit_idx -> DecomposedMLP (saved separately)
    decomposed_subcircuits: dict[str, dict[int, "DecomposedMLP"]] = field(
        default_factory=lambda: {}
    )
    # Serializable: which subcircuit indices were decomposed per gate
    decomposed_subcircuit_indices: dict[str, list[int]] = field(
        default_factory=lambda: {}
    )

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
                gates[gate] = {
                    "test_acc": gm.test_acc,
                    "best": [
                        {
                            "idx": i,
                            "acc": by_idx[i].accuracy,
                            "sim": by_idx[i].bit_similarity,
                            **({"viz": viz[i]} if i in viz else {}),
                        }
                        for i in bests
                        if i in by_idx
                    ],
                }
            summary["trials"][trial_id] = {
                "status": trial.status,
                "test_acc": trial.metrics.test_acc,
                "gates": gates,
            }

        return json.dumps(summary, indent=2)

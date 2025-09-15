import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from identifiability_toy_study.mi_identifiability.utils import (
    deterministic_id_from_dataclass,
)


@dataclass
class DataClass:
    def get_id(self) -> str:
        return deterministic_id_from_dataclass(self)

    def __str__(self) -> str:
        result_dict = asdict(self)
        return json.dumps(result_dict, indent=4)


@dataclass
class DataParams(DataClass):
    n_samples_train: int
    n_samples_val: int
    n_samples_test: int
    noise_std: float = 0.0
    skewed_distribution: bool = False


@dataclass
class ModelParams(DataClass):
    logic_gates: list[str] = field(default_factory=list)
    width: int = 3
    depth: int = 2


@dataclass
class TrainParams(DataClass):
    learning_rate: float
    loss_target: float
    acc_target: float
    batch_size: int
    epochs: int
    val_frequency: int


@dataclass
class IdentifiabilityConstraints(DataClass):
    min_sparsity: float = 0.0
    acc_threshold: float = 0.99
    is_perfect_circuit: bool = True
    is_causal_abstraction: bool = False
    non_transport_stable: bool = False
    param_decomp: bool = False


@dataclass
class TrialSetup(DataClass):
    seed: int
    model_params: ModelParams
    train_params: TrainParams
    data_params: DataParams
    iden_constraints: IdentifiabilityConstraints

    def __str__(self) -> str:
        setup_dict = asdict(self)
        setup_dict["trial_id"] = self.get_id()
        return json.dumps(setup_dict, indent=4)


@dataclass
class CircuitMetrics:
    circuit_idx: int
    accuracy: float
    logit_similarity: float
    bit_similarity: float
    sparsity: tuple[float, float, float]


@dataclass
class GateMetrics:
    num_total_circuits: int
    test_acc: float
    per_circuit: dict[int, CircuitMetrics] = field(default_factory=dict)
    faithful_circuits_idx: list = field(default_factory=list)


@dataclass
class Metrics:
    # Train info
    avg_loss: Optional[float] = None
    val_acc: Optional[float] = None
    test_acc: Optional[float] = None

    # Circuit Info
    per_gate: dict[str, GateMetrics] = field(default_factory=dict)


@dataclass
class ProfilingData:
    device: str
    train_secs: int
    circuit_secs: int
    iden_secs: int


@dataclass
class TrialResult:
    # Basic info
    setup: TrialSetup
    status: str = "UNKNOWN"
    metrics: Metrics = field(default_factory=Metrics)
    profiling: Optional[ProfilingData] = None

    # Each TrialSetup defines an deterministic id
    trial_id: str = field(init=False)

    def __post_init__(self):
        self.trial_id = self.setup.get_id()


@dataclass
class Dataset:
    x: torch.tensor
    y: torch.tensor


@dataclass
class TrialData:
    train: Dataset
    val: Dataset
    test: Dataset

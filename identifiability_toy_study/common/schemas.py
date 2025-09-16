import copy
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

import torch

from .utils import (
    deterministic_id_from_dataclass,
)


@dataclass
class SchemaClass:
    # Each schema gets unique id based on values
    def get_id(self) -> str:
        return deterministic_id_from_dataclass(self)

    # For logging ease
    def __str__(self) -> str:
        result_dict = asdict(self)
        return json.dumps(result_dict, indent=4)

    # Each trial should have their own set of params
    # We want to make sure schemas are unique and immutable
    def __post_init__(self):
        for f in fields(self):
            setattr(self, f.name, copy.deepcopy(getattr(self, f.name)))

    def __copy__(self):
        return self.__deepcopy__({})

    def __deepcopy__(self, memo):
        cls = self.__class__
        kwargs = {
            f.name: copy.deepcopy(getattr(self, f.name), memo) for f in fields(self)
        }
        return cls(**kwargs)

    def __setattr__(self, name, value):
        super().__setattr__(name, copy.deepcopy(value))


@dataclass
class DataParams(SchemaClass):
    n_samples_train: int = 2048
    n_samples_val: int = 128
    n_samples_test: int = 128
    noise_std: float = 0.0
    skewed_distribution: bool = False


@dataclass
class ModelParams(SchemaClass):
    logic_gates: list[str] = field(default_factory=lambda: ["XOR"])
    width: int = 3
    depth: int = 2


@dataclass
class TrainParams(SchemaClass):
    learning_rate: float = 0.001
    loss_target: float = 0.001
    acc_target: float = 0.99
    batch_size: int = 2048
    epochs: int = 1000
    val_frequency: int = 1


@dataclass
class IdentifiabilityConstraints(SchemaClass):
    min_sparsity: float = 0.0
    acc_threshold: float = 0.99

    is_perfect_circuit: bool = True
    is_causal_abstraction: bool = False
    non_transport_stable: bool = False
    param_decomp: bool = False

    require_commutation: bool = False
    commutation_atol: float = 0.0

    faithfulness_min: float = 0.0

    iia_min: float = 0.0
    iia_num_pairs: int = 128
    iia_max_vars_per_layer: Optional[int] = None


@dataclass
class CircuitMetrics(SchemaClass):
    #
    accuracy: float
    sparsity: tuple[float, float, float]

    # Observational
    logit_similarity: float
    bit_similarity: float

    # Interventional
    commutes: Optional[bool] = None
    comm_gap: Optional[float] = None
    faithfulness: Optional[float] = None
    iia: Optional[float] = None


@dataclass
class GateMetrics(SchemaClass):
    num_total_circuits: int
    test_acc: float
    per_circuit: dict[int, CircuitMetrics] = field(default_factory=dict)
    faithful_circuits_idx: list = field(default_factory=list)


@dataclass
class Metrics(SchemaClass):
    # Train info
    avg_loss: Optional[float] = None
    val_acc: Optional[float] = None
    test_acc: Optional[float] = None

    # Circuit Info
    per_gate: dict[str, GateMetrics] = field(default_factory=dict)


@dataclass
class ProfilingData(SchemaClass):
    device: str
    train_secs: int
    analysis_secs: int


@dataclass
class TrialSetup(SchemaClass):
    seed: int = 0
    data_params: DataParams = field(default_factory=DataParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    train_params: TrainParams = field(default_factory=TrainParams)
    iden_constraints: IdentifiabilityConstraints = field(
        default_factory=IdentifiabilityConstraints
    )

    def __str__(self) -> str:
        setup_dict = asdict(self)
        setup_dict["trial_id"] = self.get_id()
        return json.dumps(setup_dict, indent=4)

    def __post_init__(self):
        super().__post_init__
        for f in fields(self):
            val = getattr(self, f.name)
            setattr(self, f.name, copy.deepcopy(val))


@dataclass
class TrialResult(SchemaClass):
    # Basic info
    setup: TrialSetup
    status: str = "UNKNOWN"
    metrics: Metrics = field(default_factory=Metrics)
    profiling: Optional[ProfilingData] = None

    # Each TrialSetup defines an deterministic id
    trial_id: str = field(init=False)

    def __post_init__(self):
        super().__post_init__
        self.trial_id = self.setup.get_id()


@dataclass
class ExperimentConfig(SchemaClass):
    logger: Optional[Any] = None
    from_scratch: bool = False
    model_dir: str = ""
    debug: bool = False
    device: str = "mps"

    base_trial: TrialSetup = field(default_factory=TrialSetup)

    widths: list[int] = field(default_factory=lambda: [ModelParams().width])
    depths: list[int] = field(default_factory=lambda: [ModelParams().depth])
    loss_targets: list[int] = field(default_factory=lambda: [TrainParams().loss_target])
    learning_rates: list[int] = field(
        default_factory=lambda: [TrainParams().learning_rate]
    )

    target_logic_gates: list[str] = field(
        default_factory=lambda: [*ModelParams().logic_gates]
    )
    num_gates_per_run: list[int] = field(default_factory=lambda: [1])
    num_runs: int = 1

    def __str__(self) -> str:
        setup_dict = asdict(self)
        setup_dict["experiment_id"] = self.get_id()
        return json.dumps(setup_dict, indent=4)


@dataclass
class ExperimentResult(SchemaClass):
    config: ExperimentConfig
    trials: dict[str, TrialResult] = field(default_factory=dict)

    # Each ExperimentConfig defines an deterministic id
    experiment_id: str = field(init=False)

    def __post_init__(self):
        super().__post_init__
        self.experiment_id = self.config.get_id()


# ---- NOT SCHEMAS ----
# But still useful here


@dataclass
class Dataset:
    x: torch.tensor
    y: torch.tensor


@dataclass
class TrialData:
    train: Dataset
    val: Dataset
    test: Dataset

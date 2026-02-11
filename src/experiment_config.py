"""Configuration schema classes.

Contains all configuration-related dataclasses:
- DataParams: Data generation parameters
- ModelParams: Neural network architecture parameters
- TrainParams: Training hyperparameters
- FaithfulnessConfig: Faithfulness analysis configuration
- IdentifiabilityConstraints: Constraints for identifiability analysis
- TrialSetup: Complete trial configuration
- ExperimentConfig: Multi-trial experiment configuration

Note: ParallelConfig is in src.infra.parallel to avoid circular imports.
"""

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

from src.schema_class import SchemaClass

###############
# TEST HELPER
###############

_FAST_TEST_MODE = False
_FAST_TEST_GATES_IDX = 0

# Test gate configurations:
#   0: ["XOR"]                   - single 2-input gate (fastest)
#   1: ["OR", "AND"]             - two 2-input gates
#   2: ["XOR", "XOR"]            - duplicate gates
#   3: ["ID", "IMP"]             - identity gate test
#   4: ["ID", "MAJORITY"]        - 3-input gate test
#   5: ["XOR", "MAJORITY"]       - mixed 2-input and 3-input


def set_test_mode_global(val: bool, test_gates_idx: int = 0):
    global _FAST_TEST_MODE, _FAST_TEST_GATES_IDX
    _FAST_TEST_MODE = val
    _FAST_TEST_GATES_IDX = test_gates_idx


def get_default_train_n_samples():
    if _FAST_TEST_MODE:
        return 2**8
    return 2**14


def get_default_train_noise():
    if _FAST_TEST_MODE:
        return 0.00001
    return 0.1


def get_default_logic_gates():
    ALL_GATES = ["XOR", "XOR", "AND", "OR", "IMP", "MAJORITY"]
    if _FAST_TEST_MODE:
        if _FAST_TEST_GATES_IDX == 0:
            return ["XOR"]
        if _FAST_TEST_GATES_IDX == 1:
            return ["OR", "AND"]
        if _FAST_TEST_GATES_IDX == 2:
            return ["XOR", "XOR"]
        if _FAST_TEST_GATES_IDX == 3:
            return ["ID", "IMP"]
        if _FAST_TEST_GATES_IDX == 4:
            return ["ID", "MAJORITY"]
        if _FAST_TEST_GATES_IDX == 5:
            return ["XOR", "MAJORITY"]
    return ALL_GATES


def get_default_max_subcircuits_per_gate():
    if _FAST_TEST_MODE:
        return 1
    return 10


def get_default_max_edge_variations_per_subcircuits():
    if _FAST_TEST_MODE:
        return 1
    return 10


def get_default_epsilon():
    if _FAST_TEST_MODE:
        return 0.01
    return 0.2


def get_default_faith_n_samples():
    if _FAST_TEST_MODE:
        return 5
    return 200


def get_default_num_gates_per_run():
    if _FAST_TEST_MODE:
        return [2]
    return [1, 2, 3]


def get_default_activations():
    if _FAST_TEST_MODE:
        return ["leaky_relu"]
    return ["leaky_relu", "relu"]


###############
# CONFIGS
###############


@dataclass
class DataParams(SchemaClass):
    n_samples_train: int = get_default_train_n_samples()
    n_samples_val: int = get_default_train_n_samples() // 8
    n_samples_test: int = get_default_train_n_samples() // 8
    noise_std: float = get_default_train_noise()
    skewed_distribution: bool = False


@dataclass
class ModelParams(SchemaClass):
    logic_gates: list[str] = field(default_factory=lambda: get_default_logic_gates())
    width: int = 3
    depth: int = 2
    activation: str = "leaky_relu"  # Single activation for the model


@dataclass
class TrainParams(SchemaClass):
    learning_rate: float = 1e-2
    batch_size: int = DataParams().n_samples_train // 8
    epochs: int = 4000
    val_frequency: int = 10


@dataclass
class IdentifiabilityConstraints(SchemaClass):
    epsilon: float = get_default_epsilon()


@dataclass
class FaithfulnessConfig(SchemaClass):
    """Configuration for faithfulness analysis."""

    max_subcircuits_per_gate: int = get_default_max_subcircuits_per_gate()
    max_edge_variations_per_subcircuits: int = (
        get_default_max_edge_variations_per_subcircuits()
    )
    n_interventions_per_patch: int = get_default_faith_n_samples()
    n_counterfactual_pairs: int = get_default_faith_n_samples()


@dataclass
class TrialSetup(SchemaClass):
    seed: int = 0
    data_params: DataParams = field(default_factory=DataParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    train_params: TrainParams = field(default_factory=TrainParams)
    constraints: IdentifiabilityConstraints = field(
        default_factory=IdentifiabilityConstraints
    )
    faithfulness_config: FaithfulnessConfig = field(default_factory=FaithfulnessConfig)

    def __str__(self) -> str:
        setup_dict = asdict(self)
        setup_dict["trial_id"] = self.get_id()
        return json.dumps(setup_dict, indent=4)

    def __post_init__(self):
        super().__post_init__()
        for f in fields(self):
            val = getattr(self, f.name)
            setattr(self, f.name, copy.deepcopy(val))


@dataclass
class ExperimentConfig(SchemaClass):
    logger: Optional[Any] = None
    debug: bool = False
    device: str = "cpu"

    base_trial: TrialSetup = field(default_factory=TrialSetup)

    widths: list[int] = field(default_factory=lambda: [ModelParams().width])
    depths: list[int] = field(default_factory=lambda: [ModelParams().depth])
    activations: list[str] = field(default_factory=get_default_activations)
    learning_rates: list[float] = field(
        default_factory=lambda: [TrainParams().learning_rate]
    )

    target_logic_gates: list[str] = field(
        default_factory=lambda: [*ModelParams().logic_gates]
    )

    # None = use all gates from target_logic_gates
    num_gates_per_run: int | list[int] | None = field(
        default_factory=lambda: get_default_num_gates_per_run()
    )

    num_runs: int = 1

    def __str__(self) -> str:
        setup_dict = asdict(self)
        setup_dict["experiment_id"] = self.get_id()
        return json.dumps(setup_dict, indent=4)

"""Configuration schema classes.

Contains all configuration-related dataclasses:
- DataParams: Data generation parameters
- ModelParams: Neural network architecture parameters
- TrainParams: Training hyperparameters
- FaithfulnessConfig: Faithfulness analysis configuration
- ParallelConfig: Parallelization configuration
- IdentifiabilityConstraints: Constraints for identifiability analysis
- TrialSetup: Complete trial configuration
- ExperimentConfig: Multi-trial experiment configuration
"""

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

from .schema_class import SchemaClass


@dataclass
class DataParams(SchemaClass):
    n_samples_train: int = 2**14
    n_samples_val: int = 2**10
    n_samples_test: int = 2**10
    noise_std: float = 0.1
    skewed_distribution: bool = False


@dataclass
class ModelParams(SchemaClass):
    # logic_gates: list[str] = field(default_factory=lambda: ["XOR", "AND", "OR", "IMP"])
    logic_gates: list[str] = field(default_factory=lambda: ["XOR"])
    width: int = 3
    depth: int = 2


@dataclass
class TrainParams(SchemaClass):
    learning_rate: float = 0.005  # Higher LR needed for fast convergence
    batch_size: int = DataParams().n_samples_train // 4
    epochs: int = 2000
    val_frequency: int = 10


@dataclass
class IdentifiabilityConstraints(SchemaClass):
    # Max deviation from bit_similarity=1.0 to be considered "best"
    # 0.01 = only 99%+ similar, 0.1 = 90%+ similar, 0.2 = 80%+ similar
    epsilon: float = 0.1


@dataclass
class FaithfulnessConfig(SchemaClass):
    """Configuration for faithfulness analysis."""

    max_subcircuits_per_gate: int = 5
    max_edge_variations_per_subcircuits: int = 5
    n_interventions_per_patch: int = 100
    n_counterfactual_pairs: int = 100


@dataclass
class ParallelConfig(SchemaClass):
    """Configuration for parallelization and compute optimization.

    Optimized defaults based on M4 Max benchmarks:
    - MPS precomputed is 2.7x faster than CPU for batched eval
    - Sequential structure analysis is FASTER than parallel (thread overhead dominates)
    - Larger batch sizes improve throughput

    PyTorch GPU ops are NOT thread-safe, so we avoid threading for GPU work.
    """

    # Device selection for batched circuit evaluation
    eval_device: str = "mps"  # "cpu" or "mps" - MPS is 2.7x faster with precompute
    use_mps_if_available: bool = True

    # Structure analysis parallelization
    # BENCHMARK RESULT: Sequential is FASTER (77ms vs 130ms) because
    # thread overhead exceeds computation time per circuit
    max_workers_structure: int = 1  # 1 = sequential (fastest based on benchmark)
    enable_parallel_structure: bool = False  # Disabled - sequential is faster

    # Batched evaluation settings (GPU)
    precompute_masks: bool = True  # Pre-stack masks: 5.8ms vs 9.6ms on MPS

    # Robustness/Faithfulness - these involve GPU, so threading is risky
    # KEEP FALSE to avoid GPU thread safety issues
    enable_parallel_robustness: bool = False
    enable_parallel_faithfulness: bool = False

    # Memory optimization - use more memory for speed
    cache_subcircuit_models: bool = True  # Cache models for best subcircuits


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
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)

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
    learning_rates: list[float] = field(
        default_factory=lambda: [TrainParams().learning_rate]
    )

    target_logic_gates: list[str] = field(
        default_factory=lambda: [*ModelParams().logic_gates]
    )
    num_gates_per_run: list[int] | None = (
        None  # None = use all gates from target_logic_gates
    )
    num_runs: int = 1

    def __str__(self) -> str:
        setup_dict = asdict(self)
        setup_dict["experiment_id"] = self.get_id()
        return json.dumps(setup_dict, indent=4)

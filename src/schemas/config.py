"""Configuration schema classes.

Contains all configuration-related dataclasses:
- DataParams: Data generation parameters
- ModelParams: Neural network architecture parameters
- TrainParams: Training hyperparameters
- SPDConfig: Stochastic Parameter Decomposition configuration
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
class SPDConfig(SchemaClass):
    """Config for Stochastic Parameter Decomposition.

    Based on SPD paper (arXiv:2506.20790) and Goodfire's TMS experiments:

    1. n_components: 2-4x max(n_functions, hidden_width)
       - For 2->3->1 network: [4, 8, 15, 20]
       - Paper TMS uses 20 subcomponents

    2. importance_coeff: Critical for sparsity (controls g values)
       - Paper uses 3e-3 for TMS experiments
       - Too low -> all g~1 (no sparsity), too high -> all g->0
       - Sweep: [1e-4, 1e-3, 3e-3]

    3. steps: Training duration
       - Paper uses 40k for TMS but tiny models converge much faster
       - For 2->3->1: 1000 steps is usually sufficient
       - Check loss convergence to tune

    4. importance_p (pnorm): Shape of sparsity penalty
       - p=1.0: L1 (paper default, recommended)
       - p=0.5: extreme sparsity
       - p=2.0: L2 (softer, less sparse)
    """

    # Number of components per module
    # Rule of thumb: 2-4x max(n_functions, max_hidden_width)
    n_components: int = 20  # Increased default for better decomposition

    # Training settings
    steps: int = (
        1000  # 1000 is enough for tiny 2->3->1 networks (paper uses 40k for larger)
    )
    batch_size: int = 4096  # Large batch for MPS throughput
    eval_batch_size: int = 4096
    n_eval_steps: int = 10
    learning_rate: float = 1e-3  # Standard Adam LR with cosine schedule

    # Data generation
    feature_probability: float = 0.5
    data_generation_type: str = "at_least_zero_active"

    # Loss coefficients
    importance_coeff: float = 1e-3  # Key tuning param for sparsity (paper uses 3e-3)
    importance_p: float = 1.0  # pnorm: 0.5=extreme, 1.0=L1 (paper default), 2.0=L2
    recon_coeff: float = 1.0

    # Analysis settings (for post-training clustering)
    activation_threshold: float = 0.5  # Threshold for "active" component
    n_clusters: Optional[int] = None  # Auto-detect if None

    def get_config_id(self) -> str:
        """Short ID for this config based on key params."""
        return f"c{self.n_components}_s{self.steps}_i{self.importance_coeff:.0e}_p{self.importance_p}"


@dataclass
class TrialSetup(SchemaClass):
    seed: int = 0
    data_params: DataParams = field(default_factory=DataParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    train_params: TrainParams = field(default_factory=TrainParams)
    constraints: IdentifiabilityConstraints = field(
        default_factory=IdentifiabilityConstraints
    )
    spd_config: SPDConfig = field(default_factory=SPDConfig)
    # Optional: list of additional SPD configs to run (for parameter sweeps)
    # If provided, SPD runs for each config and stores results per config_id
    spd_sweep_configs: Optional[list[SPDConfig]] = None
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
    spd_device: str = (
        "cpu"  # CPU is fastest for small models (no GPU transfer overhead)
    )
    run_spd: bool = False  # Enable SPD decomposition analysis

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

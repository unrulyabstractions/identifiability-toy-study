import copy
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Optional

import torch

from .utils import deterministic_id_from_dataclass, filter_non_serializable

# Avoid circular import - these are only needed for type hints
if TYPE_CHECKING:
    from .neural_model import MLP, DecomposedMLP


@dataclass
class SchemaClass:
    # Each schema gets unique id based on values
    def get_id(self) -> str:
        return deterministic_id_from_dataclass(self)

    # For logging ease - filters out non-serializable fields like nn.Module
    def __str__(self) -> str:
        result_dict = asdict(self)
        filtered = filter_non_serializable(result_dict)
        return json.dumps(filtered, indent=4)

    # Each trial should have their own set of params
    # We want to make sure schemas are unique and immutable
    def __post_init__(self):
        for f in fields(self):
            if hasattr(self, f.name):
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
class SPDConfig(SchemaClass):
    """Config for Stochastic Parameter Decomposition."""

    # Number of components per module
    n_components: int = 10

    # Training - optimized for M4 Max with 48GB (MPS)
    steps: int = 1000  # Fast iteration with larger batches
    batch_size: int = 4096  # Large batch for MPS throughput
    eval_batch_size: int = 4096
    n_eval_steps: int = 10
    learning_rate: float = 5e-3  # Higher LR for larger batches

    # Data generation
    feature_probability: float = 0.5
    data_generation_type: str = "at_least_zero_active"

    # Loss coefficients
    importance_coeff: float = 3e-3
    recon_coeff: float = 1.0

    # Limit subcircuits to decompose (0 = decompose all)
    # Higher = more subcircuits analyzed but slower
    max_subcircuits: int = 1


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
    # Max deviation from bit_similarity=1.0 to be considered "best"
    # 0.01 = only 99%+ similar, 0.1 = 90%+ similar, 0.2 = 80%+ similar
    epsilon: float = 0.01  # More lenient to get more best circuits


@dataclass
class SubcircuitMetrics(SchemaClass):
    idx: int
    # Simple
    accuracy: float  # to gt, not full circuit (target)

    # Observational
    logit_similarity: float
    bit_similarity: float


@dataclass
class GateMetrics(SchemaClass):
    test_acc: float
    subcircuit_metrics: list[SubcircuitMetrics] = field(default_factory=list)


@dataclass
class InterventionSample(SchemaClass):
    """Result of a single intervention test (like RobustnessSample for faithfulness)."""

    # Patch info
    patch_key: str  # String representation of PatchShape
    patch_layer: int  # Layer index
    patch_indices: list[int]  # Neuron indices

    # Intervention values (what we patched in)
    intervention_values: list[float]

    # Outputs
    gate_output: float  # Output from full gate model under intervention
    subcircuit_output: float  # Output from subcircuit under intervention

    # Agreement metrics
    logit_similarity: float  # 1 - MSE between outputs
    bit_agreement: bool  # round(gate_output) == round(subcircuit_output)
    mse: float  # (gate_output - subcircuit_output)^2


@dataclass
class PatchStatistics(SchemaClass):
    """Statistics for a single patch's intervention effects."""

    mean_logit_similarity: float = 0.0
    std_logit_similarity: float = 0.0
    mean_bit_similarity: float = 0.0
    std_bit_similarity: float = 0.0
    mean_best_similarity: float = 0.0
    std_best_similarity: float = 0.0
    n_interventions: int = 0

    # Individual samples for visualization (optional, may be large)
    samples: list[InterventionSample] = field(default_factory=list)


@dataclass
class CounterfactualEffect(SchemaClass):
    """Result of a single counterfactual test."""

    faithfulness_score: float  # (y_sc - y_corrupted) / (y_clean - y_corrupted)

    # Clean/corrupted input info
    clean_input: list[float] = field(default_factory=list)  # e.g., [0, 1]
    corrupted_input: list[float] = field(default_factory=list)  # e.g., [1, 0]

    # Expected outputs
    expected_clean_output: float = 0.0  # y_clean (before patching)
    expected_corrupted_output: float = 0.0  # y_corrupted (counterfactual target)

    # Actual output after patching corrupted activations
    actual_output: float = 0.0  # y_sc (subcircuit output with patched activations)

    # Did patching change output toward corrupted?
    output_changed_to_corrupted: bool = False  # round(actual) == round(corrupted)


@dataclass
class FaithfulnessConfig(SchemaClass):
    """Configuration for faithfulness analysis."""

    n_interventions_per_patch: int = 100
    n_counterfactual_pairs: int = 10


@dataclass
class FaithfulnessMetrics(SchemaClass):
    """Comprehensive faithfulness metrics for a subcircuit."""

    # Per-patch statistics for in-circuit and out-circuit interventions
    in_circuit_stats: dict[str, PatchStatistics] = field(default_factory=dict)
    out_circuit_stats: dict[str, PatchStatistics] = field(default_factory=dict)

    # Aggregate statistics
    mean_in_circuit_similarity: float = 0.0
    mean_out_circuit_similarity: float = 0.0

    # Counterfactual analysis results - separate for in-circuit and out-circuit
    # Out-circuit: patch out-of-circuit neurons with corrupted values (tests sufficiency)
    # In-circuit: patch in-circuit neurons with corrupted values (tests necessity)
    out_counterfactual_effects: list[CounterfactualEffect] = field(default_factory=list)
    in_counterfactual_effects: list[CounterfactualEffect] = field(default_factory=list)

    # Legacy field for backwards compatibility (combines both)
    counterfactual_effects: list[CounterfactualEffect] = field(default_factory=list)

    mean_faithfulness_score: float = 0.0
    std_faithfulness_score: float = 0.0

    # Overall faithfulness score (higher is better)
    overall_faithfulness: float = 0.0


@dataclass
class RobustnessSample(SchemaClass):
    """Result of a single robustness test on one input."""

    input_values: list[float]  # The perturbed input [x0, x1]
    base_input: list[float]  # The original binary input [0, 1]
    noise_magnitude: float  # L2 norm of actual noise added: ||perturbed - base||
    ground_truth: float  # GT output for base input (e.g., XOR(0,1)=1)

    # Outputs from both models on the SAME perturbed input
    gate_output: float  # Output from gate_model(perturbed_input)
    subcircuit_output: float  # Output from subcircuit(perturbed_input)

    # Accuracy to ground truth
    gate_correct: bool  # round(gate_output) == ground_truth
    subcircuit_correct: bool  # round(subcircuit_output) == ground_truth

    # Agreement between models
    agreement_bit: bool  # round(gate_output) == round(subcircuit_output)
    mse: float  # (gate_output - subcircuit_output)^2


@dataclass
class RobustnessMetrics(SchemaClass):
    """Comprehensive robustness metrics for a subcircuit."""

    # All samples (for scatter plots by actual noise magnitude)
    noise_samples: list[RobustnessSample] = field(default_factory=list)
    ood_samples: list[RobustnessSample] = field(default_factory=list)

    # Aggregate statistics
    noise_gate_accuracy: float = 0.0
    noise_subcircuit_accuracy: float = 0.0
    noise_agreement_bit: float = 0.0
    noise_mse_mean: float = 0.0

    ood_gate_accuracy: float = 0.0
    ood_subcircuit_accuracy: float = 0.0
    ood_agreement_bit: float = 0.0
    ood_mse_mean: float = 0.0

    overall_robustness: float = 0.0  # Combined score


@dataclass
class Metrics(SchemaClass):
    # Train info
    avg_loss: Optional[float] = None
    val_acc: Optional[float] = None
    test_acc: Optional[float] = None

    # Circuit Info
    per_gate_metrics: dict[str, GateMetrics] = field(default_factory=dict)
    # Index of subcircuits that produces best result
    per_gate_bests: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # RobustnessMetrics for each best subcircuit per gate
    per_gate_bests_robust: dict[str, list["RobustnessMetrics"]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # FaithfulnessMetrics for each per_gate_metrics
    per_gate_bests_faith: dict[str, list[FaithfulnessMetrics]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class ProfilingEvent(SchemaClass):
    """A single profiling event with timing info."""

    status: str
    timestamp_ms: float  # Milliseconds since trial start
    elapsed_ms: float  # Milliseconds since last event


@dataclass
class ProfilingData(SchemaClass):
    """Profiling data for a trial."""

    device: str = "cpu"
    start_time_ms: float = 0.0  # Unix timestamp in ms when trial started
    total_duration_ms: float = 0.0  # Total trial duration in ms
    events: list[ProfilingEvent] = field(default_factory=list)

    # Aggregated phase durations (computed from events)
    phase_durations_ms: dict[str, float] = field(default_factory=dict)


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

    # Models stored at runtime (saved as model.pt, not in JSON)
    model: Optional["MLP"] = None
    decomposed_model: Optional["DecomposedMLP"] = None  # Full multi-gate model
    subcircuits: list = field(default_factory=list)
    subcircuit_structure_analysis: list = field(default_factory=list)
    # Maps gate_name -> DecomposedMLP for full single-gate model (saved separately)
    decomposed_gate_models: dict[str, "DecomposedMLP"] = field(default_factory=dict)
    # Maps gate_name -> subcircuit_idx -> DecomposedMLP (saved separately)
    decomposed_subcircuits: dict[str, dict[int, "DecomposedMLP"]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # Serializable: which subcircuit indices were decomposed per gate
    decomposed_subcircuit_indices: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Each TrialSetup defines an deterministic id
    trial_id: str = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.trial_id = self.setup.get_id()


@dataclass
class ExperimentConfig(SchemaClass):
    logger: Optional[Any] = None
    debug: bool = False
    device: str = "cpu"
    spd_device: str = (
        "cpu"  # CPU is fastest for small models (no GPU transfer overhead)
    )

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

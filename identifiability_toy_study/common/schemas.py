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
    """Config for Stochastic Parameter Decomposition.

    Based on SPD paper (arXiv:2506.20790) and Goodfire's TMS experiments:

    1. n_components: 2-4× max(n_functions, hidden_width)
       - For 2→3→1 network: [4, 8, 15, 20]
       - Paper TMS uses 20 subcomponents

    2. importance_coeff: Critical for sparsity (controls g values)
       - Paper uses 3e-3 for TMS experiments
       - Too low → all g≈1 (no sparsity), too high → all g→0
       - Sweep: [1e-4, 1e-3, 3e-3]

    3. steps: Training duration
       - Paper uses 40k for TMS but tiny models converge much faster
       - For 2→3→1: 1000 steps is usually sufficient
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


def generate_spd_sweep_configs(
    base_config: "SPDConfig" = None,
    n_components_list: list[int] = None,
    importance_coeff_list: list[float] = None,
    steps_list: list[int] = None,
    importance_p_list: list[float] = None,
) -> list["SPDConfig"]:
    """
    Generate multiple SPD configs for parameter sweep.

    Args:
        base_config: Base config to modify (uses defaults if None)
        n_components_list: List of n_components values to try
        importance_coeff_list: List of importance_coeff values to try
        steps_list: List of steps values to try
        importance_p_list: List of importance_p (pnorm) values to try

    Returns:
        List of SPDConfig objects for sweep
    """
    import copy
    from itertools import product

    if base_config is None:
        base_config = SPDConfig()

    # Default sweep values if not specified
    n_components_list = n_components_list or [base_config.n_components]
    importance_coeff_list = importance_coeff_list or [base_config.importance_coeff]
    steps_list = steps_list or [base_config.steps]
    importance_p_list = importance_p_list or [base_config.importance_p]

    configs = []
    for n_comp, imp_coeff, steps, imp_p in product(
        n_components_list, importance_coeff_list, steps_list, importance_p_list
    ):
        cfg = copy.deepcopy(base_config)
        cfg.n_components = n_comp
        cfg.importance_coeff = imp_coeff
        cfg.steps = steps
        cfg.importance_p = imp_p
        configs.append(cfg)

    return configs


@dataclass
class TrainParams(SchemaClass):
    learning_rate: float = 0.001
    batch_size: int = 2048
    epochs: int = 1000
    val_frequency: int = 1


@dataclass
class IdentifiabilityConstraints(SchemaClass):
    # Max deviation from bit_similarity=1.0 to be considered "best"
    # 0.01 = only 99%+ similar, 0.1 = 90%+ similar, 0.2 = 80%+ similar
    epsilon: float = 0.001  # More lenient to get more best circuits


@dataclass
class SubcircuitMetrics(SchemaClass):
    idx: int
    # Simple
    accuracy: float  # to gt, not full circuit (target)

    # Observational
    logit_similarity: float
    bit_similarity: float
    best_similarity: float = 0.0  # After clamping to binary [0,1]


@dataclass
class GateMetrics(SchemaClass):
    test_acc: float
    subcircuit_metrics: list[SubcircuitMetrics] = field(default_factory=list)


@dataclass
class InterventionSample(SchemaClass):
    """Result of a single intervention test (like RobustnessSample for faithfulness).

    NOTE: Activations are pre-computed here so visualization code
    NEVER needs to run models. Visualization is READ-ONLY.
    """

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

    # Pre-computed activations for visualization (NO model runs during viz!)
    # These are the activations AFTER the intervention/patch is applied
    gate_activations: list[list[float]] = field(default_factory=list)
    subcircuit_activations: list[list[float]] = field(default_factory=list)
    # Original activations BEFORE the intervention (for showing two-value display)
    original_gate_activations: list[list[float]] = field(default_factory=list)
    original_subcircuit_activations: list[list[float]] = field(default_factory=list)


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
    """Result of a single counterfactual test from the 2x2 patching matrix.

    The 2x2 matrix tests circuit faithfulness:

    |                | IN-Circuit Patch     | OUT-Circuit Patch      |
    |----------------|----------------------|------------------------|
    | DENOISING      | Sufficiency          | Completeness           |
    | (run corrupt,  | (recovery)           | (1 - recovery)         |
    | patch clean)   |                      |                        |
    |----------------|----------------------|------------------------|
    | NOISING        | Necessity            | Independence           |
    | (run clean,    | (disruption)         | (1 - disruption)       |
    | patch corrupt) |                      |                        |

    NOTE: Activations are pre-computed here so visualization code
    NEVER needs to run models. Visualization is READ-ONLY.
    """

    faithfulness_score: float  # Score depends on score_type

    # Experiment type: which direction are we patching?
    # - "denoising": Run corrupted input, patch with clean activations (Src: clean, Dest: corrupt)
    # - "noising": Run clean input, patch with corrupted activations (Src: corrupt, Dest: clean)
    experiment_type: str = "noising"

    # Score type: which of the 4 experiments?
    # - "sufficiency": Denoise in-circuit → tests if circuit can produce behavior
    # - "completeness": Denoise out-circuit → tests if anything is missing from circuit
    # - "necessity": Noise in-circuit → tests if circuit is required
    # - "independence": Noise out-circuit → tests if circuit is self-contained
    score_type: str = "necessity"

    # Clean/corrupted input info
    clean_input: list[float] = field(default_factory=list)  # e.g., [0, 1]
    corrupted_input: list[float] = field(default_factory=list)  # e.g., [1, 0]

    # Expected outputs (from original clean/corrupted runs, no intervention)
    expected_clean_output: float = 0.0  # y_clean (full model on clean input)
    expected_corrupted_output: float = (
        0.0  # y_corrupted (full model on corrupted input)
    )

    # Actual output from FULL MODEL with intervention (patched activations)
    actual_output: float = 0.0  # model(x_base, intervention=patch)

    # Did patching change output? (interpretation depends on experiment type)
    output_changed_to_corrupted: bool = False  # round(actual) == round(corrupted)

    # Pre-computed activations for visualization (NO model runs during viz!)
    # These are from the ORIGINAL clean/corrupted runs (reference)
    clean_activations: list[list[float]] = field(default_factory=list)
    corrupted_activations: list[list[float]] = field(default_factory=list)

    # Activations from the actual intervention run (FULL MODEL with patches)
    # This is what the visualization should show for the counterfactual
    intervened_activations: list[list[float]] = field(default_factory=list)


@dataclass
class FaithfulnessConfig(SchemaClass):
    """Configuration for faithfulness analysis."""

    max_subcircuits_per_gate: int = 1
    n_interventions_per_patch: int = 200  # High count for robust statistics
    n_counterfactual_pairs: int = 50  # Increased for better coverage


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
class FaithfulnessMetrics(SchemaClass):
    """Comprehensive faithfulness metrics for a subcircuit.

    Implements the 2x2 patching matrix:

    |                | IN-Circuit Patch     | OUT-Circuit Patch      |
    |----------------|----------------------|------------------------|
    | DENOISING      | Sufficiency          | Completeness           |
    | (run corrupt,  | (recovery → 1)       | (disruption → 1)       |
    | patch clean)   |                      |                        |
    |----------------|----------------------|------------------------|
    | NOISING        | Necessity            | Independence           |
    | (run clean,    | (disruption → 1)     | (recovery → 1)         |
    | patch corrupt) |                      |                        |

    A faithful circuit should score high on all 4 tests.
    """

    # Per-patch statistics for in-circuit and out-circuit interventions
    in_circuit_stats: dict[str, PatchStatistics] = field(default_factory=dict)
    out_circuit_stats: dict[str, PatchStatistics] = field(default_factory=dict)

    # OOD (out-of-distribution) intervention statistics
    in_circuit_stats_ood: dict[str, PatchStatistics] = field(default_factory=dict)
    out_circuit_stats_ood: dict[str, PatchStatistics] = field(default_factory=dict)

    # Aggregate statistics (in-distribution)
    mean_in_circuit_similarity: float = 0.0
    mean_out_circuit_similarity: float = 0.0

    # Aggregate statistics (out-of-distribution)
    mean_in_circuit_similarity_ood: float = 0.0
    mean_out_circuit_similarity_ood: float = 0.0

    # ===== 2x2 Matrix Counterfactual Effects =====
    # Denoising experiments (run corrupted, patch with clean)
    sufficiency_effects: list[CounterfactualEffect] = field(
        default_factory=list
    )  # Denoise in-circuit
    completeness_effects: list[CounterfactualEffect] = field(
        default_factory=list
    )  # Denoise out-circuit

    # Noising experiments (run clean, patch with corrupted)
    necessity_effects: list[CounterfactualEffect] = field(
        default_factory=list
    )  # Noise in-circuit
    independence_effects: list[CounterfactualEffect] = field(
        default_factory=list
    )  # Noise out-circuit

    # Legacy fields for backwards compatibility
    out_counterfactual_effects: list[CounterfactualEffect] = field(default_factory=list)
    in_counterfactual_effects: list[CounterfactualEffect] = field(default_factory=list)
    counterfactual_effects: list[CounterfactualEffect] = field(default_factory=list)

    # ===== Aggregate Scores (2x2 Matrix) =====
    mean_sufficiency: float = 0.0  # Denoise in-circuit: recovery
    mean_completeness: float = 0.0  # Denoise out-circuit: 1 - recovery
    mean_necessity: float = 0.0  # Noise in-circuit: disruption
    mean_independence: float = 0.0  # Noise out-circuit: 1 - disruption

    # Legacy aggregate scores
    mean_faithfulness_score: float = 0.0
    std_faithfulness_score: float = 0.0

    # Overall faithfulness score (higher is better)
    overall_faithfulness: float = 0.0


@dataclass
class RobustnessSample(SchemaClass):
    """Result of a single robustness test on one input.

    NOTE: Activations are pre-computed here so visualization code
    NEVER needs to run models. Visualization is READ-ONLY.

    Sample types:
    - noise: Gaussian noise perturbation
    - multiply_positive: Scale by factor > 1
    - multiply_negative: Scale by factor < 0
    - add: Add large positive value
    - subtract: Subtract large value
    - bimodal: Map [0,1] -> [-1,1] order-preserving (0->-1, 1->1)
    - bimodal_inv: Map [0,1] -> [-1,1] inverted (0->1, 1->-1)
    """

    input_values: list[float]  # The perturbed input [x0, x1]
    base_input: list[float]  # The original binary input [0, 1]
    noise_magnitude: float  # L2 norm of noise or transformation magnitude
    ground_truth: float  # GT output for base input (e.g., XOR(0,1)=1)

    # Outputs from both models on the SAME perturbed input
    gate_output: float  # Output from gate_model(perturbed_input)
    subcircuit_output: float  # Output from subcircuit(perturbed_input)

    # Accuracy to ground truth
    gate_correct: bool  # round(gate_output) == ground_truth
    subcircuit_correct: bool  # round(subcircuit_output) == ground_truth

    # Agreement between models
    agreement_bit: bool  # round(gate_output) == round(subcircuit_output)
    agreement_best: bool  # clamp_to_binary(gate) == clamp_to_binary(subcircuit)
    mse: float  # (gate_output - subcircuit_output)^2

    # Sample type for organizing visualizations
    sample_type: str = "noise"  # noise, multiply_positive, multiply_negative, add, subtract, bimodal, bimodal_inv

    # Pre-computed activations for visualization (NO model runs during viz!)
    # Each is a list of lists: [[layer0_acts], [layer1_acts], ...]
    gate_activations: list[list[float]] = field(default_factory=list)
    subcircuit_activations: list[list[float]] = field(default_factory=list)


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
    noise_agreement_best: float = 0.0
    noise_mse_mean: float = 0.0

    ood_gate_accuracy: float = 0.0
    ood_subcircuit_accuracy: float = 0.0
    ood_agreement_bit: float = 0.0
    ood_agreement_best: float = 0.0
    ood_mse_mean: float = 0.0

    overall_robustness: float = 0.0  # Combined score


@dataclass
class ObservationalMetrics(SchemaClass):
    """Aggregated observational (robustness) metrics for result.json."""

    # Noise perturbation metrics
    noise_gate_accuracy: float = 0.0
    noise_subcircuit_accuracy: float = 0.0
    noise_agreement_bit: float = 0.0
    noise_agreement_best: float = 0.0
    noise_mse_mean: float = 0.0
    noise_n_samples: int = 0

    # Per-type OOD metrics
    multiply_positive_agreement: float = 0.0
    multiply_positive_n_samples: int = 0
    multiply_negative_agreement: float = 0.0
    multiply_negative_n_samples: int = 0

    add_agreement: float = 0.0
    add_n_samples: int = 0

    subtract_agreement: float = 0.0
    subtract_n_samples: int = 0

    bimodal_agreement: float = 0.0
    bimodal_n_samples: int = 0
    bimodal_inv_agreement: float = 0.0
    bimodal_inv_n_samples: int = 0

    # Overall
    overall_observational: float = 0.0


@dataclass
class InterventionalMetrics(SchemaClass):
    """Aggregated interventional metrics for result.json."""

    # In-circuit (in-distribution)
    in_circuit_mean_bit_similarity: float = 0.0
    in_circuit_mean_logit_similarity: float = 0.0
    in_circuit_n_interventions: int = 0

    # In-circuit (out-of-distribution)
    in_circuit_ood_mean_bit_similarity: float = 0.0
    in_circuit_ood_mean_logit_similarity: float = 0.0
    in_circuit_ood_n_interventions: int = 0

    # Out-circuit (in-distribution)
    out_circuit_mean_bit_similarity: float = 0.0
    out_circuit_mean_logit_similarity: float = 0.0
    out_circuit_n_interventions: int = 0

    # Out-circuit (out-of-distribution)
    out_circuit_ood_mean_bit_similarity: float = 0.0
    out_circuit_ood_mean_logit_similarity: float = 0.0
    out_circuit_ood_n_interventions: int = 0

    # Overall
    overall_interventional: float = 0.0


@dataclass
class CounterfactualMetrics(SchemaClass):
    """Aggregated counterfactual metrics for result.json (2x2 matrix)."""

    # Denoising experiments
    mean_sufficiency: float = 0.0  # Denoise in-circuit
    mean_completeness: float = 0.0  # Denoise out-circuit
    n_denoising_pairs: int = 0

    # Noising experiments
    mean_necessity: float = 0.0  # Noise in-circuit
    mean_independence: float = 0.0  # Noise out-circuit
    n_noising_pairs: int = 0

    # Overall
    overall_counterfactual: float = 0.0


@dataclass
class FaithfulnessCategoryScore(SchemaClass):
    """Score and epsilon for a single faithfulness category."""
    score: float = 0.0
    epsilon: float = 0.0  # min(1.0 - component_scores), always positive


@dataclass
class FaithfulnessSummary(SchemaClass):
    """Summary of all faithfulness metrics for summary.json.

    Each category has:
    - score: Average of component scores
    - epsilon: Minimum margin from 1.0 across component scores (always positive)
    """

    observational: FaithfulnessCategoryScore = field(default_factory=FaithfulnessCategoryScore)
    interventional: FaithfulnessCategoryScore = field(default_factory=FaithfulnessCategoryScore)
    counterfactual: FaithfulnessCategoryScore = field(default_factory=FaithfulnessCategoryScore)
    overall: float = 0.0  # Combined overall score


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

"""SPD type definitions.

SPD (Stochastic Parameter Decomposition) breaks neural network weights into
interpretable "components" - each component is a rank-1 matrix (outer product
of vectors U and V) that contributes to the original weight matrix.

The key insight: when we train SPD with sparsity pressure, components that
implement the same function tend to activate together on the same inputs.
By clustering components based on their co-activation patterns, we can
discover the functional subcircuits in the network.

Hierarchy of types:
    SPDConfig           # How to run SPD decomposition
         ↓
    DecomposedMLP       # The trained decomposition (U, V matrices per layer)
         ↓
    SPDAnalysisResult   # Importance matrix, clustering, validation metrics
         ↓
    SPDSubcircuitEstimate  # Cluster assignments mapped to boolean functions
         ↓
    SpdTrialResult      # Per-trial: decomposed model + subcircuit estimate
         ↓
    SpdResults          # All trials + config
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.schema_class import SchemaClass

if TYPE_CHECKING:
    from src.model import DecomposedMLP


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SPDConfig(SchemaClass):
    """Configuration for Stochastic Parameter Decomposition.

    SPD decomposes weight matrix W into sum of rank-1 components:
        W ≈ Σ_c g_c * (U_c ⊗ V_c)

    where g_c is a learned "gate" value in [0,1] that indicates how much
    component c contributes. Training with sparsity pressure on g values
    encourages the model to use few components per input.

    Based on SPD paper (arXiv:2506.20790) and Goodfire's TMS experiments.

    Key tuning parameters:
    - n_components: How many rank-1 components per layer. Rule of thumb:
      2-4x the number of functions you expect to find. Paper uses 20.

    - importance_coeff: Sparsity pressure on gate values g. Higher = sparser.
      Paper uses 3e-3. Too low → all g≈1 (no sparsity). Too high → all g→0.

    - importance_p: The p in Lp norm for sparsity. 1.0 = L1 (paper default).
      0.5 = more extreme sparsity. 2.0 = softer, less sparse.

    - steps: Training iterations. 1000 is enough for tiny 2→3→1 networks.
      Paper uses 40k for larger models.
    """

    # Component count per layer (rule: 2-4x expected functions)
    n_components: int = 20

    # Training
    steps: int = 1000
    batch_size: int = 4096
    eval_batch_size: int = 4096
    n_eval_steps: int = 10
    learning_rate: float = 1e-3

    # Data generation for training SPD
    feature_probability: float = 0.5
    data_generation_type: str = "at_least_zero_active"

    # Loss coefficients
    importance_coeff: float = 1e-3  # Sparsity pressure on g values
    importance_p: float = 1.0       # Lp norm (1.0 = L1, paper default)
    recon_coeff: float = 1.0        # Weight on reconstruction loss

    # Post-training analysis
    activation_threshold: float = 0.5  # Component "active" if importance > this
    n_clusters: Optional[int] = None   # Auto-detect if None

    def get_config_id(self) -> str:
        """Short ID for this config based on key params."""
        return f"c{self.n_components}_s{self.steps}_i{self.importance_coeff:.0e}_p{self.importance_p}"


# =============================================================================
# Analysis Results
# =============================================================================


@dataclass
class ClusterInfo:
    """Information about a single cluster of SPD components.

    Components that activate together on the same inputs are grouped into
    clusters. Each cluster ideally corresponds to one boolean function
    (e.g., XOR, AND) that the network implements.

    Attributes:
        cluster_idx: Which cluster this is (0, 1, 2, ...)
        component_indices: Which components belong to this cluster
        component_labels: Human-readable labels like "layers.0.0:3"
        mean_importance: Average causal importance across all inputs
        function_mapping: Matched boolean function (e.g., "XOR (0.95)")
        robustness_score: How stable under input noise (0-1)
        faithfulness_score: How much ablating this cluster affects output (0-1)
    """

    cluster_idx: int
    component_indices: list[int]
    component_labels: list[str]
    mean_importance: float = 0.0
    robustness_score: float = 0.0
    faithfulness_score: float = 0.0
    function_mapping: str = ""


@dataclass
class SPDAnalysisResult:
    """Complete result of analyzing an SPD decomposition.

    After training SPD, we analyze it by:
    1. Computing causal importance (CI) for each component on each input
    2. Building a coactivation matrix (which components fire together)
    3. Clustering components based on coactivation patterns
    4. Mapping clusters to boolean functions by matching activation patterns

    Validation metrics (from SPD paper):
    - MMCS (Mean Max Cosine Similarity): Do components align with original weights?
      Should be ~1.0 for good decomposition.
    - ML2R (Mean L2 Ratio): Is magnitude preserved? Should be ~1.0.
    - Faithfulness loss: MSE between original and reconstructed weights.

    Component health:
    - Dead components have negligible weight norm (the decomposition learned
      not to use them). Having some dead components is good - it means
      n_components was high enough to capture everything.
    """

    # Structure
    n_components: int = 0
    n_layers: int = 0
    n_clusters: int = 0

    # Validation metrics (higher is better, 1.0 is perfect)
    mmcs: float = 0.0              # Directional alignment
    ml2r: float = 0.0              # Magnitude preservation
    faithfulness_loss: float = 0.0  # Lower is better

    # Component health (dead = not used by the decomposition)
    n_alive_components: int = 0
    n_dead_components: int = 0
    dead_component_labels: list[str] = field(default_factory=list)

    # Clustering results
    cluster_assignments: list[int] = field(default_factory=list)  # component_idx → cluster_idx
    clusters: list[ClusterInfo] = field(default_factory=list)

    # Raw data arrays
    importance_matrix: Optional[np.ndarray] = None    # [n_inputs, n_components]
    coactivation_matrix: Optional[np.ndarray] = None  # [n_components, n_components]
    component_labels: list[str] = field(default_factory=list)

    # Paths to generated visualizations
    visualization_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class SPDSubcircuitEstimate:
    """Subcircuits discovered by clustering SPD components.

    This is the main output of SPD analysis - it tells you:
    - How many functional subcircuits the network has (n_clusters)
    - Which components belong to each subcircuit (cluster_assignments)
    - What boolean function each subcircuit implements (cluster_functions)

    Example: A network trained on XOR might have 2 clusters:
    - Cluster 0: Components 0,1,5 → "XOR (0.92)" (matches XOR truth table)
    - Cluster 1: Components 2,3,4 → "INACTIVE" (dead components grouped together)

    Attributes:
        cluster_assignments: List where index i gives cluster ID for component i
        n_clusters: How many clusters were found
        cluster_sizes: How many components in each cluster
        cluster_functions: Map from cluster_idx to matched function name
        component_importance: Mean importance per component
        coactivation_matrix: How often component pairs fire together
    """

    cluster_assignments: list[int] = field(default_factory=list)
    n_clusters: int = 0
    cluster_sizes: list[int] = field(default_factory=list)
    component_importance: np.ndarray | None = None
    coactivation_matrix: np.ndarray | None = None
    component_labels: list[str] = field(default_factory=list)
    cluster_functions: dict[int, str] = field(default_factory=dict)
    full_analysis: Optional[SPDAnalysisResult] = None


# =============================================================================
# Experiment Results
# =============================================================================


@dataclass
class SpdTrialResult(SchemaClass):
    """SPD analysis results for a single trial.

    Each trial in an experiment gets its own SPD analysis because the trained
    model weights differ between trials (different random seeds, etc.).

    Attributes:
        trial_id: UUID of the trial this analysis is for
        decomposed_model: The trained SPD decomposition (U, V matrices)
        spd_subcircuit_estimate: Discovered subcircuits and their functions
    """

    trial_id: str = ""
    decomposed_model: Optional["DecomposedMLP"] = None
    spd_subcircuit_estimate: Optional[SPDSubcircuitEstimate] = None


@dataclass
class SpdResults(SchemaClass):
    """Complete SPD analysis results for an entire experiment.

    Contains SPD results for all trials, plus the config used to run SPD.

    Usage:
        spd_results = run_spd(experiment_result, run_dir)
        for trial_id, trial_result in spd_results.per_trial.items():
            estimate = trial_result.spd_subcircuit_estimate
            print(f"Trial {trial_id}: {estimate.n_clusters} subcircuits found")
            for cluster_idx, func in estimate.cluster_functions.items():
                print(f"  Cluster {cluster_idx}: {func}")
    """

    config: SPDConfig = field(default_factory=SPDConfig)
    per_trial: dict[str, SpdTrialResult] = field(default_factory=dict)

    def print_summary(self) -> str:
        """Print human-readable summary of SPD results."""
        lines = ["SPD Results Summary", "=" * 40]
        lines.append(f"Config: {self.config.get_config_id()}")
        lines.append(f"Trials: {len(self.per_trial)}")

        for trial_id, trial_result in self.per_trial.items():
            lines.append(f"\n  Trial: {trial_id[:8]}...")
            if trial_result.decomposed_model is not None:
                n_comps = trial_result.decomposed_model.get_n_components()
                lines.append(f"    Components: {n_comps}")
            if trial_result.spd_subcircuit_estimate is not None:
                n_clusters = trial_result.spd_subcircuit_estimate.n_clusters
                lines.append(f"    Clusters: {n_clusters}")

        return "\n".join(lines)

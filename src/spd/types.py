"""SPD type definitions.

All dataclasses for SPD analysis results and configuration.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.schemas.schema_class import SchemaClass

if TYPE_CHECKING:
    from src.model import DecomposedMLP


@dataclass
class SPDConfig(SchemaClass):
    """Configuration for Stochastic Parameter Decomposition.

    Based on SPD paper (arXiv:2506.20790) and Goodfire's TMS experiments.

    Key parameters:
    - n_components: 2-4x max(n_functions, hidden_width). Paper uses 20 for TMS.
    - importance_coeff: Sparsity pressure on g values. Paper uses 3e-3.
    - importance_p: pnorm for sparsity (1.0 = L1, paper default).
    - steps: Training duration. 1000 is sufficient for tiny networks.
    """

    n_components: int = 20
    steps: int = 1000
    batch_size: int = 4096
    eval_batch_size: int = 4096
    n_eval_steps: int = 10
    learning_rate: float = 1e-3
    feature_probability: float = 0.5
    data_generation_type: str = "at_least_zero_active"
    importance_coeff: float = 1e-3
    importance_p: float = 1.0
    recon_coeff: float = 1.0
    activation_threshold: float = 0.5
    n_clusters: Optional[int] = None

    def get_config_id(self) -> str:
        """Short ID for this config based on key params."""
        return f"c{self.n_components}_s{self.steps}_i{self.importance_coeff:.0e}_p{self.importance_p}"


@dataclass
class ClusterInfo:
    """Information about a single component cluster."""

    cluster_idx: int
    component_indices: list[int]
    component_labels: list[str]
    mean_importance: float = 0.0
    robustness_score: float = 0.0
    faithfulness_score: float = 0.0
    function_mapping: str = ""


@dataclass
class SPDAnalysisResult:
    """Complete result of SPD analysis including clustering and metrics."""

    n_components: int = 0
    n_layers: int = 0
    n_clusters: int = 0

    # Validation metrics (from SPD paper)
    mmcs: float = 0.0
    ml2r: float = 0.0
    faithfulness_loss: float = 0.0

    # Component health
    n_alive_components: int = 0
    n_dead_components: int = 0
    dead_component_labels: list[str] = field(default_factory=list)

    # Clustering results
    cluster_assignments: list[int] = field(default_factory=list)
    clusters: list[ClusterInfo] = field(default_factory=list)

    # Raw data arrays
    importance_matrix: Optional[np.ndarray] = None
    coactivation_matrix: Optional[np.ndarray] = None
    component_labels: list[str] = field(default_factory=list)

    # Visualization paths (relative to spd/ folder)
    visualization_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class SPDSubcircuitEstimate:
    """Result of SPD-based subcircuit estimation."""

    cluster_assignments: list[int] = field(default_factory=list)
    n_clusters: int = 0
    cluster_sizes: list[int] = field(default_factory=list)
    component_importance: np.ndarray | None = None
    coactivation_matrix: np.ndarray | None = None
    component_labels: list[str] = field(default_factory=list)
    cluster_functions: dict[int, str] = field(default_factory=dict)
    full_analysis: Optional[SPDAnalysisResult] = None


@dataclass
class SpdTrialResult(SchemaClass):
    """SPD analysis for a single trial."""

    trial_id: str = ""
    decomposed_model: Optional["DecomposedMLP"] = None
    spd_subcircuit_estimate: Optional[SPDSubcircuitEstimate] = None


@dataclass
class SpdResults(SchemaClass):
    """Complete SPD analysis results for an experiment."""

    config: SPDConfig = field(default_factory=SPDConfig)
    per_trial: dict[str, SpdTrialResult] = field(default_factory=dict)

    def print_summary(self) -> str:
        """Print summary of SPD results."""
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

"""
SPD type definitions.

Contains:
- SPDConfig: Configuration for SPD decomposition (moved from schemas/config.py)
- SpdTrialResult: SPD analysis results for a single trial
- SpdResults: Complete SPD results for an experiment
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from src.schemas.schema_class import SchemaClass

if TYPE_CHECKING:
    from src.model import DecomposedMLP


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
class SpdTrialResult(SchemaClass):
    """SPD analysis for a single trial."""

    trial_id: str = ""

    # Primary decomposition (first config)
    decomposed_model: Optional["DecomposedMLP"] = None
    spd_subcircuit_estimate: Optional[Any] = None

    # Multi-config SPD sweep results: maps config_id -> DecomposedMLP/estimate
    decomposed_models_sweep: dict[str, "DecomposedMLP"] = field(default_factory=dict)
    spd_subcircuit_estimates_sweep: dict[str, Any] = field(default_factory=dict)

    # Per-gate decompositions (optional)
    decomposed_gate_models: dict[str, "DecomposedMLP"] = field(default_factory=dict)
    decomposed_subcircuits: dict[str, dict[int, "DecomposedMLP"]] = field(
        default_factory=lambda: {}
    )


@dataclass
class SpdResults(SchemaClass):
    """Complete SPD analysis results for an experiment."""

    config: SPDConfig = field(default_factory=SPDConfig)
    sweep_configs: list[SPDConfig] = field(default_factory=list)
    per_trial: dict[str, SpdTrialResult] = field(default_factory=dict)

    def print_summary(self) -> str:
        """Print summary of SPD results."""
        lines = ["SPD Results Summary", "=" * 40]
        lines.append(f"Config: {self.config.get_config_id()}")
        lines.append(f"Sweep configs: {len(self.sweep_configs)}")
        lines.append(f"Trials: {len(self.per_trial)}")

        for trial_id, trial_result in self.per_trial.items():
            lines.append(f"\n  Trial: {trial_id[:8]}...")
            if trial_result.decomposed_model is not None:
                n_comps = trial_result.decomposed_model.get_n_components()
                lines.append(f"    Components: {n_comps}")
            if trial_result.spd_subcircuit_estimate is not None:
                n_clusters = trial_result.spd_subcircuit_estimate.n_clusters
                lines.append(f"    Clusters: {n_clusters}")
            lines.append(f"    Sweep results: {len(trial_result.decomposed_models_sweep)}")

        return "\n".join(lines)

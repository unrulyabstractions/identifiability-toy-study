"""Data types for training analysis and epiplexity estimation."""

from dataclasses import dataclass, field

from src.schema_class import SchemaClass


@dataclass
class StepRecord(SchemaClass):
    """Stored once per training step during Phase 1.

    Records the loss BEFORE gradient update, which is needed for
    computing epiplexity (structure learned during training).
    """

    step: int  # Training step index k = 0, 1, ...
    batch_indices: list[int]  # Which dataset rows were in this batch
    loss_pre_bits: float  # l_k: loss BEFORE update, in bits (sum over batch)
    # Filled in during Phase 2:
    loss_final_bits: float = 0.0  # l_k^final: final model's loss on same batch


@dataclass
class StepDiagnostics(SchemaClass):
    """Per-step derived metrics for plotting (computed in Phase 3)."""

    step: int
    loss_pre_bits: float  # l_k
    loss_final_bits: float  # l_k^final
    info_absorbed: float  # max(l_k - l_k^final, 0)
    cumulative_S_T: float  # Running sum of info_absorbed up to this step


@dataclass
class TrainingRecord(SchemaClass):
    """Complete record of training for epiplexity analysis.

    Populated during training (Phase 1) and completed during
    final model evaluation (Phase 2).
    """

    avg_loss: float = 0.0
    batch_size: int = 0
    step_records: list[StepRecord] = field(default_factory=list)

    @property
    def total_steps(self) -> int:
        return len(self.step_records)

    @property
    def total_samples(self) -> int:
        return self.total_steps * self.batch_size


@dataclass
class EpiplexityResult(SchemaClass):
    """Results of epiplexity estimation.

    S_T (Epiplexity): Total structural information absorbed during training.
        Computed as sum of max(l_k - l_k^final, 0) over all steps k.
        Higher S_T = more learnable structure in the data.

    H_T (Time-bounded entropy): Residual noise not captured by the model.
        Computed as sum of l_k^final over all steps k.
        Higher H_T = more noise/randomness in data.

    MDL (Minimum Description Length): Total = S_T + H_T
    """

    S_T: float = 0.0  # Epiplexity (structure learned)
    H_T: float = 0.0  # Time-bounded entropy (noise remaining)
    MDL: float = 0.0  # Total description length = S_T + H_T

    # Derived metrics
    structure_ratio: float = 0.0  # S_T / MDL, in [0, 1]
    total_steps: int = 0
    total_samples: int = 0
    S_T_per_sample: float = 0.0
    H_T_per_sample: float = 0.0

    def summary(self) -> str:
        """Human-readable summary of epiplexity results."""
        return (
            f"Epiplexity Analysis:\n"
            f"  S_T (structure): {self.S_T:.2f} bits\n"
            f"  H_T (noise):     {self.H_T:.2f} bits\n"
            f"  MDL (total):     {self.MDL:.2f} bits\n"
            f"  Structure ratio: {self.structure_ratio:.2%}\n"
            f"  Per-sample S_T:  {self.S_T_per_sample:.4f} bits\n"
            f"  Per-sample H_T:  {self.H_T_per_sample:.4f} bits"
        )


@dataclass
class InterpretationLabel(SchemaClass):
    """Human-readable interpretation of epiplexity results.

    Quadrant interpretation:
    - HIGH_S_LOW_H: Rich, clean data - prefer for transfer/OOD
    - HIGH_S_HIGH_H: Rich but noisy - structure exists but buried
    - LOW_S_LOW_H: Simple, clean - trivial problem
    - LOW_S_HIGH_H: Unlearnable noise - avoid for training
    """

    structure_label: str  # e.g., "High structure"
    noise_label: str  # e.g., "Low noise"
    data_quality: str  # e.g., "Rich, clean - prefer for transfer"
    quadrant: str  # e.g., "HIGH_S_LOW_H"


@dataclass
class TrainingAnalysis(SchemaClass):
    """Complete training analysis results.

    Contains the epiplexity estimation results and interpretation.
    """

    epi_result: EpiplexityResult = field(default_factory=EpiplexityResult)
    interpretation: InterpretationLabel | None = None
    diagnostics: list[StepDiagnostics] = field(default_factory=list)

"""Schema classes for the identifiability toy study.

This module provides all dataclass schemas used throughout the codebase.

Usage:
    from src.schemas import TrialResult, ExperimentConfig, ...
"""

# Base class
from .schema_class import SchemaClass

# Configuration classes
from .config import (
    DataParams,
    ExperimentConfig,
    FaithfulnessConfig,
    IdentifiabilityConstraints,
    ModelParams,
    ParallelConfig,
    SPDConfig,
    TrainParams,
    TrialSetup,
)

# Faithfulness classes
from .faithfulness import (
    CounterfactualEffect,
    CounterfactualMetrics,
    CounterfactualSample,
    FaithfulnessCategoryScore,
    FaithfulnessMetrics,
    FaithfulnessSummary,
    InterventionalMetrics,
    ObservationalMetrics,
    PatchStatistics,
    RobustnessMetrics,
)

# Evaluation classes (renamed from metrics)
from .evaluation import (
    GateMetrics,
    Metrics,
    ProfilingData,
    ProfilingEvent,
    SubcircuitMetrics,
)

# Result classes
from .results import (
    Dataset,
    ExperimentResult,
    TrialData,
    TrialResult,
)

# Sample classes
from .samples import (
    InterventionSample,
    ObservationalSample,
    RobustnessSample,
    SampleType,
)

__all__ = [
    # Base
    "SchemaClass",
    # Config
    "DataParams",
    "ModelParams",
    "SPDConfig",
    "TrainParams",
    "IdentifiabilityConstraints",
    "FaithfulnessConfig",
    "ParallelConfig",
    "TrialSetup",
    "ExperimentConfig",
    # Evaluation (renamed from Metrics)
    "SubcircuitMetrics",
    "GateMetrics",
    "Metrics",
    "ProfilingEvent",
    "ProfilingData",
    # Samples
    "InterventionSample",
    "RobustnessSample",
    "ObservationalSample",
    "SampleType",
    # Faithfulness
    "PatchStatistics",
    "CounterfactualEffect",
    "CounterfactualSample",
    "FaithfulnessMetrics",
    "RobustnessMetrics",
    "ObservationalMetrics",
    "InterventionalMetrics",
    "CounterfactualMetrics",
    "FaithfulnessCategoryScore",
    "FaithfulnessSummary",
    # Results
    "Dataset",
    "TrialData",
    "TrialResult",
    "ExperimentResult",
]

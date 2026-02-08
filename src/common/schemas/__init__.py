"""Schema classes for the identifiability toy study.

This module provides all dataclass schemas used throughout the codebase.
All classes are re-exported here for backwards compatibility with existing imports.

Usage:
    from src.common.schemas import TrialResult, ExperimentConfig, ...
    # or
    from .common.schemas import TrialResult, ExperimentConfig, ...
"""

# Base class
from .base import SchemaClass

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
    FaithfulnessCategoryScore,
    FaithfulnessMetrics,
    FaithfulnessSummary,
    InterventionalMetrics,
    ObservationalMetrics,
    PatchStatistics,
    RobustnessMetrics,
)

# Metrics classes
from .metrics import (
    GateMetrics,
    Metrics,
    ProfilingData,
    ProfilingEvent,
    SubcircuitMetrics,
    TrialMetrics,
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

# Re-export generate_spd_sweep_configs from its new location for backwards compatibility
from src.spd_internal.config import generate_spd_sweep_configs

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
    # Metrics
    "SubcircuitMetrics",
    "GateMetrics",
    "Metrics",
    "TrialMetrics",
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
    # Utilities (backwards compat)
    "generate_spd_sweep_configs",
]

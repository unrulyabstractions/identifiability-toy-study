"""Visualization-related schema classes.

Contains schema classes for pre-computed visualization data:
- MonteCarloData: Data from Monte Carlo sampling for decision boundary plots
- GridData: Data from grid sampling for 1D/2D decision boundary plots
"""

from dataclasses import dataclass, field

import numpy as np

from src.schema_class import SchemaClass


@dataclass
class CornerData(SchemaClass):
    """Binary corner coordinates and predictions.

    For n_inputs, there are 2^n_inputs corners (binary combinations).
    """

    corners: np.ndarray = field(default_factory=lambda: np.array([]))
    corner_predictions: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_corners(self) -> int:
        return len(self.corners)


@dataclass
class MonteCarloData(SchemaClass):
    """Pre-computed Monte Carlo sampling data for decision boundary visualization.

    Used for 3+ input dimensions where grid sampling is infeasible.

    Attributes:
        samples: Input samples [n_samples, n_inputs]
        predictions: Model predictions (probabilities) [n_samples]
        corners: Binary corner coordinates [2^n_inputs, n_inputs]
        corner_predictions: Predictions at corners [2^n_inputs]
        n_inputs: Number of input dimensions
        gate_idx: Which gate output was evaluated
        low: Lower bound of sampling range
        high: Upper bound of sampling range
    """

    samples: np.ndarray = field(default_factory=lambda: np.array([]))
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    corners: np.ndarray = field(default_factory=lambda: np.array([]))
    corner_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    n_inputs: int = 0
    gate_idx: int = 0
    low: float = -3.0
    high: float = 3.0

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_corners(self) -> int:
        return len(self.corners)


@dataclass
class GridData(SchemaClass):
    """Pre-computed grid sampling data for 1D/2D decision boundary visualization.

    Only valid for n_inputs <= 2 (grid is too large for higher dimensions).

    Attributes:
        grid_axes: List of 1D arrays for each axis
        grid_predictions: Predictions on the grid, shape [resolution] or [resolution, resolution]
        corners: Binary corner coordinates [2^n_inputs, n_inputs]
        corner_predictions: Predictions at corners [2^n_inputs]
        n_inputs: Number of input dimensions (1 or 2)
        gate_idx: Which gate output was evaluated
        resolution: Number of points per dimension
        low: Lower bound of grid range
        high: Upper bound of grid range
    """

    grid_axes: list[np.ndarray] = field(default_factory=list)
    grid_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    corners: np.ndarray = field(default_factory=lambda: np.array([]))
    corner_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    n_inputs: int = 0
    gate_idx: int = 0
    resolution: int = 100
    low: float = -3.0
    high: float = 3.0

    @property
    def n_corners(self) -> int:
        return len(self.corners)

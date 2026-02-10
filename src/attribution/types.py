"""Attribution type definitions.

Contains dataclasses for attribution results:
- AttributionResult: Results from patching-based attribution methods
- InputAttribution: Gradient/integrated gradients for input features
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class AttributionResult:
    """Result from activation patching or edge attribution.

    Attributes:
        method: Name of the attribution method used
        layer_attributions: Per-layer attribution scores
        edge_attributions: Edge-level attribution scores (for EAP)
        total_effect: Total causal effect of the patched component
        normalized_scores: Normalized attribution scores (sum to 1)
    """
    method: str = ""
    layer_attributions: dict[int, np.ndarray] = field(default_factory=dict)
    edge_attributions: dict[tuple[int, int, int, int], float] = field(default_factory=dict)
    total_effect: float = 0.0
    normalized_scores: Optional[np.ndarray] = None

    def get_top_edges(self, k: int = 10) -> list[tuple[tuple[int, int, int, int], float]]:
        """Get top-k edges by attribution score."""
        sorted_edges = sorted(
            self.edge_attributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_edges[:k]


@dataclass
class InputAttribution:
    """Attribution scores for input features.

    Attributes:
        method: Attribution method used (gradient, integrated_gradients, etc.)
        attributions: Attribution scores per input feature [n_samples, n_inputs]
        input_samples: The input samples used [n_samples, n_inputs]
        gate_idx: Which output gate was analyzed
        baseline: Baseline used for integrated gradients (if applicable)
    """
    method: str = ""
    attributions: Optional[np.ndarray] = None
    input_samples: Optional[np.ndarray] = None
    gate_idx: int = 0
    baseline: Optional[np.ndarray] = None

    def get_mean_attribution(self) -> np.ndarray:
        """Get mean attribution across samples."""
        if self.attributions is None:
            return np.array([])
        return np.mean(np.abs(self.attributions), axis=0)

    def get_relative_importance(self) -> np.ndarray:
        """Get relative importance of each input feature."""
        mean_attr = self.get_mean_attribution()
        total = np.sum(mean_attr)
        if total == 0:
            return mean_attr
        return mean_attr / total

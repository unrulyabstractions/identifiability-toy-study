"""Probing type definitions.

Contains dataclasses for probe training and evaluation results.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ProbeResult:
    """Result from training a linear probe.

    Attributes:
        layer_idx: Which layer the probe was trained on
        target_name: Name of the target concept being probed
        accuracy: Classification accuracy on test set
        train_accuracy: Classification accuracy on training set
        weights: Learned probe weights [n_classes, n_features]
        bias: Learned probe bias [n_classes]
        n_samples: Number of training samples used
        n_features: Number of input features (hidden dimension)
        loss_history: Training loss over epochs
    """
    layer_idx: int = 0
    target_name: str = ""
    accuracy: float = 0.0
    train_accuracy: float = 0.0
    weights: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    n_samples: int = 0
    n_features: int = 0
    loss_history: list[float] = field(default_factory=list)

    def get_feature_importance(self) -> np.ndarray:
        """Get importance of each hidden feature based on weight magnitude."""
        if self.weights is None:
            return np.array([])
        # For binary classification, use absolute weight values
        if self.weights.ndim == 1:
            return np.abs(self.weights)
        # For multi-class, use L2 norm across classes
        return np.linalg.norm(self.weights, axis=0)

    def summary(self) -> str:
        """Return a summary string of the probe result."""
        return (
            f"ProbeResult(layer={self.layer_idx}, target='{self.target_name}', "
            f"acc={self.accuracy:.3f}, train_acc={self.train_accuracy:.3f}, "
            f"n_samples={self.n_samples}, n_features={self.n_features})"
        )


@dataclass
class ProbeAnalysis:
    """Complete analysis of probing across layers.

    Attributes:
        probe_results: Dictionary mapping layer_idx to ProbeResult
        target_name: Name of the concept being probed
        best_layer: Layer with highest accuracy
        best_accuracy: Highest accuracy achieved
        layer_accuracies: List of (layer_idx, accuracy) for plotting
    """
    probe_results: dict[int, ProbeResult] = field(default_factory=dict)
    target_name: str = ""
    best_layer: int = -1
    best_accuracy: float = 0.0
    layer_accuracies: list[tuple[int, float]] = field(default_factory=list)

    def summary(self) -> str:
        """Return a summary string of the probe analysis."""
        lines = [f"ProbeAnalysis for '{self.target_name}'"]
        lines.append(f"  Best layer: {self.best_layer} (acc={self.best_accuracy:.3f})")
        lines.append("  All layers:")
        for layer_idx, acc in sorted(self.layer_accuracies):
            lines.append(f"    Layer {layer_idx}: {acc:.3f}")
        return "\n".join(lines)

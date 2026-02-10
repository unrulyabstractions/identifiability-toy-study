"""Geometric analysis type definitions.

Contains dataclasses for representational similarity analysis and CKA.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RDMResult:
    """Result from Representational Dissimilarity Matrix computation.

    Attributes:
        layer_idx: Which layer this RDM is for
        rdm: The dissimilarity matrix [n_samples, n_samples]
        distance_metric: The distance metric used
        n_samples: Number of samples
        n_features: Number of features at this layer
    """
    layer_idx: int = 0
    rdm: Optional[np.ndarray] = None
    distance_metric: str = "correlation"
    n_samples: int = 0
    n_features: int = 0

    def get_upper_triangle(self) -> np.ndarray:
        """Get the upper triangular values (excluding diagonal)."""
        if self.rdm is None:
            return np.array([])
        return self.rdm[np.triu_indices_from(self.rdm, k=1)]


@dataclass
class RSAResult:
    """Result from Representational Similarity Analysis.

    Attributes:
        rdm1_layer: Layer index of first RDM
        rdm2_layer: Layer index of second RDM
        similarity: RSA similarity (correlation of upper triangles)
        p_value: Statistical significance (if computed)
        method: Correlation method used
    """
    rdm1_layer: int = 0
    rdm2_layer: int = 0
    similarity: float = 0.0
    p_value: Optional[float] = None
    method: str = "spearman"


@dataclass
class CKAResult:
    """Result from Centered Kernel Alignment.

    Attributes:
        layer1_idx: First layer index
        layer2_idx: Second layer index
        cka: The CKA similarity value [0, 1]
        hsic_xy: HSIC(X, Y) value
        hsic_xx: HSIC(X, X) value
        hsic_yy: HSIC(Y, Y) value
        kernel: Kernel type used
    """
    layer1_idx: int = 0
    layer2_idx: int = 0
    cka: float = 0.0
    hsic_xy: float = 0.0
    hsic_xx: float = 0.0
    hsic_yy: float = 0.0
    kernel: str = "linear"


@dataclass
class GeometricAnalysis:
    """Complete geometric analysis of a model.

    Attributes:
        rdm_results: RDM for each layer
        rsa_matrix: Layer-to-layer RSA similarities
        cka_matrix: Layer-to-layer CKA similarities
        layer_names: Names for each layer
        n_layers: Number of layers analyzed
    """
    rdm_results: dict[int, RDMResult] = field(default_factory=dict)
    rsa_matrix: Optional[np.ndarray] = None
    cka_matrix: Optional[np.ndarray] = None
    layer_names: list[str] = field(default_factory=list)
    n_layers: int = 0

    def summary(self) -> str:
        """Return a summary string of the analysis."""
        lines = ["Geometric Analysis Summary"]
        lines.append(f"  Layers: {self.n_layers}")

        if self.cka_matrix is not None:
            lines.append("\n  CKA Matrix (diagonal):")
            for i in range(min(self.n_layers, len(self.layer_names))):
                name = self.layer_names[i] if i < len(self.layer_names) else f"Layer_{i}"
                lines.append(f"    {name}: {self.cka_matrix[i, i]:.3f}")

        return "\n".join(lines)

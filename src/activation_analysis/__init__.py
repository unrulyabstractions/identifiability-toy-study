"""Activation analysis for interpretability.

This module provides tools for analyzing activation patterns in neural networks:
- Activation statistics (mean, variance, correlation)
- Activation clustering
- Visualization utilities
"""

from src.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals(), recursive=True)

"""Training pipeline.

This module provides training functionality:
- train_mlp: Core training loop
- generate_dataset, generate_trial_data: Data generation utilities
- train_model: High-level training orchestration
"""

from .train_loop import train_mlp
from .data_gen import generate_dataset, generate_trial_data
from .orchestrator import train_model

__all__ = [
    "train_mlp",
    "generate_dataset",
    "generate_trial_data",
    "train_model",
]

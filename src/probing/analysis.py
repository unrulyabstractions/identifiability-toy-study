"""Probe analysis utilities.

Functions for analyzing and comparing probe results across layers.
"""

from typing import TYPE_CHECKING
import os

import numpy as np
import matplotlib.pyplot as plt

from .types import ProbeResult, ProbeAnalysis
from .linear_probe import train_probes_all_layers

if TYPE_CHECKING:
    from src.model import MLP


def analyze_probes(
    probe_results: dict[int, ProbeResult],
    target_name: str = "",
) -> ProbeAnalysis:
    """Analyze probe results across layers.

    Args:
        probe_results: Dictionary mapping layer_idx to ProbeResult
        target_name: Name of the target concept

    Returns:
        ProbeAnalysis with summary statistics
    """
    layer_accuracies = [
        (layer_idx, result.accuracy)
        for layer_idx, result in sorted(probe_results.items())
    ]

    if layer_accuracies:
        best_layer, best_accuracy = max(layer_accuracies, key=lambda x: x[1])
    else:
        best_layer, best_accuracy = -1, 0.0

    return ProbeAnalysis(
        probe_results=probe_results,
        target_name=target_name,
        best_layer=best_layer,
        best_accuracy=best_accuracy,
        layer_accuracies=layer_accuracies,
    )


def probe_for_gate(
    model: "MLP",
    gate_idx: int,
    n_samples: int = 1000,
    train_ratio: float = 0.8,
    n_epochs: int = 100,
    lr: float = 0.01,
    device: str = "cpu",
) -> ProbeAnalysis:
    """Probe all layers for a specific output gate.

    Generates random binary inputs and uses the model's output
    as the target for probing.

    Args:
        model: The MLP model to probe
        gate_idx: Which output gate to probe for
        n_samples: Number of samples to generate
        train_ratio: Fraction of samples for training
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Device to run on

    Returns:
        ProbeAnalysis for the specified gate
    """
    import torch
    from src.math import logits_to_binary

    model = model.to(device)

    # Generate random binary inputs
    x = torch.randint(0, 2, (n_samples, model.input_size), dtype=torch.float32, device=device)

    # Get model predictions as targets
    with torch.no_grad():
        logits = model(x)
        y = logits_to_binary(logits)[:, gate_idx]

    # Split into train/test
    n_train = int(n_samples * train_ratio)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Get gate name if available
    gate_name = (
        model.gate_names[gate_idx]
        if model.gate_names and gate_idx < len(model.gate_names)
        else f"Gate_{gate_idx}"
    )

    # Train probes on all layers
    probe_results = train_probes_all_layers(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_epochs=n_epochs,
        lr=lr,
        device=device,
        target_name=gate_name,
    )

    return analyze_probes(probe_results, target_name=gate_name)


def probe_for_input(
    model: "MLP",
    input_idx: int,
    n_samples: int = 1000,
    train_ratio: float = 0.8,
    n_epochs: int = 100,
    lr: float = 0.01,
    device: str = "cpu",
) -> ProbeAnalysis:
    """Probe all layers for a specific input feature.

    Tests if the input feature is linearly decodable from each layer.

    Args:
        model: The MLP model to probe
        input_idx: Which input feature to probe for
        n_samples: Number of samples to generate
        train_ratio: Fraction of samples for training
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Device to run on

    Returns:
        ProbeAnalysis for the input feature
    """
    import torch

    model = model.to(device)

    # Generate random binary inputs
    x = torch.randint(0, 2, (n_samples, model.input_size), dtype=torch.float32, device=device)

    # Target is the input feature itself
    y = x[:, input_idx]

    # Split into train/test
    n_train = int(n_samples * train_ratio)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    target_name = f"Input_{input_idx}"

    # Train probes on all layers
    probe_results = train_probes_all_layers(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_epochs=n_epochs,
        lr=lr,
        device=device,
        target_name=target_name,
    )

    return analyze_probes(probe_results, target_name=target_name)


def plot_probe_accuracy(
    analysis: ProbeAnalysis,
    output_path: str,
    title: str | None = None,
) -> str:
    """Plot probe accuracy across layers.

    Args:
        analysis: ProbeAnalysis to plot
        output_path: Path to save the figure
        title: Optional title for the plot

    Returns:
        Path to the saved figure
    """
    layers = [x[0] for x in analysis.layer_accuracies]
    accuracies = [x[1] for x in analysis.layer_accuracies]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(layers, accuracies, color='steelblue', alpha=0.8)

    # Highlight best layer
    best_idx = layers.index(analysis.best_layer)
    bars[best_idx].set_color('red')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title(title or f"Probe Accuracy for '{analysis.target_name}'")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_probe_comparison(
    analyses: list[ProbeAnalysis],
    output_path: str,
    title: str | None = None,
) -> str:
    """Plot probe accuracies for multiple targets.

    Args:
        analyses: List of ProbeAnalysis to compare
        output_path: Path to save the figure
        title: Optional title for the plot

    Returns:
        Path to the saved figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_analyses = len(analyses)
    width = 0.8 / n_analyses

    for i, analysis in enumerate(analyses):
        layers = [x[0] for x in analysis.layer_accuracies]
        accuracies = [x[1] for x in analysis.layer_accuracies]

        x_positions = np.array(layers) + i * width - (n_analyses - 1) * width / 2
        ax.bar(x_positions, accuracies, width, label=analysis.target_name, alpha=0.8)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title(title or "Probe Accuracy Comparison")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_feature_importance(
    result: ProbeResult,
    output_path: str,
    title: str | None = None,
) -> str:
    """Plot feature importance from probe weights.

    Args:
        result: ProbeResult with trained weights
        output_path: Path to save the figure
        title: Optional title for the plot

    Returns:
        Path to the saved figure
    """
    importance = result.get_feature_importance()

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(importance))
    ax.bar(x, importance, color='steelblue', alpha=0.8)

    ax.set_xlabel('Hidden Feature')
    ax.set_ylabel('Importance (|weight|)')
    ax.set_title(title or f"Feature Importance - Layer {result.layer_idx}")
    ax.set_xticks(x)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

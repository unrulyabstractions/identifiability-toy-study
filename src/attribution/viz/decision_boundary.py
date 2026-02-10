"""Decision boundary visualization using Monte Carlo sampling.

Visualizes model decision boundaries by uniformly sampling the input space
and plotting predictions with transparent overlapping markers.
"""

from typing import TYPE_CHECKING
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.model import MLP


def plot_decision_boundary_predictions(
    model: "MLP",
    output_path: str,
    x_range: tuple[float, float] = (-2, 2),
    y_range: tuple[float, float] = (-2, 2),
    n_samples: int = 10000,
    marker_size: float = 50,
    alpha: float = 0.3,
    gate_idx: int = 0,
    device: str = "cpu",
) -> str:
    """Plot decision boundary based on model predictions.

    Samples uniformly from x_range x y_range, plots with transparent markers.
    Color based on pred = logits_to_binary(mlp(sample))[gate_idx].

    Args:
        model: The MLP model to visualize
        output_path: Path to save the figure
        x_range: Range for x-axis sampling
        y_range: Range for y-axis sampling
        n_samples: Number of samples to draw
        marker_size: Size of scatter plot markers
        alpha: Transparency of markers
        gate_idx: Which output gate to visualize
        device: Device to run inference on

    Returns:
        Path to the saved figure
    """
    from src.math import logits_to_binary

    # Sample uniformly
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    y = np.random.uniform(y_range[0], y_range[1], n_samples)
    samples = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32, device=device)

    # Get predictions
    model = model.to(device)
    with torch.inference_mode():
        logits = model(samples)
        preds = logits_to_binary(logits)[:, gate_idx].cpu().numpy()

    # Plot with overlapping transparent circles
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['blue' if p == 0 else 'red' for p in preds]
    ax.scatter(x, y, c=colors, s=marker_size, alpha=alpha)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_title('Decision Boundary (Prediction)')
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_decision_boundary_logits(
    model: "MLP",
    output_path: str,
    x_range: tuple[float, float] = (-2, 2),
    y_range: tuple[float, float] = (-2, 2),
    resolution: int = 200,
    gate_idx: int = 0,
    device: str = "cpu",
    cmap: str = "RdBu",
) -> str:
    """Plot decision boundary using a dense grid with logit values.

    Creates a filled contour plot showing the raw logit values,
    with the decision boundary at logit=0.

    Args:
        model: The MLP model to visualize
        output_path: Path to save the figure
        x_range: Range for x-axis
        y_range: Range for y-axis
        resolution: Grid resolution (resolution x resolution points)
        gate_idx: Which output gate to visualize
        device: Device to run inference on
        cmap: Colormap to use

    Returns:
        Path to the saved figure
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x, y)

    # Flatten and convert to tensor
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

    # Get logits
    model = model.to(device)
    with torch.inference_mode():
        logits = model(grid_tensor)[:, gate_idx].cpu().numpy()

    # Reshape to grid
    logits_grid = logits.reshape(resolution, resolution)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Filled contour
    vmax = max(abs(logits_grid.min()), abs(logits_grid.max()))
    contour = ax.contourf(xx, yy, logits_grid, levels=50, cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar(contour, ax=ax, label='Logit')

    # Decision boundary (logit = 0)
    ax.contour(xx, yy, logits_grid, levels=[0], colors='black', linewidths=2)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_title('Decision Boundary (Logit)')
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_decision_boundary_comparison(
    model: "MLP",
    output_path: str,
    x_range: tuple[float, float] = (-2, 2),
    y_range: tuple[float, float] = (-2, 2),
    n_samples: int = 5000,
    resolution: int = 100,
    gate_idx: int = 0,
    device: str = "cpu",
) -> str:
    """Plot both scatter and contour decision boundaries side by side.

    Args:
        model: The MLP model to visualize
        output_path: Path to save the figure
        x_range: Range for x-axis
        y_range: Range for y-axis
        n_samples: Number of samples for scatter plot
        resolution: Grid resolution for contour plot
        gate_idx: Which output gate to visualize
        device: Device to run inference on

    Returns:
        Path to the saved figure
    """
    from src.math import logits_to_binary

    model = model.to(device)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Scatter plot
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    y = np.random.uniform(y_range[0], y_range[1], n_samples)
    samples = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32, device=device)

    with torch.inference_mode():
        logits = model(samples)
        preds = logits_to_binary(logits)[:, gate_idx].cpu().numpy()

    colors = ['blue' if p == 0 else 'red' for p in preds]
    axes[0].scatter(x, y, c=colors, s=30, alpha=0.3)
    axes[0].set_xlim(x_range)
    axes[0].set_ylim(y_range)
    axes[0].set_xlabel('Input 1')
    axes[0].set_ylabel('Input 2')
    axes[0].set_title('Prediction Scatter')

    # Right: Contour plot
    x_grid = np.linspace(x_range[0], x_range[1], resolution)
    y_grid = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

    with torch.inference_mode():
        grid_logits = model(grid_tensor)[:, gate_idx].cpu().numpy()

    logits_grid = grid_logits.reshape(resolution, resolution)
    vmax = max(abs(logits_grid.min()), abs(logits_grid.max()))
    contour = axes[1].contourf(xx, yy, logits_grid, levels=50, cmap='RdBu', vmin=-vmax, vmax=vmax)
    plt.colorbar(contour, ax=axes[1], label='Logit')
    axes[1].contour(xx, yy, logits_grid, levels=[0], colors='black', linewidths=2)
    axes[1].set_xlim(x_range)
    axes[1].set_ylim(y_range)
    axes[1].set_xlabel('Input 1')
    axes[1].set_ylabel('Input 2')
    axes[1].set_title('Logit Contour')

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

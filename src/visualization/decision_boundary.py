"""Decision boundary visualization using Monte Carlo sampling.

This module provides plotting functions that visualize pre-computed decision boundary data.
The data generation should happen during experiment execution (see src.trial modules).

Supports different input dimensions:
- 1-input: Line plot (y = model output vs x)
- 2-input: 2D contour/heatmap
- 3-input: Colored volume + 2D projections
- 4+ input: 2D and 3D projections
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    pass


# =============================================================================
# Data Generation Functions (called during experiment execution)
# =============================================================================


def generate_monte_carlo_data(
    model,
    n_inputs: int,
    gate_idx: int = 0,
    n_samples: int = 10000,
    low: float = -0.5,
    high: float = 1.5,
    device: str = "cpu",
) -> dict:
    """Generate Monte Carlo samples and model predictions for decision boundary visualization.

    This should be called during experiment execution to pre-compute all data.

    Args:
        model: MLP model to evaluate
        n_inputs: Number of input dimensions
        gate_idx: Which gate output to evaluate
        n_samples: Number of Monte Carlo samples
        low: Lower bound for sampling
        high: Upper bound for sampling
        device: Device for computation

    Returns:
        Dict containing:
            - samples: np.ndarray [n_samples, n_inputs] - input samples
            - predictions: np.ndarray [n_samples] - model predictions (probabilities)
            - corners: np.ndarray [2^n_inputs, n_inputs] - binary corner coordinates
            - corner_predictions: np.ndarray [2^n_inputs] - predictions at corners
    """
    import torch

    # Generate random samples
    samples = torch.rand(n_samples, n_inputs, device=device) * (high - low) + low

    # Evaluate model
    model.eval()
    with torch.no_grad():
        logits = model(samples)
        predictions = torch.sigmoid(logits[:, gate_idx])

    # Generate binary corners
    n_corners = 2**n_inputs
    corners = torch.zeros(n_corners, n_inputs, device=device)
    for i in range(n_corners):
        for j in range(n_inputs):
            corners[i, j] = (i >> j) & 1

    # Evaluate corners
    with torch.no_grad():
        corner_logits = model(corners)
        corner_predictions = torch.sigmoid(corner_logits[:, gate_idx])

    return {
        "samples": samples.cpu().numpy(),
        "predictions": predictions.cpu().numpy(),
        "corners": corners.cpu().numpy(),
        "corner_predictions": corner_predictions.cpu().numpy(),
        "n_inputs": n_inputs,
        "gate_idx": gate_idx,
        "low": low,
        "high": high,
    }


def generate_grid_data(
    model,
    n_inputs: int,
    gate_idx: int = 0,
    resolution: int = 100,
    low: float = -0.5,
    high: float = 1.5,
    device: str = "cpu",
) -> dict:
    """Generate grid samples and model predictions for 1D/2D decision boundary visualization.

    Only works for n_inputs <= 2 (grid is too large for higher dimensions).

    Args:
        model: MLP model to evaluate
        n_inputs: Number of input dimensions (must be 1 or 2)
        gate_idx: Which gate output to evaluate
        resolution: Number of points per dimension
        low: Lower bound
        high: Upper bound
        device: Device for computation

    Returns:
        Dict containing:
            - grid_axes: list of np.ndarray - axes for each dimension
            - grid_predictions: np.ndarray - predictions on the grid
            - corners: np.ndarray [2^n_inputs, n_inputs] - binary corner coordinates
            - corner_predictions: np.ndarray [2^n_inputs] - predictions at corners
    """
    import torch

    if n_inputs > 2:
        raise ValueError(f"Grid data only supports n_inputs <= 2, got {n_inputs}")

    # Create grid
    axes = [
        torch.linspace(low, high, resolution, device=device) for _ in range(n_inputs)
    ]

    if n_inputs == 1:
        grid_points = axes[0].unsqueeze(1)
    else:
        xx, yy = torch.meshgrid(axes[0], axes[1], indexing="ij")
        grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Evaluate model on grid
    model.eval()
    with torch.no_grad():
        logits = model(grid_points)
        predictions = torch.sigmoid(logits[:, gate_idx])

    # Generate binary corners
    n_corners = 2**n_inputs
    corners = torch.zeros(n_corners, n_inputs, device=device)
    for i in range(n_corners):
        for j in range(n_inputs):
            corners[i, j] = (i >> j) & 1

    # Evaluate corners
    with torch.no_grad():
        corner_logits = model(corners)
        corner_predictions = torch.sigmoid(corner_logits[:, gate_idx])

    return {
        "grid_axes": [ax.cpu().numpy() for ax in axes],
        "grid_predictions": predictions.cpu().numpy().reshape([resolution] * n_inputs),
        "corners": corners.cpu().numpy(),
        "corner_predictions": corner_predictions.cpu().numpy(),
        "n_inputs": n_inputs,
        "gate_idx": gate_idx,
        "resolution": resolution,
        "low": low,
        "high": high,
    }


# =============================================================================
# Plotting Functions (visualization only, no model calls)
# =============================================================================


def plot_decision_boundary_1d_from_data(
    data: dict,
    gate_name: str = "Gate",
    output_path: str = None,
    show: bool = False,
) -> str:
    """Plot decision boundary for 1-input case from pre-computed data.

    Args:
        data: Dict from generate_grid_data with n_inputs=1
        gate_name: Name for title
        output_path: Path to save figure
        show: Whether to display

    Returns:
        Path to saved figure
    """
    x = data["grid_axes"][0]
    y = data["grid_predictions"]
    corners = data["corners"]
    corner_preds = data["corner_predictions"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, y, color="steelblue", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Decision boundary")
    ax.axvline(0.5, color="red", linestyle=":", alpha=0.5, label="Input threshold")

    # Mark corners
    for i, (cx, cy) in enumerate(zip(corners.flatten(), corner_preds)):
        color = "blue" if cy < 0.5 else "red"
        ax.scatter([cx], [cy], c=color, s=150, marker="s", edgecolors="black", zorder=5)
        ax.annotate(
            f"({int(cx)})\n{cy:.2f}",
            (cx, cy),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=9,
        )

    ax.set_xlabel("Input")
    ax.set_ylabel("P(output=1)")
    ax.set_title(f"{gate_name}: Decision Boundary (1D)")
    ax.set_xlim(data["low"], data["high"])
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path or ""


def plot_decision_boundary_2d_from_data(
    data: dict,
    gate_name: str = "Gate",
    output_path: str = None,
    show: bool = False,
    mc_data: dict = None,
) -> str:
    """Plot decision boundary for 2-input case from pre-computed data.

    Args:
        data: Dict from generate_grid_data with n_inputs=2
        gate_name: Name for title
        output_path: Path to save figure
        show: Whether to display
        mc_data: Optional Monte Carlo data for overlay

    Returns:
        Path to saved figure
    """
    x_axis = data["grid_axes"][0]
    y_axis = data["grid_axes"][1]
    zz = data["grid_predictions"]
    corners = data["corners"]
    corner_preds = data["corner_predictions"]

    xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")

    fig, ax = plt.subplots(figsize=(8, 7))

    # Contour fill - let colors speak for themselves (no explicit boundary line)
    cf = ax.contourf(xx, yy, zz, levels=50, cmap="RdYlBu_r", alpha=0.9)

    # Monte Carlo overlay
    if mc_data is not None:
        samples = mc_data["samples"]
        preds = mc_data["predictions"]
        ax.scatter(
            samples[:, 0], samples[:, 1], c=preds, cmap="RdYlBu_r", s=5, alpha=0.3
        )

    # Mark binary corners
    for i in range(len(corners)):
        cx, cy = corners[i]
        pred = corner_preds[i]
        color = "blue" if pred < 0.5 else "red"
        ax.scatter([cx], [cy], c=color, s=200, marker="s", edgecolors="black", zorder=5)
        ax.annotate(
            f"({int(cx)},{int(cy)})\n{pred:.2f}",
            (cx, cy),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=8,
        )

    plt.colorbar(cf, ax=ax, label="P(output=1)")
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_title(f"{gate_name}: Decision Boundary (2D)")
    ax.set_xlim(data["low"], data["high"])
    ax.set_ylim(data["low"], data["high"])
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_path:
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path or ""


def plot_decision_boundary_3d_from_data(
    data: dict,
    gate_name: str = "Gate",
    output_path: str = None,
    show: bool = False,
) -> dict[str, str]:
    """Plot decision boundary for 3-input case from pre-computed Monte Carlo data.

    Creates:
    - 3D scatter plot
    - 2D projections onto each pair of axes

    Args:
        data: Dict from generate_monte_carlo_data with n_inputs=3
        gate_name: Name for title
        output_path: Base path for figures
        show: Whether to display

    Returns:
        Dict of paths to saved figures
    """
    samples = data["samples"]
    predictions = data["predictions"]
    corners = data["corners"]
    corner_preds = data["corner_predictions"]

    paths = {}
    base_path = Path(output_path) if output_path else Path(".")
    base_dir = base_path.parent
    base_name = base_path.stem
    os.makedirs(base_dir, exist_ok=True)

    # 1. 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        samples[:, 0],
        samples[:, 1],
        samples[:, 2],
        c=predictions,
        cmap="RdYlBu_r",
        s=10,
        alpha=0.6,
    )

    # Mark binary corners
    for i in range(len(corners)):
        corner = corners[i]
        pred = corner_preds[i]
        color = "blue" if pred < 0.5 else "red"
        ax.scatter(
            [corner[0]],
            [corner[1]],
            [corner[2]],
            c=color,
            s=100,
            marker="s",
            edgecolors="black",
        )

    plt.colorbar(scatter, ax=ax, label="P(output=1)", shrink=0.6)
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_zlabel("Input 3")
    ax.set_title(f"{gate_name}: Decision Boundary (3D Monte Carlo)")

    plt.tight_layout()
    path_3d = base_dir / f"{base_name}_3d.png"
    plt.savefig(path_3d, dpi=150, bbox_inches="tight")
    paths["3d"] = str(path_3d)

    if show:
        plt.show()
    else:
        plt.close(fig)

    # 2. 2D projections
    projections = [
        (0, 1, "Input 1", "Input 2", "x1_x2"),
        (0, 2, "Input 1", "Input 3", "x1_x3"),
        (1, 2, "Input 2", "Input 3", "x2_x3"),
    ]

    for dim1, dim2, xlabel, ylabel, suffix in projections:
        fig, ax = plt.subplots(figsize=(8, 7))

        scatter = ax.scatter(
            samples[:, dim1],
            samples[:, dim2],
            c=predictions,
            cmap="RdYlBu_r",
            s=10,
            alpha=0.5,
        )

        # Mark projected corners
        for i in range(len(corners)):
            corner = corners[i]
            pred = corner_preds[i]
            color = "blue" if pred < 0.5 else "red"
            ax.scatter(
                [corner[dim1]],
                [corner[dim2]],
                c=color,
                s=150,
                marker="s",
                edgecolors="black",
                zorder=5,
            )

        plt.colorbar(scatter, ax=ax, label="P(output=1)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{gate_name}: 2D Projection ({suffix})")
        ax.set_xlim(data["low"], data["high"])
        ax.set_ylim(data["low"], data["high"])
        ax.set_aspect("equal")

        plt.tight_layout()
        path_proj = base_dir / f"{base_name}_proj_{suffix}.png"
        plt.savefig(path_proj, dpi=150, bbox_inches="tight")
        paths[f"proj_{suffix}"] = str(path_proj)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return paths


def plot_decision_boundary_nd_from_data(
    data: dict,
    gate_name: str = "Gate",
    output_path: str = None,
    show: bool = False,
) -> dict[str, str]:
    """Plot decision boundary for 4+ input case from pre-computed Monte Carlo data.

    Creates:
    - 2D projections for all pairs of dimensions
    - 3D projections for selected triplets

    Args:
        data: Dict from generate_monte_carlo_data with n_inputs >= 4
        gate_name: Name for title
        output_path: Base path for figures
        show: Whether to display

    Returns:
        Dict of paths to saved figures
    """
    samples = data["samples"]
    predictions = data["predictions"]
    n_inputs = data["n_inputs"]

    paths = {}
    base_path = Path(output_path) if output_path else Path(".")
    base_dir = base_path.parent
    base_name = base_path.stem
    os.makedirs(base_dir, exist_ok=True)

    # 2D projections for all pairs
    for i in range(n_inputs):
        for j in range(i + 1, n_inputs):
            fig, ax = plt.subplots(figsize=(7, 6))

            scatter = ax.scatter(
                samples[:, i],
                samples[:, j],
                c=predictions,
                cmap="RdYlBu_r",
                s=8,
                alpha=0.4,
            )

            plt.colorbar(scatter, ax=ax, label="P(output=1)")
            ax.set_xlabel(f"Input {i + 1}")
            ax.set_ylabel(f"Input {j + 1}")
            ax.set_title(f"{gate_name}: 2D Projection (x{i + 1}_x{j + 1})")
            ax.set_xlim(data["low"], data["high"])
            ax.set_ylim(data["low"], data["high"])
            ax.set_aspect("equal")

            plt.tight_layout()
            suffix = f"x{i + 1}_x{j + 1}"
            path_proj = base_dir / f"{base_name}_2d_{suffix}.png"
            plt.savefig(path_proj, dpi=150, bbox_inches="tight")
            paths[f"2d_{suffix}"] = str(path_proj)

            if show:
                plt.show()
            else:
                plt.close(fig)

    # 3D projections for first few triplets
    triplets = []
    for i in range(min(n_inputs, 4)):
        for j in range(i + 1, min(n_inputs, 4)):
            for k in range(j + 1, min(n_inputs, 4)):
                triplets.append((i, j, k))

    for dim1, dim2, dim3 in triplets[:4]:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            samples[:, dim1],
            samples[:, dim2],
            samples[:, dim3],
            c=predictions,
            cmap="RdYlBu_r",
            s=8,
            alpha=0.5,
        )

        plt.colorbar(scatter, ax=ax, label="P(output=1)", shrink=0.6)
        ax.set_xlabel(f"Input {dim1 + 1}")
        ax.set_ylabel(f"Input {dim2 + 1}")
        ax.set_zlabel(f"Input {dim3 + 1}")
        ax.set_title(
            f"{gate_name}: 3D Projection (x{dim1 + 1}_x{dim2 + 1}_x{dim3 + 1})"
        )

        plt.tight_layout()
        suffix = f"x{dim1 + 1}_x{dim2 + 1}_x{dim3 + 1}"
        path_proj = base_dir / f"{base_name}_3d_{suffix}.png"
        plt.savefig(path_proj, dpi=150, bbox_inches="tight")
        paths[f"3d_{suffix}"] = str(path_proj)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return paths


# =============================================================================
# Convenience Functions
# =============================================================================


def plot_decision_boundary_from_data(
    data: dict,
    gate_name: str = "Gate",
    output_path: str = None,
    show: bool = False,
) -> str | dict[str, str]:
    """Plot decision boundary from pre-computed data - dispatches based on n_inputs.

    Args:
        data: Dict from generate_grid_data (1D/2D) or generate_monte_carlo_data (3D+)
        gate_name: Name for title
        output_path: Path to save figure(s)
        show: Whether to display

    Returns:
        Path (for 1D/2D) or dict of paths (for 3D+)
    """
    n_inputs = data["n_inputs"]

    if n_inputs == 1:
        return plot_decision_boundary_1d_from_data(data, gate_name, output_path, show)
    elif n_inputs == 2:
        return plot_decision_boundary_2d_from_data(data, gate_name, output_path, show)
    elif n_inputs == 3:
        return plot_decision_boundary_3d_from_data(data, gate_name, output_path, show)
    else:
        return plot_decision_boundary_nd_from_data(data, gate_name, output_path, show)


def visualize_all_gates_from_data(
    gate_data: dict[str, dict],
    output_dir: str,
    show: bool = False,
) -> dict[str, str | dict[str, str]]:
    """Visualize decision boundaries for all gates from pre-computed data.

    Args:
        gate_data: Dict mapping gate_name -> data dict
        output_dir: Directory to save figures
        show: Whether to display

    Returns:
        Dict mapping gate_name -> path(s)
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for gate_name, data in gate_data.items():
        output_path = os.path.join(output_dir, f"{gate_name}_decision_boundary")
        n_inputs = data["n_inputs"]
        if n_inputs <= 2:
            output_path += ".png"
        results[gate_name] = plot_decision_boundary_from_data(
            data=data,
            gate_name=gate_name,
            output_path=output_path,
            show=show,
        )

    return results

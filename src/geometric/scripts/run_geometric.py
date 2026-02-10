"""Run geometric analysis on a saved MLP.

Usage:
    python -m src.geometric.scripts.run_geometric [SAVED_MLP_PATH] [--output-dir OUTPUT_DIR]

Example:
    python -m src.geometric.scripts.run_geometric tmp/test_xor_and.pt --output-dir tmp/geometric_output
"""

import argparse
import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import MLP
from src.geometric.rsa import compute_all_rdms, compute_rsa_matrix
from src.geometric.cka import compare_layers_cka, compute_cka_result
from src.geometric.types import GeometricAnalysis


def plot_similarity_matrix(
    matrix: np.ndarray,
    layer_names: list[str],
    output_path: str,
    title: str = "Similarity Matrix",
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> str:
    """Plot a similarity matrix as a heatmap.

    Args:
        matrix: Similarity matrix [n_layers, n_layers]
        layer_names: Names for each layer
        output_path: Path to save the figure
        title: Title for the plot
        cmap: Colormap to use
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap

    Returns:
        Path to the saved figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                          ha="center", va="center", color="white" if matrix[i, j] > 0.5 else "black",
                          fontsize=8)

    ax.set_xticks(range(len(layer_names)))
    ax.set_yticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_yticklabels(layer_names)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Layer')
    ax.set_title(title)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_rdm(
    rdm: np.ndarray,
    output_path: str,
    layer_idx: int,
    title: str | None = None,
) -> str:
    """Plot a Representational Dissimilarity Matrix.

    Args:
        rdm: RDM matrix [n_samples, n_samples]
        output_path: Path to save the figure
        layer_idx: Layer index for title
        title: Optional title

    Returns:
        Path to the saved figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(rdm, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Dissimilarity')

    ax.set_xlabel('Sample')
    ax.set_ylabel('Sample')
    ax.set_title(title or f'RDM - Layer {layer_idx}')

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_geometric_analysis(
    model_path: str,
    output_dir: str | None = None,
    device: str = "cpu",
    n_samples: int = 100,
) -> GeometricAnalysis:
    """Run comprehensive geometric analysis on a saved MLP.

    Args:
        model_path: Path to saved MLP file
        output_dir: Directory for output files
        device: Device to run on
        n_samples: Number of samples for analysis

    Returns:
        GeometricAnalysis with all results
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = MLP.load_from_file(model_path, device=device)
    print(f"  Hidden sizes: {model.hidden_sizes}")
    print(f"  Gate names: {model.gate_names}")
    print(f"  Layer sizes: {model.layer_sizes}")

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.dirname(model_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate test data
    print(f"\nGenerating {n_samples} random samples...")
    x = torch.randn(n_samples, model.input_size, device=device)

    # Create layer names
    layer_names = [f"L{i}" for i in range(len(model.layer_sizes))]

    # 1. Compute RDMs
    print("\n" + "=" * 50)
    print("Computing Representational Dissimilarity Matrices")
    print("=" * 50)

    rdm_results = compute_all_rdms(model, x, device=device)

    for layer_idx, result in rdm_results.items():
        print(f"  Layer {layer_idx}: {result.n_features} features, {result.n_samples} samples")

        # Plot RDM
        if result.rdm is not None:
            path = plot_rdm(
                result.rdm,
                os.path.join(output_dir, f"rdm_layer{layer_idx}.png"),
                layer_idx,
            )
            print(f"    Saved: {path}")

    # 2. Compute RSA matrix
    print("\n" + "=" * 50)
    print("Computing RSA Similarity Matrix")
    print("=" * 50)

    if len(rdm_results) > 1:
        rsa_matrix = compute_rsa_matrix(rdm_results)
        print(f"  RSA matrix shape: {rsa_matrix.shape}")
        print(f"  RSA matrix:\n{np.array2string(rsa_matrix, precision=3)}")

        # Use only the layers that have RDMs
        rsa_layer_names = [layer_names[i] for i in sorted(rdm_results.keys())]

        path = plot_similarity_matrix(
            rsa_matrix,
            rsa_layer_names,
            os.path.join(output_dir, "rsa_matrix.png"),
            title="RSA Similarity Matrix",
            vmin=-1.0,
            vmax=1.0,
            cmap="RdBu_r",
        )
        print(f"  Saved: {path}")
    else:
        rsa_matrix = None
        print("  Not enough layers with valid RDMs")

    # 3. Compute CKA matrix
    print("\n" + "=" * 50)
    print("Computing CKA Similarity Matrix")
    print("=" * 50)

    cka_matrix = compare_layers_cka(model, x, kernel="linear", device=device)
    print(f"  CKA matrix shape: {cka_matrix.shape}")
    print(f"  CKA matrix:\n{np.array2string(cka_matrix, precision=3)}")

    path = plot_similarity_matrix(
        cka_matrix,
        layer_names,
        os.path.join(output_dir, "cka_matrix.png"),
        title="CKA Similarity Matrix (Linear Kernel)",
    )
    print(f"  Saved: {path}")

    # 4. Also compute RBF CKA for comparison
    print("\n" + "=" * 50)
    print("Computing RBF CKA Similarity Matrix")
    print("=" * 50)

    cka_rbf_matrix = compare_layers_cka(model, x, kernel="rbf", device=device)
    print(f"  RBF CKA matrix:\n{np.array2string(cka_rbf_matrix, precision=3)}")

    path = plot_similarity_matrix(
        cka_rbf_matrix,
        layer_names,
        os.path.join(output_dir, "cka_rbf_matrix.png"),
        title="CKA Similarity Matrix (RBF Kernel)",
    )
    print(f"  Saved: {path}")

    # 5. Detailed CKA results for adjacent layers
    print("\n" + "=" * 50)
    print("Adjacent Layer CKA Details")
    print("=" * 50)

    for i in range(len(model.layer_sizes) - 1):
        result = compute_cka_result(model, x, i, i+1, kernel="linear", device=device)
        print(f"  Layer {i} -> Layer {i+1}:")
        print(f"    CKA: {result.cka:.4f}")
        print(f"    HSIC(X,Y): {result.hsic_xy:.6f}")
        print(f"    HSIC(X,X): {result.hsic_xx:.6f}")
        print(f"    HSIC(Y,Y): {result.hsic_yy:.6f}")

    print("\n" + "=" * 50)
    print("Geometric analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 50)

    return GeometricAnalysis(
        rdm_results=rdm_results,
        rsa_matrix=rsa_matrix,
        cka_matrix=cka_matrix,
        layer_names=layer_names,
        n_layers=len(model.layer_sizes),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run geometric analysis on a saved MLP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="tmp/test_xor_and.pt",
        help="Path to saved MLP file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu/cuda)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for analysis",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    run_geometric_analysis(
        args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()

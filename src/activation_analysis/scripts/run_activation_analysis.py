"""Run activation analysis on a saved MLP.

Usage:
    python -m src.activation_analysis.scripts.run_activation_analysis [SAVED_MLP_PATH] [--output-dir OUTPUT_DIR] [--no-viz]

Example:
    python -m src.activation_analysis.scripts.run_activation_analysis tmp/test_xor_and.pt --output-dir tmp/activation_output
    python -m src.activation_analysis.scripts.run_activation_analysis tmp/test_xor_and.pt --no-viz
"""

import argparse
import os
import sys

import torch

from src.model import MLP
from src.activation_analysis.statistics import (
    compute_activation_statistics,
    compute_all_correlations,
)
from src.activation_analysis.clustering import (
    cluster_neurons_kmeans,
    cluster_neurons_hierarchical,
    find_optimal_clusters,
)
from src.activation_analysis.types import ActivationAnalysisResult
from src.activation_analysis.viz.plots import (
    plot_layer_statistics,
    plot_correlation_heatmap,
    plot_correlation_summary,
    plot_clustering_result,
    plot_silhouette_scores,
)


def run_activation_analysis(
    model_path: str,
    output_dir: str | None = None,
    device: str = "cpu",
    n_samples: int = 1000,
    n_clusters: int | None = None,
    show_viz: bool = True,
) -> ActivationAnalysisResult:
    """Run comprehensive activation analysis on a saved MLP.

    Args:
        model_path: Path to saved MLP file
        output_dir: Directory for output files
        device: Device to run on
        n_samples: Number of samples for analysis
        n_clusters: Number of clusters (if None, auto-detect)
        show_viz: Whether to generate visualizations

    Returns:
        ActivationAnalysisResult with all analysis results
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = MLP.load_from_file(model_path, device=device)
    print(f"  Hidden sizes: {model.hidden_sizes}")
    print(f"  Gate names: {model.gate_names}")
    print(f"  Layer sizes: {model.layer_sizes}")

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join("tmp", "activation_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate random input samples
    print(f"\nGenerating {n_samples} random samples...")
    x = torch.randint(0, 2, (n_samples, model.input_size), dtype=torch.float32, device=device)

    result = ActivationAnalysisResult(model_path=model_path)

    # 1. Activation Statistics
    print("\n" + "=" * 50)
    print("1. Activation Statistics")
    print("=" * 50)

    statistics = compute_activation_statistics(model, x, device=device)
    result.statistics = statistics
    print(statistics.summary())

    if show_viz:
        path = plot_layer_statistics(
            statistics,
            os.path.join(output_dir, "activation_statistics.png"),
            title="Activation Statistics",
        )
        print(f"Saved: {path}")

    # 2. Correlation Analysis
    print("\n" + "=" * 50)
    print("2. Correlation Analysis")
    print("=" * 50)

    correlations = compute_all_correlations(model, x, device=device)
    result.correlations = correlations

    for layer_idx, corr in correlations.items():
        print(corr.summary())

        if show_viz and corr.correlation_matrix is not None:
            if corr.correlation_matrix.shape[0] > 1:
                path = plot_correlation_heatmap(
                    corr,
                    os.path.join(output_dir, f"correlation_layer{layer_idx}.png"),
                )
                print(f"  Saved: {path}")

    if show_viz and len(correlations) > 1:
        path = plot_correlation_summary(
            correlations,
            os.path.join(output_dir, "correlation_summary.png"),
        )
        print(f"Saved summary: {path}")

    # 3. Clustering Analysis
    print("\n" + "=" * 50)
    print("3. Clustering Analysis")
    print("=" * 50)

    for layer_idx in range(1, len(model.layer_sizes) - 1):  # Hidden layers only
        n_neurons = model.layer_sizes[layer_idx]
        print(f"\nLayer {layer_idx} ({n_neurons} neurons):")

        if n_neurons < 2:
            print("  Skipping: not enough neurons for clustering")
            continue

        # Find optimal number of clusters
        if n_clusters is None:
            print("  Finding optimal number of clusters...")
            optimal_k, scores = find_optimal_clusters(model, x, layer_idx, device=device)
            print(f"  Optimal k: {optimal_k}")

            if show_viz and len(scores) > 1:
                path = plot_silhouette_scores(
                    scores,
                    os.path.join(output_dir, f"silhouette_layer{layer_idx}.png"),
                    title=f"Silhouette Scores - Layer {layer_idx}",
                )
                print(f"  Saved: {path}")

            k_to_use = optimal_k
        else:
            k_to_use = min(n_clusters, n_neurons - 1)
            if k_to_use < 2:
                k_to_use = 2

        # K-means clustering
        print(f"  Running K-means (k={k_to_use})...")
        kmeans_result = cluster_neurons_kmeans(
            model, x, layer_idx, n_clusters=k_to_use, device=device
        )
        print(f"  {kmeans_result.summary()}")
        result.clustering[layer_idx] = kmeans_result

        if show_viz:
            path = plot_clustering_result(
                kmeans_result,
                os.path.join(output_dir, f"clustering_layer{layer_idx}.png"),
                title=f"Neuron Clustering - Layer {layer_idx}",
            )
            print(f"  Saved: {path}")

        # Hierarchical clustering
        print(f"  Running hierarchical clustering...")
        hier_result = cluster_neurons_hierarchical(
            model, x, layer_idx, n_clusters=k_to_use, device=device
        )
        print(f"  Hierarchical: {hier_result.summary()}")

    # Summary
    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print("=" * 50)
    print(f"Results saved to: {output_dir}")
    print("\nSummary:")
    print(result.summary())

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run activation analysis on a saved MLP",
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
        help="Output directory for results (default: tmp/activation_output)",
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
        default=1000,
        help="Number of samples for analysis",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters (default: auto-detect)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    run_activation_analysis(
        args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        n_samples=args.n_samples,
        n_clusters=args.n_clusters,
        show_viz=not args.no_viz,
    )


if __name__ == "__main__":
    main()

"""Run SPD analysis on a saved MLP model.

This script provides a command-line interface for running SPD (Stochastic
Parameter Decomposition) analysis on a trained MLP model. It decomposes
the model weights into interpretable components and identifies functional
subcircuits.

Usage:
    python -m src.spd.scripts.run_spd MODEL_PATH [OPTIONS]

Examples:
    # Basic usage
    python -m src.spd.scripts.run_spd tmp/test_xor_and.pt

    # Skip visualizations
    python -m src.spd.scripts.run_spd tmp/test_xor_and.pt --no-viz

    # Custom configuration
    python -m src.spd.scripts.run_spd tmp/model.pt --n-components 30 --steps 1000

    # Save results to specific directory
    python -m src.spd.scripts.run_spd tmp/model.pt --output-dir results/spd_analysis
"""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Run SPD analysis on a saved MLP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.spd.scripts.run_spd tmp/test_xor_and.pt
  python -m src.spd.scripts.run_spd tmp/model.pt --no-viz
  python -m src.spd.scripts.run_spd tmp/model.pt --n-components 30 --steps 1000
        """,
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved MLP (.pt file)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization (useful for headless servers)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tmp/spd",
        help="Output directory for results (default: tmp/spd)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=20,
        help="Number of SPD components per layer (default: 20)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of SPD training steps (default: 500)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (default: cpu)",
    )
    parser.add_argument(
        "--importance-coeff",
        type=float,
        default=1e-3,
        help="Sparsity coefficient for importance loss (default: 1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for SPD training (default: 4096)",
    )
    args = parser.parse_args()

    # Check model path exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\n{'='*60}")
    print("SPD Analysis")
    print(f"{'='*60}")
    print(f"\nLoading model from: {model_path}")

    from src.model import MLP

    try:
        model = MLP.load_from_file(model_path, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"  Architecture: {model.input_size} -> {model.hidden_sizes} -> {model.output_size}")
    print(f"  Activation: {model.activation}")
    if model.gate_names:
        print(f"  Gates: {model.gate_names}")

    # Generate training data using all 4 canonical boolean inputs
    print("\nGenerating training data...")
    n_inputs = model.input_size
    n_total_inputs = 2**n_inputs

    # Create all binary input combinations
    all_inputs = torch.zeros(n_total_inputs, n_inputs, device=args.device)
    for i in range(n_total_inputs):
        for j in range(n_inputs):
            all_inputs[i, j] = (i >> j) & 1

    # Repeat inputs for batch diversity during SPD training
    n_repeats = max(1, args.batch_size // n_total_inputs)
    x = all_inputs.repeat(n_repeats, 1)

    # Get model outputs
    with torch.no_grad():
        y = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")

    # Configure SPD
    from src.spd.types import SPDConfig

    spd_config = SPDConfig(
        n_components=args.n_components,
        steps=args.steps,
        batch_size=args.batch_size,
        importance_coeff=args.importance_coeff,
    )

    print(f"\nSPD Configuration:")
    print(f"  Components per layer: {spd_config.n_components}")
    print(f"  Training steps: {spd_config.steps}")
    print(f"  Batch size: {spd_config.batch_size}")
    print(f"  Importance coefficient: {spd_config.importance_coeff}")

    # Run SPD decomposition
    print(f"\nRunning SPD decomposition...")
    from src.spd.decomposition import decompose_mlp

    decomposed = decompose_mlp(x, y, model, args.device, spd_config)

    n_components = decomposed.get_n_components()
    print(f"  Total components: {n_components}")

    # Run analysis
    print(f"\nAnalyzing decomposition...")
    from src.spd.analysis import run_spd_analysis

    analysis = run_spd_analysis(
        decomposed_model=decomposed,
        target_model=model,
        n_inputs=n_inputs,
        gate_names=model.gate_names,
        device=args.device,
    )

    # Print results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")

    print(f"\nValidation Metrics:")
    print(f"  MMCS (directional alignment): {analysis.mmcs:.4f}")
    print(f"  ML2R (magnitude ratio): {analysis.ml2r:.4f}")
    print(f"  Faithfulness loss: {analysis.faithfulness_loss:.6f}")

    print(f"\nComponent Health:")
    print(f"  Alive components: {analysis.n_alive_components}")
    print(f"  Dead components: {analysis.n_dead_components}")
    if analysis.dead_component_labels:
        print(f"  Dead: {', '.join(analysis.dead_component_labels[:5])}")
        if len(analysis.dead_component_labels) > 5:
            print(f"         ... and {len(analysis.dead_component_labels) - 5} more")

    print(f"\nClustering:")
    print(f"  Number of clusters: {analysis.n_clusters}")

    if analysis.clusters:
        print(f"\nCluster Details:")
        for cluster in analysis.clusters:
            n_components = len(cluster.component_indices)
            func = cluster.function_mapping or "UNKNOWN"
            print(f"  Cluster {cluster.cluster_idx}: {n_components} components -> {func}")
            if cluster.component_labels:
                labels = cluster.component_labels[:3]
                suffix = f" ... +{n_components - 3} more" if n_components > 3 else ""
                print(f"    Components: {', '.join(labels)}{suffix}")

    # Show importance matrix summary
    if analysis.importance_matrix is not None:
        imp = analysis.importance_matrix
        print(f"\nImportance Matrix [{imp.shape[0]} inputs x {imp.shape[1]} components]:")
        print(f"  Mean: {imp.mean():.3f}, Std: {imp.std():.3f}")
        print(f"  Min: {imp.min():.3f}, Max: {imp.max():.3f}")

        # Show per-input activation patterns
        input_labels = [f"({(i>>0)&1},{(i>>1)&1})" for i in range(min(4, imp.shape[0]))]
        print(f"\n  Per-input mean CI:")
        for i, label in enumerate(input_labels):
            active_count = (imp[i] > 0.5).sum()
            print(f"    {label}: mean={imp[i].mean():.3f}, active={active_count}/{imp.shape[1]}")

    # Visualizations
    if not args.no_viz:
        print(f"\nGenerating visualizations...")
        try:
            from src.spd.runner import analyze_and_visualize_spd

            viz_result = analyze_and_visualize_spd(
                decomposed_model=decomposed,
                target_model=model,
                output_dir=output_dir,
                gate_names=model.gate_names,
                n_inputs=n_inputs,
                device=args.device,
            )
            print(f"  Saved to: {output_dir}/")
            for name, path in viz_result.visualization_paths.items():
                print(f"    - {path}")
        except Exception as e:
            print(f"  Warning: Visualization failed: {e}")
    else:
        print(f"\nSkipping visualizations (--no-viz)")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

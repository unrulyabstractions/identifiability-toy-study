"""Run probing analysis on a saved MLP.

Usage:
    python -m src.probing.scripts.run_probing [SAVED_MLP_PATH] [--output-dir OUTPUT_DIR] [--no-viz]

Example:
    python -m src.probing.scripts.run_probing tmp/test_xor_and.pt --output-dir tmp/probing_output
    python -m src.probing.scripts.run_probing tmp/test_xor_and.pt --no-viz
"""

import argparse
import os
import sys

from src.model import MLP
from src.probing.analysis import (
    probe_for_gate,
    probe_for_input,
)
from src.probing.viz.plots import (
    plot_probe_accuracy,
    plot_probe_comparison,
    plot_feature_importance,
    plot_layer_accuracy_heatmap,
)


def run_probing_analysis(
    model_path: str,
    output_dir: str | None = None,
    device: str = "cpu",
    n_samples: int = 1000,
    n_epochs: int = 100,
    show_viz: bool = True,
) -> dict:
    """Run comprehensive probing analysis on a saved MLP.

    Args:
        model_path: Path to saved MLP file
        output_dir: Directory for output files
        device: Device to run on
        n_samples: Number of samples for probing
        n_epochs: Number of training epochs
        show_viz: Whether to generate visualizations

    Returns:
        Dictionary with all probing results
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = MLP.load_from_file(model_path, device=device)
    print(f"  Hidden sizes: {model.hidden_sizes}")
    print(f"  Gate names: {model.gate_names}")
    print(f"  Layer sizes: {model.layer_sizes}")

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join("tmp", "probing_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    results = {}

    # Probe for each output gate
    print("\n" + "=" * 50)
    print("Probing for Output Gates")
    print("=" * 50)

    gate_analyses = []
    n_gates = model.output_size
    gate_names = model.gate_names or [f"Gate_{i}" for i in range(n_gates)]

    for gate_idx in range(n_gates):
        gate_name = gate_names[gate_idx] if gate_idx < len(gate_names) else f"Gate_{gate_idx}"
        print(f"\nProbing for {gate_name}...")

        analysis = probe_for_gate(
            model=model,
            gate_idx=gate_idx,
            n_samples=n_samples,
            n_epochs=n_epochs,
            device=device,
        )

        print(analysis.summary())
        gate_analyses.append(analysis)
        results[f"gate_{gate_name}"] = analysis

        # Plot accuracy
        if show_viz:
            path = plot_probe_accuracy(
                analysis,
                os.path.join(output_dir, f"probe_accuracy_{gate_name}.png"),
            )
            print(f"  Saved: {path}")

            # Plot feature importance for best layer
            if analysis.best_layer in analysis.probe_results:
                best_result = analysis.probe_results[analysis.best_layer]
                path = plot_feature_importance(
                    best_result,
                    os.path.join(output_dir, f"feature_importance_{gate_name}_layer{analysis.best_layer}.png"),
                )
                print(f"  Saved: {path}")

    # Compare gate probes
    if show_viz and len(gate_analyses) > 1:
        path = plot_probe_comparison(
            gate_analyses,
            os.path.join(output_dir, "probe_comparison_gates.png"),
            title="Probe Accuracy: Output Gates",
        )
        print(f"\nSaved comparison: {path}")

    # Probe for each input feature
    print("\n" + "=" * 50)
    print("Probing for Input Features")
    print("=" * 50)

    input_analyses = []
    for input_idx in range(model.input_size):
        print(f"\nProbing for Input_{input_idx}...")

        analysis = probe_for_input(
            model=model,
            input_idx=input_idx,
            n_samples=n_samples,
            n_epochs=n_epochs,
            device=device,
        )

        print(analysis.summary())
        input_analyses.append(analysis)
        results[f"input_{input_idx}"] = analysis

        # Plot accuracy
        if show_viz:
            path = plot_probe_accuracy(
                analysis,
                os.path.join(output_dir, f"probe_accuracy_input{input_idx}.png"),
            )
            print(f"  Saved: {path}")

    # Compare input probes
    if show_viz and len(input_analyses) > 1:
        path = plot_probe_comparison(
            input_analyses,
            os.path.join(output_dir, "probe_comparison_inputs.png"),
            title="Probe Accuracy: Input Features",
        )
        print(f"\nSaved comparison: {path}")

    # Combined heatmap
    if show_viz:
        all_analyses = gate_analyses + input_analyses
        if len(all_analyses) > 1:
            path = plot_layer_accuracy_heatmap(
                all_analyses,
                os.path.join(output_dir, "probe_accuracy_heatmap.png"),
                title="Probe Accuracy by Target and Layer",
            )
            print(f"\nSaved heatmap: {path}")

    print("\n" + "=" * 50)
    print("Probing analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 50)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run probing analysis on a saved MLP",
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
        help="Output directory for results (default: tmp/probing_output)",
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
        help="Number of samples for probing",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
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

    run_probing_analysis(
        args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        n_samples=args.n_samples,
        n_epochs=args.n_epochs,
        show_viz=not args.no_viz,
    )


if __name__ == "__main__":
    main()

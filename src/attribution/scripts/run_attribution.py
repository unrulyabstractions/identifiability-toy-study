"""Run attribution analysis on a saved MLP.

Usage:
    python -m src.attribution.scripts.run_attribution [SAVED_MLP_PATH] [--output-dir OUTPUT_DIR]

Example:
    python -m src.attribution.scripts.run_attribution tmp/test_xor_and.pt --output-dir tmp/attribution_output
"""

import argparse
import os
import sys

import torch
import numpy as np

from src.model import MLP
from src.attribution.patching import compute_activation_patching, compute_mean_ablation
from src.attribution.eap import compute_eap, compute_direct_eap
from src.attribution.eap_ig import compute_eap_ig_fast
from src.attribution.input_attribution import (
    compute_gradient_attribution,
    compute_gradient_x_input,
    compute_integrated_gradients_fast,
)
from src.attribution.viz.decision_boundary import (
    plot_decision_boundary_predictions,
    plot_decision_boundary_logits,
    plot_decision_boundary_comparison,
)
from src.attribution.viz.heatmaps import (
    plot_layer_attribution_heatmap,
    plot_input_attribution,
)


def run_attribution_analysis(
    model_path: str,
    output_dir: str | None = None,
    device: str = "cpu",
    n_samples: int = 100,
) -> dict:
    """Run comprehensive attribution analysis on a saved MLP.

    Args:
        model_path: Path to saved MLP file
        output_dir: Directory for output files (default: model_path directory)
        device: Device to run on
        n_samples: Number of samples for analysis

    Returns:
        Dictionary with all attribution results
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

    # Clean and corrupted inputs for patching
    clean_input = torch.zeros(n_samples, model.input_size, device=device)
    corrupted_input = torch.randn(n_samples, model.input_size, device=device)

    results = {}

    # Run for each gate
    n_gates = model.output_size
    gate_names = model.gate_names or [f"Gate_{i}" for i in range(n_gates)]

    for gate_idx in range(n_gates):
        gate_name = gate_names[gate_idx] if gate_idx < len(gate_names) else f"Gate_{gate_idx}"
        print(f"\n{'='*50}")
        print(f"Analyzing {gate_name} (gate_idx={gate_idx})")
        print('='*50)

        gate_results = {}

        # 1. Decision Boundary Visualization
        print("\n1. Decision Boundary Visualization...")
        try:
            path = plot_decision_boundary_predictions(
                model,
                os.path.join(output_dir, f"decision_boundary_scatter_{gate_name}.png"),
                gate_idx=gate_idx,
                device=device,
            )
            print(f"   Saved: {path}")
            gate_results["decision_boundary_scatter"] = path

            path = plot_decision_boundary_logits(
                model,
                os.path.join(output_dir, f"decision_boundary_logits_{gate_name}.png"),
                gate_idx=gate_idx,
                device=device,
            )
            print(f"   Saved: {path}")
            gate_results["decision_boundary_logits"] = path

            path = plot_decision_boundary_comparison(
                model,
                os.path.join(output_dir, f"decision_boundary_comparison_{gate_name}.png"),
                gate_idx=gate_idx,
                device=device,
            )
            print(f"   Saved: {path}")
            gate_results["decision_boundary_comparison"] = path
        except Exception as e:
            print(f"   Error in decision boundary: {e}")

        # 2. Input Attribution
        print("\n2. Input Attribution...")
        try:
            grad_result = compute_gradient_attribution(model, x, gate_idx=gate_idx, device=device)
            print(f"   Gradient attribution mean: {grad_result.get_mean_attribution()}")
            gate_results["gradient"] = grad_result

            grad_x_input = compute_gradient_x_input(model, x, gate_idx=gate_idx, device=device)
            print(f"   Gradient*Input mean: {grad_x_input.get_mean_attribution()}")
            gate_results["gradient_x_input"] = grad_x_input

            ig_result = compute_integrated_gradients_fast(model, x, gate_idx=gate_idx, device=device)
            print(f"   Integrated Gradients mean: {ig_result.get_mean_attribution()}")
            gate_results["integrated_gradients"] = ig_result

            # Plot input attribution
            path = plot_input_attribution(
                ig_result,
                os.path.join(output_dir, f"input_attribution_{gate_name}.png"),
                title=f"Input Attribution - {gate_name}",
            )
            print(f"   Saved: {path}")
        except Exception as e:
            print(f"   Error in input attribution: {e}")

        # 3. Activation Patching
        print("\n3. Activation Patching...")
        try:
            patching_result = compute_activation_patching(
                model, clean_input, corrupted_input, gate_idx=gate_idx, device=device
            )
            print(f"   Total effect: {patching_result.total_effect:.4f}")
            for layer_idx, scores in patching_result.layer_attributions.items():
                print(f"   Layer {layer_idx}: {scores}")
            gate_results["activation_patching"] = patching_result

            # Plot
            if patching_result.layer_attributions:
                path = plot_layer_attribution_heatmap(
                    patching_result,
                    os.path.join(output_dir, f"patching_heatmap_{gate_name}.png"),
                    title=f"Activation Patching - {gate_name}",
                )
                print(f"   Saved: {path}")
        except Exception as e:
            print(f"   Error in activation patching: {e}")

        # 4. Mean Ablation
        print("\n4. Mean Ablation...")
        try:
            ablation_result = compute_mean_ablation(model, x, gate_idx=gate_idx, device=device)
            for layer_idx, scores in ablation_result.layer_attributions.items():
                print(f"   Layer {layer_idx}: {scores}")
            gate_results["mean_ablation"] = ablation_result

            # Plot
            if ablation_result.layer_attributions:
                path = plot_layer_attribution_heatmap(
                    ablation_result,
                    os.path.join(output_dir, f"ablation_heatmap_{gate_name}.png"),
                    title=f"Mean Ablation - {gate_name}",
                )
                print(f"   Saved: {path}")
        except Exception as e:
            print(f"   Error in mean ablation: {e}")

        # 5. EAP
        print("\n5. Edge Attribution Patching...")
        try:
            eap_result = compute_direct_eap(model, x, gate_idx=gate_idx, device=device)
            top_edges = eap_result.get_top_edges(5)
            print(f"   Top 5 edges:")
            for edge, score in top_edges:
                print(f"     {edge}: {score:.4f}")
            gate_results["eap"] = eap_result

            # Plot
            if eap_result.layer_attributions:
                path = plot_layer_attribution_heatmap(
                    eap_result,
                    os.path.join(output_dir, f"eap_heatmap_{gate_name}.png"),
                    title=f"Edge Attribution - {gate_name}",
                )
                print(f"   Saved: {path}")
        except Exception as e:
            print(f"   Error in EAP: {e}")

        # 6. EAP with Integrated Gradients
        print("\n6. EAP with Integrated Gradients...")
        try:
            eap_ig_result = compute_eap_ig_fast(model, x, gate_idx=gate_idx, device=device)
            top_edges = eap_ig_result.get_top_edges(5)
            print(f"   Top 5 edges:")
            for edge, score in top_edges:
                print(f"     {edge}: {score:.4f}")
            gate_results["eap_ig"] = eap_ig_result
        except Exception as e:
            print(f"   Error in EAP-IG: {e}")

        results[gate_name] = gate_results

    print(f"\n{'='*50}")
    print("Attribution analysis complete!")
    print(f"Results saved to: {output_dir}")
    print('='*50)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run attribution analysis on a saved MLP",
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

    run_attribution_analysis(
        args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()

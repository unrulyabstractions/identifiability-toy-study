"""Run SPD analysis on a saved MLP model.

Usage:
    python -m src.spd.scripts.run_spd MODEL_PATH [OPTIONS]

Examples:
    python -m src.spd.scripts.run_spd tmp/model.pt
    python -m src.spd.scripts.run_spd tmp/model.pt --no-viz
    python -m src.spd.scripts.run_spd tmp/model.pt --n-components 30 --steps 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from src.infra import print_profile, profile_fn
from src.model import MLP
from src.spd.analysis import run_spd_analysis
from src.spd.decomposition import decompose_mlp
from src.spd.spd_executor import analyze_and_visualize_spd
from src.spd.types import SPDConfig, SPDAnalysisResult


def get_args() -> Any:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SPD analysis on a saved MLP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved MLP (.pt file)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation",
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
    return parser.parse_args()


def validate_model_path(model_path: str) -> Path:
    """Validate that model path exists."""
    path = Path(model_path)
    if not path.exists():
        print(f"Error: Model file not found: {path}")
        sys.exit(1)
    return path


@profile_fn("load_model")
def load_model(model_path: Path, device: str) -> MLP:
    """Load MLP model from file."""
    try:
        return MLP.load_from_file(model_path, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def generate_training_data(model: MLP, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate training data from all canonical boolean inputs."""
    n_inputs = model.input_size
    n_total_inputs = 2**n_inputs

    # Create all binary input combinations
    all_inputs = torch.zeros(n_total_inputs, n_inputs, device=device)
    for i in range(n_total_inputs):
        for j in range(n_inputs):
            all_inputs[i, j] = (i >> j) & 1

    # Repeat for batch diversity
    n_repeats = max(1, batch_size // n_total_inputs)
    x = all_inputs.repeat(n_repeats, 1)

    with torch.no_grad():
        y = model(x)

    return x, y


@profile_fn("spd_decomposition")
def run_decomposition(
    x: torch.Tensor,
    y: torch.Tensor,
    model: MLP,
    device: str,
    config: SPDConfig,
) -> Any:
    """Run SPD decomposition on model."""
    return decompose_mlp(x, y, model, device, config)


@profile_fn("spd_analysis")
def run_analysis(
    decomposed: Any,
    model: MLP,
    device: str,
) -> SPDAnalysisResult:
    """Analyze decomposed model."""
    return run_spd_analysis(
        decomposed_model=decomposed,
        target_model=model,
        n_inputs=model.input_size,
        gate_names=model.gate_names,
        device=device,
    )


@profile_fn("spd_visualization")
def run_visualization(
    decomposed: Any,
    model: MLP,
    output_dir: Path,
    device: str,
) -> dict[str, str]:
    """Generate visualizations."""
    try:
        viz_result = analyze_and_visualize_spd(
            decomposed_model=decomposed,
            target_model=model,
            output_dir=output_dir,
            gate_names=model.gate_names,
            n_inputs=model.input_size,
            device=device,
        )
        return viz_result.visualization_paths
    except Exception as e:
        print(f"  Warning: Visualization failed: {e}")
        return {}


def print_model_info(model: MLP) -> None:
    """Print model architecture information."""
    print(f"  Architecture: {model.input_size} -> {model.hidden_sizes} -> {model.output_size}")
    print(f"  Activation: {model.activation}")
    if model.gate_names:
        print(f"  Gates: {model.gate_names}")


def print_config(config: SPDConfig) -> None:
    """Print SPD configuration."""
    print(f"  Components/layer: {config.n_components}")
    print(f"  Training steps: {config.steps}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Importance coeff: {config.importance_coeff}")


def print_analysis_results(analysis: SPDAnalysisResult) -> None:
    """Print analysis results."""
    print(f"\n  Validation Metrics:")
    print(f"    MMCS (directional): {analysis.mmcs:.4f}")
    print(f"    ML2R (magnitude): {analysis.ml2r:.4f}")
    print(f"    Faithfulness loss: {analysis.faithfulness_loss:.6f}")

    print(f"\n  Component Health:")
    print(f"    Alive: {analysis.n_alive_components}")
    print(f"    Dead: {analysis.n_dead_components}")

    print(f"\n  Clustering:")
    print(f"    Clusters: {analysis.n_clusters}")

    if analysis.clusters:
        for cluster in analysis.clusters[:5]:
            n_comp = len(cluster.component_indices)
            func = cluster.function_mapping or "UNKNOWN"
            print(f"    Cluster {cluster.cluster_idx}: {n_comp} components -> {func}")

    if analysis.importance_matrix is not None:
        imp = analysis.importance_matrix
        print(f"\n  Importance Matrix [{imp.shape[0]}x{imp.shape[1]}]:")
        print(f"    Mean: {imp.mean():.3f}, Std: {imp.std():.3f}")


def main() -> None:
    args = get_args()

    # Validate inputs
    model_path = validate_model_path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Header
    print(f"\n{'='*60}")
    print("  SPD Analysis")
    print(f"{'='*60}")

    # Load model
    print(f"\nLoading model: {model_path}")
    model = load_model(model_path, args.device)
    print_model_info(model)

    # Generate data
    print(f"\nGenerating training data...")
    x, y = generate_training_data(model, args.batch_size, args.device)
    print(f"  Shape: {x.shape} -> {y.shape}")

    # Configure SPD
    config = SPDConfig(
        n_components=args.n_components,
        steps=args.steps,
        batch_size=args.batch_size,
        importance_coeff=args.importance_coeff,
    )
    print(f"\nSPD Config:")
    print_config(config)

    # Run decomposition
    print(f"\nRunning decomposition...")
    decomposed = run_decomposition(x, y, model, args.device, config)
    print(f"  Components: {decomposed.get_n_components()}")

    # Run analysis
    print(f"\nAnalyzing...")
    analysis = run_analysis(decomposed, model, args.device)
    print_analysis_results(analysis)

    # Visualizations
    if not args.no_viz:
        print(f"\nGenerating visualizations...")
        paths = run_visualization(decomposed, model, output_dir, args.device)
        if paths:
            print(f"  Output: {output_dir}/")
            for name, path in list(paths.items())[:5]:
                print(f"    - {Path(path).name}")
    else:
        print(f"\nSkipping visualizations (--no-viz)")

    # Footer
    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}")

    print_profile()


if __name__ == "__main__":
    main()

import argparse
import datetime
import json
import os
from pathlib import Path

from identifiability_toy_study.experiment import run_experiment
from identifiability_toy_study.mi_identifiability.logic_gates import ALL_LOGIC_GATES
from identifiability_toy_study.mi_identifiability.utils import setup_logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to store the data on. Use mps (Apple Silicon), cuda:0 (NVIDIA), or cpu",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train instead of loading mlp model",
    )

    parser.add_argument(
        "--n-samples-train",
        type=int,
        default=2048,
        help="Number of samples for training",
    )
    parser.add_argument(
        "--n-samples-val",
        type=int,
        default=128,
        help="Number of samples for validation",
    )
    parser.add_argument(
        "--n-samples-test", type=int, default=128, help="Number of samples for testing"
    )

    parser.add_argument(
        "--loss-target",
        type=float,
        nargs="+",
        default=[0.001],
        help="Target loss value for training",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.99,
        help="The accuracy threshold for circuit search",
    )
    parser.add_argument(
        "--val-frequency",
        type=int,
        default=1,
        help="Frequency of validation during training",
    )

    parser.add_argument(
        "--min-sparsity",
        type=float,
        default=0.0,
        help="Minimum sparsity for circuit search",
    )

    # Params for MLP architecture

    # Params for Training
    parser.add_argument("--seed", type=int, default=0, help="Starting random seed")
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        nargs="+",
        default=[0.001],
        help="Learning rate for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs for training"
    )

    # Params for Task Difficulty
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Standard deviation of the Gaussian noise",
    )
    parser.add_argument(
        "--skewed-distribution",
        action="store_true",
        help="Whether to use a skewed distribution for training data",
    )

    # Params for Multiple Trials
    parser.add_argument(
        "--n-experiments",
        type=int,
        default=1,
        help="Runs each experiment n times with different seeds",
    )
    parser.add_argument(
        "--n-gates",
        type=int,
        nargs="+",
        default=[1],
        help="Number of logic gates used as target functions when n-experiments > 1",
    )
    parser.add_argument(
        "--target-logic-gates",
        type=str,
        nargs="+",
        default=["XOR"],
        choices=ALL_LOGIC_GATES.keys(),
        help="The allowed target logic gates",
    )

    args = parser.parse_args()

    # We identify run by time
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Dir paths
    output_dir = Path("runs")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(model_dir, exist_ok=True)

    # File paths
    log_path = os.path.join(run_dir, "log.txt")
    results_path = os.path.join(run_dir, "results.json")

    # Run Experiment with many trials
    logger = setup_logging(log_path, args.debug)
    result = run_experiment(args, logger=logger, model_dir=model_dir)

    # Save results
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    logger.info(f"\n\n\n run_{timestamp} \n")
    logger.info(json.dumps(result, indent=4))

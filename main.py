import argparse
import datetime
import json
import os
from dataclasses import asdict
from pathlib import Path

from identifiability_toy_study.common.schemas import (
    ExperimentConfig,
    ExperimentResult,
)
from identifiability_toy_study.common.utils import setup_logging
from identifiability_toy_study.experiment import run_experiment

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

    # Logger
    logger = setup_logging(log_path, args.debug)

    # Run Experiment with many trials
    cfg = ExperimentConfig(
        from_scratch=args.from_scratch, model_dir=model_dir, debug=args.debug
    )
    result: ExperimentResult = run_experiment(cfg, logger=logger)

    # Save results
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=4, ensure_ascii=False)

    # Print in console
    logger.info(f"\n\n\n run_{timestamp} \n")
    logger.info(json.dumps(asdict(result), indent=4))

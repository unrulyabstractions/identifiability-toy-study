# data_gen.py
"""Data generation utilities for training."""

import random

from src.domain import (
    ALL_LOGIC_GATES,
    generate_noisy_multi_gate_data,
)
from src.schemas import (
    DataParams,
    Dataset,
    TrialData,
)


def generate_dataset(
    gate_names: list[str],
    device: str,
    n_repeats: int = 1,
    noise_std: float = 0.0,
    weights=None,
) -> Dataset:
    """
    Generate a dataset for the given logic gates.

    Args:
        gate_names: List of gate names to include
        device: Device to create tensors on
        n_repeats: Number of times to repeat the base data
        noise_std: Standard deviation of noise to add
        weights: Optional weights for skewed distribution

    Returns:
        Dataset with x and y tensors
    """
    gates = [ALL_LOGIC_GATES[gate_name] for gate_name in gate_names]
    n_gates = len(gates)
    if n_gates > 1:
        x, y = generate_noisy_multi_gate_data(
            gates,
            n_repeats=n_repeats,
            noise_std=noise_std,
            device=device,
        )
    else:
        x, y = gates[0].generate_noisy_data(
            n_repeats=n_repeats,
            weights=weights,
            noise_std=noise_std,
            device=device,
        )
    return Dataset(x=x, y=y)


def generate_trial_data(
    data_params: DataParams,
    logic_gates: list[str],
    device: str,
    logger=None,
    debug: bool = False,
) -> TrialData:
    """
    Generate train, validation, and test datasets for a trial.

    Args:
        data_params: Configuration for data generation
        logic_gates: List of gate names to generate data for
        device: Device to create tensors on
        logger: Optional logger for debug output
        debug: Whether to log debug information

    Returns:
        TrialData containing train, val, and test datasets
    """
    weights = None
    if data_params.skewed_distribution:
        n_inputs = 2  # We assume all have same n_inputs
        logger.info("using skewed_distribution")
        weights = [random.random() for _ in range(2**n_inputs)]

    train_data = generate_dataset(
        logic_gates,
        n_repeats=data_params.n_samples_train,
        weights=weights,
        noise_std=data_params.noise_std,
        device=device,
    )
    val_data = generate_dataset(
        logic_gates,
        n_repeats=data_params.n_samples_val,
        weights=weights,
        noise_std=data_params.noise_std,
        device=device,
    )
    test_data = generate_dataset(
        logic_gates,
        n_repeats=data_params.n_samples_test,
        weights=weights,
        noise_std=data_params.noise_std,
        device=device,
    )

    if logger is not None and debug:
        logger.info(
            f"checking_device: x_train: {train_data.x.device}, y_train: {train_data.y.device}"
        )
        logger.info(
            f"checking_device: x_val: {val_data.x.device}, y_val: {val_data.y.device}"
        )
        logger.info(
            f"checking_device: x_test: {test_data.x.device}, y_test: {test_data.y.device}"
        )

    return TrialData(
        train=train_data,
        val=val_data,
        test=test_data,
    )

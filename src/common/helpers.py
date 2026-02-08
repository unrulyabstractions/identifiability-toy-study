import random
import time
from typing import Optional

import numpy as np
import torch

from .logic_gates import (
    ALL_LOGIC_GATES,
    generate_noisy_multi_gate_data,
)
from .neural_model import MLP
from .profiler import profile_fn
from .schemas import (
    DataParams,
    Dataset,
    ModelParams,
    ProfilingEvent,
    TrainParams,
    TrialData,
    TrialResult,
)


@torch.no_grad()
def calculate_match_rate(y_pred, y_gt):
    """Compare two binary (0/1) tensors. Callers must threshold first."""
    y_pred = y_pred.reshape(-1)
    y_gt = y_gt.reshape(-1)
    return y_pred.eq(y_gt).float().mean()


@torch.no_grad()
def calculate_match_rate_batched(
    y_preds: torch.Tensor, y_gt: torch.Tensor
) -> np.ndarray:
    """y_preds has leading batch dim, y_gt does not."""
    return y_preds.eq(y_gt.unsqueeze(0)).float().mean(dim=(1, 2)).cpu().numpy()


@torch.no_grad()
def logits_to_binary(y: torch.Tensor) -> torch.Tensor:
    """Convert raw logits to binary predictions. Decision boundary at 0."""
    return (y > 0).float()


@torch.no_grad()
def calculate_mse(y_target: torch.Tensor, y_proxy: torch.Tensor) -> torch.Tensor:
    """Calculate mean squared error between two tensors."""
    return ((y_target - y_proxy) ** 2).mean()


@torch.no_grad()
def calculate_logit_similarity(
    y_target: torch.Tensor, y_proxy: torch.Tensor
) -> torch.Tensor:
    """R²-like similarity: 1.0 = perfect match, 0.0 = predicting the mean."""
    mse = calculate_mse(y_target, y_proxy)
    var = y_target.var().clamp(min=1e-8)
    return 1 - mse / var


@torch.no_grad()
def calculate_logit_similarity_batched(
    y_target: torch.Tensor, y_proxies: torch.Tensor
) -> np.ndarray:
    """Batched R²: y_proxies has leading batch dim, y_target does not."""
    mse = ((y_proxies - y_target.unsqueeze(0)) ** 2).mean(dim=(1, 2))
    var = y_target.var().clamp(min=1e-8)
    return (1 - mse / var).detach().cpu().numpy()


@torch.no_grad()
def calculate_best_match_rate(y_target, y_proxy):
    """Calculate match rate from raw logits."""
    return calculate_match_rate(logits_to_binary(y_target), logits_to_binary(y_proxy))


def update_status_fx(trial_result: TrialResult, logger=None, device: str = "cpu"):
    """Create a status updater that tracks timing in trial_result.profiling."""
    start_time_ms = time.time() * 1000
    last_time_ms = start_time_ms

    # Initialize profiling
    trial_result.profiling.device = device
    trial_result.profiling.start_time_ms = start_time_ms

    def update_status(status: str, mssg: Optional[str] = None):
        nonlocal last_time_ms

        trial_result.status = status

        # Record timing
        current_time_ms = time.time() * 1000
        timestamp_ms = current_time_ms - start_time_ms
        elapsed_ms = current_time_ms - last_time_ms
        last_time_ms = current_time_ms

        # Add event
        event = ProfilingEvent(
            status=status,
            timestamp_ms=round(timestamp_ms, 2),
            elapsed_ms=round(elapsed_ms, 2),
        )
        trial_result.profiling.events.append(event)

        # Update total duration
        trial_result.profiling.total_duration_ms = round(timestamp_ms, 2)

        # Aggregate phase durations for STARTED_*/ENDED_* or STARTED_*/FINISHED_* pairs
        if status.startswith("ENDED_") or status.startswith("FINISHED_"):
            # Find matching start event
            phase_prefix = status.split("_", 1)[1]  # e.g., "MLP_TRAINING" or "GATE:0"
            for prev_event in reversed(trial_result.profiling.events[:-1]):
                if (
                    prev_event.status.startswith("STARTED_")
                    and phase_prefix in prev_event.status
                ):
                    phase_name = phase_prefix
                    phase_duration = timestamp_ms - prev_event.timestamp_ms
                    trial_result.profiling.phase_durations_ms[phase_name] = round(
                        phase_duration, 2
                    )
                    break

        if logger:
            if mssg:
                logger.info(
                    f" trial_id:{trial_result.trial_id} trial_status:{status} "
                    f"[{elapsed_ms:.0f}ms] mssg:{mssg}"
                )
            else:
                logger.info(
                    f" trial_id:{trial_result.trial_id} trial_status:{status} "
                    f"[{elapsed_ms:.0f}ms]"
                )

    return update_status


def generate_dataset(
    gate_names: list[str],
    device: str,
    n_repeats: int = 1,
    noise_std: float = 0.0,
    weights=None,
) -> Dataset:
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


@profile_fn("Train Model")
def train_model(
    train_params: TrainParams,
    model_params: ModelParams,
    data: TrialData,
    device: str,
    logger=None,
    debug: bool = False,
    input_size: int = 2,
    output_size: int = 1,
):
    model = MLP(
        hidden_sizes=([model_params.width] * model_params.depth),
        input_size=input_size,
        output_size=output_size,
        device=device,
        debug=debug,
        logger=logger,
    )

    avg_loss = model.do_train(
        x=data.train.x,
        y=data.train.y,
        x_val=data.val.x,
        y_val=data.val.y,
        batch_size=train_params.batch_size,
        learning_rate=train_params.learning_rate,
        epochs=train_params.epochs,
        val_frequency=train_params.val_frequency,
        logger=logger,
    )

    val_acc = calculate_match_rate(
        logits_to_binary(model(data.val.x)), data.val.y
    ).item()
    return model, avg_loss, val_acc

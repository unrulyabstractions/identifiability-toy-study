# loop.py
"""Training functions for neural network models."""

import math

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from src.math import calculate_loss, calculate_match_rate, logits_to_binary
from src.training_analysis.types import StepRecord, TrainingRecord

# Convert nats (natural log) to bits (log base 2)
NATS_TO_BITS = 1.0 / math.log(2)


def train_mlp(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    loss_target: float = 0.032,  # default for bse (eq to 1e-3 mse target)
    val_frequency: int = 1,
    early_stopping_steps: int = 10,
    logger=None,
    record_training: bool = True,
) -> TrainingRecord:
    """
    Train an MLP model using the given data and hyperparameters.

    Args:
        model: The MLP model to train
        x: Training inputs [n_samples, input_size]
        y: Training targets [n_samples, output_size]
        x_val: Validation inputs
        y_val: Validation targets
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        epochs: Maximum number of epochs
        loss_target: Stop training if loss falls below this
        val_frequency: How often to log validation metrics
        early_stopping_steps: Stop if loss doesn't improve for this many epochs
        logger: Optional logger for progress output
        record_training: Whether to record step-by-step losses for epiplexity analysis

    Returns:
        TrainingRecord with training history and losses for epiplexity analysis
    """
    device = model.device

    # Create dataset with indices for tracking
    n_samples = len(x)
    indices = torch.arange(n_samples)
    dataset = TensorDataset(indices, x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function with SUM reduction for epiplexity (total bits per batch)
    epiplexity_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_loss = float("inf")
    bad_epochs = 0
    val_acc = 0.0
    avg_loss = 0.0

    # Phase 1: Record losses before each gradient update
    step_records: list[StepRecord] = []
    global_step = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = []

        for batch_indices, inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_indices_list = batch_indices.tolist()

            # Phase 1: Record loss BEFORE gradient update (for epiplexity)
            if record_training:
                model.eval()
                with torch.no_grad():
                    logits_pre = model(inputs)
                    loss_pre_nats = epiplexity_loss_fn(logits_pre, targets)
                    loss_pre_bits = loss_pre_nats.item() * NATS_TO_BITS

                step_records.append(
                    StepRecord(
                        step=global_step,
                        batch_indices=batch_indices_list,
                        loss_pre_bits=loss_pre_bits,
                    )
                )
                model.train()

            # Standard training step
            optimizer.zero_grad()
            logits = model(inputs)
            loss = calculate_loss(model, logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            global_step += 1

        avg_loss = float(np.mean(epoch_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        # Progress / validation
        if (epoch + 1) % val_frequency == 0 and logger is not None:
            model.eval()
            with torch.no_grad():
                train_logits = model(x.to(device))
                train_predictions = logits_to_binary(train_logits)
                train_acc = calculate_match_rate(y, train_predictions).item()

                val_logits = model(x_val.to(device))
                val_loss = calculate_loss(model, val_logits, y_val.to(device)).item()
                val_predictions = logits_to_binary(val_logits)
                val_acc = calculate_match_rate(y_val, val_predictions).item()

                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Train Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.4f}"
                )
                logger.info(
                    f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Bad Epochs: {bad_epochs}"
                )

        # Early stopping
        if avg_loss < loss_target or bad_epochs >= early_stopping_steps:
            break

    # Return TrainingRecord with Phase 1 data
    training_record = TrainingRecord(
        avg_loss=avg_loss,
        batch_size=batch_size,
        step_records=step_records,
    )
    return training_record

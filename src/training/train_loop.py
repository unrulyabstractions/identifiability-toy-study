# loop.py
"""Training functions for neural network models."""

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from src.math import calculate_loss, calculate_match_rate, logits_to_binary


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
) -> float:
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
        l1_lambda: L1 regularization coefficient
        logger: Optional logger for progress output

    Returns:
        Final average training loss
    """
    device = model.device

    # DataLoader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_loss = float("inf")
    bad_epochs = 0
    val_acc = 0.0

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            loss = calculate_loss(model, logits, targets)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

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

    return avg_loss

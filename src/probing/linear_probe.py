"""Linear probe training for interpretability.

Trains linear classifiers on hidden layer activations to test if
concepts are linearly decodable (a key indicator of interpretable
representations).
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .types import ProbeResult

if TYPE_CHECKING:
    from src.model import MLP


class LinearProbe(nn.Module):
    """A simple linear classifier for probing."""

    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        if n_classes == 2:
            # Binary classification: single output with sigmoid
            self.linear = nn.Linear(n_features, 1)
            self.n_classes = 2
        else:
            self.linear = nn.Linear(n_features, n_classes)
            self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_probe(
    model: "MLP",
    layer_idx: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor | None = None,
    y_test: torch.Tensor | None = None,
    n_epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    device: str = "cpu",
    target_name: str = "",
) -> ProbeResult:
    """Train a linear probe on hidden layer activations.

    Args:
        model: The MLP model to probe
        layer_idx: Which layer's activations to probe (0=input, 1=first hidden, etc.)
        x_train: Training inputs [n_samples, n_inputs]
        y_train: Training labels [n_samples] (binary or multi-class)
        x_test: Test inputs (optional, uses train if None)
        y_test: Test labels (optional, uses train if None)
        n_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization weight
        device: Device to run on
        target_name: Name of the target concept

    Returns:
        ProbeResult with trained probe weights and accuracy
    """
    model = model.to(device)
    model.eval()

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    if x_test is None:
        x_test = x_train
        y_test = y_train
    else:
        x_test = x_test.to(device)
        y_test = y_test.to(device)

    # Get activations at the target layer
    with torch.no_grad():
        train_acts = model(x_train, return_activations=True)
        test_acts = model(x_test, return_activations=True)

        h_train = train_acts[layer_idx]  # [n_train, n_features]
        h_test = test_acts[layer_idx]    # [n_test, n_features]

    n_features = h_train.shape[-1]
    n_classes = int(y_train.max().item()) + 1

    # Create and train probe
    probe = LinearProbe(n_features, n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = []

    for epoch in range(n_epochs):
        probe.train()
        optimizer.zero_grad()

        logits = probe(h_train)

        if n_classes == 2:
            # Binary classification
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1), y_train.float()
            )
        else:
            # Multi-class classification
            loss = F.cross_entropy(logits, y_train.long())

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    # Evaluate
    probe.eval()
    with torch.no_grad():
        train_logits = probe(h_train)
        test_logits = probe(h_test)

        if n_classes == 2:
            train_preds = (train_logits.squeeze(-1) > 0).float()
            test_preds = (test_logits.squeeze(-1) > 0).float()
        else:
            train_preds = train_logits.argmax(dim=-1)
            test_preds = test_logits.argmax(dim=-1)

        train_accuracy = (train_preds == y_train).float().mean().item()
        test_accuracy = (test_preds == y_test).float().mean().item()

    # Extract weights
    weights = probe.linear.weight.detach().cpu().numpy()
    bias = probe.linear.bias.detach().cpu().numpy()

    return ProbeResult(
        layer_idx=layer_idx,
        target_name=target_name,
        accuracy=test_accuracy,
        train_accuracy=train_accuracy,
        weights=weights,
        bias=bias,
        n_samples=len(x_train),
        n_features=n_features,
        loss_history=loss_history,
    )


def train_probes_all_layers(
    model: "MLP",
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor | None = None,
    y_test: torch.Tensor | None = None,
    n_epochs: int = 100,
    lr: float = 0.01,
    device: str = "cpu",
    target_name: str = "",
) -> dict[int, ProbeResult]:
    """Train probes on all layers of the model.

    Args:
        model: The MLP model to probe
        x_train: Training inputs
        y_train: Training labels
        x_test: Test inputs (optional)
        y_test: Test labels (optional)
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Device to run on
        target_name: Name of the target concept

    Returns:
        Dictionary mapping layer_idx to ProbeResult
    """
    results = {}

    # Probe each layer (including input and output)
    n_layers = len(model.layer_sizes)

    for layer_idx in range(n_layers):
        result = train_linear_probe(
            model=model,
            layer_idx=layer_idx,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            n_epochs=n_epochs,
            lr=lr,
            device=device,
            target_name=target_name,
        )
        results[layer_idx] = result

    return results

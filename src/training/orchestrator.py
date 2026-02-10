# orchestrator.py
"""High-level training orchestration."""

from src.infra import profile_fn
from src.model import MLP
from src.experiment_config import ModelParams, TrainParams
from src.schemas import TrialData
from src.math import (
    calculate_match_rate,
    logits_to_binary,
)

from .train_loop import train_mlp


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
    """
    Train an MLP model with the given parameters.

    Args:
        train_params: Training hyperparameters
        model_params: Model architecture parameters
        data: Training, validation, and test data
        device: Device to train on
        logger: Optional logger for output
        debug: Whether to enable debug mode
        input_size: Number of input features
        output_size: Number of output features

    Returns:
        Tuple of (model, avg_loss, val_acc)
    """
    model = MLP(
        hidden_sizes=([model_params.width] * model_params.depth),
        input_size=input_size,
        output_size=output_size,
        device=device,
        debug=debug,
        logger=logger,
        gate_names=list(model_params.logic_gates) if model_params.logic_gates else None,
    )

    avg_loss = train_mlp(
        model=model,
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
        data.val.y, logits_to_binary(model(data.val.x))
    ).item()
    return model, avg_loss, val_acc

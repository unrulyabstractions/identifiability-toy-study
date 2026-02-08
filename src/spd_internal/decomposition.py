"""
Parameter decomposition module for applying SPD to MLPs.

SPD (Stochastic Parameter Decomposition) decomposes model weights into
interpretable components using stochastic masking and causal importance functions.
"""

import tempfile
from pathlib import Path

import torch
from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.run_spd import expand_module_patterns, optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader

from ..common.neural_model import MLP, DecomposedMLP
from ..common.schemas import SPDConfig


class SimpleDataset:
    """Dataset wrapper for SPD training."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]

    def __len__(self):
        return 2**31  # Effectively infinite

    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate batch by random sampling."""
        indices = torch.randint(0, self.n_samples, (batch_size,), device=self.x.device)
        return self.x[indices], self.y[indices]


def decompose_mlp(
    x: torch.Tensor,
    y: torch.Tensor,
    target_model: MLP,
    device: str,
    spd_config: SPDConfig,
) -> DecomposedMLP:
    """
    Decompose an MLP using Stochastic Parameter Decomposition.

    Args:
        x: Input data tensor
        y: Target output tensor (from the full model)
        target_model: The MLP to decompose
        device: Device to run on
        spd_config: Configuration for SPD

    Returns:
        DecomposedMLP containing the trained decomposition
    """
    dataset = SimpleDataset(x, y)
    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=spd_config.batch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=spd_config.eval_batch_size, shuffle=False
    )

    # Build module info for each layer
    module_info = [
        {"module_pattern": f"layers.{i}.0", "C": spd_config.n_components}
        for i in range(target_model.num_layers)
    ]

    config = Config(
        seed=0,
        steps=spd_config.steps,
        batch_size=spd_config.batch_size,
        eval_batch_size=spd_config.eval_batch_size,
        n_eval_steps=spd_config.n_eval_steps,
        lr_schedule={"start_val": spd_config.learning_rate, "fn_type": "constant"},
        module_info=module_info,
        loss_metric_configs=[
            {
                "classname": "ImportanceMinimalityLoss",
                "coeff": spd_config.importance_coeff,
                "pnorm": spd_config.importance_p,
                "beta": 0.0,
            },
            {"classname": "StochasticReconLoss", "coeff": spd_config.recon_coeff},
        ],
        n_mask_samples=1,
        output_loss_type="mse",
        train_log_freq=100,
        eval_freq=500,
        slow_eval_freq=500,
        n_examples_until_dead=10000,
        pretrained_model_class="src.common.neural_model.MLP",
        task_config={
            "task_name": "tms",
            "feature_probability": spd_config.feature_probability,
            "data_generation_type": spd_config.data_generation_type,
        },
    )

    # Freeze target model (required by SPD)
    target_model.requires_grad_(False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        optimize(
            target_model=target_model,
            config=config,
            device=device,
            train_loader=train_loader,
            eval_loader=eval_loader,
            n_eval_steps=spd_config.n_eval_steps,
            out_dir=out_dir,
        )

        # Create ComponentModel and load trained weights
        module_path_info = expand_module_patterns(target_model, config.all_module_info)
        cm_config = {
            "ci_fn_type": config.ci_fn_type,
            "ci_fn_hidden_dims": config.ci_fn_hidden_dims,
            "sigmoid_type": config.sigmoid_type,
            "pretrained_model_output_attr": config.pretrained_model_output_attr,
        }
        component_model = ComponentModel(
            target_model=target_model,
            module_path_info=module_path_info,
            **cm_config,
        )
        component_model.to(device)

        checkpoint_path = out_dir / f"model_{spd_config.steps}.pth"
        assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        component_model.load_state_dict(state_dict)

    return DecomposedMLP(
        component_model=component_model,
        target_model=target_model,
        cm_config=cm_config,
    )

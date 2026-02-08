# decomposed.py
"""DecomposedMLP - wrapper for SPD decomposition results."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add submodule to path
_submodule_path = str(Path(__file__).parent.parent.parent / "submodules" / "spd")
if _submodule_path not in sys.path:
    sys.path.insert(0, _submodule_path)

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .mlp import MLP


class DecomposedMLP(nn.Module):
    """
    Wrapper holding the result of SPD decomposition on an MLP.
    Stores the component model and provides methods to inspect/visualize components.
    """

    # Default ComponentModel config (used if not provided)
    DEFAULT_CM_CONFIG = {
        "ci_fn_type": "vector_mlp",
        "ci_fn_hidden_dims": [8],
        "sigmoid_type": "leaky_hard",
        "pretrained_model_output_attr": None,
    }

    def __init__(
        self,
        component_model=None,
        target_model: "MLP" = None,
        cm_config: dict = None,
    ):
        super().__init__()
        self.component_model = component_model  # SPD ComponentModel after optimization
        self.target_model = target_model  # Original MLP that was decomposed
        self.cm_config = cm_config or self.DEFAULT_CM_CONFIG.copy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decomposed model."""
        assert self.component_model is not None, "No component model available"
        return self.component_model(x)

    def get_n_components(self) -> int:
        """Return number of components per layer."""
        assert self.component_model is not None
        first_component = next(iter(self.component_model.components.values()))
        return first_component.C

    def save(self, path: str):
        """Save decomposed model to file."""
        assert self.component_model is not None
        assert self.target_model is not None

        # Extract module_info for reconstruction
        module_info = [
            {"module_pattern": name, "C": comp.C}
            for name, comp in self.component_model.components.items()
        ]

        torch.save(
            {
                "component_model_state": self.component_model.state_dict(),
                "target_model_state": self.target_model.state_dict(),
                "module_info": module_info,
                "component_model_config": self.cm_config,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, target_model: "MLP", device: str = "cpu"):
        """Load decomposed model from file.

        Args:
            path: Path to saved decomposed model
            target_model: Pre-loaded target model (required)
            device: Device to load model to
        """
        from spd.configs import ModulePatternInfoConfig
        from spd.models.component_model import ComponentModel
        from spd.run_spd import expand_module_patterns

        data = torch.load(path, map_location=device, weights_only=False)

        # Load target model state
        target_model.load_state_dict(data["target_model_state"])
        target_model.to(device)
        target_model.requires_grad_(False)

        # Reconstruct component model
        module_info = [
            ModulePatternInfoConfig(module_pattern=m["module_pattern"], C=m["C"])
            for m in data["module_info"]
        ]
        module_path_info = expand_module_patterns(target_model, module_info)

        cm_config = data["component_model_config"]
        component_model = ComponentModel(
            target_model=target_model,
            module_path_info=module_path_info,
            ci_fn_type=cm_config["ci_fn_type"],
            ci_fn_hidden_dims=cm_config["ci_fn_hidden_dims"],
            sigmoid_type=cm_config["sigmoid_type"],
            pretrained_model_output_attr=cm_config["pretrained_model_output_attr"],
        )
        component_model.load_state_dict(data["component_model_state"])
        component_model.to(device)

        return cls(
            component_model=component_model,
            target_model=target_model,
            cm_config=cm_config,
        )

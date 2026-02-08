"""SPD configuration utilities.

Contains helper functions for SPD configuration that aren't schema classes.
"""

import copy
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.schemas import SPDConfig


def generate_spd_sweep_configs(
    base_config: "SPDConfig" = None,
    n_components_list: list[int] = None,
    importance_coeff_list: list[float] = None,
    steps_list: list[int] = None,
    importance_p_list: list[float] = None,
) -> list["SPDConfig"]:
    """
    Generate multiple SPD configs for parameter sweep.

    Args:
        base_config: Base config to modify (uses defaults if None)
        n_components_list: List of n_components values to try
        importance_coeff_list: List of importance_coeff values to try
        steps_list: List of steps values to try
        importance_p_list: List of importance_p (pnorm) values to try

    Returns:
        List of SPDConfig objects for sweep
    """
    # Import here to avoid circular imports
    from ..common.schemas import SPDConfig

    if base_config is None:
        base_config = SPDConfig()

    # Default sweep values if not specified
    n_components_list = n_components_list or [base_config.n_components]
    importance_coeff_list = importance_coeff_list or [base_config.importance_coeff]
    steps_list = steps_list or [base_config.steps]
    importance_p_list = importance_p_list or [base_config.importance_p]

    configs = []
    for n_comp, imp_coeff, steps, imp_p in product(
        n_components_list, importance_coeff_list, steps_list, importance_p_list
    ):
        cfg = copy.deepcopy(base_config)
        cfg.n_components = n_comp
        cfg.importance_coeff = imp_coeff
        cfg.steps = steps
        cfg.importance_p = imp_p
        configs.append(cfg)

    return configs

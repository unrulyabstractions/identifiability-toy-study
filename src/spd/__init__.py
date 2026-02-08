"""SPD (Stochastic Parameter Decomposition) module.

Main exports:
- run_spd: Run SPD analysis on an experiment result
- SPDConfig: Configuration for SPD decomposition
- SpdResults: Complete SPD results for an experiment
- SpdTrialResult: SPD results for a single trial
- save_spd_results / load_spd_results: Persistence functions
"""

from .spd_experiment import run_spd
from .spd_trial import run_spd_trial
from .spd_types import SPDConfig, SpdResults, SpdTrialResult
from .persistence import load_spd_results, save_spd_results

# Also export lower-level components for advanced use
from .decomposition import decompose_mlp
from .subcircuits import SPDSubcircuitEstimate, estimate_spd_subcircuits
from .analysis import run_spd_analysis, analyze_and_visualize_spd
from .persistence import save_spd_estimate, load_spd_estimate

__all__ = [
    # Main API
    "run_spd",
    "run_spd_trial",
    "SPDConfig",
    "SpdResults",
    "SpdTrialResult",
    "save_spd_results",
    "load_spd_results",
    # Lower-level API
    "decompose_mlp",
    "SPDSubcircuitEstimate",
    "estimate_spd_subcircuits",
    "run_spd_analysis",
    "analyze_and_visualize_spd",
    "save_spd_estimate",
    "load_spd_estimate",
]

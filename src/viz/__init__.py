"""Visualization package for circuits and SPD decompositions.

##############################################################################
#                                                                            #
#   THIS MODULE DOES NOT RUN ANY MODELS!                                     #
#   ALL DATA MUST BE PRE-COMPUTED IN trial.py OR causal_analysis.py          #
#                                                                            #
#   If you need activations, add them to the relevant schema class and       #
#   compute them during analysis, NOT during visualization.                  #
#                                                                            #
#   Running models here causes ~5000 forward passes and kills performance!   #
#                                                                            #
##############################################################################

Main entry point: visualize_experiment
"""

# Main entry point
from .experiment_viz import visualize_experiment

# Activation visualization
from .activation_viz import (
    visualize_circuit_activations_from_data,
    visualize_circuit_activations_mean,
)

# Constants and utilities
from .constants import (
    COLORS,
    MARKERS,
    JITTER,
    TITLE_Y,
    SUBTITLE_PAD,
    LAYOUT_RECT_DEFAULT,
    LAYOUT_RECT_WITH_LEGEND,
    finalize_figure,
    set_subplot_title,
    GraphLayoutCache,
    _activation_to_color,
    _text_color_for_background,
    _symmetric_range,
)

# Circuit drawing
from .circuit_drawing import (
    draw_intervened_circuit,
    _build_graph_fast,
    _draw_graph,
    _draw_graph_with_output_highlight,
    _draw_circuit_from_data,
    _get_spd_component_weights,
)

# Observational visualization
from .observational_viz import (
    _generate_robustness_circuit_figure,
    visualize_robustness_circuit_samples,
    visualize_robustness_curves,
)

# Faithfulness visualization
from .faithfulness_viz import (
    _patch_key_to_filename,
    _faithfulness_score_to_color,
    visualize_faithfulness_intervention_effects,
    _generate_faithfulness_circuit_figure,
    visualize_faithfulness_circuit_samples,
)

# SPD visualization
from .spd_viz import visualize_spd_components

# Profiling visualization
from .profiling_viz import (
    visualize_profiling_timeline,
    visualize_profiling_phases,
    visualize_profiling_summary,
)

# Metrics export (renamed from metrics_export)
from .export import (
    compute_observational_metrics,
    compute_interventional_metrics,
    compute_counterfactual_metrics,
    save_faithfulness_json,
)

__all__ = [
    # Main entry point
    "visualize_experiment",
    # Activation visualization
    "visualize_circuit_activations_from_data",
    "visualize_circuit_activations_mean",
    # Base utilities and constants
    "COLORS",
    "MARKERS",
    "JITTER",
    "TITLE_Y",
    "SUBTITLE_PAD",
    "LAYOUT_RECT_DEFAULT",
    "LAYOUT_RECT_WITH_LEGEND",
    "finalize_figure",
    "set_subplot_title",
    "GraphLayoutCache",
    "_activation_to_color",
    "_text_color_for_background",
    "_symmetric_range",
    # Circuit drawing
    "draw_intervened_circuit",
    "_build_graph_fast",
    "_draw_graph",
    "_draw_graph_with_output_highlight",
    "_draw_circuit_from_data",
    "_get_spd_component_weights",
    # Observational visualization
    "_generate_robustness_circuit_figure",
    "visualize_robustness_circuit_samples",
    "visualize_robustness_curves",
    # Faithfulness visualization
    "_patch_key_to_filename",
    "_faithfulness_score_to_color",
    "visualize_faithfulness_intervention_effects",
    "_generate_faithfulness_circuit_figure",
    "visualize_faithfulness_circuit_samples",
    # SPD visualization
    "visualize_spd_components",
    # Profiling visualization
    "visualize_profiling_timeline",
    "visualize_profiling_phases",
    "visualize_profiling_summary",
    # Metrics export
    "compute_observational_metrics",
    "compute_interventional_metrics",
    "compute_counterfactual_metrics",
    "save_faithfulness_json",
]

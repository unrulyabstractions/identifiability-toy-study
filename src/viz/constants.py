"""Base visualization utilities and constants.

Contains:
- Style constants (COLORS, MARKERS, JITTER)
- Layout constants (TITLE_Y, SUBTITLE_PAD, LAYOUT_RECT_*)
- Helper functions (finalize_figure, set_subplot_title)
- GraphLayoutCache for caching node positions
- Color mapping utilities (_activation_to_color, _text_color_for_background, _symmetric_range)
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# Configure matplotlib backend BEFORE importing pyplot (critical for batch rendering)
matplotlib.use("Agg")

# Global DPI configuration (reduced from 300 for smaller file sizes)
_VIZ_DPI = 150


def get_dpi() -> int:
    """Get the current DPI setting for visualizations."""
    return _VIZ_DPI


def set_dpi(dpi: int) -> None:
    """Set the DPI for visualizations."""
    global _VIZ_DPI
    _VIZ_DPI = dpi

# Configure matplotlib for performance
plt.rcParams["path.simplify"] = True
plt.rcParams["path.simplify_threshold"] = 1.0
plt.rcParams["agg.path.chunksize"] = 10000
plt.rcParams["text.usetex"] = False

# Global styling - clean academic aesthetic with monospace fonts
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["DejaVu Sans Mono", "Courier New", "monospace"]
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.titlepad"] = 18  # More spacing between title and plot
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["figure.titlesize"] = 13
plt.rcParams["figure.titleweight"] = "bold"
plt.rcParams["hatch.linewidth"] = 3.0  # Thick hatch lines for visibility at small sizes


# ------------------ CONSTANTS ------------------

# Plot styling
COLORS = {
    "gate": "steelblue",
    "subcircuit": "coral",
    "agreement": "purple",
    "mse": "teal",
    "correct": "green",
    "incorrect": "red",
    # Faithfulness-specific (pastel)
    "in_circuit": "#77DD77",  # Pastel green
    "out_circuit": "#DA70D6",  # Pastel orchid
    "faithfulness": "#DA70D6",  # Pastel orchid (purple)
    "counterfactual": "#EF6C00",  # Orange
}
MARKERS = {"gate": "^", "subcircuit": "v", "agreement": "o", "mse": "s"}
JITTER = {"correct": 1.05, "incorrect": -0.05, "gate_correct": 1.05, "sc_correct": 0.95}

# Global layout constants for consistent title positioning
TITLE_Y = 0.98  # Y position for suptitle (high enough to not be cut off)
SUBTITLE_PAD = 12  # Padding for subplot titles (ax.set_title pad parameter)
LAYOUT_RECT_DEFAULT = [0, 0.02, 1, 0.90]  # [left, bottom, right, top] - leaves room for title
LAYOUT_RECT_WITH_LEGEND = [0, 0.06, 1, 0.90]  # Extra bottom space for legend


def finalize_figure(fig, title: str, has_legend_below: bool = False, fontsize: int = 13):
    """Apply consistent layout and title positioning to a figure.

    Call this INSTEAD of manually calling tight_layout + suptitle.
    Ensures titles are never cut off across all visualizations.

    Args:
        fig: The matplotlib figure
        title: The title text
        has_legend_below: If True, leaves extra space at bottom for legend
        fontsize: Title font size (default 13)
    """
    rect = LAYOUT_RECT_WITH_LEGEND if has_legend_below else LAYOUT_RECT_DEFAULT
    plt.tight_layout(rect=rect)
    fig.suptitle(title, fontsize=fontsize, fontweight="bold", y=TITLE_Y)


def set_subplot_title(ax, title: str, fontsize: int = 11):
    """Set subplot title with consistent padding.

    Use this instead of ax.set_title() directly to ensure consistent spacing.
    """
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=SUBTITLE_PAD)


# ------------------ LAYOUT CACHE ------------------


class GraphLayoutCache:
    """Cache graph layouts by structure to avoid recomputation.

    Positions depend only on layer sizes, not activation values.
    Computing positions once and reusing saves significant time.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
        return cls._instance

    def get_positions(self, layer_sizes: tuple) -> dict:
        """Get or compute node positions for given layer structure.

        Uses same convention as Circuit.visualize(): node 0 at bottom, higher nodes above.
        """
        key = tuple(layer_sizes)
        if key not in self._cache:
            pos = {}
            max_width = max(layer_sizes)
            for layer_idx, n_nodes in enumerate(layer_sizes):
                y_offset = (max_width - n_nodes) / 2
                for node_idx in range(n_nodes):
                    name = f"({layer_idx},{node_idx})"
                    pos[name] = (layer_idx, y_offset + node_idx)
            self._cache[key] = pos
        return self._cache[key]

    def clear(self):
        self._cache.clear()


# Global singleton
_layout_cache = GraphLayoutCache()


# ------------------ HELPERS ------------------


def _activation_to_color(val: float, vmin: float = -2, vmax: float = 2) -> tuple:
    """
    Pastel color gradient for logit values (decision boundary at 0):
    - 0.0 = neutral cream/beige (decision boundary)
    - Positive (class 1) → cooler colors (mint, teal)
    - Negative (class 0) → warmer colors (peach, coral, rose)

    NOTE: This is the legacy function. For layer-aware coloring, use
    _input_node_color, _hidden_node_color, or _output_node_color.
    """
    # Pastel colors (RGB tuples, 0-1 range)
    colors = {
        "deep_rose": (0.85, 0.45, 0.55),      # < -2.0
        "coral": (0.95, 0.65, 0.60),           # -1.0
        "cream": (0.98, 0.95, 0.80),           # 0.0 (decision boundary)
        "mint": (0.75, 0.92, 0.78),            # 1.0
        "teal": (0.55, 0.82, 0.78),            # > 2.0
    }

    def lerp(c1, c2, t):
        return tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(3))

    if val <= -2.0:
        # Deep rose (saturated negative)
        rgb = colors["deep_rose"]
    elif val <= -1.0:
        # Deep rose to coral
        t = (val + 2.0) / 1.0  # -2 -> -1 maps to 0 -> 1
        rgb = lerp(colors["deep_rose"], colors["coral"], t)
    elif val <= 0.0:
        # Coral to cream (approaching decision boundary)
        t = (val + 1.0) / 1.0  # -1 -> 0 maps to 0 -> 1
        rgb = lerp(colors["coral"], colors["cream"], t)
    elif val <= 1.0:
        # Cream to mint (crossing into positive)
        t = val / 1.0  # 0 -> 1 maps to 0 -> 1
        rgb = lerp(colors["cream"], colors["mint"], t)
    elif val <= 2.0:
        # Mint to teal
        t = (val - 1.0) / 1.0  # 1 -> 2 maps to 0 -> 1
        rgb = lerp(colors["mint"], colors["teal"], t)
    else:
        # Teal (saturated positive)
        rgb = colors["teal"]

    return (*rgb, 1.0)


def _input_node_color(val: float) -> tuple:
    """
    Color for input layer nodes based on binary input value.
    - 0 → pastel red
    - 1 → pastel green
    """
    pastel_red = (0.95, 0.65, 0.65)    # #F2A6A6 - soft salmon/rose
    pastel_green = (0.65, 0.92, 0.65)  # #A6EBA6 - soft mint green

    # Threshold at 0.5 for binary classification
    if val < 0.5:
        return (*pastel_red, 1.0)
    else:
        return (*pastel_green, 1.0)


def _output_node_color(val: float) -> tuple:
    """
    Color for output layer nodes based on logit sign.
    - Negative → pastel red
    - Positive → pastel green
    """
    pastel_red = (0.95, 0.60, 0.60)    # Soft red/coral for negative
    pastel_green = (0.60, 0.90, 0.60)  # Soft green for positive

    if val < 0:
        return (*pastel_red, 1.0)
    else:
        return (*pastel_green, 1.0)


def _hidden_node_color(val: float) -> tuple:
    """
    Color for hidden layer nodes based on activation magnitude.

    5 distinct bands:
    - (-inf, -10]: deep purple (very negative)
    - (-10, 0]: light purple/lavender (negative)
    - (0, 1]: cream/yellow (near zero positive)
    - (1, 10]: light teal/cyan (moderate positive)
    - (10, +inf): deep teal (very positive)
    """
    # Define band colors (RGB tuples, 0-1 range)
    colors = {
        "deep_purple": (0.58, 0.44, 0.86),   # Very negative (< -10)
        "lavender": (0.80, 0.70, 0.90),      # Negative (-10 to 0)
        "cream": (0.98, 0.95, 0.75),         # Near-zero positive (0 to 1)
        "light_teal": (0.70, 0.90, 0.88),    # Moderate positive (1 to 10)
        "deep_teal": (0.35, 0.75, 0.75),     # Very positive (> 10)
    }

    if val <= -10.0:
        rgb = colors["deep_purple"]
    elif val <= 0.0:
        rgb = colors["lavender"]
    elif val <= 1.0:
        rgb = colors["cream"]
    elif val <= 10.0:
        rgb = colors["light_teal"]
    else:
        rgb = colors["deep_teal"]

    return (*rgb, 1.0)


def _node_color_by_layer(val: float, layer_idx: int, n_layers: int) -> tuple:
    """
    Get node color based on layer type and activation value.

    Args:
        val: Activation value
        layer_idx: Layer index (0 = input, n_layers-1 = output)
        n_layers: Total number of layers

    Returns:
        RGBA color tuple
    """
    if layer_idx == 0:
        # Input layer
        return _input_node_color(val)
    elif layer_idx == n_layers - 1:
        # Output layer
        return _output_node_color(val)
    else:
        # Hidden layers
        return _hidden_node_color(val)


def _text_color_for_background(bg_color: tuple) -> str:
    """Contrasting text color based on luminance."""
    r, g, b = bg_color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


def _symmetric_range(activations: list) -> tuple[float, float]:
    """Color range centered at 0."""
    all_vals = [
        act[0].detach().cpu().numpy()
        if isinstance(act, torch.Tensor)
        else np.array(act[0] if isinstance(act[0], list) else act)
        for act in activations
    ]
    vmin, vmax = min(v.min() for v in all_vals), max(v.max() for v in all_vals)
    abs_max = max(abs(vmin), abs(vmax), 0.1)
    return -abs_max, abs_max

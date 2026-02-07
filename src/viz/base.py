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
        """Get or compute node positions for given layer structure."""
        key = tuple(layer_sizes)
        if key not in self._cache:
            pos = {}
            max_width = max(layer_sizes)
            for layer_idx, n_nodes in enumerate(layer_sizes):
                y_offset = -(max_width - n_nodes) / 2
                for node_idx in range(n_nodes):
                    name = f"({layer_idx},{node_idx})"
                    pos[name] = (layer_idx, y_offset - node_idx)
            self._cache[key] = pos
        return self._cache[key]

    def clear(self):
        self._cache.clear()


# Global singleton
_layout_cache = GraphLayoutCache()


# ------------------ HELPERS ------------------


def _activation_to_color(val: float, vmin: float = -2, vmax: float = 2) -> tuple:
    """
    Pastel color gradient for activation values:
    - 0.5 = beige/cream
    - 1.0 = light mint green
    - >1.0 = green -> teal (more saturated)
    - 0.0 = light peach/orange
    - -1.0 = coral/salmon
    - <-1.0 = deeper coral -> rose
    """
    # Pastel colors (RGB tuples, 0-1 range)
    colors = {
        "deep_rose": (0.85, 0.45, 0.55),      # < -1.0
        "coral": (0.95, 0.65, 0.60),           # -1.0
        "peach": (0.98, 0.82, 0.70),           # 0.0
        "cream": (0.98, 0.95, 0.80),           # 0.5
        "mint": (0.75, 0.92, 0.78),            # 1.0
        "teal": (0.55, 0.82, 0.78),            # > 1.0
    }

    def lerp(c1, c2, t):
        return tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(3))

    if val <= -1.0:
        # Deep rose to coral
        t = min(1.0, (-1.0 - val) / 1.0)  # How far below -1
        rgb = lerp(colors["coral"], colors["deep_rose"], t)
    elif val <= 0.0:
        # Coral to peach
        t = (val + 1.0) / 1.0  # -1 -> 0 maps to 0 -> 1
        rgb = lerp(colors["coral"], colors["peach"], t)
    elif val <= 0.5:
        # Peach to cream
        t = val / 0.5  # 0 -> 0.5 maps to 0 -> 1
        rgb = lerp(colors["peach"], colors["cream"], t)
    elif val <= 1.0:
        # Cream to mint
        t = (val - 0.5) / 0.5  # 0.5 -> 1 maps to 0 -> 1
        rgb = lerp(colors["cream"], colors["mint"], t)
    else:
        # Mint to teal
        t = min(1.0, (val - 1.0) / 1.0)  # How far above 1
        rgb = lerp(colors["mint"], colors["teal"], t)

    return (*rgb, 1.0)


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

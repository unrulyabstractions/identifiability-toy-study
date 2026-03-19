"""Visualization configuration.

Controls visualization levels to reduce output size and improve performance.
Default is NONE (no PNGs, only JSON/TXT summaries).
"""

from dataclasses import dataclass
from enum import IntEnum


class VizLevel(IntEnum):
    """Visualization detail levels."""

    NONE = 0  # No PNGs, only JSON/TXT summaries
    SUMMARY = 1  # Run-level summary plots only
    TOP_1 = 2  # + Best subcircuit per gate
    TOP_5 = 3  # + Top 5 subcircuits per gate
    ALL = 4  # + All analyzed subcircuits (default when --viz used)


@dataclass
class VizConfig:
    """Configuration for visualization output.

    Attributes:
        level: How much visualization to generate (NONE, SUMMARY, TOP_1, TOP_5)
        dpi: Resolution for generated PNGs (default 150, reduced from 300)
    """

    level: VizLevel = VizLevel.NONE
    dpi: int = 150  # Reduced from 300

    @property
    def skip_all_viz(self) -> bool:
        """Skip all visualization (level NONE)."""
        return self.level == VizLevel.NONE

    @property
    def skip_circuit_figures(self) -> bool:
        """Skip per-subcircuit circuit figures."""
        return self.level < VizLevel.TOP_1

    @property
    def max_subcircuits_per_gate(self) -> int:
        """Maximum number of subcircuits to visualize per gate."""
        return {
            VizLevel.NONE: 0,
            VizLevel.SUMMARY: 0,
            VizLevel.TOP_1: 1,
            VizLevel.TOP_5: 5,
            VizLevel.ALL: 9999,  # Effectively unlimited
        }[self.level]

    @property
    def generate_summary_plots(self) -> bool:
        """Whether to generate summary plots."""
        return self.level >= VizLevel.SUMMARY

    @classmethod
    def from_int(cls, level: int) -> "VizConfig":
        """Create VizConfig from integer level (0-4)."""
        viz_level = VizLevel(max(0, min(4, level)))
        return cls(level=viz_level)

"""Backwards compatibility re-export for trial module.

The trial module has been refactored into a package:
- src/trial/__init__.py - exports run_trial
- src/trial/runner.py - main run_trial function
- src/trial/phases.py - phase functions (training, SPD, circuits, etc.)
- src/trial/gate_analysis.py - per-gate analysis

Import from src.trial instead:
    from src.trial import run_trial
"""

from .trial import run_trial

__all__ = ["run_trial"]

"""Training analysis module for epiplexity estimation.

Measures how much structure (S_T) vs noise (H_T) a model learns during training
using prequential coding / MDL principles.
"""

from src.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())

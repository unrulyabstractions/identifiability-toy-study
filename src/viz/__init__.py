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
"""

from src.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())

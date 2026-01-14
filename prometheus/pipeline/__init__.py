"""
Pipeline: End-to-end problem solving.

This ties everything together:
1. Parse the problem
2. Formalize it
3. Run proof search
4. Extract the answer

The main entry point is PrometheusSolver.
"""

from prometheus.pipeline.solver import PrometheusSolver, SolverConfig
from prometheus.pipeline.aimo_adapter import AIMOAdapter

__all__ = ["PrometheusSolver", "SolverConfig", "AIMOAdapter"]

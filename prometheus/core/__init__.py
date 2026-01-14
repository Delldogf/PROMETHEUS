"""
Core abstractions for PROMETHEUS.

This module contains the fundamental building blocks:
- Formula: Mathematical expressions and logical statements
- ProofState: Current state of a proof attempt
- Tactic: Proof transformations (like chess moves for proofs)
- Strategy: High-level proof approaches (like game plans)
"""

from prometheus.core.formula import Formula, Expression, Variable, Constraint
from prometheus.core.proof_state import ProofState, Goal
from prometheus.core.tactic import Tactic, TacticResult
from prometheus.core.strategy import Strategy, StrategyType

__all__ = [
    "Formula", "Expression", "Variable", "Constraint",
    "ProofState", "Goal",
    "Tactic", "TacticResult", 
    "Strategy", "StrategyType"
]

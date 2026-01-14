"""
Strategy: High-level proof approaches.

While tactics are individual moves, strategies are game plans.
Think of it like:
- Tactic: "Move the knight to e5"
- Strategy: "Control the center and prepare for kingside attack"

In PROMETHEUS:
- Tactic: "Apply modular arithmetic mod 3"
- Strategy: "Reduce this number theory problem to checking small cases"

This is inspired by Omega's proof planning system.

WHAT THIS FILE DOES:
- Defines high-level proof strategies
- Each strategy suggests sequences of tactics
- LLM will help select which strategy to try
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Tuple

from prometheus.core.formula import FormalProblem, MathDomain
from prometheus.core.proof_state import ProofState, Goal


class StrategyType(Enum):
    """Types of high-level proof strategies."""
    
    # ===== General Strategies =====
    DIRECT = auto()           # Prove directly step by step
    CONTRADICTION = auto()    # Assume negation, derive contradiction
    CONTRAPOSITIVE = auto()   # Prove ¬Q → ¬P instead of P → Q
    CASE_ANALYSIS = auto()    # Split into exhaustive cases
    
    # ===== Induction Strategies =====
    SIMPLE_INDUCTION = auto()    # Standard P(1), P(k)→P(k+1)
    STRONG_INDUCTION = auto()    # Assume P(1)...P(k-1), prove P(k)
    STRUCTURAL_INDUCTION = auto() # Induction on structure (trees, etc.)
    
    # ===== Algebra Strategies =====
    ALGEBRAIC_MANIPULATION = auto()  # Expand, simplify, factor
    SUBSTITUTION = auto()            # Clever variable substitution
    INEQUALITY_CHAIN = auto()        # AM-GM, Cauchy-Schwarz, etc.
    
    # ===== Number Theory Strategies =====
    MODULAR_REDUCTION = auto()    # Work mod m for some m
    DIVISIBILITY_ANALYSIS = auto() # Analyze what divides what
    PRIME_FACTORIZATION = auto()   # Work with prime factors
    
    # ===== Geometry Strategies =====
    COORDINATE_GEOMETRY = auto()   # Set up coordinates
    SYNTHETIC_GEOMETRY = auto()    # Pure geometric reasoning
    TRIGONOMETRIC = auto()         # Use trig identities
    COMPLEX_NUMBERS = auto()       # Represent points as complex numbers
    
    # ===== Combinatorics Strategies =====
    COUNTING_BIJECTION = auto()    # Find a bijection
    GENERATING_FUNCTIONS = auto()  # Use generating functions
    PIGEONHOLE = auto()            # Pigeonhole principle
    INCLUSION_EXCLUSION = auto()   # PIE
    
    # ===== Meta Strategies =====
    FIND_PATTERN = auto()     # Check small cases, find pattern
    WORK_BACKWARDS = auto()   # Start from answer, work back
    GENERALIZE = auto()       # Prove something stronger
    SPECIALIZE = auto()       # First solve a special case


@dataclass
class StrategyStep:
    """
    A single step in a strategy's suggested tactic sequence.
    """
    tactic_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    optional: bool = False  # Can this step be skipped?


@dataclass
class Strategy:
    """
    A high-level proof strategy.
    
    Contains:
    - The type of strategy
    - Suggested sequence of tactics
    - Conditions for when this strategy applies
    - Estimated difficulty/time
    """
    
    strategy_type: StrategyType
    name: str
    description: str
    
    # Suggested sequence of tactics
    steps: List[StrategyStep] = field(default_factory=list)
    
    # What kinds of problems is this good for?
    applicable_domains: List[str] = field(default_factory=list)  # ["algebra", "number_theory"]
    applicable_goal_types: List[str] = field(default_factory=list)  # ["prove", "find_all"]
    
    # How confident are we this will work? (0-1)
    confidence: float = 0.5
    
    # Estimated computational cost (1-10)
    estimated_cost: int = 5
    
    # Prerequisites - other things that should be true first
    prerequisites: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"Strategy({self.name}): {self.description}"
    
    def get_steps_preview(self) -> str:
        """Get a preview of the tactic sequence."""
        if not self.steps:
            return "(no steps defined)"
        step_names = [s.tactic_name for s in self.steps[:5]]
        if len(self.steps) > 5:
            step_names.append("...")
        return " → ".join(step_names)


# ===== Predefined Strategies =====

def create_default_strategies() -> List[Strategy]:
    """Create the default set of strategies."""
    return [
        # ===== Direct Proof =====
        Strategy(
            strategy_type=StrategyType.DIRECT,
            name="direct_proof",
            description="Prove directly through a chain of implications",
            steps=[
                StrategyStep("simplify", {}, "Simplify the goal"),
                StrategyStep("apply_theorem", {}, "Apply relevant theorems"),
            ],
            applicable_domains=["all"],
            applicable_goal_types=["prove"],
            confidence=0.4,
            estimated_cost=3
        ),
        
        # ===== Proof by Contradiction =====
        Strategy(
            strategy_type=StrategyType.CONTRADICTION,
            name="proof_by_contradiction",
            description="Assume the negation and derive a contradiction",
            steps=[
                StrategyStep("contradiction", {}, "Assume the negation"),
                StrategyStep("simplify", {}, "Simplify with the assumption"),
                # Then search for contradiction...
            ],
            applicable_domains=["all"],
            applicable_goal_types=["prove"],
            confidence=0.5,
            estimated_cost=5
        ),
        
        # ===== Simple Induction =====
        Strategy(
            strategy_type=StrategyType.SIMPLE_INDUCTION,
            name="simple_induction",
            description="Standard mathematical induction on a natural number",
            steps=[
                StrategyStep("induction", {"strong": False}, "Set up induction"),
                # Subgoals will be base case and inductive step
            ],
            applicable_domains=["number_theory", "algebra", "combinatorics"],
            applicable_goal_types=["prove_forall"],
            confidence=0.6,
            estimated_cost=4
        ),
        
        # ===== Strong Induction =====
        Strategy(
            strategy_type=StrategyType.STRONG_INDUCTION,
            name="strong_induction",
            description="Strong induction - assume all smaller cases",
            steps=[
                StrategyStep("induction", {"strong": True}, "Set up strong induction"),
            ],
            applicable_domains=["number_theory", "combinatorics"],
            applicable_goal_types=["prove_forall"],
            confidence=0.55,
            estimated_cost=5
        ),
        
        # ===== Case Analysis =====
        Strategy(
            strategy_type=StrategyType.CASE_ANALYSIS,
            name="case_analysis",
            description="Split into exhaustive cases and prove each",
            steps=[
                StrategyStep("split_cases", {}, "Split into cases"),
                # Then solve each case...
            ],
            applicable_domains=["all"],
            applicable_goal_types=["prove", "find_all"],
            confidence=0.5,
            estimated_cost=4
        ),
        
        # ===== Modular Reduction =====
        Strategy(
            strategy_type=StrategyType.MODULAR_REDUCTION,
            name="modular_reduction",
            description="Analyze the problem modulo some number",
            steps=[
                StrategyStep("mod", {}, "Reduce modulo m"),
                StrategyStep("simplify", {}, "Simplify in modular arithmetic"),
            ],
            applicable_domains=["number_theory"],
            applicable_goal_types=["prove", "find_all"],
            confidence=0.6,
            estimated_cost=4
        ),
        
        # ===== Find Pattern =====
        Strategy(
            strategy_type=StrategyType.FIND_PATTERN,
            name="find_pattern",
            description="Check small cases and look for a pattern",
            steps=[
                StrategyStep("substitute", {"n": 1}, "Check n=1"),
                StrategyStep("substitute", {"n": 2}, "Check n=2"),
                StrategyStep("substitute", {"n": 3}, "Check n=3"),
                # Then generalize...
            ],
            applicable_domains=["number_theory", "algebra", "combinatorics"],
            applicable_goal_types=["find_all", "find_formula"],
            confidence=0.7,
            estimated_cost=3
        ),
        
        # ===== Algebraic Manipulation =====
        Strategy(
            strategy_type=StrategyType.ALGEBRAIC_MANIPULATION,
            name="algebraic_manipulation",
            description="Expand, simplify, factor to find structure",
            steps=[
                StrategyStep("simplify", {"expand": True}, "Expand expressions"),
                StrategyStep("simplify", {"factor": True}, "Factor expressions"),
            ],
            applicable_domains=["algebra"],
            applicable_goal_types=["prove", "simplify"],
            confidence=0.6,
            estimated_cost=2
        ),
        
        # ===== Coordinate Geometry =====
        Strategy(
            strategy_type=StrategyType.COORDINATE_GEOMETRY,
            name="coordinate_geometry",
            description="Set up coordinates and compute algebraically",
            steps=[
                StrategyStep("set_coordinates", {}, "Establish coordinate system"),
                StrategyStep("translate_to_algebra", {}, "Convert to equations"),
                StrategyStep("simplify", {}, "Solve algebraically"),
            ],
            applicable_domains=["geometry"],
            applicable_goal_types=["prove", "find_value"],
            confidence=0.7,
            estimated_cost=6
        ),
        
        # ===== Bijection Counting =====
        Strategy(
            strategy_type=StrategyType.COUNTING_BIJECTION,
            name="counting_bijection",
            description="Count by finding a bijection to something simpler",
            steps=[
                StrategyStep("find_bijection", {}, "Find a bijection"),
                StrategyStep("count", {}, "Count the simpler set"),
            ],
            applicable_domains=["combinatorics"],
            applicable_goal_types=["count", "find_formula"],
            confidence=0.5,
            estimated_cost=5
        ),
    ]


class StrategyRegistry:
    """
    Registry of all available strategies.
    """
    
    def __init__(self):
        self._strategies: Dict[str, Strategy] = {}
        self._by_type: Dict[StrategyType, List[Strategy]] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        for strategy in create_default_strategies():
            self.register(strategy)
    
    def register(self, strategy: Strategy) -> None:
        """Register a strategy."""
        self._strategies[strategy.name] = strategy
        
        if strategy.strategy_type not in self._by_type:
            self._by_type[strategy.strategy_type] = []
        self._by_type[strategy.strategy_type].append(strategy)
    
    def get(self, name: str) -> Optional[Strategy]:
        """Get strategy by name."""
        return self._strategies.get(name)
    
    def get_by_type(self, stype: StrategyType) -> List[Strategy]:
        """Get all strategies of a given type."""
        return self._by_type.get(stype, [])
    
    def get_applicable(self, problem: FormalProblem) -> List[Strategy]:
        """Get strategies that might work for this problem."""
        applicable = []
        for strategy in self._strategies.values():
            # Check domain match
            if "all" in strategy.applicable_domains:
                applicable.append(strategy)
            elif any(d in problem.domain_hints for d in strategy.applicable_domains):
                applicable.append(strategy)
            # Check goal type match
            elif problem.goal_type in strategy.applicable_goal_types:
                applicable.append(strategy)
        
        # Sort by confidence
        applicable.sort(key=lambda s: s.confidence, reverse=True)
        return applicable
    
    def list_all(self) -> List[str]:
        """List all strategy names."""
        return list(self._strategies.keys())


# Global registry instance
STRATEGY_REGISTRY = StrategyRegistry()

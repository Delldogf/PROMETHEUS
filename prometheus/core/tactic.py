"""
Tactic: Proof transformations - the "moves" in theorem proving.

Think of tactics like chess moves. Each tactic takes the current
proof state and transforms it into a new state, hopefully closer
to a complete proof.

Examples of tactics:
- "simplify": Simplify algebraic expressions
- "split_cases": Break into cases (e.g., n even vs n odd)
- "apply_theorem": Use a known theorem
- "substitute": Replace a variable with an expression
- "induction": Set up mathematical induction

WHAT THIS FILE DOES:
- Defines the base Tactic class that all tactics inherit from
- Provides common tactics used in olympiad problems
- Handles the mechanics of applying tactics to proof states
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Callable, Tuple

from prometheus.core.formula import (
    Formula, Constraint, Expression, Variable, 
    RelationType, ExpressionType, MathDomain
)
from prometheus.core.proof_state import ProofState, Goal, Fact, GoalStatus


class TacticStatus(Enum):
    """Result status of applying a tactic."""
    SUCCESS = auto()         # Tactic worked, made progress
    PARTIAL = auto()         # Tactic partially worked
    FAILURE = auto()         # Tactic couldn't be applied
    SOLVED = auto()          # Tactic completely solved the goal!
    NEEDS_SUBGOALS = auto()  # Tactic created subgoals to solve


@dataclass
class TacticResult:
    """
    The result of applying a tactic.
    
    Contains:
    - Whether it worked
    - The new proof state (if it worked)
    - Any messages/explanations
    - New subgoals created
    """
    status: TacticStatus
    new_state: Optional[ProofState] = None
    message: str = ""
    new_goals: List[Goal] = field(default_factory=list)
    new_facts: List[Fact] = field(default_factory=list)
    confidence: float = 1.0  # How confident are we this was the right move?
    
    def succeeded(self) -> bool:
        return self.status in [TacticStatus.SUCCESS, TacticStatus.SOLVED, TacticStatus.PARTIAL]
    
    def __str__(self) -> str:
        status_emoji = {
            TacticStatus.SUCCESS: "✅",
            TacticStatus.PARTIAL: "🔶",
            TacticStatus.FAILURE: "❌",
            TacticStatus.SOLVED: "🎉",
            TacticStatus.NEEDS_SUBGOALS: "🔀"
        }
        return f"{status_emoji[self.status]} {self.message}"


class Tactic(ABC):
    """
    Base class for all proof tactics.
    
    A tactic is a proof transformation. You give it a proof state
    and a goal to work on, and it tries to make progress.
    
    This is inspired by tactics in Lean, Isabelle, and Coq,
    but simplified for our olympiad-focused use case.
    """
    
    # Human-readable name
    name: str = "base_tactic"
    
    # Description of what this tactic does
    description: str = "Base tactic class"
    
    # What kinds of goals can this tactic handle?
    applicable_goal_types: List[str] = []
    
    @abstractmethod
    def apply(self, state: ProofState, goal: Goal, **kwargs) -> TacticResult:
        """
        Apply this tactic to try to prove/progress on the given goal.
        
        Args:
            state: Current proof state
            goal: The specific goal to work on
            **kwargs: Additional parameters (e.g., which theorem to apply)
            
        Returns:
            TacticResult describing what happened
        """
        pass
    
    def can_apply(self, state: ProofState, goal: Goal) -> bool:
        """
        Check if this tactic might be applicable.
        Quick check before trying the full apply.
        """
        return goal.is_open()
    
    def __str__(self) -> str:
        return f"Tactic({self.name})"


# ===== Concrete Tactics =====

class SimplifyTactic(Tactic):
    """
    Simplify algebraic expressions in the goal.
    
    Examples:
    - x + 0 → x
    - x * 1 → x
    - (a + b)² → a² + 2ab + b²
    """
    name = "simplify"
    description = "Simplify algebraic expressions"
    
    def apply(self, state: ProofState, goal: Goal, **kwargs) -> TacticResult:
        # This will use SymPy for actual simplification
        # For now, return a placeholder
        new_state = state.clone()
        return TacticResult(
            status=TacticStatus.SUCCESS,
            new_state=new_state,
            message="Simplified expressions"
        )


class SplitCasesTactic(Tactic):
    """
    Split a proof into cases.
    
    Examples:
    - Split on "n even or n odd"
    - Split on "x > 0, x = 0, or x < 0"
    - Split on "n = 1, n = 2, or n ≥ 3"
    """
    name = "split_cases"
    description = "Split proof into multiple cases"
    
    def apply(self, state: ProofState, goal: Goal, cases: List[Constraint] = None, **kwargs) -> TacticResult:
        if not cases:
            return TacticResult(
                status=TacticStatus.FAILURE,
                message="No cases provided for split"
            )
        
        new_state = state.clone()
        new_goals = []
        
        # Create a subgoal for each case
        for i, case in enumerate(cases):
            case_goal = Goal(
                statement=goal.statement,  # Same thing to prove
                origin=f"case_{i+1}: {case}",
                parent_id=goal.id
            )
            new_goals.append(case_goal)
            new_state.goals.append(case_goal)
        
        # Mark original goal as needing subgoals
        for g in new_state.goals:
            if g.id == goal.id:
                g.status = GoalStatus.ABANDONED
                g.proof_method = f"split into {len(cases)} cases"
        
        return TacticResult(
            status=TacticStatus.NEEDS_SUBGOALS,
            new_state=new_state,
            new_goals=new_goals,
            message=f"Split into {len(cases)} cases"
        )


class ApplyTheoremTactic(Tactic):
    """
    Apply a known theorem to the goal.
    
    This is one of the most powerful tactics - it uses
    our knowledge base of mathematical theorems.
    """
    name = "apply_theorem"
    description = "Apply a theorem from the knowledge base"
    
    def apply(self, state: ProofState, goal: Goal, theorem_name: str = "", **kwargs) -> TacticResult:
        if not theorem_name:
            return TacticResult(
                status=TacticStatus.FAILURE,
                message="No theorem specified"
            )
        
        # TODO: Look up theorem, check if applicable, apply it
        new_state = state.clone()
        return TacticResult(
            status=TacticStatus.SUCCESS,
            new_state=new_state,
            message=f"Applied theorem: {theorem_name}"
        )


class InductionTactic(Tactic):
    """
    Set up mathematical induction.
    
    For proving P(n) for all positive integers n:
    1. Base case: Prove P(1)
    2. Inductive step: Prove P(k) → P(k+1)
    
    Can also do strong induction:
    - Prove: if P(1)...P(k-1) all true, then P(k) true
    """
    name = "induction"
    description = "Set up mathematical induction"
    
    def apply(self, state: ProofState, goal: Goal, 
              variable: str = "n", 
              base: int = 1,
              strong: bool = False,
              **kwargs) -> TacticResult:
        
        new_state = state.clone()
        new_goals = []
        
        # Create base case goal
        base_goal = Goal(
            statement=goal.statement,  # Will be specialized to n=base
            origin=f"induction_base: {variable}={base}",
            parent_id=goal.id
        )
        new_goals.append(base_goal)
        new_state.goals.append(base_goal)
        
        # Create inductive step goal
        step_type = "strong_induction_step" if strong else "induction_step"
        step_goal = Goal(
            statement=goal.statement,  # Will assume P(k), prove P(k+1)
            origin=f"{step_type}: {variable}=k → {variable}=k+1",
            parent_id=goal.id
        )
        new_goals.append(step_goal)
        new_state.goals.append(step_goal)
        
        # Mark original as handled
        for g in new_state.goals:
            if g.id == goal.id:
                g.status = GoalStatus.ABANDONED
                g.proof_method = f"{'strong ' if strong else ''}induction on {variable}"
        
        return TacticResult(
            status=TacticStatus.NEEDS_SUBGOALS,
            new_state=new_state,
            new_goals=new_goals,
            message=f"Set up {'strong ' if strong else ''}induction on {variable}"
        )


class ContradictionTactic(Tactic):
    """
    Prove by contradiction.
    
    To prove P:
    1. Assume ¬P (not P)
    2. Derive a contradiction
    3. Conclude P must be true
    """
    name = "contradiction"
    description = "Prove by deriving a contradiction"
    
    def apply(self, state: ProofState, goal: Goal, **kwargs) -> TacticResult:
        new_state = state.clone()
        
        # Add negation of goal as a new fact (our assumption)
        negated = Constraint(
            left=goal.statement.right,  # Swap sides for negation
            relation=self.negate_relation(goal.statement.relation),
            right=goal.statement.left
        )
        
        new_state.add_fact(negated, source="contradiction_assumption")
        
        # New goal: derive False (a contradiction)
        contradiction_goal = Goal(
            statement=Constraint(
                left=Expression.constant("False"),
                relation=RelationType.EQUALS,
                right=Expression.constant("True")
            ),
            origin="derive_contradiction",
            parent_id=goal.id
        )
        new_state.goals.append(contradiction_goal)
        
        return TacticResult(
            status=TacticStatus.NEEDS_SUBGOALS,
            new_state=new_state,
            new_goals=[contradiction_goal],
            message="Attempting proof by contradiction"
        )
    
    def negate_relation(self, rel: RelationType) -> RelationType:
        """Get the negation of a relation."""
        negations = {
            RelationType.EQUALS: RelationType.NOT_EQUALS,
            RelationType.NOT_EQUALS: RelationType.EQUALS,
            RelationType.LESS_THAN: RelationType.GREATER_EQUAL,
            RelationType.GREATER_EQUAL: RelationType.LESS_THAN,
            RelationType.LESS_EQUAL: RelationType.GREATER_THAN,
            RelationType.GREATER_THAN: RelationType.LESS_EQUAL,
        }
        return negations.get(rel, RelationType.NOT_EQUALS)


class SubstituteTactic(Tactic):
    """
    Substitute a value or expression for a variable.
    
    Examples:
    - Substitute n = 1 to check base case
    - Substitute x = y + 1 to simplify
    - Substitute a = b·q + r (division algorithm)
    """
    name = "substitute"
    description = "Substitute an expression for a variable"
    
    def apply(self, state: ProofState, goal: Goal,
              variable: str = "",
              expression: str = "",
              **kwargs) -> TacticResult:
        
        if not variable or not expression:
            return TacticResult(
                status=TacticStatus.FAILURE,
                message="Must specify variable and expression for substitution"
            )
        
        new_state = state.clone()
        # TODO: Actually perform the substitution
        
        return TacticResult(
            status=TacticStatus.SUCCESS,
            new_state=new_state,
            message=f"Substituted {variable} = {expression}"
        )


class ModularArithmeticTactic(Tactic):
    """
    Work modulo some number.
    
    Very powerful for number theory problems.
    Example: Consider equation mod 4 to find constraints on n.
    """
    name = "mod"
    description = "Work in modular arithmetic"
    
    def apply(self, state: ProofState, goal: Goal,
              modulus: int = 0,
              **kwargs) -> TacticResult:
        
        if modulus <= 0:
            return TacticResult(
                status=TacticStatus.FAILURE,
                message="Must specify a positive modulus"
            )
        
        new_state = state.clone()
        # TODO: Convert goal to work mod m
        
        return TacticResult(
            status=TacticStatus.SUCCESS,
            new_state=new_state,
            message=f"Working modulo {modulus}"
        )


# ===== Tactic Registry =====

class TacticRegistry:
    """
    Registry of all available tactics.
    
    This lets us look up tactics by name and see what's available.
    """
    
    def __init__(self):
        self._tactics: Dict[str, Tactic] = {}
        self._register_default_tactics()
    
    def _register_default_tactics(self):
        """Register the built-in tactics."""
        defaults = [
            SimplifyTactic(),
            SplitCasesTactic(),
            ApplyTheoremTactic(),
            InductionTactic(),
            ContradictionTactic(),
            SubstituteTactic(),
            ModularArithmeticTactic(),
        ]
        for tactic in defaults:
            self.register(tactic)
    
    def register(self, tactic: Tactic) -> None:
        """Register a tactic."""
        self._tactics[tactic.name] = tactic
    
    def get(self, name: str) -> Optional[Tactic]:
        """Get a tactic by name."""
        return self._tactics.get(name)
    
    def list_all(self) -> List[str]:
        """List all registered tactic names."""
        return list(self._tactics.keys())
    
    def get_applicable(self, state: ProofState, goal: Goal) -> List[Tactic]:
        """Get all tactics that might apply to this goal."""
        return [t for t in self._tactics.values() if t.can_apply(state, goal)]


# Global registry instance
TACTIC_REGISTRY = TacticRegistry()

"""
ProofState: The current state of a proof attempt.

Think of this like a "snapshot" of where we are in solving a problem.
Just like in a chess game you have the current board position,
in a proof we have:
- What we're trying to prove (goals)
- What we already know (facts)
- What we've tried so far (history)

WHAT THIS FILE DOES:
- Tracks the current state of our proof attempt
- Manages goals (things we still need to prove)
- Manages facts (things we've established)
- Keeps history so we can backtrack if needed
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
import copy

from prometheus.core.formula import Formula, Constraint, Variable, Expression, FormalProblem


class GoalStatus(Enum):
    """Status of a proof goal."""
    OPEN = auto()       # Still needs to be proven
    PROVEN = auto()     # Successfully proven
    FAILED = auto()     # We know it can't be proven (contradiction found)
    ABANDONED = auto()  # Gave up on this approach


@dataclass
class Goal:
    """
    A single thing we need to prove.
    
    In a proof, we often have multiple sub-goals. For example:
    - To prove "A and B", we need two goals: prove A, prove B
    - To prove "A or B", we need one goal: prove either A or B
    
    Think of goals like items on a to-do list for the proof.
    """
    
    # What we need to prove
    statement: Constraint
    
    # Current status
    status: GoalStatus = GoalStatus.OPEN
    
    # How this goal was created (for debugging)
    origin: str = "initial"
    
    # Parent goal (if this is a sub-goal)
    parent_id: Optional[str] = None
    
    # Unique identifier
    id: str = ""
    
    # If proven, how?
    proof_method: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            # Generate a simple unique ID
            import uuid
            self.id = str(uuid.uuid4())[:8]
    
    def __str__(self) -> str:
        status_emoji = {
            GoalStatus.OPEN: "❓",
            GoalStatus.PROVEN: "✅",
            GoalStatus.FAILED: "❌",
            GoalStatus.ABANDONED: "⏭️"
        }
        return f"{status_emoji[self.status]} Goal[{self.id}]: {self.statement}"
    
    def is_open(self) -> bool:
        return self.status == GoalStatus.OPEN


@dataclass
class Fact:
    """
    Something we know to be true.
    
    Facts come from:
    - The problem statement (given facts)
    - Theorems we've applied
    - Things we've derived during the proof
    """
    statement: Constraint
    source: str  # Where did this fact come from?
    confidence: float = 1.0  # 1.0 = definitely true, <1.0 = possibly true
    
    def __str__(self) -> str:
        return f"Fact: {self.statement} [{self.source}]"


@dataclass
class ProofStep:
    """
    A single step in the proof history.
    
    This lets us track what we've done and potentially undo it.
    """
    action: str  # What tactic/action was taken
    before_goals: List[str]  # Goal IDs before
    after_goals: List[str]   # Goal IDs after
    new_facts: List[Fact]    # Facts established by this step
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    notes: str = ""
    
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.action}: {len(self.before_goals)} goals → {len(self.after_goals)} goals"


@dataclass
class ProofState:
    """
    The complete state of a proof attempt.
    
    This is the central data structure that tactics operate on.
    Think of it as the "game board" for theorem proving.
    
    Key idea from MRPPS: The search strategy works by evaluating
    proof states and deciding which one to explore next.
    """
    
    # The original problem we're solving
    problem: Optional[FormalProblem] = None
    
    # Current goals (things we need to prove)
    goals: List[Goal] = field(default_factory=list)
    
    # Known facts (things we can use)
    facts: List[Fact] = field(default_factory=list)
    
    # Variables in scope
    variables: Dict[str, Variable] = field(default_factory=dict)
    
    # History of proof steps (for backtracking)
    history: List[ProofStep] = field(default_factory=list)
    
    # Current depth in the proof tree
    depth: int = 0
    
    # Unique identifier for this state
    state_id: str = ""
    
    # For answer-finding problems: candidate answers found
    candidate_answers: List[Any] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.state_id:
            import uuid
            self.state_id = str(uuid.uuid4())[:8]
    
    # ===== State Queries =====
    
    def is_solved(self) -> bool:
        """Are all goals proven?"""
        return len(self.goals) > 0 and all(g.status == GoalStatus.PROVEN for g in self.goals)
    
    def is_stuck(self) -> bool:
        """Are we stuck (some goal failed)?"""
        return any(g.status == GoalStatus.FAILED for g in self.goals)
    
    def open_goals(self) -> List[Goal]:
        """Get all goals that still need proving."""
        return [g for g in self.goals if g.is_open()]
    
    def has_open_goals(self) -> bool:
        """Do we have work left to do?"""
        return len(self.open_goals()) > 0
    
    # ===== State Modifications =====
    
    def add_goal(self, statement: Constraint, origin: str = "derived") -> Goal:
        """Add a new goal to prove."""
        goal = Goal(statement=statement, origin=origin)
        self.goals.append(goal)
        return goal
    
    def add_fact(self, statement: Constraint, source: str = "derived") -> Fact:
        """Add a new known fact."""
        fact = Fact(statement=statement, source=source)
        self.facts.append(fact)
        return fact
    
    def mark_goal_proven(self, goal_id: str, method: str = "") -> bool:
        """Mark a goal as successfully proven."""
        for goal in self.goals:
            if goal.id == goal_id:
                goal.status = GoalStatus.PROVEN
                goal.proof_method = method
                return True
        return False
    
    def mark_goal_failed(self, goal_id: str) -> bool:
        """Mark a goal as failed (contradiction found or impossible)."""
        for goal in self.goals:
            if goal.id == goal_id:
                goal.status = GoalStatus.FAILED
                return True
        return False
    
    def add_candidate_answer(self, answer: Any) -> None:
        """Add a candidate answer (for 'find all' type problems)."""
        if answer not in self.candidate_answers:
            self.candidate_answers.append(answer)
    
    # ===== Cloning for Search =====
    
    def clone(self) -> ProofState:
        """Create a deep copy of this state for branching search."""
        new_state = copy.deepcopy(self)
        import uuid
        new_state.state_id = str(uuid.uuid4())[:8]
        new_state.depth = self.depth + 1
        return new_state
    
    # ===== Display =====
    
    def __str__(self) -> str:
        lines = [
            f"═══ ProofState [{self.state_id}] (depth={self.depth}) ═══",
            f"Goals ({len(self.goals)}):"
        ]
        for g in self.goals[:5]:  # Show first 5 goals
            lines.append(f"  {g}")
        if len(self.goals) > 5:
            lines.append(f"  ... and {len(self.goals) - 5} more")
        
        lines.append(f"Facts ({len(self.facts)}):")
        for f in self.facts[:3]:  # Show first 3 facts
            lines.append(f"  {f}")
        if len(self.facts) > 3:
            lines.append(f"  ... and {len(self.facts) - 3} more")
        
        if self.candidate_answers:
            lines.append(f"Candidate answers: {self.candidate_answers}")
        
        status = "SOLVED ✅" if self.is_solved() else ("STUCK ❌" if self.is_stuck() else "IN PROGRESS 🔄")
        lines.append(f"Status: {status}")
        
        return "\n".join(lines)
    
    def summary(self) -> str:
        """One-line summary of state."""
        open_count = len(self.open_goals())
        proven_count = len([g for g in self.goals if g.status == GoalStatus.PROVEN])
        return f"State[{self.state_id}]: {proven_count} proven, {open_count} open, {len(self.facts)} facts"

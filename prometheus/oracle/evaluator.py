"""
Evaluator: Assess proof state quality and progress.

This is the modernized Q* algorithm from MRPPS.
Instead of hand-coded merit functions, we use the LLM to evaluate
how promising a proof state is.

WHAT THIS FILE DOES:
- Evaluates proof states to guide search
- Combines multiple signals (LLM, heuristics, complexity)
- Helps the search prioritize promising branches
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
import math

from prometheus.core.proof_state import ProofState, GoalStatus


@dataclass
class Evaluation:
    """
    Evaluation of a proof state.
    
    The evaluation combines multiple signals:
    - LLM assessment of promise
    - Structural heuristics (goal count, fact count, etc.)
    - Progress indicators
    """
    # Overall score (0 to 1)
    score: float
    
    # Component scores (for debugging/analysis)
    components: Dict[str, float] = field(default_factory=dict)
    
    # Explanation
    explanation: str = ""
    
    # Confidence in this evaluation
    confidence: float = 1.0


class Evaluator:
    """
    Evaluates proof states to guide search.
    
    This is like a chess engine's position evaluation -
    it tells us which proof states are worth exploring further.
    
    The original MRPPS Q* algorithm used:
    - Cost-so-far (like A* search)
    - Estimated cost-to-go (heuristic)
    - Merit function combining these
    
    We modernize this by:
    - Using LLM for cost-to-go estimation
    - Adding learned heuristics
    - Considering problem-specific features
    """
    
    def __init__(self, oracle=None):
        """
        Initialize the Evaluator.
        
        Args:
            oracle: Optional LLMOracle for neural evaluation
        """
        self.oracle = oracle
        
        # Weights for combining different signals
        self.weights = {
            "llm_score": 0.4,         # LLM's assessment
            "goal_progress": 0.2,     # How many goals closed
            "complexity": 0.15,       # Simpler is better
            "depth_penalty": 0.1,     # Prefer shorter proofs
            "fact_utility": 0.15,     # Are facts useful?
        }
    
    async def evaluate(self, state: ProofState) -> Evaluation:
        """
        Evaluate a proof state.
        
        Returns a score from 0 (hopeless) to 1 (very promising).
        """
        components = {}
        
        # 1. LLM evaluation (if available)
        if self.oracle:
            state_dict = self._state_to_dict(state)
            llm_score = await self.oracle.evaluate_state(state_dict)
            components["llm_score"] = llm_score
        else:
            components["llm_score"] = 0.5  # Neutral if no LLM
        
        # 2. Goal progress
        components["goal_progress"] = self._evaluate_goal_progress(state)
        
        # 3. Complexity (lower is better)
        components["complexity"] = self._evaluate_complexity(state)
        
        # 4. Depth penalty
        components["depth_penalty"] = self._evaluate_depth(state)
        
        # 5. Fact utility
        components["fact_utility"] = self._evaluate_fact_utility(state)
        
        # Combine scores
        total_score = sum(
            self.weights.get(key, 0.1) * value
            for key, value in components.items()
        )
        
        # Clamp to [0, 1]
        total_score = max(0.0, min(1.0, total_score))
        
        return Evaluation(
            score=total_score,
            components=components,
            explanation=self._generate_explanation(state, components)
        )
    
    def evaluate_sync(self, state: ProofState) -> Evaluation:
        """
        Synchronous evaluation using only heuristics.
        Use when LLM call isn't needed/possible.
        """
        components = {}
        
        components["goal_progress"] = self._evaluate_goal_progress(state)
        components["complexity"] = self._evaluate_complexity(state)
        components["depth_penalty"] = self._evaluate_depth(state)
        components["fact_utility"] = self._evaluate_fact_utility(state)
        
        # Without LLM, reweight
        total = (
            0.35 * components["goal_progress"] +
            0.25 * components["complexity"] +
            0.2 * components["depth_penalty"] +
            0.2 * components["fact_utility"]
        )
        
        return Evaluation(
            score=max(0.0, min(1.0, total)),
            components=components,
            confidence=0.7  # Lower confidence without LLM
        )
    
    def _evaluate_goal_progress(self, state: ProofState) -> float:
        """
        How much progress have we made on goals?
        
        Score from 0 (no progress) to 1 (all done).
        """
        if not state.goals:
            return 0.5  # No goals yet
        
        proven = sum(1 for g in state.goals if g.status == GoalStatus.PROVEN)
        failed = sum(1 for g in state.goals if g.status == GoalStatus.FAILED)
        total = len(state.goals)
        
        if failed > 0:
            return 0.1  # Failure is bad
        
        return proven / total if total > 0 else 0.5
    
    def _evaluate_complexity(self, state: ProofState) -> float:
        """
        Simpler states are more promising.
        
        Penalizes:
        - Too many open goals
        - Complex expressions
        - Deep nesting
        """
        open_goals = len(state.open_goals())
        
        # More open goals = more complex = lower score
        # Use exponential decay
        complexity_penalty = math.exp(-0.2 * open_goals)
        
        return complexity_penalty
    
    def _evaluate_depth(self, state: ProofState) -> float:
        """
        Prefer shorter proofs (lower depth).
        
        But don't penalize too harshly - sometimes deep proofs are necessary.
        """
        # Soft penalty for depth
        depth_score = 1.0 / (1.0 + 0.1 * state.depth)
        return depth_score
    
    def _evaluate_fact_utility(self, state: ProofState) -> float:
        """
        Are we accumulating useful facts?
        
        Having facts is good, but too many might mean we're flailing.
        """
        fact_count = len(state.facts)
        
        # Sweet spot: 2-8 facts
        if fact_count == 0:
            return 0.3
        elif fact_count <= 8:
            return 0.7 + 0.3 * (fact_count / 8)
        else:
            # Too many facts might mean we're stuck
            return max(0.5, 1.0 - 0.05 * (fact_count - 8))
    
    def _state_to_dict(self, state: ProofState) -> Dict[str, Any]:
        """Convert a ProofState to a dictionary for the LLM."""
        return {
            "goals": [
                {"statement": str(g.statement), "status": g.status.name}
                for g in state.goals[:10]  # Limit for context
            ],
            "facts": [
                {"statement": str(f.statement), "source": f.source}
                for f in state.facts[:10]
            ],
            "depth": state.depth,
            "candidate_answers": state.candidate_answers,
            "is_solved": state.is_solved(),
            "open_goal_count": len(state.open_goals())
        }
    
    def _generate_explanation(
        self, 
        state: ProofState, 
        components: Dict[str, float]
    ) -> str:
        """Generate a human-readable explanation of the evaluation."""
        parts = []
        
        if components.get("goal_progress", 0) > 0.8:
            parts.append("Strong goal progress")
        elif components.get("goal_progress", 0) < 0.3:
            parts.append("Little goal progress")
        
        if components.get("complexity", 0) > 0.7:
            parts.append("manageable complexity")
        elif components.get("complexity", 0) < 0.3:
            parts.append("high complexity")
        
        if components.get("depth_penalty", 0) < 0.5:
            parts.append("deep proof tree")
        
        return "; ".join(parts) if parts else "Neutral evaluation"
    
    def compare(self, state1: ProofState, state2: ProofState) -> int:
        """
        Compare two states.
        Returns: -1 if state1 better, 1 if state2 better, 0 if equal.
        """
        eval1 = self.evaluate_sync(state1)
        eval2 = self.evaluate_sync(state2)
        
        if eval1.score > eval2.score + 0.05:
            return -1
        elif eval2.score > eval1.score + 0.05:
            return 1
        else:
            return 0

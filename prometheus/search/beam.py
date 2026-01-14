"""
Beam Search: Keep track of multiple promising proof paths.

Beam search is simpler than MCTS but can be effective:
- Maintain a "beam" of the k best states
- At each step, expand all states and keep the top k

This is a good fallback when we need faster, more predictable search.

WHAT THIS FILE DOES:
- Implements beam search through proof space
- Maintains multiple candidate proof paths
- Uses LLM Oracle to rank states
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import asyncio

from prometheus.core.proof_state import ProofState, Goal, GoalStatus
from prometheus.core.tactic import Tactic, TacticResult, TACTIC_REGISTRY
from prometheus.oracle.evaluator import Evaluator


@dataclass
class BeamConfig:
    """Configuration for beam search."""
    
    # Beam parameters
    beam_width: int = 10             # How many states to keep
    max_depth: int = 30              # Maximum proof depth
    max_time_seconds: float = 30.0   # Time limit
    
    # Expansion
    tactics_per_state: int = 5       # How many tactics to try per state
    
    # Pruning
    min_score_threshold: float = 0.1  # Don't keep states below this


@dataclass
class BeamState:
    """A state in the beam with its score and history."""
    state: ProofState
    score: float
    path: List[str]  # Tactics applied to get here
    
    def __lt__(self, other: BeamState) -> bool:
        return self.score < other.score


@dataclass
class BeamResult:
    """Result of beam search."""
    success: bool
    best_state: Optional[ProofState] = None
    proof_path: List[str] = field(default_factory=list)
    depth_reached: int = 0
    time_taken: float = 0.0
    states_explored: int = 0
    candidate_answers: List[Any] = field(default_factory=list)


class BeamSearch:
    """
    Beam search for proof finding.
    
    Simpler than MCTS, keeps a fixed number of best candidates
    and expands them in parallel.
    """
    
    def __init__(
        self,
        config: Optional[BeamConfig] = None,
        evaluator: Optional[Evaluator] = None,
        oracle=None
    ):
        self.config = config or BeamConfig()
        self.evaluator = evaluator or Evaluator()
        self.oracle = oracle
    
    async def search(self, initial_state: ProofState) -> BeamResult:
        """
        Run beam search from an initial proof state.
        """
        start_time = datetime.now()
        states_explored = 0
        
        # Initialize beam with just the starting state
        initial_eval = self.evaluator.evaluate_sync(initial_state)
        beam: List[BeamState] = [
            BeamState(state=initial_state, score=initial_eval.score, path=[])
        ]
        
        # Available tactics
        available_tactics = TACTIC_REGISTRY.list_all()
        
        # Best solution found
        best_solution: Optional[BeamState] = None
        
        for depth in range(self.config.max_depth):
            # Check time limit
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.config.max_time_seconds:
                break
            
            # Check for solutions in current beam
            for beam_state in beam:
                if beam_state.state.is_solved():
                    if best_solution is None or beam_state.score > best_solution.score:
                        best_solution = beam_state
            
            # If we found a solution, we can stop
            if best_solution is not None:
                break
            
            # Expand all states in the beam
            candidates: List[BeamState] = []
            
            for beam_state in beam:
                # Skip terminal states
                if beam_state.state.is_solved() or beam_state.state.is_stuck():
                    candidates.append(beam_state)
                    continue
                
                # Get goal to work on
                goals = beam_state.state.open_goals()
                if not goals:
                    continue
                goal = goals[0]
                
                # Try multiple tactics
                tactics_to_try = self._select_tactics(
                    beam_state.state, 
                    available_tactics,
                    self.config.tactics_per_state
                )
                
                for tactic_name in tactics_to_try:
                    tactic = TACTIC_REGISTRY.get(tactic_name)
                    if not tactic:
                        continue
                    
                    result = tactic.apply(beam_state.state, goal)
                    states_explored += 1
                    
                    if result.succeeded() and result.new_state:
                        # Evaluate new state
                        new_eval = self.evaluator.evaluate_sync(result.new_state)
                        
                        if new_eval.score >= self.config.min_score_threshold:
                            candidates.append(BeamState(
                                state=result.new_state,
                                score=new_eval.score,
                                path=beam_state.path + [tactic_name]
                            ))
            
            # Keep top k candidates
            candidates.sort(reverse=True)  # Higher score first
            beam = candidates[:self.config.beam_width]
            
            # If beam is empty, we're stuck
            if not beam:
                break
        
        # Prepare result
        time_taken = (datetime.now() - start_time).total_seconds()
        
        if best_solution:
            return BeamResult(
                success=True,
                best_state=best_solution.state,
                proof_path=best_solution.path,
                depth_reached=len(best_solution.path),
                time_taken=time_taken,
                states_explored=states_explored,
                candidate_answers=best_solution.state.candidate_answers
            )
        
        # Return best partial result
        if beam:
            best_partial = max(beam, key=lambda s: s.score)
            return BeamResult(
                success=False,
                best_state=best_partial.state,
                proof_path=best_partial.path,
                depth_reached=len(best_partial.path),
                time_taken=time_taken,
                states_explored=states_explored,
                candidate_answers=best_partial.state.candidate_answers
            )
        
        return BeamResult(
            success=False,
            best_state=initial_state,
            proof_path=[],
            depth_reached=0,
            time_taken=time_taken,
            states_explored=states_explored
        )
    
    def _select_tactics(
        self, 
        state: ProofState, 
        available: List[str],
        k: int
    ) -> List[str]:
        """
        Select which tactics to try.
        
        TODO: Use Oracle for smarter selection.
        For now, just return all or first k.
        """
        applicable = TACTIC_REGISTRY.get_applicable(state, state.open_goals()[0]) if state.open_goals() else []
        
        if applicable:
            return [t.name for t in applicable[:k]]
        else:
            return available[:k]

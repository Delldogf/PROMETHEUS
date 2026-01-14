"""
MCTS: Monte Carlo Tree Search for proof finding.

This is inspired by AlphaGo/AlphaProof's approach:
- Tree structure where nodes are proof states
- Edges are tactic applications
- Use LLM to guide which branches to explore

The key insight from MRPPS Q* algorithm:
- Evaluate states not just by depth, but by "promise"
- Balance exploration (trying new things) vs exploitation (following promising paths)

WHAT THIS FILE DOES:
- Implements tree search through proof space
- Uses LLM Oracle to evaluate positions
- Balances exploration vs exploitation with UCB1
- Returns the best proof found
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import math
import random
from datetime import datetime, timedelta
import asyncio

from prometheus.core.proof_state import ProofState, Goal, GoalStatus
from prometheus.core.tactic import Tactic, TacticResult, TACTIC_REGISTRY
from prometheus.core.strategy import Strategy, STRATEGY_REGISTRY
from prometheus.oracle.evaluator import Evaluator, Evaluation


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""
    
    # Search budget
    max_iterations: int = 1000       # Maximum tree expansions
    max_time_seconds: float = 60.0   # Time limit
    max_depth: int = 50              # Maximum proof depth
    
    # MCTS parameters
    exploration_constant: float = 1.414  # UCB1 exploration (sqrt(2) is common)
    
    # Rollout settings
    rollout_depth: int = 5           # How deep to simulate
    num_rollouts: int = 3            # Rollouts per expansion
    
    # Pruning
    min_visit_threshold: int = 5     # Visits before considering pruning
    prune_score_threshold: float = 0.1  # Prune nodes below this score
    
    # Parallelism
    num_parallel_leaves: int = 4     # Virtual loss parallelism


@dataclass
class MCTSNode:
    """
    A node in the MCTS tree.
    
    Each node represents a proof state, and edges to children
    represent tactic applications.
    """
    
    # The proof state at this node
    state: ProofState
    
    # The tactic that led to this state (None for root)
    tactic_used: Optional[str] = None
    tactic_params: Dict[str, Any] = field(default_factory=dict)
    
    # MCTS statistics
    visits: int = 0
    total_value: float = 0.0
    
    # Tree structure
    parent: Optional[MCTSNode] = None
    children: Dict[str, MCTSNode] = field(default_factory=dict)  # tactic_key -> child
    
    # Cache evaluation
    prior_value: Optional[float] = None  # LLM evaluation
    
    # Is this a terminal node?
    is_terminal: bool = False
    is_solved: bool = False
    
    def __post_init__(self):
        self.is_terminal = self.state.is_solved() or self.state.is_stuck()
        self.is_solved = self.state.is_solved()
    
    @property
    def q_value(self) -> float:
        """Average value (exploitation term)."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    def ucb1(self, exploration_constant: float, parent_visits: int) -> float:
        """
        UCB1 score for node selection.
        
        Balances exploitation (high q_value) with exploration (rarely visited).
        """
        if self.visits == 0:
            return float('inf')  # Always try unvisited nodes
        
        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)
        
        return exploitation + exploration
    
    def is_fully_expanded(self, available_tactics: List[str]) -> bool:
        """Check if all tactics have been tried."""
        return len(self.children) >= len(available_tactics)
    
    def best_child(self, exploration_constant: float = 1.414) -> Optional[MCTSNode]:
        """Select the best child using UCB1."""
        if not self.children:
            return None
        
        return max(
            self.children.values(),
            key=lambda c: c.ucb1(exploration_constant, self.visits)
        )
    
    def most_visited_child(self) -> Optional[MCTSNode]:
        """Return the most visited child (used for final selection)."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda c: c.visits)
    
    def __str__(self) -> str:
        return f"MCTSNode(visits={self.visits}, Q={self.q_value:.3f}, tactic={self.tactic_used})"


@dataclass
class MCTSResult:
    """Result of MCTS search."""
    success: bool
    best_state: Optional[ProofState] = None
    proof_path: List[str] = field(default_factory=list)  # Sequence of tactics
    iterations: int = 0
    time_taken: float = 0.0
    nodes_explored: int = 0
    candidate_answers: List[Any] = field(default_factory=list)


class MCTSSearch:
    """
    Monte Carlo Tree Search for proof finding.
    
    This is the main search algorithm, modernized from MRPPS's Q* approach.
    
    The search alternates between:
    1. SELECT: Walk down the tree using UCB1
    2. EXPAND: Add a new child node (try a new tactic)
    3. EVALUATE: Get the LLM's assessment of the new state
    4. BACKPROPAGATE: Update visit counts and values up the tree
    """
    
    def __init__(
        self, 
        config: Optional[MCTSConfig] = None,
        evaluator: Optional[Evaluator] = None,
        oracle=None
    ):
        self.config = config or MCTSConfig()
        self.evaluator = evaluator or Evaluator()
        self.oracle = oracle
        
        # Statistics
        self.nodes_created = 0
        self.evaluations_made = 0
    
    async def search(self, initial_state: ProofState) -> MCTSResult:
        """
        Run MCTS from an initial proof state.
        
        Returns the best result found within the budget.
        """
        start_time = datetime.now()
        
        # Create root node
        root = MCTSNode(state=initial_state)
        self.nodes_created = 1
        
        # Get available tactics
        available_tactics = TACTIC_REGISTRY.list_all()
        
        # Track best solution found
        best_solution: Optional[MCTSNode] = None
        best_score = -1.0
        
        iteration = 0
        while iteration < self.config.max_iterations:
            # Check time limit
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.config.max_time_seconds:
                break
            
            # === SELECT ===
            node = self._select(root)
            
            # === EXPAND ===
            if not node.is_terminal and not node.is_fully_expanded(available_tactics):
                node = await self._expand(node, available_tactics)
            
            # === EVALUATE ===
            if node.prior_value is None:
                value = await self._evaluate(node)
                node.prior_value = value
            else:
                value = node.prior_value
            
            # Check if we found a solution
            if node.is_solved:
                if value > best_score:
                    best_score = value
                    best_solution = node
            
            # === BACKPROPAGATE ===
            self._backpropagate(node, value)
            
            iteration += 1
        
        # Prepare result
        time_taken = (datetime.now() - start_time).total_seconds()
        
        if best_solution:
            return MCTSResult(
                success=True,
                best_state=best_solution.state,
                proof_path=self._extract_path(best_solution),
                iterations=iteration,
                time_taken=time_taken,
                nodes_explored=self.nodes_created,
                candidate_answers=best_solution.state.candidate_answers
            )
        
        # No complete solution - return best partial result
        best_node = self._find_best_node(root)
        return MCTSResult(
            success=False,
            best_state=best_node.state if best_node else initial_state,
            proof_path=self._extract_path(best_node) if best_node else [],
            iterations=iteration,
            time_taken=time_taken,
            nodes_explored=self.nodes_created,
            candidate_answers=best_node.state.candidate_answers if best_node else []
        )
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node to expand using UCB1.
        
        Walk down the tree, always choosing the best child.
        """
        current = node
        
        while current.children and not current.is_terminal:
            best = current.best_child(self.config.exploration_constant)
            if best is None:
                break
            current = best
        
        return current
    
    async def _expand(self, node: MCTSNode, available_tactics: List[str]) -> MCTSNode:
        """
        Expand a node by trying a new tactic.
        """
        # Find tactics not yet tried
        tried = set(node.children.keys())
        untried = [t for t in available_tactics if t not in tried]
        
        if not untried:
            return node  # Fully expanded
        
        # Choose tactic (could use Oracle for smarter selection)
        if self.oracle:
            # Ask LLM which tactic to try
            state_dict = self._state_to_dict(node.state)
            suggestion = await self.oracle.select_tactic(state_dict, untried)
            tactic_name = suggestion.get("tactic", untried[0])
            params = suggestion.get("parameters", {})
        else:
            # Random selection
            tactic_name = random.choice(untried)
            params = {}
        
        # Get the tactic and apply it
        tactic = TACTIC_REGISTRY.get(tactic_name)
        if not tactic:
            # Mark as tried but failed
            node.children[tactic_name] = MCTSNode(
                state=node.state.clone(),
                tactic_used=tactic_name,
                parent=node,
                is_terminal=True
            )
            return node.children[tactic_name]
        
        # Try to apply the tactic
        goal = node.state.open_goals()[0] if node.state.open_goals() else None
        if goal is None:
            return node
        
        result = tactic.apply(node.state, goal, **params)
        
        if result.succeeded() and result.new_state:
            child = MCTSNode(
                state=result.new_state,
                tactic_used=tactic_name,
                tactic_params=params,
                parent=node
            )
            node.children[tactic_name] = child
            self.nodes_created += 1
            return child
        else:
            # Failed application - create a "dead" node
            child = MCTSNode(
                state=node.state.clone(),
                tactic_used=tactic_name,
                parent=node,
                is_terminal=True
            )
            child.prior_value = 0.1  # Low value for failures
            node.children[tactic_name] = child
            return child
    
    async def _evaluate(self, node: MCTSNode) -> float:
        """
        Evaluate a node using the Evaluator.
        """
        if node.is_solved:
            return 1.0  # Perfect score for solutions
        if node.is_terminal and not node.is_solved:
            return 0.0  # Dead end
        
        if self.evaluator:
            evaluation = await self.evaluator.evaluate(node.state)
            self.evaluations_made += 1
            return evaluation.score
        else:
            # Simple heuristic if no evaluator
            open_goals = len(node.state.open_goals())
            proven = sum(1 for g in node.state.goals if g.status == GoalStatus.PROVEN)
            total = len(node.state.goals)
            
            if total == 0:
                return 0.5
            return proven / total
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagate the value up the tree.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent
    
    def _extract_path(self, node: MCTSNode) -> List[str]:
        """Extract the sequence of tactics from root to node."""
        path = []
        current = node
        while current.parent is not None:
            if current.tactic_used:
                path.append(current.tactic_used)
            current = current.parent
        path.reverse()
        return path
    
    def _find_best_node(self, root: MCTSNode) -> Optional[MCTSNode]:
        """Find the best node in the tree (most promising partial solution)."""
        best = root
        best_score = root.q_value
        
        def traverse(node: MCTSNode):
            nonlocal best, best_score
            if node.q_value > best_score:
                best = node
                best_score = node.q_value
            for child in node.children.values():
                traverse(child)
        
        traverse(root)
        return best
    
    def _state_to_dict(self, state: ProofState) -> Dict[str, Any]:
        """Convert state to dict for Oracle."""
        return {
            "goals": [
                {"statement": str(g.statement), "status": g.status.name}
                for g in state.goals[:10]
            ],
            "facts": [
                {"statement": str(f.statement), "source": f.source}
                for f in state.facts[:10]
            ],
            "depth": state.depth,
            "candidate_answers": state.candidate_answers
        }

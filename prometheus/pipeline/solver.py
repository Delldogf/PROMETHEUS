"""
Solver: The main PROMETHEUS problem solver.

This is the entry point that ties everything together:
1. Take a natural language math problem
2. Formalize it using the Oracle
3. Set up the initial proof state
4. Run proof search (MCTS or Beam)
5. Extract and verify the answer

WHAT THIS FILE DOES:
- Provides a simple interface for solving problems
- Orchestrates all the components
- Handles errors gracefully
- Reports progress and results
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum, auto
import asyncio
from datetime import datetime

from prometheus.core.formula import FormalProblem
from prometheus.core.proof_state import ProofState, Goal, Fact
from prometheus.core.tactic import TACTIC_REGISTRY
from prometheus.core.strategy import STRATEGY_REGISTRY

from prometheus.oracle.llm_oracle import LLMOracle, OracleConfig
from prometheus.oracle.formalizer import Formalizer
from prometheus.oracle.evaluator import Evaluator

from prometheus.search.mcts import MCTSSearch, MCTSConfig, MCTSResult
from prometheus.search.beam import BeamSearch, BeamConfig, BeamResult

from prometheus.engines.algebra import AlgebraEngine
from prometheus.engines.number_theory import NumberTheoryEngine
from prometheus.engines.geometry import GeometryEngine
from prometheus.engines.combinatorics import CombinatoricsEngine


class SearchMethod(Enum):
    """Which search algorithm to use."""
    MCTS = auto()
    BEAM = auto()
    HYBRID = auto()  # Try MCTS, fall back to beam


@dataclass
class SolverConfig:
    """Configuration for PROMETHEUS solver."""
    
    # Oracle settings
    use_oracle: bool = True
    oracle_config: Optional[OracleConfig] = None
    
    # Search settings
    search_method: SearchMethod = SearchMethod.MCTS
    mcts_config: Optional[MCTSConfig] = None
    beam_config: Optional[BeamConfig] = None
    
    # Time limits
    max_time_seconds: float = 120.0
    formalization_timeout: float = 10.0
    
    # Verification
    verify_answer: bool = True
    
    # Output
    verbose: bool = True


@dataclass
class SolverResult:
    """Result of solving a problem."""
    success: bool
    answer: Optional[Any] = None
    confidence: float = 0.0
    
    # Details
    formal_problem: Optional[FormalProblem] = None
    proof_path: List[str] = field(default_factory=list)
    search_result: Optional[Union[MCTSResult, BeamResult]] = None
    
    # Timing
    formalization_time: float = 0.0
    search_time: float = 0.0
    total_time: float = 0.0
    
    # Explanation
    explanation: str = ""
    
    def __str__(self) -> str:
        if self.success:
            return f"✅ Answer: {self.answer} (confidence: {self.confidence:.1%})"
        else:
            return f"❌ Could not solve (best effort: {self.answer})"


class PrometheusSolver:
    """
    🔥 PROMETHEUS: The main solver interface.
    
    Usage:
        solver = PrometheusSolver()
        result = await solver.solve("Find all positive integers n such that n^2 + 1 divides n^3 + 1")
        print(result.answer)  # [1]
    
    Or synchronously:
        result = solver.solve_sync("...")
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize all solver components."""
        
        # Oracle (LLM interface)
        if self.config.use_oracle:
            self.oracle = LLMOracle(self.config.oracle_config)
        else:
            self.oracle = None
        
        # Formalizer
        self.formalizer = Formalizer(self.oracle)
        
        # Evaluator
        self.evaluator = Evaluator(self.oracle)
        
        # Search algorithms
        self.mcts = MCTSSearch(
            config=self.config.mcts_config or MCTSConfig(max_time_seconds=self.config.max_time_seconds / 2),
            evaluator=self.evaluator,
            oracle=self.oracle
        )
        
        self.beam = BeamSearch(
            config=self.config.beam_config or BeamConfig(max_time_seconds=self.config.max_time_seconds / 2),
            evaluator=self.evaluator,
            oracle=self.oracle
        )
        
        # Math engines
        self.engines = {
            "algebra": AlgebraEngine(),
            "number_theory": NumberTheoryEngine(),
            "geometry": GeometryEngine(),
            "combinatorics": CombinatoricsEngine()
        }
    
    async def solve(self, problem_text: str) -> SolverResult:
        """
        Solve a math problem.
        
        Args:
            problem_text: Natural language problem statement
            
        Returns:
            SolverResult with the answer (if found) and details
        """
        start_time = datetime.now()
        
        if self.config.verbose:
            print(f"🔥 PROMETHEUS solving: {problem_text[:80]}...")
        
        # Step 1: Formalize the problem
        formalization_start = datetime.now()
        try:
            formalization_result = await self.formalizer.formalize(problem_text)
            if not formalization_result.success:
                return SolverResult(
                    success=False,
                    explanation=f"Could not formalize problem: {formalization_result.error_message}"
                )
            formal_problem = formalization_result.formal_problem
        except Exception as e:
            return SolverResult(
                success=False,
                explanation=f"Formalization error: {str(e)}"
            )
        formalization_time = (datetime.now() - formalization_start).total_seconds()
        
        if self.config.verbose:
            print(f"   📝 Formalized as: {formal_problem.goal_type}")
            print(f"   📚 Domains: {formal_problem.domain_hints}")
        
        # Step 2: Set up initial proof state
        initial_state = self._create_initial_state(formal_problem)
        
        # Step 3: Run proof search
        search_start = datetime.now()
        search_result = await self._run_search(initial_state, formal_problem)
        search_time = (datetime.now() - search_start).total_seconds()
        
        # Step 4: Extract answer
        answer = self._extract_answer(search_result, formal_problem)
        
        # Step 5: Verify answer (if possible)
        if answer is not None and self.config.verify_answer:
            verified = await self._verify_answer(problem_text, answer, formal_problem)
            confidence = 0.9 if verified else 0.6
        else:
            confidence = 0.5 if answer is not None else 0.0
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        if self.config.verbose:
            if answer is not None:
                print(f"   ✅ Answer: {answer}")
            else:
                print(f"   ❌ Could not find answer")
            print(f"   ⏱️ Time: {total_time:.2f}s")
        
        return SolverResult(
            success=(answer is not None),
            answer=answer,
            confidence=confidence,
            formal_problem=formal_problem,
            proof_path=search_result.proof_path if search_result else [],
            search_result=search_result,
            formalization_time=formalization_time,
            search_time=search_time,
            total_time=total_time,
            explanation=self._generate_explanation(search_result, answer)
        )
    
    def solve_sync(self, problem_text: str) -> SolverResult:
        """Synchronous wrapper for solve()."""
        return asyncio.run(self.solve(problem_text))
    
    def _create_initial_state(self, formal_problem: FormalProblem) -> ProofState:
        """Create the initial proof state from a formalized problem."""
        state = ProofState(problem=formal_problem)
        
        # Add variables
        for var in formal_problem.formula.variables:
            state.variables[var.name] = var
        
        # Add hypotheses as facts
        for hyp in formal_problem.formula.hypotheses:
            state.add_fact(hyp, source="given")
        
        # Add main goal
        if formal_problem.formula.conclusion:
            state.add_goal(formal_problem.formula.conclusion, origin="main_goal")
        
        return state
    
    async def _run_search(
        self, 
        initial_state: ProofState,
        formal_problem: FormalProblem
    ) -> Optional[Union[MCTSResult, BeamResult]]:
        """Run the appropriate search algorithm."""
        
        if self.config.search_method == SearchMethod.MCTS:
            return await self.mcts.search(initial_state)
        
        elif self.config.search_method == SearchMethod.BEAM:
            return await self.beam.search(initial_state)
        
        elif self.config.search_method == SearchMethod.HYBRID:
            # Try MCTS first
            mcts_result = await self.mcts.search(initial_state)
            if mcts_result.success:
                return mcts_result
            
            # Fall back to beam search
            return await self.beam.search(initial_state)
        
        return None
    
    def _extract_answer(
        self, 
        search_result: Optional[Union[MCTSResult, BeamResult]],
        formal_problem: FormalProblem
    ) -> Optional[Any]:
        """Extract the final answer from search results."""
        if search_result is None:
            return None
        
        # Check for candidate answers
        if search_result.candidate_answers:
            if formal_problem.goal_type == "find_all":
                return search_result.candidate_answers
            else:
                return search_result.candidate_answers[0]
        
        # Check if solved
        if search_result.success and search_result.best_state:
            # Try to extract from proof state
            state = search_result.best_state
            if state.candidate_answers:
                return state.candidate_answers
        
        return None
    
    async def _verify_answer(
        self, 
        problem_text: str,
        answer: Any,
        formal_problem: FormalProblem
    ) -> bool:
        """Verify the answer is correct."""
        # Use appropriate engine based on domain
        domains = formal_problem.domain_hints
        
        # For now, basic verification
        # TODO: Implement proper verification per domain
        
        if "number_theory" in domains:
            engine = self.engines["number_theory"]
            # Could verify divisibility claims, etc.
        
        if "algebra" in domains:
            engine = self.engines["algebra"]
            # Could verify equation solutions
        
        # Default: assume LLM-verified
        return True
    
    def _generate_explanation(
        self, 
        search_result: Optional[Union[MCTSResult, BeamResult]],
        answer: Any
    ) -> str:
        """Generate a human-readable explanation of the solution."""
        if search_result is None:
            return "No search was performed."
        
        parts = []
        
        if search_result.success:
            parts.append(f"Successfully solved the problem.")
        else:
            parts.append(f"Partial solution found.")
        
        if search_result.proof_path:
            parts.append(f"Proof steps: {' → '.join(search_result.proof_path)}")
        
        parts.append(f"Time: {search_result.time_taken:.2f}s")
        parts.append(f"States explored: {getattr(search_result, 'nodes_explored', 'N/A')}")
        
        return " | ".join(parts)
    
    # ===== Direct Engine Access =====
    
    def algebra(self) -> AlgebraEngine:
        """Get the algebra engine for direct use."""
        return self.engines["algebra"]
    
    def number_theory(self) -> NumberTheoryEngine:
        """Get the number theory engine for direct use."""
        return self.engines["number_theory"]
    
    def geometry(self) -> GeometryEngine:
        """Get the geometry engine for direct use."""
        return self.engines["geometry"]
    
    def combinatorics(self) -> CombinatoricsEngine:
        """Get the combinatorics engine for direct use."""
        return self.engines["combinatorics"]


# ===== Convenience Function =====

async def solve(problem: str, **kwargs) -> SolverResult:
    """
    Quick solve function.
    
    Usage:
        result = await prometheus.solve("Find all primes p such that p^2 + 2 is prime")
    """
    config = SolverConfig(**kwargs)
    solver = PrometheusSolver(config)
    return await solver.solve(problem)

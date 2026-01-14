"""
AIMO Adapter: Format conversion for the AIMO 3 competition.

The AIMO competition has specific input/output formats.
This adapter handles:
- Parsing competition problem format
- Formatting answers for submission
- Handling the 50-problem test set format

WHAT THIS FILE DOES:
- Converts between AIMO format and PROMETHEUS format
- Ensures answers meet competition requirements
- Provides batch solving interface
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import re
import asyncio

from prometheus.pipeline.solver import PrometheusSolver, SolverConfig, SolverResult


@dataclass
class AIMOProblem:
    """A problem in AIMO competition format."""
    id: str
    problem_text: str
    category: Optional[str] = None  # algebra, number_theory, geometry, combinatorics
    difficulty: Optional[int] = None  # estimated difficulty 1-10


@dataclass
class AIMOSubmission:
    """A submission in AIMO competition format."""
    id: str
    answer: int  # AIMO expects integer answers
    confidence: float = 1.0


class AIMOAdapter:
    """
    Adapter for AIMO 3 competition format.
    
    Handles:
    - Reading problem sets
    - Formatting submissions
    - Batch solving
    """
    
    def __init__(self, solver: Optional[PrometheusSolver] = None):
        self.solver = solver or PrometheusSolver()
    
    def parse_problem(self, problem_dict: Dict[str, Any]) -> AIMOProblem:
        """
        Parse a problem from AIMO dataset format.
        
        Expected format:
        {
            "id": "problem_1",
            "problem": "Find all positive integers...",
            "category": "number_theory"  # optional
        }
        """
        return AIMOProblem(
            id=str(problem_dict.get("id", "unknown")),
            problem_text=problem_dict.get("problem", ""),
            category=problem_dict.get("category"),
            difficulty=problem_dict.get("difficulty")
        )
    
    def format_answer(self, result: SolverResult) -> int:
        """
        Format a solver result as an AIMO answer.
        
        AIMO expects integer answers (or list of integers for "find all").
        For "find all" problems, we typically return the sum or count.
        """
        if result.answer is None:
            return 0  # Default for failed solutions
        
        answer = result.answer
        
        # Handle list answers (e.g., "find all n")
        if isinstance(answer, list):
            if len(answer) == 0:
                return 0
            elif len(answer) == 1:
                return self._to_int(answer[0])
            else:
                # For multiple answers, return sum (common convention)
                # Or could return count depending on problem
                try:
                    return sum(self._to_int(x) for x in answer)
                except (ValueError, TypeError):
                    return len(answer)  # Return count if sum fails
        
        return self._to_int(answer)
    
    def _to_int(self, value: Any) -> int:
        """Convert a value to integer."""
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                try:
                    return int(float(value))
                except ValueError:
                    return 0
        return 0
    
    async def solve_problem(self, problem: AIMOProblem) -> AIMOSubmission:
        """
        Solve a single AIMO problem.
        """
        result = await self.solver.solve(problem.problem_text)
        answer = self.format_answer(result)
        
        return AIMOSubmission(
            id=problem.id,
            answer=answer,
            confidence=result.confidence
        )
    
    async def solve_batch(
        self, 
        problems: List[AIMOProblem],
        max_concurrent: int = 4
    ) -> List[AIMOSubmission]:
        """
        Solve a batch of problems with controlled concurrency.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def solve_with_limit(problem: AIMOProblem) -> AIMOSubmission:
            async with semaphore:
                return await self.solve_problem(problem)
        
        tasks = [solve_with_limit(p) for p in problems]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        submissions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle failures
                submissions.append(AIMOSubmission(
                    id=problems[i].id,
                    answer=0,
                    confidence=0.0
                ))
            else:
                submissions.append(result)
        
        return submissions
    
    def create_submission_csv(
        self, 
        submissions: List[AIMOSubmission],
        output_path: str = "submission.csv"
    ) -> str:
        """
        Create a CSV file for Kaggle submission.
        
        Format:
        id,answer
        problem_1,42
        problem_2,17
        ...
        """
        lines = ["id,answer"]
        for sub in submissions:
            lines.append(f"{sub.id},{sub.answer}")
        
        csv_content = "\n".join(lines)
        
        with open(output_path, "w") as f:
            f.write(csv_content)
        
        return output_path
    
    def estimate_difficulty(self, problem_text: str) -> int:
        """
        Estimate problem difficulty (1-10).
        
        Uses heuristics based on problem text.
        """
        text_lower = problem_text.lower()
        
        # Length is a rough indicator
        length_score = min(len(problem_text) / 200, 3)
        
        # Keyword indicators
        hard_keywords = ["prove that", "show that", "determine all", "find all", 
                        "characterize", "classify"]
        easy_keywords = ["calculate", "compute", "find the value", "what is"]
        
        keyword_score = 0
        for kw in hard_keywords:
            if kw in text_lower:
                keyword_score += 1
        for kw in easy_keywords:
            if kw in text_lower:
                keyword_score -= 0.5
        
        # Topic complexity
        complex_topics = ["elliptic", "projective", "galois", "modular form",
                         "diophantine", "functional equation"]
        topic_score = sum(1 for t in complex_topics if t in text_lower)
        
        total = 3 + length_score + keyword_score + topic_score
        return max(1, min(10, int(total)))


# ===== Kaggle Integration =====

def create_kaggle_submission(
    test_data: List[Dict[str, Any]],
    output_path: str = "submission.csv",
    verbose: bool = True
) -> str:
    """
    Create a Kaggle submission from test data.
    
    Usage in Kaggle notebook:
        import prometheus
        submission = prometheus.create_kaggle_submission(test.to_dict('records'))
    """
    adapter = AIMOAdapter()
    
    # Parse problems
    problems = [adapter.parse_problem(p) for p in test_data]
    
    if verbose:
        print(f"🔥 PROMETHEUS solving {len(problems)} problems...")
    
    # Solve all problems
    submissions = asyncio.run(adapter.solve_batch(problems))
    
    # Create submission file
    output = adapter.create_submission_csv(submissions, output_path)
    
    if verbose:
        # Report statistics
        solved = sum(1 for s in submissions if s.confidence > 0.5)
        print(f"   ✅ Solved with confidence: {solved}/{len(problems)}")
        print(f"   📄 Submission saved to: {output}")
    
    return output

"""
Formalizer: Convert natural language math problems to formal representations.

This is the "translation layer" that bridges human-readable problems
and PROMETHEUS's internal representation.

WHAT THIS FILE DOES:
- Parses natural language math problems
- Extracts variables, constraints, and goals
- Creates FormalProblem objects for the proof engine
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re

from prometheus.core.formula import (
    Formula, Variable, Expression, Constraint,
    MathDomain, RelationType, FormalProblem
)


@dataclass
class FormalizationResult:
    """Result of formalizing a problem."""
    success: bool
    formal_problem: Optional[FormalProblem] = None
    raw_llm_output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    confidence: float = 0.0


class Formalizer:
    """
    Converts natural language problems to formal representations.
    
    Two modes:
    1. LLM-assisted: Use the Oracle for complex problems
    2. Rule-based: Use pattern matching for common formats
    """
    
    # Pattern matching for common problem types
    PROBLEM_PATTERNS = {
        "find_all": [
            r"find all (?:positive )?(?:integers?|numbers?) (\w+)",
            r"determine all (\w+)",
            r"for which (?:values of )?(\w+)",
        ],
        "prove": [
            r"prove that",
            r"show that",
            r"demonstrate that",
        ],
        "compute": [
            r"calculate",
            r"compute",
            r"find the value of",
            r"evaluate",
        ],
        "find_minimum": [
            r"find the minimum",
            r"minimize",
            r"smallest (?:value|number)",
        ],
        "find_maximum": [
            r"find the maximum",
            r"maximize",
            r"largest (?:value|number)",
        ],
    }
    
    DOMAIN_PATTERNS = {
        MathDomain.POSITIVE_INTEGERS: [
            r"positive integers?",
            r"natural numbers?",
            r"n (?:≥|>=|>) 0",
        ],
        MathDomain.INTEGERS: [
            r"integers?",
            r"whole numbers?",
        ],
        MathDomain.REALS: [
            r"real numbers?",
            r"reals?",
        ],
        MathDomain.RATIONALS: [
            r"rationals?",
            r"fractions?",
        ],
    }
    
    def __init__(self, oracle=None):
        """
        Initialize the Formalizer.
        
        Args:
            oracle: Optional LLMOracle for complex problems
        """
        self.oracle = oracle
    
    def detect_problem_type(self, text: str) -> str:
        """Detect what kind of problem this is."""
        text_lower = text.lower()
        
        for problem_type, patterns in self.PROBLEM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return problem_type
        
        return "unknown"
    
    def detect_domain(self, text: str) -> MathDomain:
        """Detect the mathematical domain of the variable."""
        text_lower = text.lower()
        
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return domain
        
        return MathDomain.REALS  # Default
    
    def extract_variables(self, text: str) -> List[Variable]:
        """Extract variable names from problem text."""
        # Common single-letter variables in math
        potential_vars = set(re.findall(r'\b([a-z])\b', text.lower()))
        
        # Remove common non-variable letters
        non_vars = {'a', 'i', 'e'}  # articles, common words
        potential_vars -= non_vars
        
        domain = self.detect_domain(text)
        
        return [Variable(name=v, domain=domain) for v in sorted(potential_vars)]
    
    def detect_math_domains(self, text: str) -> List[str]:
        """Detect which mathematical areas this problem involves."""
        text_lower = text.lower()
        domains = []
        
        # Number theory indicators
        nt_patterns = ["divides", "divisible", "mod", "prime", "gcd", "lcm", 
                       "congruent", "coprime", "factor"]
        if any(p in text_lower for p in nt_patterns):
            domains.append("number_theory")
        
        # Algebra indicators
        alg_patterns = ["polynomial", "equation", "quadratic", "root", 
                        "expression", "simplify", "expand", "factor"]
        if any(p in text_lower for p in alg_patterns):
            domains.append("algebra")
        
        # Geometry indicators
        geo_patterns = ["triangle", "circle", "angle", "point", "line",
                        "perpendicular", "parallel", "inscribed", "area"]
        if any(p in text_lower for p in geo_patterns):
            domains.append("geometry")
        
        # Combinatorics indicators
        combo_patterns = ["count", "arrange", "choose", "permutation",
                          "combination", "sequence", "subset", "partition"]
        if any(p in text_lower for p in combo_patterns):
            domains.append("combinatorics")
        
        # Default to algebra if nothing detected
        if not domains:
            domains.append("algebra")
        
        return domains
    
    async def formalize(self, problem_text: str) -> FormalizationResult:
        """
        Formalize a problem.
        
        Uses LLM if available, otherwise falls back to rule-based parsing.
        """
        # Try LLM-assisted formalization first
        if self.oracle:
            try:
                llm_result = await self.oracle.formalize(problem_text)
                if "error" not in llm_result:
                    formal_problem = self._build_from_llm(problem_text, llm_result)
                    return FormalizationResult(
                        success=True,
                        formal_problem=formal_problem,
                        raw_llm_output=llm_result,
                        confidence=0.9
                    )
            except Exception as e:
                pass  # Fall back to rule-based
        
        # Rule-based fallback
        return self._rule_based_formalize(problem_text)
    
    def _build_from_llm(
        self, 
        original_text: str, 
        llm_output: Dict[str, Any]
    ) -> FormalProblem:
        """Build a FormalProblem from LLM output."""
        
        # Parse variables
        variables = []
        for v in llm_output.get("variables", []):
            domain_str = v.get("domain", "reals")
            domain_map = {
                "positive_integers": MathDomain.POSITIVE_INTEGERS,
                "integers": MathDomain.INTEGERS,
                "reals": MathDomain.REALS,
                "rationals": MathDomain.RATIONALS,
                "naturals": MathDomain.NATURALS,
            }
            domain = domain_map.get(domain_str, MathDomain.REALS)
            variables.append(Variable(
                name=v.get("name", "x"),
                domain=domain,
                constraints=v.get("constraints", [])
            ))
        
        # Build formula
        formula = Formula(
            variables=variables,
            natural_language=llm_output.get("goal", original_text),
            problem_type=llm_output.get("problem_type", "unknown")
        )
        
        return FormalProblem(
            original_text=original_text,
            formula=formula,
            goal_type=llm_output.get("problem_type", "unknown"),
            answer_format=llm_output.get("suggested_answer_format", "integer"),
            domain_hints=llm_output.get("mathematical_domain", ["algebra"])
        )
    
    def _rule_based_formalize(self, problem_text: str) -> FormalizationResult:
        """Rule-based formalization when LLM isn't available."""
        
        problem_type = self.detect_problem_type(problem_text)
        variables = self.extract_variables(problem_text)
        domains = self.detect_math_domains(problem_text)
        
        formula = Formula(
            variables=variables,
            natural_language=problem_text,
            problem_type=problem_type
        )
        
        formal_problem = FormalProblem(
            original_text=problem_text,
            formula=formula,
            goal_type=problem_type,
            answer_format="integer",  # Default
            domain_hints=domains
        )
        
        return FormalizationResult(
            success=True,
            formal_problem=formal_problem,
            confidence=0.5  # Lower confidence for rule-based
        )

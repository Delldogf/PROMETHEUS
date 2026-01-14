"""
Formula: Mathematical expressions and logical statements.

Think of this like the "language" PROMETHEUS uses internally.
Just as humans write "x² + 1 = 0", PROMETHEUS has its own structured way
to represent mathematical ideas that it can manipulate precisely.

WHAT THIS FILE DOES:
- Defines how we represent mathematical expressions (like x² + 2x + 1)
- Defines how we represent logical statements (like "for all x > 0, x² > 0")
- Defines variables, constants, and constraints
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Union, Set
from abc import ABC, abstractmethod


class MathDomain(Enum):
    """
    What "universe" are we working in?
    
    Think of this like the context for a math problem:
    - INTEGERS: whole numbers (..., -2, -1, 0, 1, 2, ...)
    - POSITIVE_INTEGERS: counting numbers (1, 2, 3, ...)
    - REALS: all real numbers (including decimals, π, √2, etc.)
    - RATIONALS: fractions (1/2, 3/4, -7/3, etc.)
    """
    INTEGERS = auto()
    POSITIVE_INTEGERS = auto()
    NON_NEGATIVE_INTEGERS = auto()
    REALS = auto()
    POSITIVE_REALS = auto()
    RATIONALS = auto()
    NATURALS = auto()  # 0, 1, 2, 3, ...
    COMPLEX = auto()


class ExpressionType(Enum):
    """Types of mathematical expressions."""
    VARIABLE = auto()      # x, y, n
    CONSTANT = auto()      # 1, 2, π
    OPERATION = auto()     # x + y, x * y
    FUNCTION = auto()      # f(x), sin(x), gcd(a,b)
    POWER = auto()         # x^2, 2^n
    RELATION = auto()      # x = y, x < y, x | y (divides)


@dataclass
class Variable:
    """
    A mathematical variable.
    
    Example: In "Find all positive integers n such that...",
    n is a Variable with domain=POSITIVE_INTEGERS
    
    Attributes:
        name: The variable name (like "n", "x", "k")
        domain: What kind of number it is (integer, real, etc.)
        constraints: Additional restrictions (like "n > 5")
    """
    name: str
    domain: MathDomain = MathDomain.REALS
    constraints: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return self.name
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Variable):
            return self.name == other.name
        return False


@dataclass
class Expression:
    """
    A mathematical expression - the building block of formulas.
    
    Examples:
    - Simple: Variable "x"
    - Compound: "x² + 2x + 1"
    - Function: "gcd(a, b)"
    
    This is a tree structure. For "x² + 1":
    - Root: Addition (+)
      - Left child: Power (x, 2)  
      - Right child: Constant (1)
    """
    expr_type: ExpressionType
    value: Any  # Could be number, string, operator symbol
    children: List[Expression] = field(default_factory=list)
    
    # Store the original string representation for debugging
    original: Optional[str] = None
    
    @classmethod
    def variable(cls, name: str) -> Expression:
        """Create a variable expression."""
        return cls(ExpressionType.VARIABLE, name)
    
    @classmethod
    def constant(cls, value: Union[int, float, str]) -> Expression:
        """Create a constant expression."""
        return cls(ExpressionType.CONSTANT, value)
    
    @classmethod
    def operation(cls, operator: str, left: Expression, right: Expression) -> Expression:
        """Create a binary operation (like +, -, *, /)."""
        return cls(ExpressionType.OPERATION, operator, [left, right])
    
    @classmethod
    def function(cls, name: str, *args: Expression) -> Expression:
        """Create a function application (like gcd(a,b), sin(x))."""
        return cls(ExpressionType.FUNCTION, name, list(args))
    
    @classmethod
    def power(cls, base: Expression, exponent: Expression) -> Expression:
        """Create a power expression (like x^2, 2^n)."""
        return cls(ExpressionType.POWER, "^", [base, exponent])
    
    def __str__(self) -> str:
        if self.original:
            return self.original
        if self.expr_type == ExpressionType.VARIABLE:
            return str(self.value)
        elif self.expr_type == ExpressionType.CONSTANT:
            return str(self.value)
        elif self.expr_type == ExpressionType.OPERATION:
            return f"({self.children[0]} {self.value} {self.children[1]})"
        elif self.expr_type == ExpressionType.FUNCTION:
            args = ", ".join(str(c) for c in self.children)
            return f"{self.value}({args})"
        elif self.expr_type == ExpressionType.POWER:
            return f"({self.children[0]})^({self.children[1]})"
        return repr(self)
    
    def get_variables(self) -> Set[str]:
        """Find all variables in this expression."""
        if self.expr_type == ExpressionType.VARIABLE:
            return {str(self.value)}
        elif self.expr_type == ExpressionType.CONSTANT:
            return set()
        else:
            result = set()
            for child in self.children:
                result.update(child.get_variables())
            return result


class RelationType(Enum):
    """Types of mathematical relations."""
    EQUALS = "="
    NOT_EQUALS = "≠"
    LESS_THAN = "<"
    LESS_EQUAL = "≤"
    GREATER_THAN = ">"
    GREATER_EQUAL = "≥"
    DIVIDES = "|"          # a | b means "a divides b"
    CONGRUENT = "≡"        # a ≡ b (mod m)
    ELEMENT_OF = "∈"       # x ∈ S


@dataclass
class Constraint:
    """
    A constraint on variables - a statement that must be true.
    
    Examples:
    - "n > 0" (positivity constraint)
    - "x² + y² = 1" (equation constraint)
    - "a | b" (divisibility constraint)
    """
    left: Expression
    relation: RelationType
    right: Expression
    modulus: Optional[Expression] = None  # For congruences: a ≡ b (mod m)
    
    def __str__(self) -> str:
        base = f"{self.left} {self.relation.value} {self.right}"
        if self.modulus:
            base += f" (mod {self.modulus})"
        return base


class QuantifierType(Enum):
    """Logical quantifiers."""
    FOR_ALL = "∀"      # "for all x"
    EXISTS = "∃"       # "there exists x"
    EXISTS_UNIQUE = "∃!"  # "there exists a unique x"


@dataclass 
class Formula:
    """
    A complete mathematical formula or statement.
    
    This is the main unit PROMETHEUS works with. It combines:
    - Variables with their domains
    - Constraints (what must be true)
    - The main statement to prove or the question to answer
    
    Example: "For all positive integers n, if n² + 1 divides n³ + 1, 
              then n ∈ {1}"
    
    Would be represented as:
    - variables: [n with domain POSITIVE_INTEGERS]
    - hypotheses: [Constraint(n² + 1 | n³ + 1)]
    - conclusion: n = 1
    """
    
    # The variables involved
    variables: List[Variable] = field(default_factory=list)
    
    # Assumptions/hypotheses (things we're told are true)
    hypotheses: List[Constraint] = field(default_factory=list)
    
    # What we want to prove or find
    conclusion: Optional[Constraint] = None
    
    # Quantifier structure (e.g., "for all n, there exists k")
    quantifiers: List[tuple[QuantifierType, Variable]] = field(default_factory=list)
    
    # Natural language description (for debugging/display)
    natural_language: Optional[str] = None
    
    # What type of problem is this?
    problem_type: Optional[str] = None  # "find_all", "prove", "compute"
    
    def __str__(self) -> str:
        parts = []
        
        # Quantifiers
        for qtype, var in self.quantifiers:
            parts.append(f"{qtype.value}{var}")
        
        # Hypotheses
        if self.hypotheses:
            hyp_str = " ∧ ".join(str(h) for h in self.hypotheses)
            parts.append(f"[{hyp_str}]")
        
        # Conclusion
        if self.conclusion:
            parts.append(f"⟹ {self.conclusion}")
        
        if parts:
            return " ".join(parts)
        elif self.natural_language:
            return self.natural_language
        else:
            return "Formula()"
    
    def get_all_variables(self) -> Set[str]:
        """Get all variable names in this formula."""
        result = {v.name for v in self.variables}
        for h in self.hypotheses:
            result.update(h.left.get_variables())
            result.update(h.right.get_variables())
        if self.conclusion:
            result.update(self.conclusion.left.get_variables())
            result.update(self.conclusion.right.get_variables())
        return result


@dataclass
class FormalProblem:
    """
    A fully formalized math problem ready for PROMETHEUS to solve.
    
    This is what we get after the LLM parses the natural language problem.
    
    Attributes:
        original_text: The original problem statement
        formula: The formal logical representation
        goal_type: What kind of answer we need (find_all, prove, compute_value)
        answer_format: How to format the final answer
    """
    original_text: str
    formula: Formula
    goal_type: str  # "find_all", "prove", "compute", "find_minimum", etc.
    answer_format: str = "integer"  # "integer", "list", "expression"
    domain_hints: List[str] = field(default_factory=list)  # ["number_theory", "algebra"]
    difficulty_estimate: Optional[int] = None  # 1-10 scale
    
    def __str__(self) -> str:
        return f"FormalProblem({self.goal_type}): {self.formula}"

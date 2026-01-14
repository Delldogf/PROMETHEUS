"""
Techniques: Problem-solving techniques and strategies.

While theorems are specific mathematical facts,
techniques are general problem-solving approaches.

Examples:
- "Check small cases first"
- "Work backwards from the answer"
- "Introduce auxiliary elements"

WHAT THIS FILE DOES:
- Catalogs problem-solving techniques
- Suggests techniques based on problem type
- Provides guidance on applying techniques
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Technique:
    """
    A problem-solving technique.
    
    Attributes:
        name: Short identifier
        title: Full name
        description: What it is and when to use it
        steps: How to apply the technique
        applicable_to: Problem types it works for
        keywords: Terms suggesting this technique
        examples: Example applications
    """
    name: str
    title: str
    description: str
    steps: List[str] = field(default_factory=list)
    applicable_to: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    importance: int = 5  # 1-10


# ===== Technique Library =====

TECHNIQUES = [
    # ===== General Problem Solving =====
    Technique(
        name="small_cases",
        title="Check Small Cases",
        description="Try small values first to understand the pattern",
        steps=[
            "Substitute n=1, n=2, n=3, etc.",
            "Look for patterns in the results",
            "Form a conjecture",
            "Prove the conjecture holds generally"
        ],
        applicable_to=["find_all", "find_formula", "prove_forall"],
        keywords=["positive integers", "for all", "find all", "pattern"],
        examples=["Find all n such that n! + 1 is a perfect square"],
        importance=10
    ),
    
    Technique(
        name="work_backwards",
        title="Work Backwards",
        description="Start from the desired conclusion and work towards the hypotheses",
        steps=[
            "Assume what you want to prove",
            "Derive necessary conditions",
            "Check if those conditions are satisfied",
            "Reverse the logic for the proof"
        ],
        applicable_to=["prove", "find_conditions"],
        keywords=["show that", "prove that", "if and only if"],
        importance=8
    ),
    
    Technique(
        name="auxiliary_element",
        title="Introduce Auxiliary Element",
        description="Add a helpful construction (point, line, variable) that isn't in the original problem",
        steps=[
            "Identify what relationship would help",
            "Construct an element exhibiting that relationship",
            "Use the new element to bridge the proof"
        ],
        applicable_to=["geometry", "algebra"],
        keywords=["construct", "let", "define", "consider"],
        importance=8
    ),
    
    # ===== Number Theory Techniques =====
    Technique(
        name="mod_analysis",
        title="Modular Analysis",
        description="Analyze the problem modulo various numbers",
        steps=[
            "Identify relevant moduli (often small primes)",
            "Reduce the equation/expression mod m",
            "Look for contradictions or restrictions",
            "Combine restrictions to limit solutions"
        ],
        applicable_to=["number_theory", "diophantine"],
        keywords=["divisible", "remainder", "mod", "congruent"],
        examples=["Check n³ + n² + n + 1 mod 2, 3, 4 to find constraints"],
        importance=9
    ),
    
    Technique(
        name="infinite_descent",
        title="Infinite Descent",
        description="Prove no solution exists by showing any solution implies a smaller solution",
        steps=[
            "Assume a solution exists",
            "Derive a strictly smaller solution",
            "This contradicts minimality, so no solution exists"
        ],
        applicable_to=["prove_nonexistence", "diophantine"],
        keywords=["no solution", "impossible", "fermat"],
        importance=7
    ),
    
    Technique(
        name="factorization_trick",
        title="Clever Factorization",
        description="Factor expressions in non-obvious ways",
        steps=[
            "Look for patterns like a² - b², SFFT, Sophie Germain",
            "Apply the factorization",
            "Analyze factors"
        ],
        applicable_to=["number_theory", "algebra"],
        keywords=["factor", "divisor", "product"],
        examples=["a⁴ + 4b⁴ = (a²+2b²+2ab)(a²+2b²-2ab)"],
        importance=8
    ),
    
    # ===== Algebra Techniques =====
    Technique(
        name="substitution",
        title="Variable Substitution",
        description="Replace complex expressions with simpler variables",
        steps=[
            "Identify repeated or complex expressions",
            "Substitute u = expression",
            "Solve in terms of u",
            "Substitute back"
        ],
        applicable_to=["algebra", "equations"],
        keywords=["let", "substitute", "replace", "variable"],
        importance=9
    ),
    
    Technique(
        name="symmetry",
        title="Use Symmetry",
        description="Exploit symmetric structure in the problem",
        steps=[
            "Identify symmetric functions or conditions",
            "Apply symmetric operations",
            "Use that symmetric optima often occur at symmetric points"
        ],
        applicable_to=["algebra", "inequalities", "optimization"],
        keywords=["symmetric", "equal", "interchange", "cyclic"],
        importance=8
    ),
    
    Technique(
        name="homogenization",
        title="Homogenization",
        description="Make an expression homogeneous by scaling",
        steps=[
            "Identify the degrees of terms",
            "Multiply/divide to make all terms same degree",
            "Apply inequality or simplify"
        ],
        applicable_to=["inequalities"],
        keywords=["homogeneous", "degree", "scale"],
        importance=6
    ),
    
    # ===== Geometry Techniques =====
    Technique(
        name="coordinate_bash",
        title="Coordinate Bash",
        description="Set up coordinates and compute algebraically",
        steps=[
            "Choose a good coordinate system (often one vertex at origin)",
            "Express all points in coordinates",
            "Compute the required quantities algebraically",
            "Simplify"
        ],
        applicable_to=["geometry"],
        keywords=["point", "distance", "angle", "verify"],
        importance=8
    ),
    
    Technique(
        name="angle_chasing",
        title="Angle Chasing",
        description="Track angles through the figure using inscribed angles, etc.",
        steps=[
            "Label known angles",
            "Use properties: inscribed angles, supplementary, etc.",
            "Chase angles to relate distant parts of the figure"
        ],
        applicable_to=["geometry"],
        keywords=["angle", "inscribed", "cyclic", "equal"],
        importance=8
    ),
    
    Technique(
        name="power_of_point",
        title="Power of a Point",
        description="Use the power of a point theorem for circles",
        steps=[
            "Identify the point and circle(s)",
            "Write power equations for intersecting chords/secants",
            "Relate the powers"
        ],
        applicable_to=["geometry"],
        keywords=["circle", "power", "chord", "tangent", "secant"],
        importance=7
    ),
    
    # ===== Combinatorics Techniques =====
    Technique(
        name="bijection",
        title="Bijection/Counting Correspondence",
        description="Count one set by finding a bijection to an easier set",
        steps=[
            "Identify what you need to count",
            "Find a set in bijection that's easier to count",
            "Establish the bijection explicitly",
            "Count the easier set"
        ],
        applicable_to=["combinatorics", "counting"],
        keywords=["count", "number of", "how many", "bijection"],
        importance=9
    ),
    
    Technique(
        name="double_counting",
        title="Double Counting",
        description="Count the same quantity in two different ways",
        steps=[
            "Identify a quantity to count",
            "Find two different ways to count it",
            "Set them equal for an equation"
        ],
        applicable_to=["combinatorics", "graph_theory"],
        keywords=["count", "sum", "total"],
        importance=8
    ),
    
    Technique(
        name="generating_functions",
        title="Generating Functions",
        description="Encode a sequence as coefficients of a power series",
        steps=[
            "Write the generating function for the sequence",
            "Manipulate the GF algebraically",
            "Extract the coefficient you need"
        ],
        applicable_to=["combinatorics", "recurrences"],
        keywords=["sequence", "recurrence", "coefficient", "series"],
        importance=7
    ),
    
    # ===== Proof Techniques =====
    Technique(
        name="contradiction",
        title="Proof by Contradiction",
        description="Assume the negation and derive something impossible",
        steps=[
            "Assume the opposite of what you want to prove",
            "Derive logical consequences",
            "Reach a contradiction",
            "Conclude the original statement is true"
        ],
        applicable_to=["prove", "prove_nonexistence"],
        keywords=["suppose not", "assume", "contradiction", "impossible"],
        importance=9
    ),
    
    Technique(
        name="induction",
        title="Mathematical Induction",
        description="Prove for all n by proving base case and inductive step",
        steps=[
            "Prove P(1) (or smallest case)",
            "Assume P(k) for some k (induction hypothesis)",
            "Prove P(k+1) using the hypothesis",
            "Conclude P(n) for all n"
        ],
        applicable_to=["prove_forall", "sequences"],
        keywords=["for all", "positive integers", "natural numbers", "recursive"],
        importance=10
    ),
    
    Technique(
        name="extremal_principle",
        title="Extremal Principle",
        description="Consider the minimal or maximal element with some property",
        steps=[
            "Consider the smallest/largest element satisfying X",
            "Derive properties of this extremal element",
            "Use these properties to reach conclusion"
        ],
        applicable_to=["prove", "prove_existence"],
        keywords=["smallest", "largest", "minimum", "maximum", "extremal"],
        importance=7
    ),
]


def get_techniques() -> List[Technique]:
    """Get all available techniques."""
    return TECHNIQUES


def suggest_techniques(problem_text: str, domains: List[str]) -> List[Technique]:
    """Suggest techniques for a given problem."""
    suggestions = []
    text_lower = problem_text.lower()
    
    for technique in TECHNIQUES:
        score = 0
        
        # Domain match
        if any(d in technique.applicable_to for d in domains):
            score += 3
        
        # Keyword match
        if any(kw in text_lower for kw in technique.keywords):
            score += 5
        
        if score > 0:
            suggestions.append((score + technique.importance, technique))
    
    suggestions.sort(reverse=True, key=lambda x: x[0])
    return [t for _, t in suggestions[:5]]

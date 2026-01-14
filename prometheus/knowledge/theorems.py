"""
Theorems: A collection of olympiad-relevant theorems.

This is the mathematical knowledge base - theorems that
PROMETHEUS can apply during proof search.

Each theorem has:
- A formal statement
- Conditions for when it applies
- How to use it in a proof

WHAT THIS FILE DOES:
- Defines a library of useful theorems
- Organizes them by mathematical domain
- Provides lookup by name and by applicability
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum, auto


class TheoremDomain(Enum):
    """Mathematical domains for theorems."""
    ALGEBRA = auto()
    NUMBER_THEORY = auto()
    GEOMETRY = auto()
    COMBINATORICS = auto()
    INEQUALITIES = auto()
    POLYNOMIALS = auto()
    GENERAL = auto()


@dataclass
class Theorem:
    """
    A mathematical theorem in the knowledge base.
    
    Attributes:
        name: Short identifier (e.g., "am_gm")
        title: Full name (e.g., "AM-GM Inequality")
        statement: The theorem statement in natural language
        formal_statement: Formal/symbolic statement
        domains: Which areas it applies to
        keywords: Terms that suggest this theorem
        prerequisites: What must be true to apply it
        produces: What you get after applying it
    """
    name: str
    title: str
    statement: str
    formal_statement: str = ""
    domains: List[TheoremDomain] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    example: str = ""
    importance: int = 5  # 1-10, how often it's useful
    
    def matches_keywords(self, text: str) -> bool:
        """Check if the text contains any of this theorem's keywords."""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.keywords)


class TheoremBase:
    """
    The theorem database.
    
    Stores all theorems and provides lookup methods.
    """
    
    def __init__(self):
        self._theorems: Dict[str, Theorem] = {}
        self._by_domain: Dict[TheoremDomain, List[Theorem]] = {}
        self._load_default_theorems()
    
    def _load_default_theorems(self):
        """Load the built-in theorem library."""
        for theorem in OLYMPIAD_THEOREMS:
            self.add(theorem)
    
    def add(self, theorem: Theorem) -> None:
        """Add a theorem to the database."""
        self._theorems[theorem.name] = theorem
        for domain in theorem.domains:
            if domain not in self._by_domain:
                self._by_domain[domain] = []
            self._by_domain[domain].append(theorem)
    
    def get(self, name: str) -> Optional[Theorem]:
        """Get a theorem by name."""
        return self._theorems.get(name)
    
    def get_by_domain(self, domain: TheoremDomain) -> List[Theorem]:
        """Get all theorems in a domain."""
        return self._by_domain.get(domain, [])
    
    def search(self, query: str) -> List[Theorem]:
        """Search for theorems matching a query."""
        results = []
        query_lower = query.lower()
        
        for theorem in self._theorems.values():
            score = 0
            
            # Name match
            if query_lower in theorem.name.lower():
                score += 10
            
            # Title match
            if query_lower in theorem.title.lower():
                score += 8
            
            # Keyword match
            if theorem.matches_keywords(query):
                score += 5
            
            # Statement match
            if query_lower in theorem.statement.lower():
                score += 3
            
            if score > 0:
                results.append((score, theorem))
        
        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in results]
    
    def suggest_for_problem(self, problem_text: str, domains: List[str]) -> List[Theorem]:
        """Suggest theorems that might be useful for a problem."""
        suggestions = []
        
        # Map string domains to enum
        domain_map = {
            "algebra": TheoremDomain.ALGEBRA,
            "number_theory": TheoremDomain.NUMBER_THEORY,
            "geometry": TheoremDomain.GEOMETRY,
            "combinatorics": TheoremDomain.COMBINATORICS,
        }
        
        # Get theorems from relevant domains
        for domain_str in domains:
            domain = domain_map.get(domain_str)
            if domain:
                suggestions.extend(self.get_by_domain(domain))
        
        # Also check keyword matches
        for theorem in self._theorems.values():
            if theorem.matches_keywords(problem_text) and theorem not in suggestions:
                suggestions.append(theorem)
        
        # Sort by importance
        suggestions.sort(key=lambda t: t.importance, reverse=True)
        
        return suggestions[:10]  # Top 10
    
    def list_all(self) -> List[str]:
        """List all theorem names."""
        return list(self._theorems.keys())


# ===== Olympiad Theorem Library =====

OLYMPIAD_THEOREMS = [
    # ===== Inequalities =====
    Theorem(
        name="am_gm",
        title="AM-GM Inequality",
        statement="For non-negative real numbers, the arithmetic mean is at least the geometric mean",
        formal_statement="(a₁ + a₂ + ... + aₙ)/n ≥ (a₁·a₂·...·aₙ)^(1/n), equality iff all aᵢ equal",
        domains=[TheoremDomain.INEQUALITIES, TheoremDomain.ALGEBRA],
        keywords=["mean", "average", "product", "sum", "inequality", "minimum", "maximum"],
        prerequisites=["all terms non-negative"],
        produces=["inequality between sum and product"],
        example="x + y ≥ 2√(xy) for x,y ≥ 0",
        importance=10
    ),
    
    Theorem(
        name="cauchy_schwarz",
        title="Cauchy-Schwarz Inequality",
        statement="The square of the dot product is at most the product of squared magnitudes",
        formal_statement="(∑aᵢbᵢ)² ≤ (∑aᵢ²)(∑bᵢ²), equality iff aᵢ/bᵢ constant",
        domains=[TheoremDomain.INEQUALITIES, TheoremDomain.ALGEBRA],
        keywords=["sum", "product", "square", "cauchy", "schwarz"],
        produces=["upper bound on sum of products"],
        example="(a₁b₁ + a₂b₂)² ≤ (a₁² + a₂²)(b₁² + b₂²)",
        importance=9
    ),
    
    Theorem(
        name="qm_am",
        title="QM-AM Inequality",
        statement="Quadratic mean is at least arithmetic mean",
        formal_statement="√((a₁² + ... + aₙ²)/n) ≥ (a₁ + ... + aₙ)/n",
        domains=[TheoremDomain.INEQUALITIES],
        keywords=["quadratic", "mean", "square", "rms"],
        importance=6
    ),
    
    Theorem(
        name="jensen",
        title="Jensen's Inequality",
        statement="For convex functions, f(average) ≤ average of f",
        formal_statement="f((x₁+...+xₙ)/n) ≤ (f(x₁)+...+f(xₙ))/n for convex f",
        domains=[TheoremDomain.INEQUALITIES, TheoremDomain.ALGEBRA],
        keywords=["convex", "concave", "jensen", "function"],
        prerequisites=["function is convex (or concave)"],
        importance=7
    ),
    
    # ===== Number Theory =====
    Theorem(
        name="fermat_little",
        title="Fermat's Little Theorem",
        statement="a^p ≡ a (mod p) for prime p",
        formal_statement="If p is prime and gcd(a,p)=1, then a^(p-1) ≡ 1 (mod p)",
        domains=[TheoremDomain.NUMBER_THEORY],
        keywords=["prime", "power", "modulo", "fermat", "congruence"],
        prerequisites=["p is prime", "a not divisible by p"],
        produces=["simplification of powers mod p"],
        example="2^6 ≡ 1 (mod 7)",
        importance=10
    ),
    
    Theorem(
        name="euler_theorem",
        title="Euler's Theorem",
        statement="a^φ(n) ≡ 1 (mod n) for gcd(a,n)=1",
        formal_statement="If gcd(a,n)=1, then a^φ(n) ≡ 1 (mod n)",
        domains=[TheoremDomain.NUMBER_THEORY],
        keywords=["euler", "phi", "totient", "power", "modulo"],
        prerequisites=["a and n coprime"],
        produces=["simplification of powers mod n"],
        importance=9
    ),
    
    Theorem(
        name="crt",
        title="Chinese Remainder Theorem",
        statement="System of congruences with coprime moduli has unique solution",
        formal_statement="If m₁,...,mₖ pairwise coprime, x≡aᵢ(mod mᵢ) has unique solution mod M=∏mᵢ",
        domains=[TheoremDomain.NUMBER_THEORY],
        keywords=["chinese", "remainder", "congruence", "modulo", "system"],
        prerequisites=["moduli are pairwise coprime"],
        produces=["unique solution modulo product"],
        importance=9
    ),
    
    Theorem(
        name="wilson",
        title="Wilson's Theorem",
        statement="(p-1)! ≡ -1 (mod p) iff p is prime",
        formal_statement="p is prime ⟺ (p-1)! ≡ -1 (mod p)",
        domains=[TheoremDomain.NUMBER_THEORY],
        keywords=["factorial", "prime", "wilson"],
        produces=["primality test", "factorial value mod p"],
        importance=6
    ),
    
    Theorem(
        name="zsigmondy",
        title="Zsigmondy's Theorem",
        statement="a^n - b^n has a prime factor not dividing a^k - b^k for k < n",
        formal_statement="For a>b≥1, gcd(a,b)=1, n≥1: a^n-b^n has primitive prime divisor (exceptions: n=1,2,6)",
        domains=[TheoremDomain.NUMBER_THEORY],
        keywords=["prime", "divisor", "power", "difference", "zsigmondy"],
        importance=7
    ),
    
    Theorem(
        name="lte",
        title="Lifting the Exponent Lemma",
        statement="For odd p dividing a-b: vₚ(a^n-b^n) = vₚ(a-b) + vₚ(n)",
        formal_statement="If p odd, p∤a, p∤b, p|a-b: vₚ(aⁿ-bⁿ)=vₚ(a-b)+vₚ(n)",
        domains=[TheoremDomain.NUMBER_THEORY],
        keywords=["exponent", "power", "valuation", "prime", "divisibility", "lte"],
        importance=8
    ),
    
    # ===== Geometry =====
    Theorem(
        name="ptolemy",
        title="Ptolemy's Theorem",
        statement="For cyclic quadrilateral: AC·BD = AB·CD + AD·BC",
        formal_statement="ABCD cyclic ⟹ |AC|·|BD| = |AB|·|CD| + |AD|·|BC|",
        domains=[TheoremDomain.GEOMETRY],
        keywords=["cyclic", "quadrilateral", "ptolemy", "diagonal", "product"],
        prerequisites=["quadrilateral is cyclic"],
        importance=7
    ),
    
    Theorem(
        name="power_of_point",
        title="Power of a Point",
        statement="Product of signed distances from point to circle intersections is constant",
        formal_statement="For point P and circle: PA·PB = PC·PD for any chords through P",
        domains=[TheoremDomain.GEOMETRY],
        keywords=["power", "circle", "chord", "intersection", "radical"],
        importance=8
    ),
    
    Theorem(
        name="menelaus",
        title="Menelaus' Theorem",
        statement="Points on sides of triangle are collinear iff product condition holds",
        formal_statement="D,E,F on lines BC,CA,AB: collinear ⟺ (BD/DC)(CE/EA)(AF/FB)=-1",
        domains=[TheoremDomain.GEOMETRY],
        keywords=["collinear", "triangle", "ratio", "menelaus"],
        importance=7
    ),
    
    Theorem(
        name="ceva",
        title="Ceva's Theorem",
        statement="Cevians are concurrent iff product of ratios equals 1",
        formal_statement="AD,BE,CF cevians concurrent ⟺ (BD/DC)(CE/EA)(AF/FB)=1",
        domains=[TheoremDomain.GEOMETRY],
        keywords=["concurrent", "cevian", "triangle", "ratio", "ceva"],
        importance=7
    ),
    
    Theorem(
        name="stewart",
        title="Stewart's Theorem",
        statement="Relates cevian length to side lengths",
        formal_statement="If AD is cevian to BC in △ABC: b²·m + c²·n - a·m·n = a·d²",
        domains=[TheoremDomain.GEOMETRY],
        keywords=["cevian", "length", "triangle", "stewart"],
        importance=6
    ),
    
    # ===== Combinatorics =====
    Theorem(
        name="pigeonhole",
        title="Pigeonhole Principle",
        statement="If n+1 items go into n boxes, some box has ≥2 items",
        formal_statement="If |A| > n·|B| and f:A→B, then some b∈B has |f⁻¹(b)| > n",
        domains=[TheoremDomain.COMBINATORICS],
        keywords=["pigeonhole", "box", "bucket", "count", "at least"],
        produces=["existence of overcrowded container"],
        importance=10
    ),
    
    Theorem(
        name="vandermonde",
        title="Vandermonde's Identity",
        statement="C(m+n,r) = Σ C(m,k)C(n,r-k)",
        formal_statement="C(m+n,r) = Σₖ₌₀ʳ C(m,k)C(n,r-k)",
        domains=[TheoremDomain.COMBINATORICS],
        keywords=["binomial", "coefficient", "choose", "vandermonde", "sum"],
        importance=7
    ),
    
    Theorem(
        name="hockey_stick",
        title="Hockey Stick Identity",
        statement="Sum of diagonal binomial coefficients",
        formal_statement="Σᵢ₌₀ⁿ C(i,r) = C(n+1,r+1)",
        domains=[TheoremDomain.COMBINATORICS],
        keywords=["binomial", "sum", "diagonal", "pascal", "hockey"],
        importance=6
    ),
    
    # ===== Algebra/Polynomials =====
    Theorem(
        name="vieta",
        title="Vieta's Formulas",
        statement="Relate polynomial coefficients to roots",
        formal_statement="For P(x)=xⁿ+a₁xⁿ⁻¹+...+aₙ with roots r₁,...,rₙ: Σrᵢ=-a₁, Σrᵢrⱼ=a₂, ...",
        domains=[TheoremDomain.ALGEBRA, TheoremDomain.POLYNOMIALS],
        keywords=["root", "polynomial", "coefficient", "vieta", "sum", "product"],
        importance=9
    ),
    
    Theorem(
        name="remainder",
        title="Polynomial Remainder Theorem",
        statement="P(a) equals the remainder when dividing P by (x-a)",
        formal_statement="P(x) = (x-a)Q(x) + P(a)",
        domains=[TheoremDomain.ALGEBRA, TheoremDomain.POLYNOMIALS],
        keywords=["polynomial", "remainder", "division", "factor", "root"],
        importance=8
    ),
    
    Theorem(
        name="factor_theorem",
        title="Factor Theorem",
        statement="(x-a) divides P(x) iff P(a)=0",
        formal_statement="(x-a)|P(x) ⟺ P(a)=0",
        domains=[TheoremDomain.ALGEBRA, TheoremDomain.POLYNOMIALS],
        keywords=["factor", "root", "polynomial", "divides", "zero"],
        importance=8
    ),
    
    Theorem(
        name="sophie_germain",
        title="Sophie Germain Identity",
        statement="a⁴ + 4b⁴ = (a² + 2b² + 2ab)(a² + 2b² - 2ab)",
        formal_statement="a⁴ + 4b⁴ = (a² + 2b² + 2ab)(a² + 2b² - 2ab)",
        domains=[TheoremDomain.ALGEBRA, TheoremDomain.NUMBER_THEORY],
        keywords=["factorization", "fourth power", "sum", "sophie", "germain"],
        importance=6
    ),
]


def get_theorems() -> TheoremBase:
    """Get the default theorem database."""
    return TheoremBase()

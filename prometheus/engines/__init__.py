"""
Engines: Specialized mathematical reasoning engines.

Each engine handles a specific type of mathematics:
- Algebra: Polynomial manipulation, equation solving
- Number Theory: Divisibility, primes, modular arithmetic
- Geometry: Points, lines, circles, transformations
- Combinatorics: Counting, arrangements, graphs

Engines use SymPy and Z3 for verified computations.
"""

from prometheus.engines.algebra import AlgebraEngine
from prometheus.engines.number_theory import NumberTheoryEngine
from prometheus.engines.geometry import GeometryEngine
from prometheus.engines.combinatorics import CombinatoricsEngine

__all__ = [
    "AlgebraEngine",
    "NumberTheoryEngine", 
    "GeometryEngine",
    "CombinatoricsEngine"
]

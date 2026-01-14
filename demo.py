#!/usr/bin/env python3
"""
PROMETHEUS Demo Script

This demonstrates the capabilities of the PROMETHEUS system.
Run this to see each component in action!
"""

import asyncio
import sys

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Use plain print for maximum compatibility
USE_RICH = False

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_section(title):
    print(f"\n>>> {title}\n")

def print_success(text):
    print(f"  [OK] {text}")

def print_info(text):
    print(f"       {text}")


def demo_number_theory():
    """Demonstrate the number theory engine."""
    print_section("Number Theory Engine")
    
    from prometheus.engines.number_theory import NumberTheoryEngine
    nt = NumberTheoryEngine()
    
    # GCD
    result = nt.gcd(48, 18)
    print_success(f"gcd(48, 18) = {result.result}")
    print_info(f"   Steps: {result.steps}")
    
    # Extended GCD
    result = nt.extended_gcd(35, 15)
    gcd, x, y = result.result
    print_success(f"Extended GCD(35, 15): gcd={gcd}, 35*{x} + 15*{y} = {gcd}")
    
    # Prime factorization
    result = nt.prime_factorization(360)
    print_success(f"360 = {result.result}")
    
    # Modular exponentiation
    result = nt.mod_pow(3, 100, 7)
    print_success(f"3^100 mod 7 = {result.result}")
    
    # Euler's phi function
    result = nt.euler_phi(36)
    print_success(f"φ(36) = {result.result}")
    
    # Solve congruence
    result = nt.solve_linear_congruence(3, 6, 9)
    print_success(f"3x ≡ 6 (mod 9): x ∈ {result.result}")


def demo_algebra():
    """Demonstrate the algebra engine."""
    print_section("Algebra Engine")
    
    from prometheus.engines.algebra import AlgebraEngine
    alg = AlgebraEngine()
    
    # Simplify
    result = alg.simplify("(x + 1)**2 - x**2 - 2*x")
    print_success(f"Simplify (x+1)² - x² - 2x = {result.result}")
    
    # Factor
    result = alg.factor("x**2 - 5*x + 6")
    print_success(f"Factor x² - 5x + 6 = {result.result}")
    
    # Solve equation
    result = alg.solve("x**2 - 5*x + 6", "x")
    print_success(f"Solve x² - 5x + 6 = 0: x ∈ {result.result}")
    
    # Solve system
    result = alg.solve_system(
        ["x + y = 10", "x - y = 4"],
        ["x", "y"]
    )
    print_success(f"System x+y=10, x-y=4: {result.result}")
    
    # Verify identity
    result = alg.verify_identity("(a + b)**2", "a**2 + 2*a*b + b**2")
    print_success(f"(a+b)² = a² + 2ab + b²: {result.result}")


def demo_geometry():
    """Demonstrate the geometry engine."""
    print_section("Geometry Engine")
    
    from prometheus.engines.geometry import GeometryEngine, Point, Triangle
    geo = GeometryEngine()
    
    # Distance
    p1 = Point(0, 0)
    p2 = Point(3, 4)
    result = geo.distance(p1, p2)
    print_success(f"Distance from (0,0) to (3,4) = {result.result}")
    
    # Triangle properties
    A = Point(0, 0, "A")
    B = Point(4, 0, "B")
    C = Point(0, 3, "C")
    tri = Triangle(A, B, C)
    print_success(f"Triangle ABC: sides = {tri.sides}")
    print_success(f"   Area = {tri.area}")
    print_success(f"   Is right triangle: {tri.is_right()}")
    
    # Angle
    result = geo.angle(Point(1, 0), Point(0, 0), Point(0, 1))
    import math
    print_success(f"Angle at origin = {math.degrees(result.result):.1f}°")


def demo_combinatorics():
    """Demonstrate the combinatorics engine."""
    print_section("Combinatorics Engine")
    
    from prometheus.engines.combinatorics import CombinatoricsEngine
    combo = CombinatoricsEngine()
    
    # Combinations
    result = combo.combinations(10, 3)
    print_success(f"C(10, 3) = {result.result}")
    
    # Catalan number
    result = combo.catalan(5)
    print_success(f"Catalan(5) = {result.result}")
    
    # Fibonacci
    result = combo.fibonacci(10)
    print_success(f"Fibonacci(10) = {result.result}")
    
    # Partition count
    result = combo.partition_count(10)
    print_success(f"p(10) = {result.result} (number of partitions)")
    
    # Derangements
    result = combo.derangement(5)
    print_success(f"D(5) = {result.result} (derangements)")


def demo_knowledge_base():
    """Demonstrate the knowledge base."""
    print_section("Knowledge Base")
    
    from prometheus.knowledge.theorems import get_theorems
    from prometheus.knowledge.techniques import get_techniques, suggest_techniques
    
    theorems = get_theorems()
    
    # Show some theorems
    print_info(f"Loaded {len(theorems.list_all())} theorems")
    
    # Search for a theorem
    results = theorems.search("prime")
    print_success(f"Theorems about 'prime': {[t.name for t in results[:3]]}")
    
    # Show a theorem
    flt = theorems.get("fermat_little")
    if flt:
        print_success(f"Fermat's Little Theorem: {flt.statement}")
        print_info(f"   Formal: {flt.formal_statement}")
    
    # Suggest techniques
    techniques = suggest_techniques(
        "Find all positive integers n such that n^2 + 1 divides n^3 + 1",
        ["number_theory"]
    )
    print_success(f"Suggested techniques: {[t.name for t in techniques]}")


def demo_proof_state():
    """Demonstrate proof state management."""
    print_section("Proof State & Tactics")
    
    from prometheus.core.proof_state import ProofState, Goal
    from prometheus.core.formula import Constraint, Expression, RelationType
    from prometheus.core.tactic import TACTIC_REGISTRY
    
    # Create a proof state
    state = ProofState()
    
    # Add a goal
    goal_statement = Constraint(
        left=Expression.variable("n"),
        relation=RelationType.GREATER_THAN,
        right=Expression.constant(0)
    )
    goal = state.add_goal(goal_statement, origin="prove positivity")
    
    print_success(f"Created proof state with goal: {goal.statement}")
    print_info(f"   {state.summary()}")
    
    # List available tactics
    tactics = TACTIC_REGISTRY.list_all()
    print_success(f"Available tactics: {tactics}")


async def demo_solver():
    """Demonstrate the main solver (simplified, no actual LLM calls)."""
    print_section("PROMETHEUS Solver (Offline Mode)")
    
    from prometheus.pipeline.solver import PrometheusSolver, SolverConfig
    
    # Create solver in offline mode
    config = SolverConfig(
        use_oracle=False,  # No LLM needed for demo
        verbose=False
    )
    solver = PrometheusSolver(config)
    
    print_success("Solver initialized successfully")
    print_info(f"   Engines: {list(solver.engines.keys())}")
    
    # Use the number theory engine directly
    nt = solver.number_theory()
    result = nt.is_prime(97)
    print_success(f"Is 97 prime? {result.result}")


def main():
    """Run all demos."""
    print_header("PROMETHEUS System Demo")
    
    print_info("This demo shows the components of PROMETHEUS working together.")
    print_info("PROMETHEUS is a neuro-symbolic system for solving olympiad math.")
    print_info("It combines symbolic reasoning engines with LLM guidance.\n")
    
    try:
        # Run component demos
        demo_number_theory()
        demo_algebra()
        demo_geometry()
        demo_combinatorics()
        demo_knowledge_base()
        demo_proof_state()
        
        # Run async demo
        asyncio.run(demo_solver())
        
        print_header("Demo Complete!")
        print_info("")
        print_info("PROMETHEUS is ready. To use the full system with LLM guidance,")
        print_info("set your API key and use the PrometheusSolver with use_oracle=True.")
        
    except ImportError as e:
        print(f"\n[!] Some dependencies are missing: {e}")
        print("Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()

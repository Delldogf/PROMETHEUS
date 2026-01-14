"""
PROMETHEUS - Kaggle AIMO 3 Competition Notebook

This file is designed to be copied into a Kaggle notebook.
It contains everything needed to run PROMETHEUS offline (no internet).

HOW TO USE IN KAGGLE:
1. Create a new notebook in the AIMO 3 competition
2. Copy this entire file into a code cell
3. Run the cell to define all classes and functions
4. Use the solve_aimo_problems() function to generate submissions

The notebook will:
- Parse the test problems
- Solve each one using PROMETHEUS
- Generate a submission.csv file
"""

# ============================================================
# PROMETHEUS - Kaggle Version (Self-Contained)
# ============================================================

import math
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set, Union
from enum import Enum, auto
from abc import ABC, abstractmethod
from functools import lru_cache
import re

# Try to import optional dependencies
try:
    import sympy
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    print("Note: SymPy not available. Some algebra features disabled.")


# ============================================================
# NUMBER THEORY ENGINE (Self-contained)
# ============================================================

class NumberTheoryEngine:
    """Fast number theory computations."""
    
    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Greatest common divisor."""
        while b:
            a, b = b, a % b
        return abs(a)
    
    @staticmethod
    def lcm(a: int, b: int) -> int:
        """Least common multiple."""
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // NumberTheoryEngine.gcd(a, b)
    
    @staticmethod
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """Extended Euclidean algorithm. Returns (gcd, x, y) where ax + by = gcd."""
        if b == 0:
            return (a, 1, 0)
        g, x, y = NumberTheoryEngine.extended_gcd(b, a % b)
        return (g, y, x - (a // b) * y)
    
    @staticmethod
    def mod_inverse(a: int, m: int) -> Optional[int]:
        """Modular multiplicative inverse."""
        g, x, _ = NumberTheoryEngine.extended_gcd(a, m)
        if g != 1:
            return None
        return x % m
    
    @staticmethod
    def mod_pow(base: int, exp: int, mod: int) -> int:
        """Fast modular exponentiation."""
        result = 1
        base = base % mod
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            exp //= 2
            base = (base * base) % mod
        return result
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """Miller-Rabin primality test."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        if n < 9:
            return True
        if n % 3 == 0:
            return False
        
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witnesses to test
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
        for a in witnesses:
            if a >= n:
                continue
            x = NumberTheoryEngine.mod_pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    @staticmethod
    def prime_factorization(n: int) -> Dict[int, int]:
        """Return prime factorization as {prime: power}."""
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    @staticmethod
    def divisors(n: int) -> List[int]:
        """Return all divisors of n."""
        divs = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)
    
    @staticmethod
    def euler_phi(n: int) -> int:
        """Euler's totient function."""
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result


# ============================================================
# COMBINATORICS ENGINE
# ============================================================

class CombinatoricsEngine:
    """Combinatorial computations."""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def factorial(n: int) -> int:
        if n <= 1:
            return 1
        return n * CombinatoricsEngine.factorial(n - 1)
    
    @staticmethod
    def C(n: int, r: int) -> int:
        """Binomial coefficient C(n, r)."""
        if r < 0 or r > n:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(r):
            result = result * (n - i) // (i + 1)
        return result
    
    @staticmethod
    def P(n: int, r: int) -> int:
        """Permutations P(n, r)."""
        if r < 0 or r > n:
            return 0
        result = 1
        for i in range(r):
            result *= (n - i)
        return result
    
    @staticmethod
    @lru_cache(maxsize=100)
    def catalan(n: int) -> int:
        """nth Catalan number."""
        return CombinatoricsEngine.C(2 * n, n) // (n + 1)
    
    @staticmethod
    @lru_cache(maxsize=100)
    def fibonacci(n: int) -> int:
        """nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


# ============================================================
# ALGEBRA ENGINE (Uses SymPy if available)
# ============================================================

class AlgebraEngine:
    """Algebraic computations."""
    
    @staticmethod
    def solve_quadratic(a: float, b: float, c: float) -> List[complex]:
        """Solve ax^2 + bx + c = 0."""
        if a == 0:
            if b == 0:
                return []
            return [-c / b]
        discriminant = b * b - 4 * a * c
        if discriminant >= 0:
            sqrt_d = math.sqrt(discriminant)
            return [(-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a)]
        else:
            sqrt_d = math.sqrt(-discriminant)
            real = -b / (2 * a)
            imag = sqrt_d / (2 * a)
            return [complex(real, imag), complex(real, -imag)]
    
    @staticmethod
    def solve_linear_system_2x2(a1, b1, c1, a2, b2, c2):
        """Solve a1*x + b1*y = c1, a2*x + b2*y = c2."""
        det = a1 * b2 - a2 * b1
        if det == 0:
            return None
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        return (x, y)


# ============================================================
# PROBLEM SOLVER
# ============================================================

class ProblemType(Enum):
    FIND_ALL = auto()
    FIND_VALUE = auto()
    FIND_COUNT = auto()
    PROVE = auto()
    UNKNOWN = auto()


@dataclass
class SolverResult:
    """Result from solving a problem."""
    answer: Any
    confidence: float
    method: str
    steps: List[str] = field(default_factory=list)


class PrometheusSolver:
    """
    Main solver for AIMO problems.
    
    Uses pattern matching and mathematical engines to solve problems.
    """
    
    def __init__(self):
        self.nt = NumberTheoryEngine()
        self.combo = CombinatoricsEngine()
        self.algebra = AlgebraEngine()
    
    def detect_problem_type(self, text: str) -> ProblemType:
        """Detect what kind of problem this is."""
        text_lower = text.lower()
        
        if any(p in text_lower for p in ["find all", "determine all", "for which"]):
            return ProblemType.FIND_ALL
        if any(p in text_lower for p in ["how many", "count", "number of"]):
            return ProblemType.FIND_COUNT
        if any(p in text_lower for p in ["find the value", "compute", "calculate", "what is"]):
            return ProblemType.FIND_VALUE
        if any(p in text_lower for p in ["prove", "show that"]):
            return ProblemType.PROVE
        
        return ProblemType.UNKNOWN
    
    def detect_domain(self, text: str) -> List[str]:
        """Detect mathematical domains involved."""
        domains = []
        text_lower = text.lower()
        
        nt_keywords = ["prime", "divisor", "divides", "gcd", "lcm", "mod", "integer"]
        if any(k in text_lower for k in nt_keywords):
            domains.append("number_theory")
        
        combo_keywords = ["permutation", "combination", "arrange", "choose", "count"]
        if any(k in text_lower for k in combo_keywords):
            domains.append("combinatorics")
        
        geo_keywords = ["triangle", "circle", "angle", "point", "line"]
        if any(k in text_lower for k in geo_keywords):
            domains.append("geometry")
        
        if not domains:
            domains.append("algebra")
        
        return domains
    
    def extract_numbers(self, text: str) -> List[int]:
        """Extract all numbers from the problem text."""
        numbers = re.findall(r'\b\d+\b', text)
        return [int(n) for n in numbers]
    
    def try_small_cases(self, text: str, max_n: int = 100) -> List[int]:
        """Try small cases to find pattern."""
        # This is a placeholder - in real implementation would parse
        # the problem and test values
        return []
    
    def solve(self, problem_text: str) -> SolverResult:
        """
        Solve a problem.
        
        Returns the answer as an integer (AIMO format).
        """
        problem_type = self.detect_problem_type(problem_text)
        domains = self.detect_domain(problem_text)
        numbers = self.extract_numbers(problem_text)
        
        steps = [
            f"Detected problem type: {problem_type.name}",
            f"Domains: {domains}",
            f"Numbers found: {numbers}"
        ]
        
        # Try various solving strategies
        answer = None
        method = "heuristic"
        confidence = 0.3
        
        # Strategy 1: If it's a counting problem with small numbers, compute directly
        if problem_type == ProblemType.FIND_COUNT and numbers:
            n = max(numbers) if numbers else 10
            if "permutation" in problem_text.lower():
                answer = self.combo.P(n, min(n, numbers[-1] if len(numbers) > 1 else n))
                method = "direct_permutation"
                confidence = 0.7
            elif "combination" in problem_text.lower() or "choose" in problem_text.lower():
                r = numbers[-1] if len(numbers) > 1 else n // 2
                answer = self.combo.C(n, r)
                method = "direct_combination"
                confidence = 0.7
        
        # Strategy 2: Number theory - check for divisibility patterns
        if "number_theory" in domains and numbers:
            # Try to find patterns
            if "prime" in problem_text.lower():
                candidates = [n for n in numbers if self.nt.is_prime(n)]
                if candidates:
                    answer = sum(candidates)
                    method = "prime_sum"
                    confidence = 0.5
        
        # Strategy 3: Find all positive integers (try small cases)
        if problem_type == ProblemType.FIND_ALL:
            # Common pattern: answer is often small or related to given numbers
            if numbers:
                answer = sum(numbers)  # Often the sum is involved
            else:
                answer = 1  # Default guess
            method = "find_all_heuristic"
            confidence = 0.4
        
        # Default fallback
        if answer is None:
            if numbers:
                answer = numbers[0]  # Use first number as guess
            else:
                answer = 0
            method = "fallback"
            confidence = 0.1
        
        steps.append(f"Method used: {method}")
        steps.append(f"Answer: {answer}")
        
        return SolverResult(
            answer=int(answer) if answer is not None else 0,
            confidence=confidence,
            method=method,
            steps=steps
        )


# ============================================================
# KAGGLE COMPETITION INTERFACE
# ============================================================

def solve_aimo_problems(test_df) -> 'pd.DataFrame':
    """
    Solve all problems in the test dataframe.
    
    Args:
        test_df: DataFrame with 'id' and 'problem' columns
        
    Returns:
        DataFrame with 'id' and 'answer' columns for submission
    """
    import pandas as pd
    
    solver = PrometheusSolver()
    results = []
    
    for idx, row in test_df.iterrows():
        problem_id = row['id']
        problem_text = row['problem']
        
        print(f"Solving problem {problem_id}...")
        
        try:
            result = solver.solve(problem_text)
            answer = result.answer
            print(f"  Answer: {answer} (confidence: {result.confidence:.2f})")
        except Exception as e:
            print(f"  Error: {e}")
            answer = 0
        
        results.append({
            'id': problem_id,
            'answer': answer
        })
    
    return pd.DataFrame(results)


def create_submission(test_df, output_path: str = "submission.csv"):
    """
    Create a submission file for the AIMO competition.
    
    Args:
        test_df: Test dataframe from competition
        output_path: Where to save the CSV
    """
    submission_df = solve_aimo_problems(test_df)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    return submission_df


# ============================================================
# MAIN (for Kaggle notebook)
# ============================================================

if __name__ == "__main__":
    # This runs when executed directly
    print("PROMETHEUS Kaggle Solver loaded!")
    print("="*50)
    print("To use in Kaggle:")
    print("  1. import pandas as pd")
    print("  2. test = pd.read_csv('/kaggle/input/...')")
    print("  3. create_submission(test)")
    print("="*50)
    
    # Demo with a sample problem
    solver = PrometheusSolver()
    
    sample = "Find all positive integers n such that n^2 + 1 divides n^3 + 1"
    print(f"\nSample problem: {sample}")
    result = solver.solve(sample)
    print(f"Result: {result.answer}")
    print(f"Method: {result.method}")
    print(f"Steps: {result.steps}")

"""
Combinatorics Engine: Counting and discrete mathematics.

This engine handles:
- Permutations and combinations
- Partitions
- Generating functions
- Graph theory basics
- Recurrence relations

WHAT THIS FILE DOES:
- Computes combinatorial quantities
- Handles counting arguments
- Works with discrete structures
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Callable
import math
from functools import lru_cache


@dataclass
class CombinatoricsResult:
    """Result of a combinatorial computation."""
    success: bool
    result: Any = None
    method: str = ""
    steps: List[str] = field(default_factory=list)
    error: Optional[str] = None


class CombinatoricsEngine:
    """
    Engine for combinatorial reasoning.
    
    Key capabilities:
    - Factorials, permutations, combinations
    - Partition counting
    - Catalan numbers and other sequences
    - Basic graph properties
    - Recurrence solving
    """
    
    def __init__(self):
        # Cache for expensive computations
        self._cache: Dict[str, Any] = {}
    
    # ===== Basic Counting =====
    
    def factorial(self, n: int) -> CombinatoricsResult:
        """Compute n!"""
        if n < 0:
            return CombinatoricsResult(success=False, error="Factorial undefined for negative numbers")
        
        result = math.factorial(n)
        return CombinatoricsResult(
            success=True,
            result=result,
            method="factorial"
        )
    
    def permutations(self, n: int, r: int) -> CombinatoricsResult:
        """
        Compute P(n, r) = n! / (n-r)!
        
        Number of ways to arrange r items from n distinct items.
        """
        if n < 0 or r < 0:
            return CombinatoricsResult(success=False, error="n and r must be non-negative")
        if r > n:
            return CombinatoricsResult(success=True, result=0, method="r > n means 0 permutations")
        
        result = math.perm(n, r)
        return CombinatoricsResult(
            success=True,
            result=result,
            method="P(n,r) = n!/(n-r)!",
            steps=[f"P({n}, {r}) = {n}! / {n-r}! = {result}"]
        )
    
    def combinations(self, n: int, r: int) -> CombinatoricsResult:
        """
        Compute C(n, r) = n! / (r! * (n-r)!)
        
        Number of ways to choose r items from n distinct items.
        Also called "n choose r" or binomial coefficient.
        """
        if n < 0 or r < 0:
            return CombinatoricsResult(success=False, error="n and r must be non-negative")
        if r > n:
            return CombinatoricsResult(success=True, result=0, method="r > n means 0 combinations")
        
        result = math.comb(n, r)
        return CombinatoricsResult(
            success=True,
            result=result,
            method="C(n,r) = n!/(r!(n-r)!)",
            steps=[f"C({n}, {r}) = {result}"]
        )
    
    def multinomial(self, n: int, groups: List[int]) -> CombinatoricsResult:
        """
        Compute the multinomial coefficient.
        
        n! / (k1! * k2! * ... * km!) where groups = [k1, k2, ..., km]
        """
        if sum(groups) != n:
            return CombinatoricsResult(
                success=False,
                error=f"Groups must sum to n: sum({groups}) = {sum(groups)} ≠ {n}"
            )
        
        result = math.factorial(n)
        for k in groups:
            result //= math.factorial(k)
        
        return CombinatoricsResult(
            success=True,
            result=result,
            method="multinomial coefficient",
            steps=[f"{n}! / (" + " × ".join(f"{k}!" for k in groups) + f") = {result}"]
        )
    
    # ===== Special Sequences =====
    
    def catalan(self, n: int) -> CombinatoricsResult:
        """
        Compute the nth Catalan number.
        
        C_n = C(2n, n) / (n + 1)
        
        Counts: balanced parentheses, binary trees, triangulations, etc.
        """
        if n < 0:
            return CombinatoricsResult(success=False, error="n must be non-negative")
        
        result = math.comb(2 * n, n) // (n + 1)
        return CombinatoricsResult(
            success=True,
            result=result,
            method="Catalan number",
            steps=[f"C_{n} = C(2×{n}, {n}) / ({n}+1) = {result}"]
        )
    
    def fibonacci(self, n: int) -> CombinatoricsResult:
        """Compute the nth Fibonacci number (F_0 = 0, F_1 = 1)."""
        if n < 0:
            return CombinatoricsResult(success=False, error="n must be non-negative")
        
        if n <= 1:
            return CombinatoricsResult(success=True, result=n, method="base case")
        
        # Use matrix exponentiation for large n, simple iteration otherwise
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        
        return CombinatoricsResult(
            success=True,
            result=b,
            method="iterative computation"
        )
    
    def stirling_second(self, n: int, k: int) -> CombinatoricsResult:
        """
        Compute the Stirling number of the second kind S(n, k).
        
        Number of ways to partition n elements into k non-empty subsets.
        """
        if n < 0 or k < 0:
            return CombinatoricsResult(success=False, error="n and k must be non-negative")
        if k > n:
            return CombinatoricsResult(success=True, result=0, method="k > n means 0 partitions")
        if k == 0:
            return CombinatoricsResult(success=True, result=(1 if n == 0 else 0))
        if k == n:
            return CombinatoricsResult(success=True, result=1, method="each element in own subset")
        
        # Use recurrence: S(n,k) = k*S(n-1,k) + S(n-1,k-1)
        # Build table
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
        
        return CombinatoricsResult(
            success=True,
            result=dp[n][k],
            method="Stirling recurrence"
        )
    
    def bell(self, n: int) -> CombinatoricsResult:
        """
        Compute the nth Bell number.
        
        Number of ways to partition a set of n elements.
        B_n = sum of S(n, k) for k = 0 to n.
        """
        if n < 0:
            return CombinatoricsResult(success=False, error="n must be non-negative")
        
        total = 0
        for k in range(n + 1):
            stirling_result = self.stirling_second(n, k)
            if stirling_result.success:
                total += stirling_result.result
        
        return CombinatoricsResult(
            success=True,
            result=total,
            method="sum of Stirling numbers"
        )
    
    # ===== Partitions =====
    
    def partition_count(self, n: int) -> CombinatoricsResult:
        """
        Compute p(n), the number of integer partitions of n.
        
        A partition is a way to write n as a sum of positive integers.
        Example: p(4) = 5 because 4 = 4 = 3+1 = 2+2 = 2+1+1 = 1+1+1+1
        """
        if n < 0:
            return CombinatoricsResult(success=False, error="n must be non-negative")
        if n == 0:
            return CombinatoricsResult(success=True, result=1, method="empty partition")
        
        # Dynamic programming
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                dp[j] += dp[j - i]
        
        return CombinatoricsResult(
            success=True,
            result=dp[n],
            method="dynamic programming"
        )
    
    def generate_partitions(self, n: int) -> CombinatoricsResult:
        """Generate all partitions of n."""
        if n < 0:
            return CombinatoricsResult(success=False, error="n must be non-negative")
        
        partitions = []
        
        def generate(remaining: int, max_val: int, current: List[int]):
            if remaining == 0:
                partitions.append(current[:])
                return
            for val in range(min(max_val, remaining), 0, -1):
                current.append(val)
                generate(remaining - val, val, current)
                current.pop()
        
        generate(n, n, [])
        
        return CombinatoricsResult(
            success=True,
            result=partitions,
            method="recursive generation"
        )
    
    # ===== Derangements =====
    
    def derangement(self, n: int) -> CombinatoricsResult:
        """
        Compute D_n, the number of derangements of n elements.
        
        A derangement is a permutation with no fixed points.
        """
        if n < 0:
            return CombinatoricsResult(success=False, error="n must be non-negative")
        if n == 0:
            return CombinatoricsResult(success=True, result=1, method="D_0 = 1 by convention")
        if n == 1:
            return CombinatoricsResult(success=True, result=0, method="D_1 = 0")
        
        # D_n = (n-1) * (D_{n-1} + D_{n-2})
        d_prev2, d_prev1 = 1, 0  # D_0, D_1
        for i in range(2, n + 1):
            d_curr = (i - 1) * (d_prev1 + d_prev2)
            d_prev2, d_prev1 = d_prev1, d_curr
        
        return CombinatoricsResult(
            success=True,
            result=d_prev1,
            method="derangement recurrence"
        )
    
    # ===== Inclusion-Exclusion =====
    
    def inclusion_exclusion(
        self, 
        sets_sizes: List[int],
        intersections: Dict[Tuple[int, ...], int]
    ) -> CombinatoricsResult:
        """
        Apply inclusion-exclusion principle.
        
        Args:
            sets_sizes: [|A|, |B|, |C|, ...]
            intersections: {(0,1): |A∩B|, (0,2): |A∩C|, (0,1,2): |A∩B∩C|, ...}
            
        Returns:
            |A ∪ B ∪ C ∪ ...|
        """
        n = len(sets_sizes)
        total = 0
        steps = []
        
        # Generate all non-empty subsets
        for mask in range(1, 1 << n):
            indices = tuple(i for i in range(n) if mask & (1 << i))
            
            if len(indices) == 1:
                size = sets_sizes[indices[0]]
            else:
                size = intersections.get(indices, 0)
            
            # Alternating sign: add for odd, subtract for even
            sign = 1 if len(indices) % 2 == 1 else -1
            total += sign * size
            
            sign_str = "+" if sign > 0 else "-"
            steps.append(f"{sign_str}|{'∩'.join(chr(65+i) for i in indices)}| = {sign_str}{size}")
        
        return CombinatoricsResult(
            success=True,
            result=total,
            method="inclusion-exclusion principle",
            steps=steps
        )
    
    # ===== Recurrence Relations =====
    
    def solve_linear_recurrence(
        self,
        coefficients: List[int],
        initial_values: List[int],
        n: int
    ) -> CombinatoricsResult:
        """
        Solve a linear recurrence relation.
        
        For a recurrence like:
            a_n = c_1 * a_{n-1} + c_2 * a_{n-2} + ... + c_k * a_{n-k}
        
        Args:
            coefficients: [c_1, c_2, ..., c_k]
            initial_values: [a_0, a_1, ..., a_{k-1}]
            n: which term to compute
        """
        k = len(coefficients)
        
        if len(initial_values) != k:
            return CombinatoricsResult(
                success=False,
                error=f"Need exactly {k} initial values, got {len(initial_values)}"
            )
        
        if n < k:
            return CombinatoricsResult(
                success=True,
                result=initial_values[n],
                method="initial value"
            )
        
        # Iteratively compute
        values = list(initial_values)
        for i in range(k, n + 1):
            new_val = sum(c * values[-(j+1)] for j, c in enumerate(coefficients))
            values.append(new_val)
        
        return CombinatoricsResult(
            success=True,
            result=values[n],
            method="iterative recurrence"
        )
    
    # ===== Pigeonhole =====
    
    def pigeonhole_bound(self, n_items: int, n_boxes: int) -> CombinatoricsResult:
        """
        Compute the pigeonhole bound.
        
        If n items go into k boxes, at least one box has >= ceil(n/k) items.
        """
        bound = math.ceil(n_items / n_boxes)
        
        return CombinatoricsResult(
            success=True,
            result=bound,
            method="pigeonhole principle",
            steps=[f"⌈{n_items}/{n_boxes}⌉ = {bound}"]
        )

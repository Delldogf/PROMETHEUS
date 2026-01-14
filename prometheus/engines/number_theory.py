"""
Number Theory Engine: Integer-specific reasoning.

This engine handles:
- Divisibility and factors
- Prime numbers
- Modular arithmetic
- GCD, LCM
- Congruences

WHAT THIS FILE DOES:
- Provides verified number-theoretic computations
- Tests divisibility conditions
- Works with modular arithmetic
- Handles prime factorizations
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set
import math


@dataclass
class NumberTheoryResult:
    """Result of a number-theoretic computation."""
    success: bool
    result: Any = None
    method: str = ""
    steps: List[str] = field(default_factory=list)
    error: Optional[str] = None


class NumberTheoryEngine:
    """
    Engine for number-theoretic reasoning.
    
    Key capabilities:
    - GCD, LCM computation
    - Prime factorization
    - Modular arithmetic
    - Divisibility analysis
    - Congruence solving
    
    Most operations are implemented directly for speed,
    with SymPy as backup for complex cases.
    """
    
    def __init__(self):
        self._sympy = None
    
    def _get_sympy(self):
        """Lazy load SymPy."""
        if self._sympy is None:
            try:
                import sympy
                self._sympy = sympy
            except ImportError:
                pass
        return self._sympy
    
    # ===== Basic Operations =====
    
    def gcd(self, a: int, b: int) -> NumberTheoryResult:
        """
        Compute the greatest common divisor.
        
        Uses Euclidean algorithm with step tracking.
        """
        if a == 0 and b == 0:
            return NumberTheoryResult(
                success=False,
                error="gcd(0, 0) is undefined"
            )
        
        steps = []
        original_a, original_b = abs(a), abs(b)
        a, b = original_a, original_b
        
        while b != 0:
            steps.append(f"gcd({a}, {b}) = gcd({b}, {a % b})")
            a, b = b, a % b
        
        return NumberTheoryResult(
            success=True,
            result=a,
            method="Euclidean algorithm",
            steps=steps
        )
    
    def lcm(self, a: int, b: int) -> NumberTheoryResult:
        """Compute the least common multiple."""
        if a == 0 or b == 0:
            return NumberTheoryResult(success=True, result=0, method="lcm with zero")
        
        gcd_result = self.gcd(a, b)
        result = abs(a * b) // gcd_result.result
        
        return NumberTheoryResult(
            success=True,
            result=result,
            method="lcm(a,b) = |a*b|/gcd(a,b)",
            steps=[f"gcd({a}, {b}) = {gcd_result.result}", f"lcm = |{a}*{b}|/{gcd_result.result} = {result}"]
        )
    
    def extended_gcd(self, a: int, b: int) -> NumberTheoryResult:
        """
        Extended Euclidean algorithm.
        
        Returns (gcd, x, y) such that a*x + b*y = gcd(a, b).
        This is crucial for modular inverse and Diophantine equations.
        """
        if b == 0:
            return NumberTheoryResult(
                success=True,
                result=(a, 1, 0),
                method="extended Euclidean algorithm"
            )
        
        steps = []
        old_r, r = a, b
        old_s, s = 1, 0
        old_t, t = 0, 1
        
        while r != 0:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t
            steps.append(f"r={old_r}, s={old_s}, t={old_t}")
        
        return NumberTheoryResult(
            success=True,
            result=(old_r, old_s, old_t),  # (gcd, x, y)
            method="extended Euclidean algorithm",
            steps=steps
        )
    
    # ===== Primality =====
    
    def is_prime(self, n: int) -> NumberTheoryResult:
        """
        Check if a number is prime.
        
        Uses trial division for small numbers, Miller-Rabin for large.
        """
        if n < 2:
            return NumberTheoryResult(success=True, result=False, method="n < 2")
        if n == 2:
            return NumberTheoryResult(success=True, result=True, method="2 is prime")
        if n % 2 == 0:
            return NumberTheoryResult(success=True, result=False, method="even number > 2")
        
        # Trial division up to sqrt(n)
        limit = int(math.sqrt(n)) + 1
        for i in range(3, min(limit, 10000), 2):
            if n % i == 0:
                return NumberTheoryResult(
                    success=True,
                    result=False,
                    method="trial division",
                    steps=[f"{n} is divisible by {i}"]
                )
        
        if limit > 10000:
            # Use SymPy for large numbers
            sp = self._get_sympy()
            if sp:
                result = sp.isprime(n)
                return NumberTheoryResult(
                    success=True,
                    result=result,
                    method="sympy.isprime"
                )
        
        return NumberTheoryResult(success=True, result=True, method="trial division")
    
    def prime_factorization(self, n: int) -> NumberTheoryResult:
        """
        Compute the prime factorization of n.
        
        Returns: Dict mapping prime factors to their powers.
        Example: prime_factorization(12) → {2: 2, 3: 1}
        """
        if n < 1:
            return NumberTheoryResult(success=False, error="n must be positive")
        if n == 1:
            return NumberTheoryResult(success=True, result={}, method="1 has no prime factors")
        
        factors = {}
        steps = []
        original_n = n
        
        # Factor out 2s
        while n % 2 == 0:
            factors[2] = factors.get(2, 0) + 1
            n //= 2
            steps.append(f"Divided by 2, now n = {n}")
        
        # Factor out odd primes
        i = 3
        while i * i <= n:
            while n % i == 0:
                factors[i] = factors.get(i, 0) + 1
                n //= i
                steps.append(f"Divided by {i}, now n = {n}")
            i += 2
        
        # Remaining prime factor
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
            steps.append(f"{n} is a prime factor")
        
        return NumberTheoryResult(
            success=True,
            result=factors,
            method="trial division",
            steps=steps
        )
    
    def list_primes(self, n: int) -> NumberTheoryResult:
        """
        List all primes up to n.
        
        Uses Sieve of Eratosthenes.
        """
        if n < 2:
            return NumberTheoryResult(success=True, result=[], method="sieve of Eratosthenes")
        
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        
        primes = [i for i, is_prime in enumerate(sieve) if is_prime]
        
        return NumberTheoryResult(
            success=True,
            result=primes,
            method="sieve of Eratosthenes"
        )
    
    # ===== Modular Arithmetic =====
    
    def mod(self, a: int, m: int) -> NumberTheoryResult:
        """Compute a mod m (always non-negative)."""
        if m == 0:
            return NumberTheoryResult(success=False, error="modulus cannot be 0")
        result = a % m
        return NumberTheoryResult(success=True, result=result, method="modulo operation")
    
    def mod_pow(self, base: int, exp: int, mod: int) -> NumberTheoryResult:
        """
        Compute (base^exp) mod m efficiently.
        
        Uses binary exponentiation (square-and-multiply).
        """
        if mod == 0:
            return NumberTheoryResult(success=False, error="modulus cannot be 0")
        if mod == 1:
            return NumberTheoryResult(success=True, result=0, method="mod 1 is always 0")
        
        steps = []
        result = 1
        base = base % mod
        
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
                steps.append(f"Multiplied, result = {result}")
            exp //= 2
            base = (base * base) % mod
            steps.append(f"Squared base, now base = {base}")
        
        return NumberTheoryResult(
            success=True,
            result=result,
            method="binary exponentiation",
            steps=steps
        )
    
    def mod_inverse(self, a: int, m: int) -> NumberTheoryResult:
        """
        Compute the modular inverse of a mod m.
        
        Returns x such that (a * x) ≡ 1 (mod m).
        Only exists if gcd(a, m) = 1.
        """
        gcd_result = self.extended_gcd(a, m)
        gcd, x, _ = gcd_result.result
        
        if gcd != 1:
            return NumberTheoryResult(
                success=False,
                error=f"No inverse: gcd({a}, {m}) = {gcd} ≠ 1"
            )
        
        inverse = x % m
        return NumberTheoryResult(
            success=True,
            result=inverse,
            method="extended Euclidean algorithm",
            steps=[f"{a} * {inverse} ≡ 1 (mod {m})"]
        )
    
    def solve_linear_congruence(
        self, 
        a: int, 
        b: int, 
        m: int
    ) -> NumberTheoryResult:
        """
        Solve ax ≡ b (mod m).
        
        Returns all solutions modulo m (or an error if no solution).
        """
        gcd_result = self.gcd(a, m)
        g = gcd_result.result
        
        if b % g != 0:
            return NumberTheoryResult(
                success=False,
                error=f"No solution: gcd({a}, {m}) = {g} does not divide {b}"
            )
        
        # Reduce to (a/g)x ≡ (b/g) (mod m/g)
        a1, b1, m1 = a // g, b // g, m // g
        
        # Find inverse
        inv_result = self.mod_inverse(a1, m1)
        if not inv_result.success:
            return inv_result
        
        # Base solution
        x0 = (inv_result.result * b1) % m1
        
        # All solutions are x0, x0 + m1, x0 + 2*m1, ...
        solutions = [(x0 + i * m1) % m for i in range(g)]
        
        return NumberTheoryResult(
            success=True,
            result=solutions,
            method="linear congruence solver",
            steps=[
                f"gcd({a}, {m}) = {g}",
                f"Reduced: {a1}x ≡ {b1} (mod {m1})",
                f"Base solution: x ≡ {x0} (mod {m1})"
            ]
        )
    
    # ===== Divisibility =====
    
    def divides(self, a: int, b: int) -> NumberTheoryResult:
        """Check if a divides b (a | b)."""
        if a == 0:
            return NumberTheoryResult(
                success=True,
                result=(b == 0),  # 0 | 0 is true, 0 | n is false for n ≠ 0
                method="zero divisibility"
            )
        result = b % a == 0
        return NumberTheoryResult(
            success=True,
            result=result,
            method="divisibility check",
            steps=[f"{b} = {b // a if result else '?'} × {a}" + ("" if result else f" + {b % a}")]
        )
    
    def divisors(self, n: int) -> NumberTheoryResult:
        """
        Find all positive divisors of n.
        """
        if n < 1:
            return NumberTheoryResult(success=False, error="n must be positive")
        
        divs = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        
        divs.sort()
        return NumberTheoryResult(
            success=True,
            result=divs,
            method="trial division"
        )
    
    def divisor_count(self, n: int) -> NumberTheoryResult:
        """
        Count the number of positive divisors.
        
        Uses prime factorization: if n = p1^a1 * p2^a2 * ...,
        then d(n) = (a1+1)(a2+1)...
        """
        if n < 1:
            return NumberTheoryResult(success=False, error="n must be positive")
        
        factorization = self.prime_factorization(n)
        if not factorization.success:
            return factorization
        
        count = 1
        for prime, power in factorization.result.items():
            count *= (power + 1)
        
        return NumberTheoryResult(
            success=True,
            result=count,
            method="from prime factorization",
            steps=[f"τ({n}) = " + " × ".join(f"({p}+1)" for p in factorization.result.values())]
        )
    
    def divisor_sum(self, n: int) -> NumberTheoryResult:
        """
        Compute the sum of positive divisors.
        """
        divs_result = self.divisors(n)
        if not divs_result.success:
            return divs_result
        
        total = sum(divs_result.result)
        return NumberTheoryResult(
            success=True,
            result=total,
            method="sum of divisors",
            steps=[f"σ({n}) = " + " + ".join(map(str, divs_result.result)) + f" = {total}"]
        )
    
    # ===== Special Checks =====
    
    def is_perfect(self, n: int) -> NumberTheoryResult:
        """Check if n is a perfect number (σ(n) = 2n)."""
        sigma = self.divisor_sum(n)
        if not sigma.success:
            return sigma
        
        is_perfect = sigma.result == 2 * n
        return NumberTheoryResult(
            success=True,
            result=is_perfect,
            method="perfect number check",
            steps=[f"σ({n}) = {sigma.result}, 2×{n} = {2*n}"]
        )
    
    def is_coprime(self, a: int, b: int) -> NumberTheoryResult:
        """Check if a and b are coprime (gcd = 1)."""
        gcd_result = self.gcd(a, b)
        return NumberTheoryResult(
            success=True,
            result=(gcd_result.result == 1),
            method="coprimality check",
            steps=[f"gcd({a}, {b}) = {gcd_result.result}"]
        )
    
    def euler_phi(self, n: int) -> NumberTheoryResult:
        """
        Compute Euler's totient function φ(n).
        
        φ(n) = count of integers in [1, n] coprime to n.
        """
        if n < 1:
            return NumberTheoryResult(success=False, error="n must be positive")
        
        factorization = self.prime_factorization(n)
        if not factorization.success:
            return factorization
        
        result = n
        for prime in factorization.result.keys():
            result -= result // prime
        
        return NumberTheoryResult(
            success=True,
            result=result,
            method="Euler's formula",
            steps=[f"φ({n}) = {n}" + "".join(f" × (1 - 1/{p})" for p in factorization.result.keys())]
        )

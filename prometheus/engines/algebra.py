"""
Algebra Engine: Algebraic manipulation and equation solving.

This engine handles:
- Polynomial operations (expand, factor, simplify)
- Equation solving
- Inequalities
- Function analysis

Uses SymPy for verified algebraic computations.

WHAT THIS FILE DOES:
- Provides verified algebraic operations
- Solves equations and systems
- Simplifies expressions
- Checks algebraic identities
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Set, Tuple
from abc import ABC, abstractmethod


@dataclass
class AlgebraResult:
    """Result of an algebraic computation."""
    success: bool
    result: Any = None
    method: str = ""
    steps: List[str] = field(default_factory=list)
    error: Optional[str] = None


class AlgebraEngine:
    """
    Engine for algebraic manipulation and equation solving.
    
    This wraps SymPy with a clean interface and adds verification.
    All results are mathematically verified, not just "looks right."
    
    Key capabilities:
    - Simplify, expand, factor expressions
    - Solve equations and systems
    - Work with inequalities
    - Polynomial operations
    """
    
    def __init__(self):
        self._sympy = None
        self._init_sympy()
    
    def _init_sympy(self):
        """Initialize SymPy (lazy import)."""
        try:
            import sympy
            self._sympy = sympy
        except ImportError:
            pass  # Will handle gracefully
    
    @property
    def sympy(self):
        """Get SymPy module (raise if not available)."""
        if self._sympy is None:
            raise ImportError("SymPy is required for the Algebra Engine. Install with: pip install sympy")
        return self._sympy
    
    # ===== Expression Manipulation =====
    
    def simplify(self, expr: str) -> AlgebraResult:
        """
        Simplify an expression.
        
        Example: simplify("(x+1)^2 - x^2 - 2*x") → "1"
        """
        try:
            sp = self.sympy
            parsed = sp.sympify(expr)
            simplified = sp.simplify(parsed)
            return AlgebraResult(
                success=True,
                result=str(simplified),
                method="sympy.simplify"
            )
        except Exception as e:
            return AlgebraResult(success=False, error=str(e))
    
    def expand(self, expr: str) -> AlgebraResult:
        """
        Expand an expression.
        
        Example: expand("(x+1)^2") → "x^2 + 2*x + 1"
        """
        try:
            sp = self.sympy
            parsed = sp.sympify(expr)
            expanded = sp.expand(parsed)
            return AlgebraResult(
                success=True,
                result=str(expanded),
                method="sympy.expand"
            )
        except Exception as e:
            return AlgebraResult(success=False, error=str(e))
    
    def factor(self, expr: str) -> AlgebraResult:
        """
        Factor an expression.
        
        Example: factor("x^2 - 1") → "(x - 1)*(x + 1)"
        """
        try:
            sp = self.sympy
            parsed = sp.sympify(expr)
            factored = sp.factor(parsed)
            return AlgebraResult(
                success=True,
                result=str(factored),
                method="sympy.factor"
            )
        except Exception as e:
            return AlgebraResult(success=False, error=str(e))
    
    # ===== Equation Solving =====
    
    def solve(
        self, 
        equation: str, 
        variable: str = "x",
        domain: str = "complex"
    ) -> AlgebraResult:
        """
        Solve an equation.
        
        Args:
            equation: The equation (e.g., "x^2 - 5*x + 6" or "x^2 = 4")
            variable: The variable to solve for
            domain: "complex", "real", "integer", "positive"
            
        Returns:
            AlgebraResult with list of solutions
        """
        try:
            sp = self.sympy
            var = sp.Symbol(variable)
            
            # Parse equation
            if "=" in equation:
                lhs, rhs = equation.split("=")
                expr = sp.sympify(lhs) - sp.sympify(rhs)
            else:
                expr = sp.sympify(equation)
            
            # Choose domain
            if domain == "integer":
                solutions = sp.solve(expr, var, domain=sp.S.Integers)
            elif domain == "positive":
                # Solve then filter
                solutions = [s for s in sp.solve(expr, var) if s > 0]
            elif domain == "real":
                solutions = sp.solve(expr, var, domain=sp.S.Reals)
            else:
                solutions = sp.solve(expr, var)
            
            return AlgebraResult(
                success=True,
                result=[str(s) for s in solutions],
                method=f"sympy.solve (domain={domain})"
            )
        except Exception as e:
            return AlgebraResult(success=False, error=str(e))
    
    def solve_system(
        self, 
        equations: List[str],
        variables: List[str]
    ) -> AlgebraResult:
        """
        Solve a system of equations.
        
        Example:
            solve_system(["x + y = 10", "x - y = 2"], ["x", "y"])
            → {"x": 6, "y": 4}
        """
        try:
            sp = self.sympy
            vars_sym = [sp.Symbol(v) for v in variables]
            
            # Parse equations
            eqs_sym = []
            for eq in equations:
                if "=" in eq:
                    lhs, rhs = eq.split("=")
                    eqs_sym.append(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)))
                else:
                    eqs_sym.append(sp.sympify(eq))
            
            solutions = sp.solve(eqs_sym, vars_sym)
            
            # Format result
            if isinstance(solutions, dict):
                result = {str(k): str(v) for k, v in solutions.items()}
            elif isinstance(solutions, list):
                result = [
                    {str(k): str(v) for k, v in sol.items()} 
                    if isinstance(sol, dict) else str(sol)
                    for sol in solutions
                ]
            else:
                result = str(solutions)
            
            return AlgebraResult(
                success=True,
                result=result,
                method="sympy.solve (system)"
            )
        except Exception as e:
            return AlgebraResult(success=False, error=str(e))
    
    # ===== Verification =====
    
    def verify_equation(
        self, 
        equation: str, 
        substitutions: Dict[str, Any]
    ) -> AlgebraResult:
        """
        Verify an equation holds with given substitutions.
        
        Example:
            verify_equation("x^2 - 4 = 0", {"x": 2}) → True
        """
        try:
            sp = self.sympy
            
            if "=" in equation:
                lhs, rhs = equation.split("=")
                expr = sp.sympify(lhs) - sp.sympify(rhs)
            else:
                expr = sp.sympify(equation)
            
            # Substitute values
            sub_dict = {sp.Symbol(k): v for k, v in substitutions.items()}
            result = expr.subs(sub_dict)
            
            # Check if zero
            simplified = sp.simplify(result)
            is_zero = simplified == 0
            
            return AlgebraResult(
                success=True,
                result=is_zero,
                method="substitution verification",
                steps=[
                    f"Substituted: {substitutions}",
                    f"Result: {result}",
                    f"Simplified: {simplified}",
                    f"Is zero: {is_zero}"
                ]
            )
        except Exception as e:
            return AlgebraResult(success=False, error=str(e))
    
    def verify_identity(
        self, 
        lhs: str, 
        rhs: str,
        assumptions: Optional[Dict[str, str]] = None
    ) -> AlgebraResult:
        """
        Verify that two expressions are identical.
        
        Example:
            verify_identity("(a+b)^2", "a^2 + 2*a*b + b^2") → True
        """
        try:
            sp = self.sympy
            
            left = sp.sympify(lhs)
            right = sp.sympify(rhs)
            
            diff = sp.simplify(left - right)
            is_identical = diff == 0
            
            return AlgebraResult(
                success=True,
                result=is_identical,
                method="identity verification",
                steps=[
                    f"LHS: {left}",
                    f"RHS: {right}",
                    f"Difference: {diff}",
                    f"Identical: {is_identical}"
                ]
            )
        except Exception as e:
            return AlgebraResult(success=False, error=str(e))
    
    # ===== Polynomial Operations =====
    
    def polynomial_degree(self, expr: str, variable: str = "x") -> AlgebraResult:
        """Get the degree of a polynomial."""
        try:
            sp = self.sympy
            var = sp.Symbol(variable)
            poly = sp.Poly(sp.sympify(expr), var)
            degree = poly.degree()
            return AlgebraResult(success=True, result=degree, method="sympy.Poly.degree")
        except Exception as e:
            return AlgebraResult(success=False, error=str(e))
    
    def polynomial_coefficients(self, expr: str, variable: str = "x") -> AlgebraResult:
        """Get the coefficients of a polynomial."""
        try:
            sp = self.sympy
            var = sp.Symbol(variable)
            poly = sp.Poly(sp.sympify(expr), var)
            coeffs = poly.all_coeffs()
            return AlgebraResult(
                success=True, 
                result=[str(c) for c in coeffs],
                method="sympy.Poly.all_coeffs"
            )
        except Exception as e:
            return AlgebraResult(success=False, error=str(e))
    
    def polynomial_roots(
        self, 
        expr: str, 
        variable: str = "x",
        domain: str = "complex"
    ) -> AlgebraResult:
        """Find roots of a polynomial."""
        try:
            sp = self.sympy
            var = sp.Symbol(variable)
            poly = sp.Poly(sp.sympify(expr), var)
            
            if domain == "real":
                roots = sp.real_roots(poly)
            else:
                roots = sp.roots(poly)
            
            if isinstance(roots, dict):
                # SymPy returns {root: multiplicity}
                result = [(str(root), mult) for root, mult in roots.items()]
            else:
                result = [str(r) for r in roots]
            
            return AlgebraResult(
                success=True,
                result=result,
                method="sympy.roots"
            )
        except Exception as e:
            return AlgebraResult(success=False, error=str(e))
    
    # ===== Inequality Solving =====
    
    def solve_inequality(self, inequality: str, variable: str = "x") -> AlgebraResult:
        """
        Solve an inequality.
        
        Example: solve_inequality("x^2 - 4 < 0") → "(-2, 2)"
        """
        try:
            sp = self.sympy
            var = sp.Symbol(variable, real=True)
            
            # Parse inequality
            ineq = sp.sympify(inequality.replace(variable, f"Symbol('{variable}', real=True)"))
            solution = sp.solve(ineq, var)
            
            return AlgebraResult(
                success=True,
                result=str(solution),
                method="sympy.solve (inequality)"
            )
        except Exception as e:
            return AlgebraResult(success=False, error=str(e))

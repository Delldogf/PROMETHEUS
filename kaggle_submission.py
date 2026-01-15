"""
PROMETHEUS + GPT-OSS-120B Kaggle Submission (AIMO 3)
=====================================================

This notebook combines:
- GPT-OSS-120B: For reasoning and tool orchestration
- PROMETHEUS: For verified mathematical computation

The LLM uses PROMETHEUS tools to solve olympiad problems with verified answers.

COMPETITION REQUIREMENTS (AIMO 3):
- Answer Format: Integer from 0 to 99999 (5 digits)
- GPU Notebook: <= 5 hours runtime
- CPU Notebook: <= 9 hours runtime
- Internet: DISABLED (offline only)
- Must call serve() within 15 minutes of script start
- Model loading should happen INSIDE predict() (lazy loading)

VLLM COMPATIBILITY NOTES:
- Use vLLM >= 0.10.2 for tool calling support
- GPT-OSS may have issues with vLLM V1 engine
- Set VLLM_USE_V1=0 if experiencing instability
- For H100: use --async-scheduling flag for better performance

DIAGNOSTICS:
- All major operations are logged with timestamps
- GPU/CPU memory usage is tracked
- Problem-level timing and answers are logged
- Tool call execution is logged

KNOWN ISSUES & FIXES:
- If you see "ModuleNotFoundError: No module named 'rpds.rpds'" - this is a Kaggle
  environment bug, not our code. See FIX_RPDS_ERROR below.
- If you see "numpy.dtype size changed, may indicate binary incompatibility" - 
  this is a numpy/sklearn version mismatch. See FIX_NUMPY_ERROR below.
"""

# ============================================================
# FIX 1: KAGGLE ENVIRONMENT BUG (rpds.rpds error)
# ============================================================
# If you encounter "ModuleNotFoundError: No module named 'rpds.rpds'", 
# this is a bug in Kaggle's aimoutility environment, not our code.
#
# WORKAROUND OPTIONS:
# 1. Create a fresh notebook and paste this code (sometimes fixes it)
# 2. Add this as the FIRST CELL of your notebook (with internet enabled):
#    !pip install --force-reinstall rpds-py referencing jsonschema
# 3. If that doesn't work, the issue is in Kaggle's papermill/nbconvert pipeline
#    and you may need to report it to Kaggle or wait for them to fix it.
#
# The error happens BEFORE your Python code runs (during notebook conversion),
# so if you see this error, the notebook isn't even starting your code yet.
# ============================================================

# ============================================================
# FIX 2: NUMPY/SKLEARN BINARY INCOMPATIBILITY
# ============================================================
# If you encounter "numpy.dtype size changed, may indicate binary incompatibility",
# this means sklearn was compiled with a different numpy version.
#
# WORKAROUND - Add this as the FIRST CELL of your notebook:
#
#   !pip install --quiet --force-reinstall numpy==1.26.4
#   !pip install --quiet --force-reinstall scikit-learn
#
# Then RESTART the notebook kernel and run again.
#
# This error happens because Kaggle's pre-installed packages are sometimes
# compiled against different versions of numpy.
# ============================================================

import os
import re
import json
import math
import time
import traceback
import gc
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from functools import lru_cache

import torch
import polars as pl
import pandas as pd

# ============================================================
# DIAGNOSTICS AND LOGGING SYSTEM
# ============================================================

class DiagnosticLogger:
    """
    Comprehensive logging for AIMO 3 competition.
    Since we can't access internet, all diagnostics go to stdout.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = time.time()
        self.problem_count = 0
        self.total_problems = 0
        self.correct_count = 0  # For local testing
        self.timings: List[float] = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp and level."""
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}][{elapsed:7.1f}s][{level:5s}]"
        print(f"{prefix} {message}")
        
    def log_system_info(self):
        """Log system information at startup."""
        self.log("=" * 60, "INFO")
        self.log("PROMETHEUS + GPT-OSS-120B for AIMO 3", "INFO")
        self.log("=" * 60, "INFO")
        
        # Python and environment
        import sys
        self.log(f"Python: {sys.version.split()[0]}", "INFO")
        self.log(f"Platform: {sys.platform}", "INFO")
        
        # Check if running in competition mode
        is_competition = os.getenv('KAGGLE_IS_COMPETITION_RERUN')
        self.log(f"Competition mode: {bool(is_competition)}", "INFO")
        
        # GPU information
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.log(f"GPUs available: {gpu_count}", "INFO")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                self.log(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)", "INFO")
        else:
            self.log("No GPU available - using CPU", "WARN")
            
        # Check vLLM availability
        self.log(f"vLLM backend: {USE_VLLM}", "INFO")
        
        self.log("=" * 60, "INFO")
        
    def log_memory(self, label: str = ""):
        """Log current memory usage."""
        if not self.verbose:
            return
            
        prefix = f"[Memory{': ' + label if label else ''}]"
        
        # CPU memory
        try:
            import psutil
            process = psutil.Process()
            cpu_mem = process.memory_info().rss / 1e9
            self.log(f"{prefix} CPU RAM: {cpu_mem:.2f} GB", "DEBUG")
        except ImportError:
            pass
            
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                self.log(f"{prefix} GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved", "DEBUG")
                
    def log_problem_start(self, problem_id: Any, problem_text: str):
        """Log when starting a new problem."""
        self.problem_count += 1
        self.log("-" * 40, "INFO")
        self.log(f"Problem {self.problem_count}: ID={problem_id}", "INFO")
        
        # Show first 200 chars of problem
        preview = problem_text[:200].replace('\n', ' ')
        if len(problem_text) > 200:
            preview += "..."
        self.log(f"Text: {preview}", "INFO")
        
    def log_problem_end(self, problem_id: Any, answer: int, duration: float):
        """Log when finishing a problem."""
        self.timings.append(duration)
        avg_time = sum(self.timings) / len(self.timings)
        
        self.log(f"Answer: {answer} (took {duration:.2f}s, avg: {avg_time:.2f}s)", "INFO")
        self.log_memory("after problem")
        
    def log_tool_call(self, tool_name: str, args: Dict, result: Any):
        """Log a tool call execution."""
        if self.verbose:
            args_str = json.dumps(args)[:100]
            result_str = str(result)[:100]
            self.log(f"Tool: {tool_name}({args_str}) -> {result_str}", "DEBUG")
            
    def log_error(self, error: Exception, context: str = ""):
        """Log an error with traceback."""
        self.log(f"ERROR in {context}: {str(error)}", "ERROR")
        self.log(traceback.format_exc(), "ERROR")
        
    def log_summary(self):
        """Log final summary statistics."""
        self.log("=" * 60, "INFO")
        self.log("FINAL SUMMARY", "INFO")
        self.log("=" * 60, "INFO")
        
        total_time = time.time() - self.start_time
        self.log(f"Total problems processed: {self.problem_count}", "INFO")
        self.log(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)", "INFO")
        
        if self.timings:
            avg_time = sum(self.timings) / len(self.timings)
            max_time = max(self.timings)
            min_time = min(self.timings)
            self.log(f"Time per problem: avg={avg_time:.1f}s, min={min_time:.1f}s, max={max_time:.1f}s", "INFO")
            
        # Estimate remaining time for 50 problems
        if self.timings:
            remaining = 50 - self.problem_count
            estimated = remaining * avg_time
            self.log(f"Estimated time for {remaining} more problems: {estimated/60:.1f} min", "INFO")
            
        self.log_memory("final")
        self.log("=" * 60, "INFO")


# Global logger instance
LOGGER = DiagnosticLogger(verbose=True)


# ============================================================
# KAGGLE ENVIRONMENT FIXES
# ============================================================

# Fix for numpy/sklearn binary incompatibility in Kaggle
# This error: "numpy.dtype size changed, may indicate binary incompatibility"
# happens when sklearn was compiled with a different numpy version
try:
    import subprocess
    import sys
    
    # Check if we're in a Kaggle environment
    if os.path.exists('/kaggle'):
        LOGGER.log("Kaggle environment detected - checking package compatibility", "INFO")
        
        # Try to fix numpy/sklearn compatibility if needed
        # This might help with the binary incompatibility error
        # Uncomment the lines below if you get numpy.dtype errors:
        # subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", 
        #                 "--force-reinstall", "numpy==1.26.4"], check=False)
        # subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
        #                 "--force-reinstall", "scikit-learn"], check=False)
except Exception as e:
    LOGGER.log(f"Environment fix check failed (non-critical): {e}", "WARN")

# ============================================================
# VLLM STABILITY SETTINGS FOR GPT-OSS
# ============================================================

# Set V1 engine off for GPT-OSS stability (if experiencing issues)
# Uncomment the line below if you experience hangs or garbled output
# os.environ["VLLM_USE_V1"] = "0"

# Disable tensorflow logging before imports (reduces noise)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Try vLLM first (preferred for tool calling), fallback to Transformers
USE_VLLM = False
try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    USE_VLLM = True
    LOGGER.log("vLLM imported successfully", "INFO")
except (ImportError, ValueError, OSError) as e:
    USE_VLLM = False
    LOGGER.log(f"vLLM not available: {type(e).__name__}: {e}", "WARN")
    LOGGER.log("Falling back to Transformers", "WARN")

# If vLLM failed, try Transformers
if not USE_VLLM:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        LOGGER.log("Transformers imported successfully", "INFO")
    except (ImportError, ValueError, OSError) as e2:
        LOGGER.log(f"Transformers also not available: {type(e2).__name__}: {e2}", "ERROR")

# ============================================================
# PROMETHEUS MATH TOOLS (Verified Computations)
# ============================================================

class NumberTheoryTools:
    """Number theory computations - always verified."""
    
    @staticmethod
    def gcd(a: int, b: int) -> Dict[str, Any]:
        """Compute greatest common divisor."""
        result = a
        while b:
            result, b = b, result % b
        return {"result": abs(result), "method": "euclidean_algorithm"}
    
    @staticmethod
    def lcm(a: int, b: int) -> Dict[str, Any]:
        """Compute least common multiple."""
        if a == 0 or b == 0:
            return {"result": 0}
        g = NumberTheoryTools.gcd(a, b)["result"]
        return {"result": abs(a * b) // g}
    
    @staticmethod
    def is_prime(n: int) -> Dict[str, Any]:
        """Check if n is prime using Miller-Rabin."""
        if n < 2:
            return {"result": False, "reason": "n < 2"}
        if n == 2:
            return {"result": True}
        if n % 2 == 0:
            return {"result": False, "reason": "even"}
        
        # Miller-Rabin with deterministic witnesses for n < 2^64
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return {"result": False, "witness": a}
        return {"result": True}
    
    @staticmethod
    def prime_factorization(n: int) -> Dict[str, Any]:
        """Return prime factorization as {prime: power}."""
        if n < 1:
            return {"error": "n must be positive"}
        factors = {}
        d = 2
        original = n
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return {"result": factors, "original": original}
    
    @staticmethod
    def mod_pow(base: int, exp: int, mod: int) -> Dict[str, Any]:
        """Compute (base^exp) mod m efficiently."""
        if mod == 0:
            return {"error": "modulus cannot be 0"}
        result = pow(base, exp, mod)
        return {"result": result}
    
    @staticmethod
    def mod_inverse(a: int, m: int) -> Dict[str, Any]:
        """Compute modular multiplicative inverse."""
        def extended_gcd(a, b):
            if b == 0:
                return a, 1, 0
            g, x, y = extended_gcd(b, a % b)
            return g, y, x - (a // b) * y
        
        g, x, _ = extended_gcd(a % m, m)
        if g != 1:
            return {"error": f"No inverse: gcd({a}, {m}) = {g}"}
        return {"result": x % m}
    
    @staticmethod
    def divisors(n: int) -> Dict[str, Any]:
        """Find all positive divisors."""
        if n < 1:
            return {"error": "n must be positive"}
        divs = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return {"result": sorted(divs), "count": len(divs)}
    
    @staticmethod
    def euler_phi(n: int) -> Dict[str, Any]:
        """Compute Euler's totient function."""
        if n < 1:
            return {"error": "n must be positive"}
        result = n
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return {"result": result}
    
    @staticmethod
    def solve_congruence(a: int, b: int, m: int) -> Dict[str, Any]:
        """Solve ax ≡ b (mod m)."""
        g = NumberTheoryTools.gcd(a, m)["result"]
        if b % g != 0:
            return {"error": f"No solution: gcd({a},{m})={g} doesn't divide {b}"}
        
        a1, b1, m1 = a // g, b // g, m // g
        inv = NumberTheoryTools.mod_inverse(a1, m1)
        if "error" in inv:
            return inv
        
        x0 = (inv["result"] * b1) % m1
        solutions = [(x0 + i * m1) % m for i in range(g)]
        return {"solutions": solutions, "modulus": m}


class AlgebraTools:
    """Algebraic computations."""
    
    @staticmethod
    def solve_quadratic(a: float, b: float, c: float) -> Dict[str, Any]:
        """Solve ax² + bx + c = 0."""
        if a == 0:
            if b == 0:
                return {"error": "Not a valid equation"}
            return {"roots": [-c / b], "type": "linear"}
        
        discriminant = b * b - 4 * a * c
        if discriminant > 0:
            sqrt_d = math.sqrt(discriminant)
            r1 = (-b + sqrt_d) / (2 * a)
            r2 = (-b - sqrt_d) / (2 * a)
            return {"roots": [r1, r2], "discriminant": discriminant, "type": "two_real"}
        elif discriminant == 0:
            r = -b / (2 * a)
            return {"roots": [r], "discriminant": 0, "type": "repeated"}
        else:
            real = -b / (2 * a)
            imag = math.sqrt(-discriminant) / (2 * a)
            return {"roots": [f"{real}+{imag}i", f"{real}-{imag}i"], "type": "complex"}
    
    @staticmethod
    def evaluate_polynomial(coeffs: List[float], x: float) -> Dict[str, Any]:
        """Evaluate polynomial with coefficients [a_n, a_{n-1}, ..., a_0] at x."""
        result = 0
        for c in coeffs:
            result = result * x + c
        return {"result": result, "x": x}
    
    @staticmethod
    def floor(x: float) -> Dict[str, Any]:
        """Floor function."""
        return {"result": math.floor(x)}
    
    @staticmethod
    def ceil(x: float) -> Dict[str, Any]:
        """Ceiling function."""
        return {"result": math.ceil(x)}


class CombinatoricsTools:
    """Combinatorial computations."""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def factorial(n: int) -> int:
        if n <= 1:
            return 1
        return n * CombinatoricsTools.factorial(n - 1)
    
    @staticmethod
    def C(n: int, r: int) -> Dict[str, Any]:
        """Binomial coefficient C(n, r)."""
        if r < 0 or r > n:
            return {"result": 0}
        if r == 0 or r == n:
            return {"result": 1}
        r = min(r, n - r)
        result = 1
        for i in range(r):
            result = result * (n - i) // (i + 1)
        return {"result": result}
    
    @staticmethod
    def P(n: int, r: int) -> Dict[str, Any]:
        """Permutations P(n, r)."""
        if r < 0 or r > n:
            return {"result": 0}
        result = 1
        for i in range(r):
            result *= (n - i)
        return {"result": result}
    
    @staticmethod
    def catalan(n: int) -> Dict[str, Any]:
        """nth Catalan number."""
        if n < 0:
            return {"error": "n must be non-negative"}
        c = CombinatoricsTools.C(2 * n, n)["result"]
        return {"result": c // (n + 1)}
    
    @staticmethod
    def fibonacci(n: int) -> Dict[str, Any]:
        """nth Fibonacci number."""
        if n < 0:
            return {"error": "n must be non-negative"}
        if n <= 1:
            return {"result": n}
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return {"result": b}
    
    @staticmethod
    def partition_count(n: int) -> Dict[str, Any]:
        """Number of integer partitions of n."""
        if n < 0:
            return {"error": "n must be non-negative"}
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                dp[j] += dp[j - i]
        return {"result": dp[n]}


# ============================================================
# TOOL DEFINITIONS FOR GPT-OSS-120B
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "gcd",
            "description": "Compute the greatest common divisor of two integers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First integer"},
                    "b": {"type": "integer", "description": "Second integer"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lcm",
            "description": "Compute the least common multiple of two integers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First integer"},
                    "b": {"type": "integer", "description": "Second integer"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "is_prime",
            "description": "Check if a number is prime",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "The number to check"}
                },
                "required": ["n"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "prime_factorization",
            "description": "Get the prime factorization of a positive integer",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "The number to factorize"}
                },
                "required": ["n"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mod_pow",
            "description": "Compute (base^exp) mod m efficiently",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "integer"},
                    "exp": {"type": "integer"},
                    "mod": {"type": "integer"}
                },
                "required": ["base", "exp", "mod"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mod_inverse",
            "description": "Compute the modular multiplicative inverse of a mod m",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "m": {"type": "integer"}
                },
                "required": ["a", "m"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "divisors",
            "description": "Find all positive divisors of n",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"}
                },
                "required": ["n"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "euler_phi",
            "description": "Compute Euler's totient function phi(n)",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"}
                },
                "required": ["n"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "solve_congruence",
            "description": "Solve the linear congruence ax ≡ b (mod m)",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                    "m": {"type": "integer"}
                },
                "required": ["a", "b", "m"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "solve_quadratic",
            "description": "Solve quadratic equation ax² + bx + c = 0",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "c": {"type": "number"}
                },
                "required": ["a", "b", "c"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "binomial",
            "description": "Compute binomial coefficient C(n, r) = n choose r",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "r": {"type": "integer"}
                },
                "required": ["n", "r"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "factorial",
            "description": "Compute n!",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"}
                },
                "required": ["n"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fibonacci",
            "description": "Compute the nth Fibonacci number",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"}
                },
                "required": ["n"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "catalan",
            "description": "Compute the nth Catalan number",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"}
                },
                "required": ["n"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "floor",
            "description": "Compute floor(x) - the greatest integer ≤ x",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"}
                },
                "required": ["x"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ceil",
            "description": "Compute ceil(x) - the smallest integer ≥ x",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"}
                },
                "required": ["x"]
            }
        }
    }
]

# Map tool names to functions
TOOL_MAP = {
    "gcd": lambda a, b: NumberTheoryTools.gcd(a, b),
    "lcm": lambda a, b: NumberTheoryTools.lcm(a, b),
    "is_prime": lambda n: NumberTheoryTools.is_prime(n),
    "prime_factorization": lambda n: NumberTheoryTools.prime_factorization(n),
    "mod_pow": lambda base, exp, mod: NumberTheoryTools.mod_pow(base, exp, mod),
    "mod_inverse": lambda a, m: NumberTheoryTools.mod_inverse(a, m),
    "divisors": lambda n: NumberTheoryTools.divisors(n),
    "euler_phi": lambda n: NumberTheoryTools.euler_phi(n),
    "solve_congruence": lambda a, b, m: NumberTheoryTools.solve_congruence(a, b, m),
    "solve_quadratic": lambda a, b, c: AlgebraTools.solve_quadratic(a, b, c),
    "binomial": lambda n, r: CombinatoricsTools.C(n, r),
    "factorial": lambda n: {"result": CombinatoricsTools.factorial(n)},
    "fibonacci": lambda n: CombinatoricsTools.fibonacci(n),
    "catalan": lambda n: CombinatoricsTools.catalan(n),
    "floor": lambda x: AlgebraTools.floor(x),
    "ceil": lambda x: AlgebraTools.ceil(x),
}


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Execute a PROMETHEUS tool and return the result as a string."""
    if name not in TOOL_MAP:
        error_result = {"error": f"Unknown tool: {name}"}
        LOGGER.log(f"Tool call FAILED: {name} - unknown tool", "WARN")
        return json.dumps(error_result)
    
    try:
        result = TOOL_MAP[name](**arguments)
        
        # Log successful tool call
        LOGGER.log_tool_call(name, arguments, result)
        
        return json.dumps(result)
    except Exception as e:
        error_result = {"error": str(e)}
        LOGGER.log(f"Tool call FAILED: {name}({arguments}) - {e}", "ERROR")
        return json.dumps(error_result)


# ============================================================
# GPT-OSS-120B SOLVER (Supports both vLLM and Transformers)
# ============================================================

class PrometheusSolver:
    """
    PROMETHEUS solver using GPT-OSS-120B with tool calling.
    
    Supports two backends:
    - vLLM (preferred): Native tool calling via llm.chat(..., tools=TOOLS)
    - Transformers (fallback): Manual tool call extraction
    
    GPT-OSS COMPATIBILITY NOTES:
    - Use vLLM 0.10.2 for best stability
    - Set VLLM_USE_V1=0 if experiencing hangs or garbled output
    - For H100: enable --async-scheduling for better performance
    - Model loads from local path (offline compatible)
    """
    
    def __init__(self, model_path: str = "/kaggle/input/gpt-oss-120b/transformers/default/1"):
        LOGGER.log(f"Initializing PrometheusSolver with model: {model_path}", "INFO")
        LOGGER.log(f"Backend: {'vLLM' if USE_VLLM else 'Transformers'}", "INFO")
        
        self.model_path = model_path
        
        # Verify model path exists (offline check)
        if os.path.exists(model_path):
            LOGGER.log(f"Model path verified: {model_path}", "INFO")
        else:
            LOGGER.log(f"WARNING: Model path not found: {model_path}", "WARN")
            LOGGER.log("This is expected if not running in Kaggle environment", "WARN")
        
        if USE_VLLM:
            LOGGER.log("Initializing vLLM backend...", "INFO")
            
            # vLLM backend - native tool calling support
            # GPT-OSS recommended settings for H100
            self.llm = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype="float16",
                # GPT-OSS optimized settings
                max_model_len=32768,
                # Uncomment for H100 with multiple GPUs:
                # tensor_parallel_size=1,
                # For stability issues, you can also try:
                # gpu_memory_utilization=0.85,
            )
            
            self.sampling_params = SamplingParams(
                max_tokens=4096,
                temperature=0.3,
                # For more deterministic outputs:
                # top_p=0.95,
            )
            
            LOGGER.log("vLLM backend initialized", "INFO")
            
        else:
            LOGGER.log("Initializing Transformers backend...", "INFO")
            
            # Transformers backend - manual tool handling
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            LOGGER.log("Transformers backend initialized", "INFO")
        
        LOGGER.log("Model loaded successfully!", "INFO")
        
        self.system_prompt = """You are PROMETHEUS, an expert mathematical olympiad solver.

You have access to verified mathematical tools that you should use for computations.
ALWAYS use tools for:
- GCD, LCM, primality testing
- Modular arithmetic
- Factorials, combinations, permutations
- Any numerical computation

Your task is to solve competition math problems. Think step by step:
1. Understand what the problem is asking
2. Identify the mathematical domain (number theory, algebra, combinatorics, geometry)
3. Plan your approach
4. Use tools for any calculations
5. Verify your answer makes sense
6. Return the final answer as a single integer from 0 to 99999

IMPORTANT: The final answer must be a 5-digit integer between 0 and 99999."""

    def solve(self, problem: str, max_iterations: int = 10) -> int:
        """
        Solve a math problem using GPT-OSS-120B with tool calling.
        
        Returns: Integer answer (0-99999 for AIMO 3)
        """
        if USE_VLLM:
            return self._solve_vllm(problem, max_iterations)
        else:
            return self._solve_transformers(problem, max_iterations)
    
    def _solve_vllm(self, problem: str, max_iterations: int = 10) -> int:
        """Solve using vLLM with native tool calling."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Solve this problem and give the final answer as an integer from 0 to 99999:\n\n{problem}"}
        ]
        
        for iteration in range(max_iterations):
            # Use vLLM's native chat interface with tools
            outputs = self.llm.chat(
                messages,
                sampling_params=self.sampling_params,
                tools=TOOLS
            )
            
            response_text = outputs[0].outputs[0].text.strip()
            
            # Check if the model made tool calls (vLLM returns JSON for tool calls)
            tool_calls = self._extract_tool_calls(response_text)
            
            if tool_calls:
                # Execute tools and add results
                messages.append({"role": "assistant", "content": response_text})
                
                tool_results = []
                for tool_call in tool_calls:
                    result = execute_tool(tool_call["name"], tool_call["arguments"])
                    tool_results.append(f"{tool_call['name']}: {result}")
                
                # Add tool results as a single message
                messages.append({
                    "role": "tool",
                    "content": "\n".join(tool_results),
                    "tool_call_id": f"call_{iteration}"
                })
            else:
                # No tool calls - extract final answer
                answer = self._extract_answer(response_text)
                return max(0, min(99999, answer))
        
        # Fallback if max iterations reached
        return 0
    
    def _solve_transformers(self, problem: str, max_iterations: int = 10) -> int:
        """Solve using Transformers (manual tool call handling)."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Solve this problem and give the final answer as an integer from 0 to 99999:\n\n{problem}"}
        ]
        
        for iteration in range(max_iterations):
            # Format messages for the model
            prompt = self._format_messages(messages)
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Check for tool calls
            tool_calls = self._extract_tool_calls(response_text)
            
            if tool_calls:
                # Execute tools and add results
                messages.append({"role": "assistant", "content": response_text})
                
                for tool_call in tool_calls:
                    result = execute_tool(tool_call["name"], tool_call["arguments"])
                    messages.append({
                        "role": "tool",
                        "name": tool_call["name"],
                        "content": result
                    })
            else:
                # No tool calls - extract final answer
                answer = self._extract_answer(response_text)
                return max(0, min(99999, answer))
        
        # Fallback if max iterations reached
        return 0
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages for the Transformers backend."""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n"
            elif role == "tool":
                name = msg.get('name', 'tool')
                formatted += f"<|tool_result|>\n{name}: {content}\n"
        
        formatted += "<|assistant|>\n"
        return formatted
    
    def _extract_tool_calls(self, response: str) -> List[Dict]:
        """Extract tool calls from model response."""
        tool_calls = []
        
        # Look for JSON-formatted tool calls
        # Pattern 1: {"name": "tool_name", "arguments": {...}}
        pattern1 = r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
        
        for match in re.finditer(pattern1, response):
            try:
                name = match.group(1)
                args_str = match.group(2)
                arguments = json.loads(args_str)
                tool_calls.append({"name": name, "arguments": arguments})
            except:
                continue
        
        # Pattern 2: Try parsing as JSON array (vLLM style)
        if not tool_calls:
            try:
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    for item in parsed:
                        if "name" in item and "arguments" in item:
                            tool_calls.append(item)
            except:
                pass
        
        return tool_calls
    
    def _extract_answer(self, response: str) -> int:
        """Extract the final numerical answer from response (AIMO 3: 0-99999)."""
        # Look for explicit answer patterns
        patterns = [
            r"(?:final\s+)?answer\s*(?:is|:)\s*(\d+)",
            r"(?:result|solution)\s*(?:is|:)\s*(\d+)",
            r"\\boxed\{(\d+)\}",  # LaTeX boxed format
            r"=\s*(\d+)\s*$",
            r"(\d{1,5})\s*$",  # Up to 5 digits at end
        ]
        
        response_lower = response.lower()
        
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                try:
                    return int(match.group(1))
                except:
                    continue
        
        # Fallback: find the last number in the response
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            return int(numbers[-1])
        
        return 0


# ============================================================
# KAGGLE SUBMISSION INTERFACE (AIMO 3 API)
# ============================================================

class ModelWrapper:
    """
    Wrapper for lazy model loading.
    
    IMPORTANT: In AIMO 3, you MUST call serve() within 15 minutes of script start.
    Model loading should happen INSIDE predict(), which has no time limit.
    This wrapper implements lazy loading - the model is only loaded on first use.
    """
    
    def __init__(self, model_path: str = "/kaggle/input/gpt-oss-120b/transformers/default/1"):
        self.model_path = model_path
        self._solver = None
        self._load_attempted = False
        
    def load(self):
        """Load the model (called lazily on first predict)."""
        if self._solver is not None:
            return self._solver
            
        if self._load_attempted:
            LOGGER.log("Model loading already attempted and failed", "WARN")
            return None
            
        self._load_attempted = True
        LOGGER.log("Starting model loading (this may take a few minutes)...", "INFO")
        LOGGER.log_memory("before model load")
        
        load_start = time.time()
        
        try:
            self._solver = PrometheusSolver(model_path=self.model_path)
            load_time = time.time() - load_start
            LOGGER.log(f"Model loaded successfully in {load_time:.1f}s", "INFO")
            LOGGER.log_memory("after model load")
            
            # Force garbage collection to free any temporary memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            LOGGER.log_error(e, "model loading")
            self._solver = None
            
        return self._solver
    
    def predict(self, problem: str) -> int:
        """Make a prediction (loads model on first call)."""
        solver = self.load()
        
        if solver is None:
            LOGGER.log("No solver available, returning 0", "ERROR")
            return 0
            
        return solver.solve(problem)


# Global model wrapper (lazy loading)
_model = ModelWrapper()


def predict(id_: pl.Series, question: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """
    Make a prediction for the AIMO 3 competition.
    
    IMPORTANT API NOTES:
    - id_: pl.Series containing the problem ID
    - question: pl.Series containing the problem text
    - Returns: DataFrame with 'id' and 'answer' columns
    - Answer must be integer from 0 to 99999
    
    This function is called by the Kaggle evaluation server.
    Model loading happens lazily on first call (no time limit).
    """
    problem_start = time.time()
    
    # Unpack values from Series (not DataFrame!)
    problem_id = id_.item(0)
    problem_text: str = question.item(0)
    
    # Log problem start
    LOGGER.log_problem_start(problem_id, problem_text)
    
    try:
        # Make prediction (model loads lazily on first call)
        answer = _model.predict(problem_text)
        
        # Ensure answer is in valid AIMO 3 range (0-99999)
        answer = max(0, min(99999, int(answer)))
        
    except Exception as e:
        LOGGER.log_error(e, "prediction")
        answer = 0
    
    # Log problem end
    duration = time.time() - problem_start
    LOGGER.log_problem_end(problem_id, answer, duration)
    
    return pl.DataFrame({'id': problem_id, 'answer': answer})


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Log system information at startup
    LOGGER.log_system_info()
    
    # ========================================
    # AIMO 3 Competition Setup
    # ========================================
    
    # Import the AIMO 3 inference server
    LOGGER.log("Importing Kaggle evaluation server...", "INFO")
    
    try:
        import kaggle_evaluation.aimo_3_inference_server as aimo_server
        InferenceServer = aimo_server.AIMO3InferenceServer
        test_csv_path = '/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv'
        LOGGER.log("Using AIMO 3 inference server", "INFO")
    except ImportError as e:
        LOGGER.log(f"AIMO 3 server not found: {e}", "WARN")
        LOGGER.log("Falling back to AIMO 2 server...", "WARN")
        try:
            import kaggle_evaluation.aimo_2_inference_server as aimo_server
            InferenceServer = aimo_server.AIMO2InferenceServer
            test_csv_path = '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv'
            LOGGER.log("Using AIMO 2 inference server (fallback)", "INFO")
        except ImportError as e2:
            LOGGER.log_error(e2, "importing evaluation server")
            raise RuntimeError("Could not import any AIMO evaluation server")
    
    # Create inference server
    LOGGER.log("Creating inference server...", "INFO")
    inference_server = InferenceServer(predict)
    
    # ========================================
    # Run Mode Selection
    # ========================================
    
    is_competition = os.getenv('KAGGLE_IS_COMPETITION_RERUN')
    
    if is_competition:
        # =====================================
        # COMPETITION MODE
        # =====================================
        LOGGER.log("=" * 60, "INFO")
        LOGGER.log("RUNNING IN COMPETITION MODE", "INFO")
        LOGGER.log("=" * 60, "INFO")
        LOGGER.log("IMPORTANT: serve() must be called within 15 minutes!", "WARN")
        LOGGER.log("Model will load lazily on first predict() call", "INFO")
        
        # Start serving (this blocks until all problems are processed)
        inference_server.serve()
        
        # Log final summary
        LOGGER.log_summary()
        
    else:
        # =====================================
        # LOCAL TESTING MODE
        # =====================================
        LOGGER.log("=" * 60, "INFO")
        LOGGER.log("RUNNING IN LOCAL TESTING MODE", "INFO")
        LOGGER.log("=" * 60, "INFO")
        LOGGER.log(f"Using test file: {test_csv_path}", "INFO")
        
        # Check if test file exists
        if os.path.exists(test_csv_path):
            LOGGER.log("Test file found", "INFO")
        else:
            LOGGER.log(f"Test file not found at {test_csv_path}", "WARN")
            LOGGER.log("Will attempt to run anyway...", "WARN")
        
        # Run local gateway for testing
        try:
            inference_server.run_local_gateway((test_csv_path,))
        except Exception as e:
            LOGGER.log_error(e, "local gateway")
        
        # Log final summary
        LOGGER.log_summary()

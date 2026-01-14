"""
PROMETHEUS + GPT-OSS-120B Kaggle Submission
============================================

This notebook combines:
- GPT-OSS-120B: For reasoning and tool orchestration
- PROMETHEUS: For verified mathematical computation

The LLM uses PROMETHEUS tools to solve olympiad problems with verified answers.
"""

import os
import re
import json
import math
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from functools import lru_cache

import torch
import polars as pl
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Execute a tool and return the result as a string."""
    if name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool: {name}"})
    
    try:
        result = TOOL_MAP[name](**arguments)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================
# GPT-OSS-120B SOLVER
# ============================================================

class PrometheusSolver:
    """
    PROMETHEUS solver using GPT-OSS-120B with tool calling.
    """
    
    def __init__(self, model_path: str = "/kaggle/input/gpt-oss-120b/transformers/default/1"):
        print("Loading GPT-OSS-120B...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        
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
6. Return the final answer as a single integer mod 1000

IMPORTANT: The final answer must be an integer between 0 and 999."""

    def solve(self, problem: str, max_iterations: int = 10) -> int:
        """
        Solve a math problem using GPT-OSS-120B with tool calling.
        
        Returns: Integer answer (0-999)
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Solve this problem and give the final answer as an integer mod 1000:\n\n{problem}"}
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
                    pad_token_id=self.tokenizer.eos_token_id,
                    tools=TOOLS  # Pass tools to the model
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
                return answer % 1000
        
        # Fallback if max iterations reached
        return 0
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages for the model."""
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
                formatted += f"<|tool_result|>\n{msg['name']}: {content}\n"
        
        formatted += "<|assistant|>\n"
        return formatted
    
    def _extract_tool_calls(self, response: str) -> List[Dict]:
        """Extract tool calls from model response."""
        tool_calls = []
        
        # Look for JSON-formatted tool calls
        # Pattern: {"name": "tool_name", "arguments": {...}}
        pattern = r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
        
        for match in re.finditer(pattern, response):
            try:
                name = match.group(1)
                args_str = match.group(2)
                arguments = json.loads(args_str)
                tool_calls.append({"name": name, "arguments": arguments})
            except:
                continue
        
        return tool_calls
    
    def _extract_answer(self, response: str) -> int:
        """Extract the final numerical answer from response."""
        # Look for explicit answer patterns
        patterns = [
            r"(?:final\s+)?answer\s*(?:is|:)\s*(\d+)",
            r"(?:result|solution)\s*(?:is|:)\s*(\d+)",
            r"=\s*(\d+)\s*$",
            r"(\d+)\s*(?:mod\s*1000)?\s*$",
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
# KAGGLE SUBMISSION INTERFACE
# ============================================================

# Global solver instance (loaded once)
_solver = None

def get_solver():
    """Get or create the global solver instance."""
    global _solver
    if _solver is None:
        _solver = PrometheusSolver()
    return _solver


def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame:
    """
    Make a prediction for the AIMO 2 competition.
    
    This function is called by the Kaggle evaluation server.
    """
    # Unpack values
    problem_id = id_.item(0)
    problem_text = question.item(0)
    
    print(f"\n{'='*60}")
    print(f"Problem {problem_id}")
    print(f"{'='*60}")
    print(problem_text[:200] + "..." if len(problem_text) > 200 else problem_text)
    
    # Get solver and solve
    solver = get_solver()
    
    try:
        answer = solver.solve(problem_text)
        print(f"\nAnswer: {answer}")
    except Exception as e:
        print(f"\nError: {e}")
        answer = 0
    
    # Ensure answer is in valid range
    answer = answer % 1000
    
    return pl.DataFrame({'id': problem_id, 'answer': answer})


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import kaggle_evaluation.aimo_2_inference_server
    
    # Create inference server
    inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)
    
    # Run in appropriate mode
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Competition mode
        inference_server.serve()
    else:
        # Local testing mode
        inference_server.run_local_gateway(
            ('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',)
        )

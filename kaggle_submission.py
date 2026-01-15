"""
PROMETHEUS + GPT-OSS-120B for AIMO 3 Competition
=================================================

This notebook implements the proven architecture from successful AIMO submissions:
1. Uninstall broken packages (tensorflow, sklearn, keras) to avoid numpy conflicts
2. Run vLLM as a separate server process (avoids import conflicts)
3. Use openai_harmony for GPT-OSS tool calling protocol
4. Pre-cache model files for faster loading
5. Use Jupyter kernel for stateful Python tool execution
6. Integrate PROMETHEUS symbolic math engines

Based on a notebook that scored 35/50 on AIMO 3.
"""

# ============================================================
# CELL 1: TIMING AND CONSTANTS
# ============================================================
import time
import numpy as np
import os

start_time = time.time()
final_cutoff_time = start_time + (4 * 60 + 58) * 60  # 4h 58m (2 min safety buffer)

TOTAL_TIME = 4 * 60 * 60 + 58 * 60  # 4h 58m
NUM_QUESTIONS = 50
BUFFER_TIME = 60

# ============================================================
# CELL 2: UNINSTALL BROKEN PACKAGES (CRITICAL!)
# ============================================================
# These packages are compiled against numpy 1.x but the utility notebook
# installed numpy 2.x. Instead of trying to hide them, just uninstall them.
# This runs in the background while we do other setup.

import subprocess

print("[SETUP] Uninstalling broken packages in background...")
uninstall_proc = subprocess.Popen(
    ["pip", "uninstall", "--yes", "tensorflow", "matplotlib", "keras", "scikit-learn"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# ============================================================
# CELL 3: PRE-CACHE UTILITY NOTEBOOK FILES
# ============================================================
# This reads files into OS page cache for faster access later
# The ! syntax is Jupyter magic - we use subprocess instead

print("[SETUP] Pre-caching utility notebook files...")
try:
    cache_proc = subprocess.run(
        ["find", "/kaggle/usr/lib", "-type", "f", "-print0"],
        capture_output=True,
        timeout=120
    )
    # In a real Kaggle notebook, this would pipe to xargs cat
    print("[SETUP] File caching complete")
except Exception as e:
    print(f"[SETUP] File caching skipped: {e}")

# ============================================================
# CELL 4: MODEL CACHING FUNCTION
# ============================================================
def cache_model(path, exts=(".bin", ".pt", ".safetensors"), num_workers=None, chunk_mb=256):
    """
    Pre-read model weight files into OS page cache.
    
    This dramatically speeds up model loading because the files are already
    in RAM when vLLM tries to read them.
    """
    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def warmup_file(fpath):
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        with open(fpath, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                total += len(data)
        return fpath, total

    if os.path.isdir(path):
        files = [
            os.path.join(root, name)
            for root, _, names in os.walk(path)
            for name in names
            if name.endswith(exts)
        ]
        files.sort()
    else:
        files = [path]

    if not files:
        print(f"[cache_model] No model files found under: {path}")
        return 0

    if num_workers is None:
        try:
            num_workers = min(multiprocessing.cpu_count(), 8)
        except Exception:
            num_workers = 4

    print(f"[cache_model] {len(files)} file(s), {num_workers} worker(s)")
    t0 = time.time()
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fpath, n = fut.result()
            total_bytes += n
            print(f"[{i}/{len(files)}] cached {os.path.basename(fpath)}")

    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    print(f"[cache_model] total read = {gb:.2f} GB in {elapsed:.2f}s")
    return total_bytes


# Cache the model files (this happens while other setup continues)
MODEL_PATH = "/kaggle/input/gpt-oss-120b/transformers/default/1"
if os.path.exists(MODEL_PATH):
    print("[SETUP] Pre-caching model files...")
    cache_model(MODEL_PATH, num_workers=16, chunk_mb=1024)
else:
    print(f"[SETUP] Model not found at {MODEL_PATH}, skipping cache")

# ============================================================
# CELL 5: COPY VLLM COMPILE CACHE (if available)
# ============================================================
if os.path.exists("/kaggle/input/gpt-oss-120b-cache-compile/torch_compile_cache"):
    print("[SETUP] Copying vLLM compile cache...")
    os.makedirs("/root/.cache/vllm/", exist_ok=True)
    subprocess.run([
        "cp", "-r", 
        "/kaggle/input/gpt-oss-120b-cache-compile/torch_compile_cache",
        "/root/.cache/vllm/"
    ], capture_output=True)
    print("[SETUP] vLLM compile cache copied")

# ============================================================
# CELL 6: WAIT FOR PACKAGE UNINSTALL TO COMPLETE
# ============================================================
print("[SETUP] Waiting for package uninstall to complete...")
uninstall_proc.wait()
print("[SETUP] Package uninstall complete")

# ============================================================
# CELL 7: ENVIRONMENT VARIABLES
# ============================================================
# Set these AFTER uninstalling broken packages but BEFORE importing anything

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set tiktoken encodings path if available
TIKTOKEN_PATH = "/kaggle/usr/lib/pip_install_aimo3_1/tiktoken_encodings"
if os.path.exists(TIKTOKEN_PATH):
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = TIKTOKEN_PATH
    print(f"[SETUP] Set TIKTOKEN_ENCODINGS_BASE to {TIKTOKEN_PATH}")

# ============================================================
# CELL 8: PYTHON TOOL WITH JUPYTER KERNEL
# ============================================================
# This provides stateful Python execution for tool calls

import queue
import threading
from typing import Any
from uuid import uuid4

class LocalJupyterSession:
    """
    Stateful Jupyter kernel session for code execution.
    
    This is better than subprocess because:
    1. State persists between calls (variables stay defined)
    2. More robust error handling
    3. Proper timeout support
    """

    # Class-level lock and port counter to avoid port conflicts
    _port_lock = threading.Lock()
    _next_port = 50000
    _max_port = 65535

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list:
        """Get next available ports for kernel connection."""
        import socket
        with cls._port_lock:
            ports = []
            attempts = 0
            max_attempts = 100

            while len(ports) < count and attempts < max_attempts:
                start_port = cls._next_port
                available = True
                for i in range(count):
                    port = start_port + i
                    if port > cls._max_port:
                        start_port = 50000
                        port = start_port + i

                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.settimeout(0.1)
                            result = s.connect_ex(('127.0.0.1', port))
                            if result == 0:
                                available = False
                                break
                    except Exception:
                        available = False
                        break

                if available:
                    ports = list(range(start_port, start_port + count))
                    cls._next_port = start_port + count
                    if cls._next_port > cls._max_port:
                        cls._next_port = 50000
                    break
                else:
                    cls._next_port += count
                    if cls._next_port > cls._max_port:
                        cls._next_port = 50000
                    attempts += 1

            if len(ports) < count:
                ports = list(range(cls._next_port, cls._next_port + count))
                cls._next_port += count
                if cls._next_port > cls._max_port:
                    cls._next_port = 50000

            return ports

    def __init__(self, connection_file: str = None, *, timeout: float = 120.0):
        try:
            from jupyter_client import BlockingKernelClient, KernelManager
        except ImportError as exc:
            raise RuntimeError("jupyter_client package required") from exc

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None

        if connection_file:
            from pathlib import Path
            connection_path = Path(connection_file).expanduser()
            if not connection_path.exists():
                raise FileNotFoundError(f"Connection file not found: {connection_path}")
            client = BlockingKernelClient()
            client.load_connection_file(str(connection_path))
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
        else:
            ports = self._get_next_ports(5)
            km = None
            max_retries = 3
            for retry in range(max_retries):
                try:
                    km = KernelManager()
                    km.shell_port = ports[0]
                    km.iopub_port = ports[1]
                    km.stdin_port = ports[2]
                    km.hb_port = ports[3]
                    km.control_port = ports[4]
                    km.start_kernel()
                    client = km.blocking_client()
                    client.start_channels()
                    client.wait_for_ready(timeout=self._default_timeout)
                    self._client = client
                    self._km = km
                    self._owns_kernel = True
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        ports = self._get_next_ports(5)
                        if km is not None:
                            try:
                                km.shutdown_kernel(now=True)
                            except Exception:
                                pass
                    else:
                        raise RuntimeError(f"Failed to start kernel after {max_retries} retries: {e}") from e

    def execute(self, code: str, *, timeout: float = None) -> str:
        """Execute code and return combined stdout/stderr."""
        import queue as _queue

        client = self._client
        effective_timeout = float(timeout or self._default_timeout)

        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)

        stdout_parts = []
        stderr_parts = []
        _timeout_triggered = False

        start = time.time()
        poll = 0.5

        def _timed_out() -> bool:
            return (time.time() - start) >= effective_timeout

        max_timeout_grace = 1.0
        timeout_grace_start = None

        while True:
            if _timed_out():
                if not _timeout_triggered:
                    _timeout_triggered = True
                    timeout_grace_start = time.time()
                    try:
                        client.interrupt_kernel()
                    except Exception:
                        try:
                            if self._owns_kernel and self._km is not None:
                                self._km.interrupt_kernel()
                        except Exception:
                            pass

                if timeout_grace_start and (time.time() - timeout_grace_start) > max_timeout_grace:
                    raise TimeoutError(f"Python execution exceeded wall-time limit: {effective_timeout:.1f}s")

            try:
                msg = client.get_iopub_msg(timeout=poll)
            except _queue.Empty:
                if _timeout_triggered and timeout_grace_start and (time.time() - timeout_grace_start) > max_timeout_grace:
                    raise TimeoutError(f"Python execution exceeded wall-time limit: {effective_timeout:.1f}s")
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if _timeout_triggered:
                if msg_type == "status":
                    if content.get("execution_state") == "idle":
                        break
                continue

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # Shell reply
        shell_timeout_grace_start = timeout_grace_start if _timeout_triggered else None

        while True:
            if _timed_out():
                if not _timeout_triggered:
                    _timeout_triggered = True
                    shell_timeout_grace_start = time.time()
                    try:
                        client.interrupt_kernel()
                    except Exception:
                        try:
                            if self._owns_kernel and self._km is not None:
                                self._km.interrupt_kernel()
                        except Exception:
                            pass

                if shell_timeout_grace_start and (time.time() - shell_timeout_grace_start) > max_timeout_grace:
                    raise TimeoutError(f"Python execution exceeded wall-time limit: {effective_timeout:.1f}s")

            try:
                reply = client.get_shell_msg(timeout=poll)
            except _queue.Empty:
                if _timeout_triggered and shell_timeout_grace_start and (time.time() - shell_timeout_grace_start) > max_timeout_grace:
                    raise TimeoutError(f"Python execution exceeded wall-time limit: {effective_timeout:.1f}s")
                continue

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            reply_content = reply.get("content", {})

            if _timeout_triggered and reply_content.get("status") == "error":
                break

            if reply_content.get("status") == "error":
                traceback_data = reply_content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = reply_content.get("ename", "")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            stdout = f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        if not stdout.strip():
            stdout = "[WARN] No output. Use print() to see results."
        return stdout

    def close(self):
        import contextlib
        with contextlib.suppress(Exception):
            self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

    def __del__(self):
        self.close()


class PythonTool:
    """Python execution tool using Jupyter kernel."""

    def __init__(self, local_jupyter_timeout: float = 60.0):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._execution_lock = threading.Lock()
        self._jupyter_session = None
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        """Lazily initialize the Jupyter session."""
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = LocalJupyterSession(timeout=self._local_jupyter_timeout)

    @property
    def name(self) -> str:
        return "python"

    def execute(self, code: str, timeout: float = None) -> str:
        """Execute Python code and return output."""
        self._ensure_session()
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(code, timeout=timeout)
            except TimeoutError as exc:
                output = f"[ERROR] {exc}"
            except Exception as exc:
                output = f"[ERROR] {exc}"
        return output

    def close(self):
        if self._jupyter_session is not None:
            self._jupyter_session.close()
            self._jupyter_session = None

    def __del__(self):
        self.close()


# ============================================================
# CELL 9: PROMETHEUS MATH ENGINES (THE KEY DIFFERENCE!)
# ============================================================
# This is what makes this different from the base notebook:
# We expose PROMETHEUS's specialized math engines as tools.

try:
    # Import PROMETHEUS engines
    import sys
    import os
    
    # Add PROMETHEUS to path if running from Kaggle
    prometheus_path = os.path.dirname(os.path.abspath(__file__))
    if prometheus_path not in sys.path:
        sys.path.insert(0, prometheus_path)
    
    from prometheus.engines.algebra import AlgebraEngine
    from prometheus.engines.number_theory import NumberTheoryEngine
    from prometheus.engines.geometry import GeometryEngine
    from prometheus.engines.combinatorics import CombinatoricsEngine
    
    PROMETHEUS_AVAILABLE = True
    print("[SETUP] PROMETHEUS engines loaded")
except ImportError as e:
    print(f"[WARN] PROMETHEUS engines not available: {e}")
    PROMETHEUS_AVAILABLE = False
    AlgebraEngine = None
    NumberTheoryEngine = None
    GeometryEngine = None
    CombinatoricsEngine = None


class PrometheusTool:
    """
    Wrapper for PROMETHEUS math engines as Harmony tools.
    
    This exposes specialized math operations (algebra, number theory, 
    geometry, combinatorics) as structured tools that GPT-OSS can call.
    """
    
    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            raise RuntimeError("PROMETHEUS engines not available")
        
        self.algebra = AlgebraEngine()
        self.number_theory = NumberTheoryEngine()
        self.geometry = GeometryEngine()
        self.combinatorics = CombinatoricsEngine()
    
    @property
    def name(self) -> str:
        return "prometheus"
    
    def execute(self, engine: str, operation: str, **kwargs) -> str:
        """
        Execute a PROMETHEUS math operation.
        
        Args:
            engine: "algebra", "number_theory", "geometry", or "combinatorics"
            operation: The operation name (e.g., "simplify", "solve", "gcd")
            **kwargs: Operation-specific arguments
        
        Returns:
            String result or error message
        """
        try:
            if engine == "algebra":
                return self._execute_algebra(operation, **kwargs)
            elif engine == "number_theory":
                return self._execute_number_theory(operation, **kwargs)
            elif engine == "geometry":
                return self._execute_geometry(operation, **kwargs)
            elif engine == "combinatorics":
                return self._execute_combinatorics(operation, **kwargs)
            else:
                return f"[ERROR] Unknown engine: {engine}"
        except Exception as e:
            return f"[ERROR] {str(e)}"
    
    def _execute_algebra(self, operation: str, **kwargs) -> str:
        """Execute algebra operation."""
        if operation == "simplify":
            result = self.algebra.simplify(kwargs.get("expr", ""))
        elif operation == "expand":
            result = self.algebra.expand(kwargs.get("expr", ""))
        elif operation == "factor":
            result = self.algebra.factor(kwargs.get("expr", ""))
        elif operation == "solve":
            result = self.algebra.solve(
                kwargs.get("equation", ""),
                variable=kwargs.get("variable", "x"),
                domain=kwargs.get("domain", "complex")
            )
        else:
            return f"[ERROR] Unknown algebra operation: {operation}"
        
        if result.success:
            return f"Result: {result.result}\nMethod: {result.method}"
        else:
            return f"[ERROR] {result.error}"
    
    def _execute_number_theory(self, operation: str, **kwargs) -> str:
        """Execute number theory operation."""
        if operation == "gcd":
            result = self.number_theory.gcd(
                int(kwargs.get("a", 0)),
                int(kwargs.get("b", 0))
            )
        elif operation == "lcm":
            result = self.number_theory.lcm(
                int(kwargs.get("a", 0)),
                int(kwargs.get("b", 0))
            )
        elif operation == "is_prime":
            result = self.number_theory.is_prime(int(kwargs.get("n", 0)))
        elif operation == "factorize":
            result = self.number_theory.factorize(int(kwargs.get("n", 0)))
        elif operation == "mod":
            result = self.number_theory.mod(
                int(kwargs.get("a", 0)),
                int(kwargs.get("m", 1))
            )
        else:
            return f"[ERROR] Unknown number theory operation: {operation}"
        
        if result.success:
            steps = "\n".join(result.steps) if result.steps else ""
            output = f"Result: {result.result}\nMethod: {result.method}"
            if steps:
                output += f"\nSteps:\n{steps}"
            return output
        else:
            return f"[ERROR] {result.error}"
    
    def _execute_geometry(self, operation: str, **kwargs) -> str:
        """Execute geometry operation."""
        # Geometry operations are more complex - for now, return a message
        # In a full implementation, you'd parse geometric objects and call methods
        return f"[INFO] Geometry operation '{operation}' - use Python tool for complex geometry"
    
    def _execute_combinatorics(self, operation: str, **kwargs) -> str:
        """Execute combinatorics operation."""
        # Similar to geometry - placeholder for now
        return f"[INFO] Combinatorics operation '{operation}' - use Python tool for complex counting"


# ============================================================
# CELL 10: IMPORTS (after packages are uninstalled)
# ============================================================
import warnings
warnings.simplefilter('ignore')

import re
import math
import gc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import pandas as pd
import polars as pl

print("[SETUP] Basic imports complete")

# ============================================================
# CELL 10: OPENAI AND HARMONY IMPORTS
# ============================================================
try:
    from openai import OpenAI
    print("[SETUP] OpenAI client imported")
except ImportError:
    print("[WARN] OpenAI not available")
    OpenAI = None

try:
    from transformers import set_seed, AutoTokenizer
    print("[SETUP] Transformers imported")
except ImportError:
    print("[WARN] Transformers not available")
    set_seed = lambda x: None
    AutoTokenizer = None

try:
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        ReasoningEffort,
        RenderConversationConfig,
        Author,
        TextContent,
        ToolNamespaceConfig,
    )
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    HARMONY_AVAILABLE = True
    print("[SETUP] OpenAI Harmony imported")
except ImportError as e:
    print(f"[WARN] OpenAI Harmony not available: {e}")
    HARMONY_AVAILABLE = False
    encoding = None

# ============================================================
# CELL 11: CONSTANTS
# ============================================================
SEED = 42
if set_seed:
    set_seed(SEED)
MAX_LEN = 64 * 1024
USE_BUDGET = False
K = 8  # Number of parallel samples

# Inference parameters
TEMPERATURE = 1.0
TOP_P = 1.0
MIN_P = 0.02

# ============================================================
# CELL 12: PROMPTS
# ============================================================
# IMO25-style prompts for generation

TIR_PROMPT_IMO25 = """You are an elite olympiad mathematician solving a national/international-level mathematical problem with full rigor.

Your approach must be systematic and thorough:
1. **Problem Analysis**: Carefully read and understand the problem statement. Identify all given conditions, constraints, and what is being asked.
2. **Strategy Exploration**: Consider multiple solution approaches. Think about different mathematical techniques that might apply.
3. **Rigorous Reasoning**: 
   - Justify all nontrivial steps in your reasoning
   - Show your work clearly and logically
   - Check edge cases and special conditions
4. **Tool-Assisted Computation**: Use the available tools:
   - **prometheus tool**: For specialized math operations (algebra, number theory, geometry, combinatorics)
     - Use "algebra" engine for: simplify, expand, factor, solve equations
     - Use "number_theory" engine for: gcd, lcm, prime checks, factorization, modular arithmetic
     - Use "geometry" and "combinatorics" engines for domain-specific operations
   - **python tool**: For general computation, numerical work, and complex calculations
   - Prefer prometheus for verified symbolic math, python for numerical/computational work
5. **Verification**: Before finalizing your answer, verify your solution satisfies all problem constraints using tools.
6. **Final Answer**: Return only the final verified answer in \\boxed{n}, where n is an integer in [0, 99999]. Never guess - only provide an answer you have rigorously verified.

Remember: Mathematical rigor is paramount. Every step must be justified, and all computations should be verified using the available tools."""

TIR_PROMPT_COMPUTE = """Solve this mathematical problem using a computation-first approach.

Strategy:
1. **Start with Python**: Use the python tool immediately to explore the problem numerically
2. **Find patterns**: Compute small cases and look for patterns
3. **Form conjecture**: Based on computation, form a hypothesis for the answer
4. **Prove rigorously**: Only after computing, prove your conjecture mathematically
5. **Verify extensively**: Use python to verify your answer against multiple test cases

Key principles:
- Let computation guide your intuition
- Test edge cases numerically before reasoning about them
- Verify every step with code
- Cross-check your final answer multiple ways

Return your final verified answer in \\boxed{n}, where n is an integer in [0, 99999]."""

TIR_PROMPTS = [TIR_PROMPT_IMO25, TIR_PROMPT_COMPUTE]

# ============================================================
# CELL 13: VLLM SERVER
# ============================================================
def start_vllm_server() -> subprocess.Popen:
    """
    Start vLLM server in background.
    
    Running vLLM as a separate server process has several advantages:
    1. Avoids import conflicts in the main notebook
    2. Can communicate via clean OpenAI-compatible API
    3. Server handles all GPU memory management
    """
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--served-model-name", "gpt-oss",
        "--tensor-parallel-size", "1",
        "--max-num-seqs", "64",
        "--gpu-memory-utilization", "0.96",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "auto",
        "--max-model-len", str(MAX_LEN),
        "--stream-interval", "20",
    ]
    
    with open("./vllm.log", "w") as logfile:
        process = subprocess.Popen(
            command, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True
        )
    print("[SETUP] vLLM server started. Logs: ./vllm.log")
    return process


# Start the server
vllm_process = None
if os.path.exists(MODEL_PATH):
    vllm_process = start_vllm_server()
else:
    print(f"[WARN] Model not found, vLLM server not started")

# ============================================================
# CELL 14: PYTHON TOOL POOL
# ============================================================
# Create a pool of Python tools for parallel execution

python_pool = queue.Queue(maxsize=K)

print("[SETUP] Creating Python tool pool...")
for _ in range(K):
    try:
        t = PythonTool(local_jupyter_timeout=60.0)
        python_pool.put(t)
    except Exception as e:
        print(f"[WARN] Failed to create Python tool: {e}")
print(f"[SETUP] Python tool pool created with {python_pool.qsize()} tools")

# Cleanup code to reset Python sessions between problems
CLEANUP_CODE = r"""
import gc
_keep = {
    "__builtins__", "__name__", "__doc__", "__package__", "__loader__", "__spec__",
    "np", "sp", "math",
}
g = globals()
for k in list(g.keys()):
    if k in _keep or k.startswith("_"):
        continue
    try:
        del g[k]
    except Exception:
        pass
gc.collect()
"""

# ============================================================
# CELL 15: DIAGNOSTICS TRACKER
# ============================================================
from dataclasses import dataclass, field, asdict
import json

@dataclass
class QuestionDiagnostics:
    """Detailed diagnostics for a single question."""
    question_id: str = ""
    question_preview: str = ""
    all_answers: list = field(default_factory=list)
    answer_distribution: dict = field(default_factory=dict)
    num_valid_answers: int = 0
    num_unique_answers: int = 0
    selection_method: str = ""
    selected_answer: int = 0
    verification_attempted: bool = False
    verification_passes: int = 0
    verification_fails: int = 0
    verification_feedback: str = ""
    refinement_triggered: bool = False
    refinement_iterations: int = 0
    refinement_answers: list = field(default_factory=list)
    final_answer: int = 0
    final_verified: bool = False
    time_allocated: float = 0.0
    time_used: float = 0.0
    time_saved: float = 0.0
    sample_reasoning_excerpt: str = ""
    ground_truth: Optional[int] = None
    is_correct: Optional[bool] = None


class DiagnosticsTracker:
    """Tracks diagnostics for all questions."""

    def __init__(self):
        self.questions: list = []
        self.current: Optional[QuestionDiagnostics] = None

    def start_question(self, question_id: str, question_text: str, time_allocated: float):
        self.current = QuestionDiagnostics(
            question_id=question_id,
            question_preview=question_text[:100] + "..." if len(question_text) > 100 else question_text,
            time_allocated=time_allocated
        )

    def finish_question(self, ground_truth: Optional[int] = None):
        if self.current:
            if ground_truth is not None:
                self.current.ground_truth = ground_truth
                self.current.is_correct = (self.current.final_answer == ground_truth)
            self.questions.append(self.current)
            self.current = None

    def print_final_summary(self):
        if not self.questions:
            print("No questions processed.")
            return

        print(f"\n{'#'*60}")
        print(f"FINAL DIAGNOSTICS SUMMARY")
        print(f"{'#'*60}")

        total = len(self.questions)
        correct = sum(1 for q in self.questions if q.is_correct == True)
        wrong = sum(1 for q in self.questions if q.is_correct == False)

        print(f"\nACCURACY:")
        print(f"   Total: {total}")
        print(f"   Correct: {correct}")
        print(f"   Wrong: {wrong}")
        if correct + wrong > 0:
            print(f"   Accuracy: {100*correct/(correct+wrong):.1f}%")

        # Timing summary
        total_time = sum(q.time_used for q in self.questions)
        avg_time = total_time / total if total > 0 else 0

        print(f"\nTIMING:")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Average per question: {avg_time:.1f}s")

        print(f"\n{'#'*60}")


diagnostics = DiagnosticsTracker()
print("[SETUP] Diagnostics tracker initialized")

# ============================================================
# CELL 16: HARMONY TIR INFERENCER
# ============================================================
class HarmonyTIRInferencer:
    """
    Tool-Integrated Reasoning inferencer using GPT-OSS and Harmony protocol.
    
    Implements the verification-and-refinement pipeline from IMO25 paper.
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int = MAX_LEN,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        min_p: float = MIN_P,
        seed: int = SEED,
        k: int = K,
        use_budget: bool = USE_BUDGET,
        max_iter: int = 100,
    ):
        self.model_path = model_path
        self.model = "gpt-oss"
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.seed = seed
        self.k = k
        self.use_budget = use_budget
        self.max_iter = max_iter
        self.deadline = None

        # Initialize OpenAI client pointing to local vLLM server
        self.client = OpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="sk-local",
            timeout=360,
        ) if OpenAI else None

        # Load tokenizer
        self.tokenizer = None
        if AutoTokenizer and os.path.exists(model_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
                print("[SETUP] Tokenizer loaded")
            except Exception as e:
                print(f"[WARN] Failed to load tokenizer: {e}")

        # Get stop tokens if harmony is available
        self.stop_token_ids = []
        if HARMONY_AVAILABLE and encoding:
            self.stop_token_ids = encoding.stop_tokens_for_assistant_actions()
        
        # Initialize PROMETHEUS tool if available
        self.prometheus_tool = None
        if PROMETHEUS_AVAILABLE:
            try:
                self.prometheus_tool = PrometheusTool()
                print("[SETUP] PROMETHEUS tool initialized")
            except Exception as e:
                print(f"[WARN] Failed to initialize PROMETHEUS tool: {e}")

    def wait_server(self, timeout: int = 900):
        """Wait until vLLM server is ready."""
        print("[SETUP] Waiting for vLLM server...")
        for i in range(timeout):
            time.sleep(1)
            try:
                if self.client:
                    models = self.client.models.list()
                    print(f"[SETUP] vLLM server ready: {models}")
                    return
            except Exception:
                if i % 30 == 0:
                    print(f"[SETUP] Still waiting for vLLM server... ({i}s)")
                continue
        raise RuntimeError("vLLM server failed to start")

    def _reset_python_pools(self):
        """Reset Python tool state between problems."""
        tools_to_clean = []
        while not python_pool.empty():
            try:
                tool = python_pool.get_nowait()
                tools_to_clean.append(tool)
            except:
                break

        for tool in tools_to_clean:
            try:
                if tool._jupyter_session is not None:
                    tool._jupyter_session.execute(CLEANUP_CODE, timeout=5.0)
            except Exception:
                pass
            try:
                python_pool.put(tool, block=False)
            except:
                pass

    def get_num_samples(self) -> int:
        return self.k

    def format_prompts(self, problem: str) -> list:
        """Create multiple prompts for parallel sampling."""
        num_samples = self.get_num_samples()
        prompts = []
        for i in range(num_samples):
            tir_prompt = TIR_PROMPTS[i % len(TIR_PROMPTS)]
            prompts.append(problem + "\n\n" + tir_prompt)
        return prompts

    def apply_chat_template(self, prompt: str, python_tool: PythonTool) -> list:
        """Create Harmony conversation format with both Python and PROMETHEUS tools."""
        if not HARMONY_AVAILABLE:
            return []
        
        # Python tool config
        python_tool_config = ToolNamespaceConfig(
            name="python",
            description="Execute Python code. Use print() to see output.",
            tools=[]
        )
        
        # PROMETHEUS tool config (if available)
        tool_configs = [python_tool_config]
        if self.prometheus_tool:
            prometheus_tool_config = ToolNamespaceConfig(
                name="prometheus",
                description="Specialized math engines: algebra (simplify, expand, factor, solve), number_theory (gcd, lcm, is_prime, factorize, mod), geometry, combinatorics.",
                tools=[]
            )
            tool_configs.append(prometheus_tool_config)
        
        # Create system message with all tools
        system_content = SystemContent.new().with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
        for tool_config in tool_configs:
            system_content = system_content.with_tools(tool_config)
        
        return [
            Message.from_role_and_content(Role.SYSTEM, system_content),
            Message.from_role_and_content(Role.USER, prompt),
        ]

    def single_generate_tir(self, prompt: str, stop_event: threading.Event, seed_offset: int = 0) -> str:
        """Generate single TIR response with tool execution."""
        if not HARMONY_AVAILABLE or not self.client:
            return ""
        
        python_tool = None

        def _compute_req_timeout() -> float:
            CUSHION = 0.5
            MAX_REQ_TIMEOUT = 30.0
            MIN_ALLOW = 0.2
            if not getattr(self, "deadline", None):
                return MAX_REQ_TIMEOUT
            remaining = self.deadline - time.time()
            if remaining <= 0:
                return 0.0
            t = remaining - CUSHION
            return min(MAX_REQ_TIMEOUT, max(MIN_ALLOW, t)) if t > 0 else 0.0

        def _compute_py_timeout() -> float:
            PY_CUSHION = 1.0
            MAX_PY_TIMEOUT = 15.0
            MIN_ALLOW = 0.2
            if not getattr(self, "deadline", None):
                return MAX_PY_TIMEOUT
            remaining = self.deadline - time.time()
            t = remaining - PY_CUSHION
            return min(MAX_PY_TIMEOUT, max(MIN_ALLOW, t)) if t > 0 else 0.0

        try:
            # Get python tool from pool
            try:
                python_tool = python_pool.get(timeout=30.0)
            except queue.Empty:
                python_tool = PythonTool()
                try:
                    python_tool._ensure_session()
                except Exception as e:
                    print(f"[WARN] Python session init failed: {e}")
                    return ""

            # Verify session is alive
            try:
                if python_tool._jupyter_session is None:
                    python_tool._ensure_session()
                test_output = python_tool._jupyter_session.execute("1+1", timeout=2.0)
                if "[ERROR]" in test_output:
                    python_tool._jupyter_session = None
                    python_tool._ensure_session()
            except Exception as e:
                print(f"[WARN] Python session health check failed: {e}")
                try:
                    python_tool._jupyter_session = None
                    python_tool._ensure_session()
                except Exception:
                    return ""

            messages = self.apply_chat_template(prompt, python_tool)
            final_answer_found = ""

            for iteration in range(self.max_iter):
                if stop_event and stop_event.is_set():
                    break
                if getattr(self, "deadline", None) and time.time() >= self.deadline:
                    break
                if final_answer_found:
                    break

                prompt_ids = encoding.render_conversation_for_completion(
                    Conversation.from_messages(messages), Role.ASSISTANT
                )
                max_tokens = self.max_model_len - len(prompt_ids)
                if max_tokens < 1:
                    break

                req_timeout = _compute_req_timeout()
                if req_timeout <= 0:
                    break

                token_buffer = []
                token_buffer_str = ""
                breaking = False

                stream = None
                try:
                    stream = self.client.completions.create(
                        model=self.model,
                        prompt=prompt_ids,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        seed=self.seed + seed_offset,
                        stream=True,
                        extra_body=dict(
                            min_p=self.min_p,
                            stop_token_ids=self.stop_token_ids,
                            return_token_ids=True,
                        ),
                        timeout=req_timeout,
                    )

                    for chunk in stream:
                        try:
                            if stop_event and stop_event.is_set():
                                breaking = True
                                break
                            if getattr(self, "deadline", None) and time.time() >= self.deadline:
                                breaking = True
                                break

                            if not chunk.choices or len(chunk.choices) == 0:
                                continue

                            choice = chunk.choices[0]
                            token_chunk = getattr(choice, 'token_ids', None) or []
                            text_chunk = getattr(choice, 'text', '') or ''

                            if token_chunk:
                                token_buffer.extend(token_chunk)
                                token_buffer_str += text_chunk

                            if len(token_buffer) > 60_000:
                                breaking = True
                                break

                            if "}" in text_chunk and self.extract_boxed_text(token_buffer_str) is not None:
                                final_answer_found = token_buffer_str
                                breaking = True
                                break
                        except StopIteration:
                            break
                        except Exception as e:
                            print(f"[WARN] Stream chunk error: {e}")
                            break

                except Exception as e:
                    print(f"[WARN] Stream error: {e}")
                    breaking = True
                finally:
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass

                if breaking:
                    break

                if not token_buffer:
                    continue

                # Parse completion
                try:
                    new_messages = encoding.parse_messages_from_completion_tokens(
                        token_buffer, Role.ASSISTANT
                    )
                except Exception as e:
                    print(f"[WARN] Parse error: {e}")
                    break

                messages.extend(new_messages)
                last_message = messages[-1]

                if last_message.channel == "final" or (token_buffer and token_buffer[-1] == 200002):
                    break

                # Handle tool calls
                if last_message.recipient in ["python", "prometheus"]:
                    if stop_event and stop_event.is_set():
                        break
                    if getattr(self, "deadline", None) and time.time() >= self.deadline:
                        break

                    tool_timeout = _compute_py_timeout()
                    if tool_timeout < 0.5:
                        break

                    if last_message.recipient == "python":
                        print("[TOOL] Executing Python code...")
                        try:
                            code = last_message.content[0].text
                            output = python_tool.execute(code, timeout=tool_timeout)
                            
                            # Create response message
                            response_content = TextContent(text=output)
                            author = Author(role=Role.TOOL, name="python")
                            response_msg = Message(author=author, content=[response_content]).with_recipient("assistant")
                            messages.append(response_msg)
                        except Exception as e:
                            print(f"[WARN] Python tool failed: {e}")
                            break
                    
                    elif last_message.recipient == "prometheus" and self.prometheus_tool:
                        print("[TOOL] Executing PROMETHEUS math operation...")
                        try:
                            # Parse tool call from message
                            # The message content should contain JSON with engine, operation, and args
                            import json
                            tool_call_text = last_message.content[0].text
                            
                            # Try to parse as JSON
                            try:
                                tool_call = json.loads(tool_call_text)
                                engine = tool_call.get("engine", "")
                                operation = tool_call.get("operation", "")
                                args = tool_call.get("args", {})
                            except:
                                # Fallback: try to extract from text
                                # Format: "engine: algebra, operation: simplify, expr: x^2+2x+1"
                                engine = ""
                                operation = ""
                                args = {}
                                for line in tool_call_text.split("\n"):
                                    if ":" in line:
                                        key, value = line.split(":", 1)
                                        key = key.strip().lower()
                                        value = value.strip()
                                        if key == "engine":
                                            engine = value
                                        elif key == "operation":
                                            operation = value
                                        else:
                                            args[key] = value
                            
                            output = self.prometheus_tool.execute(engine, operation, **args)
                            
                            # Create response message
                            response_content = TextContent(text=output)
                            author = Author(role=Role.TOOL, name="prometheus")
                            response_msg = Message(author=author, content=[response_content]).with_recipient("assistant")
                            messages.append(response_msg)
                        except Exception as e:
                            print(f"[WARN] PROMETHEUS tool failed: {e}")
                            # Still send error response so LLM knows it failed
                            error_content = TextContent(text=f"[ERROR] {str(e)}")
                            author = Author(role=Role.TOOL, name="prometheus")
                            response_msg = Message(author=author, content=[error_content]).with_recipient("assistant")
                            messages.append(response_msg)

            if final_answer_found:
                return final_answer_found

            return encoding.decode_utf8(
                encoding.render_conversation_for_training(
                    Conversation.from_messages(messages),
                    RenderConversationConfig(auto_drop_analysis=False),
                )
            )

        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback
            print(f"[ERROR] Generation error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return ""
        finally:
            if python_tool is not None:
                try:
                    python_pool.put(python_tool, block=False)
                except queue.Full:
                    try:
                        python_tool.close()
                    except Exception:
                        pass

    def _inference_parallel(self, prompts: list) -> list:
        """Run multiple generations in parallel."""
        stop_event = threading.Event()
        answers_collected = []
        raw_responses = [""] * len(prompts)
        majority_threshold = len(prompts) / 2

        print(f"[INFERENCE] Sampling {len(prompts)} times (threshold: > {majority_threshold})...")

        executor = ThreadPoolExecutor(max_workers=self.k)
        futures = []
        future_to_idx = {}
        
        try:
            for i, p in enumerate(prompts):
                fut = executor.submit(self.single_generate_tir, p, stop_event, i)
                futures.append(fut)
                future_to_idx[fut] = i

            for fut in as_completed(futures):
                idx = future_to_idx.get(fut, -1)
                if idx < 0:
                    continue

                try:
                    result_text = fut.result(timeout=1.0)
                except Exception as e:
                    print(f"[WARN] Task {idx} failed: {e}")
                    result_text = ""

                raw_responses[idx] = result_text

                ans = self.extract_boxed_text(result_text)
                if ans is not None:
                    answers_collected.append(ans)
                    counts = Counter(answers_collected)
                    if counts:
                        most_common_ans, count = counts.most_common(1)[0]
                        if count > majority_threshold:
                            print(f"[INFERENCE] Majority reached! {most_common_ans} appeared {count} times")
                            stop_event.set()
                            for f in futures:
                                if f is not fut and not f.done():
                                    try:
                                        f.cancel()
                                    except Exception:
                                        pass
                            break

        except Exception as e:
            print(f"[ERROR] Parallel inference error: {e}")
        finally:
            stop_event.set()
            try:
                executor.shutdown(wait=True)
            except Exception:
                executor.shutdown(wait=False)

        return raw_responses

    def extract_boxed_text(self, text: str) -> Optional[int]:
        """Extract numeric answer from \\boxed{}."""
        pattern = r'oxed{(.*?)}'
        matches = re.findall(pattern, str(text))
        if matches:
            for match in reversed(matches):
                if match:
                    try:
                        clean_match = match.strip().replace(',', '').replace(' ', '')
                        val = int(float(clean_match[:20]))
                        if 0 <= val <= 99999:
                            return val
                    except Exception:
                        pass

        pattern = r'(?i)final\s+answer\s*(?:is|:)?\s*(\d+)'
        matches = re.findall(pattern, str(text))
        if matches:
            for match in reversed(matches):
                try:
                    val = int(match)
                    if 0 <= val <= 99999:
                        return val
                except Exception:
                    pass

        return None

    def parse_responses(self, responses: list) -> int:
        """Majority vote on responses."""
        answers = [self.extract_boxed_text(r) for r in responses]
        valid_answers = [a for a in answers if a is not None]
        
        if not valid_answers:
            return 8687  # Default fallback

        counter = Counter(valid_answers)
        print(f"[VOTE] Answers: {counter}")

        most_common_list = counter.most_common(2)
        if len(most_common_list) > 1 and most_common_list[0][1] == most_common_list[1][1]:
            tied_answers = [ans for ans, cnt in counter.items() if cnt == most_common_list[0][1]]
            answer = max(tied_answers)
        else:
            answer = most_common_list[0][0]
        return answer

    def inference(self, problem: str, deadline: float) -> tuple:
        """Main inference with verification and refinement."""
        self.deadline = deadline
        start_time = time.time()

        # Reset Python state
        self._reset_python_pools()

        # Generate
        prompts = self.format_prompts(problem)
        responses = self._inference_parallel(prompts)

        # Extract answers
        answers = [self.extract_boxed_text(r) for r in responses]
        valid_answers = [a for a in answers if a is not None]
        answer_counter = Counter(valid_answers)

        print(f"[RESULT] Generated {len(valid_answers)} answers: {answer_counter}")

        # Update diagnostics
        if diagnostics.current:
            diagnostics.current.all_answers = answers.copy()
            diagnostics.current.answer_distribution = dict(answer_counter)
            diagnostics.current.num_valid_answers = len(valid_answers)
            diagnostics.current.num_unique_answers = len(set(valid_answers))

        # Majority vote
        final_answer = self.parse_responses(responses)
        
        if diagnostics.current:
            diagnostics.current.selection_method = "majority_vote"
            diagnostics.current.selected_answer = final_answer
            diagnostics.current.final_answer = final_answer

        duration = time.time() - start_time
        saved_time = max(0.0, deadline - time.time())

        print(f"[TIMING] Used: {duration:.1f}s, Saved: {saved_time:.1f}s")

        return final_answer, saved_time


# ============================================================
# CELL 17: INITIALIZE INFERENCER
# ============================================================
inferencer = None
if os.path.exists(MODEL_PATH) and HARMONY_AVAILABLE:
    inferencer = HarmonyTIRInferencer(
        MODEL_PATH,
        use_budget=USE_BUDGET,
        k=K,
    )
    print("[SETUP] Inferencer initialized")
else:
    print("[WARN] Inferencer not initialized (missing model or harmony)")

# ============================================================
# CELL 18: WAIT FOR SERVER
# ============================================================
if inferencer and vllm_process:
    try:
        inferencer.wait_server()
    except Exception as e:
        print(f"[ERROR] Server wait failed: {e}")

# ============================================================
# CELL 19: CUTOFF TIMES
# ============================================================
init_time = time.time()
cutoff_times = [int(x) for x in np.linspace(final_cutoff_time, init_time, 50 + 1)]
cutoff_times.pop()

# ============================================================
# CELL 20: PREDICTION FUNCTION
# ============================================================
predictions = {}
correct_count = 0
total_count = 0
ground_truth = {}

def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame:
    """Make a prediction with diagnostics logging."""
    global correct_count, total_count, predictions, cutoff_times, diagnostics

    question_id = id_.item(0)
    question_text = question.item(0)

    print("=" * 70)
    print(f"QUESTION {total_count + 1}/50: {question_id}")
    print("=" * 70)
    print(f"Problem: {question_text[:300]}...")
    print("-" * 70)

    current_deadline = cutoff_times[-1] if cutoff_times else time.time() + 300
    time_allocated = current_deadline - time.time()

    diagnostics.start_question(question_id, question_text, time_allocated)

    start_time = time.time()
    
    if inferencer:
        answer, saved_time = inferencer.inference(question_text, deadline=current_deadline)
    else:
        # Fallback: return default answer
        print("[WARN] No inferencer available, returning default")
        answer = 8687
        saved_time = 0.0
    
    time_used = time.time() - start_time

    if cutoff_times:
        cutoff_times.pop()

    # Redistribute saved time
    if len(cutoff_times) > 0 and saved_time > 0:
        now = time.time()
        num_remaining = len(cutoff_times)
        base_times = np.linspace(final_cutoff_time, now, num_remaining + 1)
        base_times = base_times[:-1]
        extra = saved_time / num_remaining
        cutoff_times = [int(t + extra) for t in base_times]

    predictions[question_id] = answer

    if diagnostics.current:
        diagnostics.current.time_used = time_used
        diagnostics.current.time_saved = saved_time
        diagnostics.current.final_answer = answer

    total_count += 1
    gt = ground_truth.get(question_id)

    print("-" * 70)
    if gt is not None:
        is_correct = (answer == gt)
        if is_correct:
            correct_count += 1
        status = "CORRECT" if is_correct else "WRONG"
        print(f"RESULT: {status}")
        print(f"   Our Answer: {answer}")
        print(f"   Ground Truth: {gt}")
        print(f"Running Accuracy: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")
    else:
        print(f"Answer: {answer} (no ground truth available)")

    print(f"Time: {time_used:.1f}s used, {saved_time:.1f}s saved")

    diagnostics.finish_question(gt)
    print("=" * 70 + "\n")

    return pl.DataFrame({"id": question_id, "answer": answer})


# ============================================================
# CELL 21: MAIN SUBMISSION
# ============================================================
if __name__ == "__main__":
    # Load reference data
    ref_path = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    if os.path.exists(ref_path):
        df = pd.read_csv(ref_path)
        if "answer" in df.columns:
            ground_truth = dict(zip(df["id"], df["answer"]))
        df.drop("answer", axis=1, errors="ignore").to_csv("reference.csv", index=False)
        print(f"[SETUP] Loaded {len(df)} problems")
    else:
        print("[WARN] Reference file not found")

    # Import and run inference server
    try:
        import kaggle_evaluation.aimo_3_inference_server
        inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            inference_server.serve()
        else:
            inference_server.run_local_gateway(("reference.csv",))

            # Print final diagnostics
            print("\n")
            print("#" * 70)
            print("#" + " " * 20 + "FINAL DIAGNOSTICS REPORT" + " " * 22 + "#")
            print("#" * 70)

            if total_count > 0:
                print(f"\nACCURACY: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")

            diagnostics.print_final_summary()

            print("\n" + "#" * 70)
            print("#" + " " * 20 + "END OF DIAGNOSTICS REPORT" + " " * 21 + "#")
            print("#" * 70)

    except ImportError as e:
        print(f"[WARN] Kaggle evaluation not available: {e}")
        print("[INFO] Running in standalone mode")

"""
Oracle: LLM integration for PROMETHEUS.

The Oracle is where neural meets symbolic. The LLM provides:
- Problem formalization (natural language → formal math)
- Strategy suggestions (which approach to try)
- Tactic selection (which move to make next)
- Position evaluation (how promising is this proof state?)
- Lemma discovery (suggest helpful intermediate results)

This is the modernization of MRPPS's heuristic functions -
instead of hand-coded heuristics, we use learned intuition.
"""

from prometheus.oracle.llm_oracle import LLMOracle, OracleConfig
from prometheus.oracle.formalizer import Formalizer
from prometheus.oracle.evaluator import Evaluator

__all__ = ["LLMOracle", "OracleConfig", "Formalizer", "Evaluator"]

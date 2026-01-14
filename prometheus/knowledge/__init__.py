"""
Knowledge: Mathematical knowledge base.

This module contains:
- Olympiad theorems and lemmas
- Problem-solving techniques
- Common patterns and tricks

This is the "textbook" PROMETHEUS has learned from.
"""

from prometheus.knowledge.theorems import TheoremBase, Theorem, get_theorems
from prometheus.knowledge.techniques import Technique, get_techniques

__all__ = ["TheoremBase", "Theorem", "get_theorems", "Technique", "get_techniques"]

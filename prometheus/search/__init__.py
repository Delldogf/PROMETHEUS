"""
Search: Proof search algorithms.

This module implements the search strategies that explore the space
of possible proofs. Inspired by MRPPS's Q* algorithm, but modernized
with techniques from AlphaGo/AlphaProof:

- MCTS (Monte Carlo Tree Search): Balance exploration vs exploitation
- Beam Search: Keep track of multiple promising paths

The key insight: Use the LLM Oracle to guide search, not just random exploration.
"""

from prometheus.search.mcts import MCTSSearch, MCTSConfig
from prometheus.search.beam import BeamSearch, BeamConfig

__all__ = ["MCTSSearch", "MCTSConfig", "BeamSearch", "BeamConfig"]

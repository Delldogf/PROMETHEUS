# 🔥 PROMETHEUS

**P**roof-**R**easoning and **O**lympiad **M**athematical **E**ngine with **T**actic-**H**euristic **E**xtended **U**nderstanding **S**ystem

A modern neuro-symbolic system for solving International Mathematical Olympiad level problems, inspired by the Maryland Refutation Proof Procedure System (MRPPS) from the 1970s.

---

## 🎯 Vision

PROMETHEUS combines the **rigor of symbolic reasoning** with the **intuition of large language models** to solve competition mathematics problems. Unlike pure LLM approaches that can hallucinate, PROMETHEUS verifies every step.

The core insight from MRPPS: **Separate WHAT (inference rules) from HOW (search strategy), and let learned heuristics guide the search.**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PROMETHEUS                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 5: PROBLEM UNDERSTANDING                                      │
│           └── LLM parses natural language → formal representation    │
│                                                                      │
│  Layer 4: STRATEGY SELECTION (Proof Planning)                        │
│           └── LLM suggests high-level approaches                     │
│                                                                      │
│  Layer 3: TACTIC ENGINE                                              │
│           └── Programmable proof tactics (like Lean/Isabelle)        │
│                                                                      │
│  Layer 2: INFERENCE CORE                                             │
│           └── Multi-logic reasoning + SMT solvers                    │
│                                                                      │
│  Layer 1: KNOWLEDGE BASE                                             │
│           └── Olympiad theorems, lemmas, techniques                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Key Innovation: LLM as Heuristic Oracle

In the original MRPPS, hand-crafted heuristics guided the search through proof space. 
PROMETHEUS replaces these with an LLM that has learned mathematical intuition from millions of examples.

The LLM provides:
1. **Problem Formalization** - Translates natural language to formal math
2. **Strategy Suggestions** - Proposes proof approaches ranked by likelihood
3. **Tactic Selection** - Chooses next proof steps within a strategy
4. **Position Evaluation** - Estimates probability of success (the Q* merit function)
5. **Lemma Discovery** - Proposes helpful intermediate results when stuck

---

## 📁 Project Structure

```
PROMETHEUS/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── prometheus/               # Main package
│   ├── __init__.py
│   │
│   ├── core/                 # Core abstractions
│   │   ├── __init__.py
│   │   ├── formula.py        # Mathematical formulas and expressions
│   │   ├── proof_state.py    # Current state of a proof attempt
│   │   ├── tactic.py         # Proof tactics (transformations)
│   │   └── strategy.py       # High-level proof strategies
│   │
│   ├── oracle/               # LLM integration
│   │   ├── __init__.py
│   │   ├── llm_oracle.py     # Main LLM interface
│   │   ├── formalizer.py     # Problem formalization
│   │   └── evaluator.py      # Position evaluation (merit function)
│   │
│   ├── engines/              # Specialized reasoning engines
│   │   ├── __init__.py
│   │   ├── algebra.py        # Algebraic manipulation
│   │   ├── number_theory.py  # Modular arithmetic, divisibility
│   │   ├── geometry.py       # Geometric reasoning
│   │   └── combinatorics.py  # Counting, bijections
│   │
│   ├── search/               # Proof search algorithms
│   │   ├── __init__.py
│   │   ├── mcts.py           # Monte Carlo Tree Search
│   │   └── beam.py           # Beam search alternative
│   │
│   ├── knowledge/            # Mathematical knowledge base
│   │   ├── __init__.py
│   │   ├── theorems.py       # Olympiad theorems
│   │   └── techniques.py     # Problem-solving techniques
│   │
│   └── pipeline/             # End-to-end pipeline
│       ├── __init__.py
│       ├── solver.py         # Main solver interface
│       └── aimo_adapter.py   # AIMO competition format adapter
│
├── tests/                    # Test suite
│   └── ...
│
└── notebooks/                # Jupyter notebooks for experimentation
    └── ...
```

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run on a sample problem
python -m prometheus.pipeline.solver "Find all positive integers n such that n^2 + 1 divides n^3 + 1"
```

---

## 🎓 Inspired By

- **MRPPS** (1970s) - Maryland Refutation Proof Procedure System
- **GPS** - General Problem Solver (means-ends analysis)
- **Omega** - Proof planning
- **Lean/Isabelle** - Tactic-based proof assistants
- **AlphaProof** - DeepMind's IMO solver approach

---

## 🏆 Target: AIMO 3 Competition

This system is being built for the AI Mathematical Olympiad 3 (AIMO 3) Kaggle competition:
- 110 original problems (Algebra, Number Theory, Geometry, Combinatorics)
- Difficulty: National Olympiad to IMO level
- Hardware: NVIDIA H100 GPUs
- Prize pool: $2.2 million

---

## 👥 Team

- **Project Lead & Architect**: Human (you!)
- **Implementation**: Claude (AI pair programmer)

---

*"Bringing fire to humanity" - PROMETHEUS steals the fire of mathematical reasoning from the gods of pure intelligence.*

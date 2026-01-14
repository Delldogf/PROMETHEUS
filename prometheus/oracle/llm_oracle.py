"""
LLM Oracle: The neural brain of PROMETHEUS.

This is the central LLM interface. All LLM calls go through here.
The Oracle answers questions like:
- "What does this problem ask for?" (formalization)
- "What strategy should I try?" (strategy selection)
- "What tactic should I use next?" (tactic selection)
- "How good is this proof state?" (evaluation)

WHAT THIS FILE DOES:
- Wraps the LLM API (supports multiple providers)
- Provides structured prompts for each type of query
- Parses LLM responses into usable data
- Caches results to avoid redundant calls
"""

from __future__ import annotations
import os
import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
from enum import Enum, auto

# These will be the actual LLM library imports
# For now we define the interface


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = auto()      # GPT-4, GPT-4-turbo
    ANTHROPIC = auto()   # Claude 3
    LOCAL = auto()       # Local model via transformers


@dataclass
class OracleConfig:
    """
    Configuration for the LLM Oracle.
    
    Attributes:
        provider: Which LLM provider to use
        model_name: Specific model name (e.g., "gpt-4", "claude-3-opus")
        temperature: Creativity (0=deterministic, 1=creative)
        max_tokens: Maximum response length
        api_key: API key (loaded from env if not provided)
    """
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-4"
    temperature: float = 0.3  # Lower = more consistent for math
    max_tokens: int = 2000
    api_key: Optional[str] = None
    
    # Caching settings
    use_cache: bool = True
    cache_dir: str = ".prometheus_cache"
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        if self.api_key is None:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")


@dataclass
class OracleQuery:
    """
    A query to the Oracle.
    """
    query_type: str  # "formalize", "strategy", "tactic", "evaluate", "lemma"
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    require_json: bool = False


@dataclass
class OracleResponse:
    """
    Response from the Oracle.
    """
    success: bool
    content: str
    parsed_data: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    tokens_used: int = 0
    from_cache: bool = False


class LLMOracle:
    """
    The LLM Oracle - neural guidance for proof search.
    
    This is where the magic happens. The Oracle uses a large language
    model to provide intuition about mathematical problems.
    
    Key methods:
    - formalize(): Convert natural language to formal math
    - suggest_strategies(): Propose proof approaches
    - select_tactic(): Choose the next proof step
    - evaluate_state(): Score how promising a state is
    - suggest_lemmas(): Propose helpful intermediate results
    """
    
    def __init__(self, config: Optional[OracleConfig] = None):
        self.config = config or OracleConfig()
        self._client = None
        self._cache: Dict[str, OracleResponse] = {}
        
        # Initialize the LLM client
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize the LLM client based on provider."""
        # This will be implemented based on which provider is configured
        # For now, we'll set up the structure
        pass
    
    def _get_cache_key(self, query: OracleQuery) -> str:
        """Generate a cache key for a query."""
        content = f"{query.query_type}:{query.prompt}:{json.dumps(query.context, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _call_llm(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Make the actual LLM API call.
        Returns the response content.
        """
        # TODO: Implement actual API calls
        # For now, return a placeholder
        return "LLM response placeholder"
    
    async def query(self, query: OracleQuery) -> OracleResponse:
        """
        Send a query to the Oracle.
        """
        # Check cache first
        if self.config.use_cache:
            cache_key = self._get_cache_key(query)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                return OracleResponse(
                    success=cached.success,
                    content=cached.content,
                    parsed_data=cached.parsed_data,
                    from_cache=True
                )
        
        # Build messages based on query type
        messages = self._build_messages(query)
        
        # Make the call
        try:
            response_text = await self._call_llm(messages)
            
            # Parse if needed
            parsed_data = None
            if query.require_json:
                try:
                    parsed_data = json.loads(response_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    parsed_data = self._extract_json(response_text)
            
            response = OracleResponse(
                success=True,
                content=response_text,
                parsed_data=parsed_data
            )
            
            # Cache the response
            if self.config.use_cache:
                self._cache[cache_key] = response
            
            return response
            
        except Exception as e:
            return OracleResponse(
                success=False,
                content=f"Error: {str(e)}"
            )
    
    def _build_messages(self, query: OracleQuery) -> List[Dict[str, str]]:
        """Build the message list for the LLM based on query type."""
        
        system_prompts = {
            "formalize": FORMALIZATION_SYSTEM_PROMPT,
            "strategy": STRATEGY_SYSTEM_PROMPT,
            "tactic": TACTIC_SYSTEM_PROMPT,
            "evaluate": EVALUATION_SYSTEM_PROMPT,
            "lemma": LEMMA_SYSTEM_PROMPT,
        }
        
        system_prompt = system_prompts.get(query.query_type, GENERAL_SYSTEM_PROMPT)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query.prompt}
        ]
        
        return messages
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract JSON from a text response."""
        # Look for JSON in code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Try the whole text
        try:
            return json.loads(text)
        except:
            pass
        
        return None
    
    # ===== High-Level Query Methods =====
    
    async def formalize(self, problem_text: str) -> Dict[str, Any]:
        """
        Formalize a natural language problem.
        
        Takes: "Find all positive integers n such that n² + 1 divides n³ + 1"
        Returns: Structured representation of the problem
        """
        query = OracleQuery(
            query_type="formalize",
            prompt=f"""Please formalize this mathematical problem:

{problem_text}

Provide your response as JSON with the following structure:
{{
    "problem_type": "find_all" | "prove" | "compute" | "find_minimum" | "find_maximum",
    "variables": [
        {{"name": "n", "domain": "positive_integers", "constraints": []}}
    ],
    "goal": "description of what we're looking for",
    "hypotheses": ["list of given conditions"],
    "mathematical_domain": ["number_theory", "algebra", "geometry", "combinatorics"],
    "key_expressions": ["n^2 + 1", "n^3 + 1"],
    "suggested_answer_format": "integer" | "list" | "expression"
}}""",
            require_json=True
        )
        
        response = await self.query(query)
        if response.success and response.parsed_data:
            return response.parsed_data
        return {"error": response.content}
    
    async def suggest_strategies(
        self, 
        problem_text: str, 
        formal_problem: Optional[Dict[str, Any]] = None,
        tried_strategies: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest proof strategies for a problem.
        
        Returns ranked list of strategies with explanations.
        """
        tried = tried_strategies or []
        tried_str = f"\n\nAlready tried: {', '.join(tried)}" if tried else ""
        
        formal_str = ""
        if formal_problem:
            formal_str = f"\n\nFormalized as: {json.dumps(formal_problem, indent=2)}"
        
        query = OracleQuery(
            query_type="strategy",
            prompt=f"""Suggest proof strategies for this problem:

{problem_text}
{formal_str}
{tried_str}

Provide your response as JSON array of strategies, ranked by likelihood of success:
[
    {{
        "strategy_name": "modular_reduction",
        "confidence": 0.8,
        "explanation": "Why this strategy is promising",
        "key_steps": ["Step 1", "Step 2", "Step 3"],
        "potential_obstacles": ["What might go wrong"]
    }}
]

Suggest 3-5 strategies.""",
            require_json=True
        )
        
        response = await self.query(query)
        if response.success and response.parsed_data:
            return response.parsed_data
        return []
    
    async def select_tactic(
        self,
        proof_state: Dict[str, Any],
        available_tactics: List[str],
        strategy_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select the next tactic to apply.
        
        This is called during proof search to decide what to do next.
        """
        strategy_str = f"\nCurrent strategy: {strategy_hint}" if strategy_hint else ""
        
        query = OracleQuery(
            query_type="tactic",
            prompt=f"""Select the next tactic to apply in this proof:

Current state:
{json.dumps(proof_state, indent=2)}

Available tactics: {', '.join(available_tactics)}
{strategy_str}

Respond with JSON:
{{
    "tactic": "name of tactic to use",
    "parameters": {{}},  // any parameters needed
    "explanation": "why this is the right move",
    "confidence": 0.8
}}""",
            require_json=True
        )
        
        response = await self.query(query)
        if response.success and response.parsed_data:
            return response.parsed_data
        return {"tactic": available_tactics[0] if available_tactics else "simplify"}
    
    async def evaluate_state(self, proof_state: Dict[str, Any]) -> float:
        """
        Evaluate how promising a proof state is.
        
        Returns a score from 0 (hopeless) to 1 (likely to succeed).
        This is the modernized Q* merit function from MRPPS.
        """
        query = OracleQuery(
            query_type="evaluate",
            prompt=f"""Evaluate how promising this proof state is:

{json.dumps(proof_state, indent=2)}

Rate from 0.0 (hopeless) to 1.0 (very likely to succeed).
Consider:
- How close are we to the goal?
- Are there obvious next steps?
- Have we made meaningful progress?
- Are there concerning contradictions or complications?

Respond with just a JSON object:
{{
    "score": 0.7,
    "reasoning": "brief explanation"
}}""",
            require_json=True
        )
        
        response = await self.query(query)
        if response.success and response.parsed_data:
            return float(response.parsed_data.get("score", 0.5))
        return 0.5
    
    async def suggest_lemmas(
        self,
        problem_text: str,
        proof_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Suggest helpful lemmas when we're stuck.
        
        This is like an experienced mathematician noticing
        "if we could just prove X, everything would follow."
        """
        query = OracleQuery(
            query_type="lemma",
            prompt=f"""We're stuck on this problem:

{problem_text}

Current proof state:
{json.dumps(proof_state, indent=2)}

Suggest helpful intermediate lemmas or results that, if true, 
would help complete the proof.

Respond with JSON:
[
    {{
        "lemma": "statement of the lemma",
        "why_helpful": "how it helps",
        "how_to_prove": "sketch of proof approach",
        "confidence": 0.7
    }}
]""",
            require_json=True
        )
        
        response = await self.query(query)
        if response.success and response.parsed_data:
            return response.parsed_data
        return []


# ===== System Prompts =====

GENERAL_SYSTEM_PROMPT = """You are PROMETHEUS, an expert mathematical reasoning system.
You specialize in International Mathematical Olympiad (IMO) level problems.
You combine deep mathematical knowledge with rigorous logical reasoning.
Always be precise and formal in your mathematical analysis."""

FORMALIZATION_SYSTEM_PROMPT = """You are PROMETHEUS's formalization engine.
Your job is to convert natural language math problems into structured formal representations.

Key responsibilities:
1. Identify the type of problem (prove, find all, compute, etc.)
2. Extract all variables and their domains
3. Identify all constraints and conditions
4. Determine the mathematical domain (number theory, algebra, geometry, combinatorics)
5. Identify key expressions that will be important

Be precise and complete. Miss nothing."""

STRATEGY_SYSTEM_PROMPT = """You are PROMETHEUS's strategy advisor.
Your job is to suggest high-level proof strategies for mathematical problems.

You know many strategies:
- Direct proof, contradiction, contrapositive
- Induction (simple, strong, structural)
- Case analysis
- Algebraic manipulation, substitution
- Modular arithmetic, divisibility analysis
- Geometric methods (coordinate, synthetic, trigonometric)
- Combinatorial methods (bijection, generating functions, pigeonhole)
- Pattern finding, working backwards

Consider what type of problem it is and what strategies historically work well.
Rank strategies by estimated probability of success."""

TACTIC_SYSTEM_PROMPT = """You are PROMETHEUS's tactic selector.
Your job is to choose the next proof step given the current state.

Think like an IMO competitor:
- What's the most promising move right now?
- What information haven't we used yet?
- Is there a clever observation that simplifies things?
- Should we compute some examples first?

Be specific about which tactic to use and with what parameters."""

EVALUATION_SYSTEM_PROMPT = """You are PROMETHEUS's position evaluator.
Your job is to assess how promising a proof state is.

Consider:
- Progress made toward the goal
- Complexity of remaining work
- Presence of useful structure or patterns
- Warning signs of dead ends
- Similarity to solved problems

Be calibrated: 0.5 is neutral, above is promising, below is concerning."""

LEMMA_SYSTEM_PROMPT = """You are PROMETHEUS's lemma discovery system.
Your job is to suggest helpful intermediate results when stuck.

A good lemma:
- Is easier to prove than the main goal
- Directly helps prove the main goal
- Captures key insight about the problem structure
- Might be independently interesting

Think about what a human mathematician would want to establish first."""

"""
Base data structures for query routing.

This module defines the core types used by the query router to classify
user intents and make routing decisions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QueryIntent(Enum):
    """
    Classification of user query intent.
    
    Different intents require different handling strategies:
    - KNOWLEDGE_QUERY: Semantic search in knowledge base
    - FACTUAL_LOOKUP: Specific fact retrieval (who, what, when)
    - COMPARISON: Compare multiple entities/concepts
    - COMPLEX_REASONING: Multi-step reasoning required
    - ADMIN_REQUEST: Administrative operations (stats, list docs)
    - CHITCHAT: Casual conversation, greetings
    - UNCLEAR: Intent cannot be determined
    """
    
    KNOWLEDGE_QUERY = "knowledge_query"
    FACTUAL_LOOKUP = "factual_lookup"
    COMPARISON = "comparison"
    COMPLEX_REASONING = "complex_reasoning"
    ADMIN_REQUEST = "admin_request"
    CHITCHAT = "chitchat"
    UNCLEAR = "unclear"


class RoutingStrategy(Enum):
    """
    Strategy for handling the query.
    
    Determines which tools and methods to use:
    - DIRECT_TOOL: Single tool call (e.g., query_knowledge_base)
    - MULTI_STEP: Multiple tool calls with reasoning
    - SIMPLE_RESPONSE: No tool needed, direct response
    - CLARIFICATION_NEEDED: Ask user for more info
    """
    
    DIRECT_TOOL = "direct_tool"
    MULTI_STEP = "multi_step"
    SIMPLE_RESPONSE = "simple_response"
    CLARIFICATION_NEEDED = "clarification_needed"


@dataclass
class RouteDecision:
    """
    Result of query routing decision.
    
    Contains all information needed to execute the query:
    - intent: Classified user intent
    - strategy: How to handle the query
    - tools: Which tools to use (if any)
    - confidence: How confident we are in this routing (0-1)
    - reasoning: Explanation of why this route was chosen
    - metadata: Additional routing metadata (filters, parameters, etc.)
    
    Attributes:
        intent: The classified query intent.
        strategy: The routing strategy to use.
        tools: List of tool names to invoke.
        confidence: Confidence score (0-1).
        reasoning: Explanation for this routing decision.
        metadata: Additional metadata for query execution.
    """
    
    intent: QueryIntent
    strategy: RoutingStrategy
    tools: list[str] = field(default_factory=list)
    confidence: float = 1.0
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate the decision."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
    
    def should_use_tools(self) -> bool:
        """
        Check if this route requires tool usage.
        
        Returns:
            bool: True if tools should be invoked.
        """
        return self.strategy in {RoutingStrategy.DIRECT_TOOL, RoutingStrategy.MULTI_STEP}
    
    def needs_clarification(self) -> bool:
        """
        Check if clarification is needed from the user.
        
        Returns:
            bool: True if we should ask for clarification.
        """
        return self.strategy == RoutingStrategy.CLARIFICATION_NEEDED
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary format.
        
        Returns:
            dict: Dictionary representation.
        """
        return {
            "intent": self.intent.value,
            "strategy": self.strategy.value,
            "tools": self.tools,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


@dataclass
class QueryAnalysis:
    """
    Detailed analysis of a user query.
    
    This is used internally by the router to make decisions.
    
    Attributes:
        query: Original user query.
        keywords: Extracted keywords.
        question_type: Type of question (what, how, when, etc.).
        entities: Named entities mentioned.
        requires_knowledge_base: Whether KB lookup is needed.
        is_greeting: Whether this is a greeting/chitchat.
        complexity_score: Estimated query complexity (0-1).
    """
    
    query: str
    keywords: list[str] = field(default_factory=list)
    question_type: Optional[str] = None
    entities: list[str] = field(default_factory=list)
    requires_knowledge_base: bool = True
    is_greeting: bool = False
    complexity_score: float = 0.5
    
    def is_simple_query(self) -> bool:
        """
        Check if this is a simple query.
        
        Returns:
            bool: True if query is simple (single-step).
        """
        return self.complexity_score < 0.5
    
    def is_complex_query(self) -> bool:
        """
        Check if this is a complex query.
        
        Returns:
            bool: True if query requires multi-step reasoning.
        """
        return self.complexity_score >= 0.7
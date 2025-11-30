"""
Query router for agent system.

This module implements the main routing logic that determines how to
handle user queries based on intent classification and context.
"""

from typing import Optional

from .base import QueryIntent, RouteDecision, RoutingStrategy
from .intent_classifier import IntentClassifier


class QueryRouter:
    """
    Routes user queries to appropriate handling strategies.
    
    The router:
    1. Analyzes the query to determine intent
    2. Decides which tools (if any) to use
    3. Determines the execution strategy
    4. Returns a routing decision with metadata
    
    Example usage:
        ```python
        router = QueryRouter()
        decision = router.route("What is the vacation policy?")
        
        if decision.should_use_tools():
            # Execute tools
            for tool_name in decision.tools:
                result = await execute_tool(tool_name, ...)
        ```
    
    Attributes:
        classifier: Intent classifier instance.
        min_confidence: Minimum confidence to proceed without clarification.
    """
    
    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        min_confidence: float = 0.7,
    ) -> None:
        """
        Initialize the query router.
        
        Args:
            classifier: Intent classifier (creates default if None).
            min_confidence: Minimum confidence threshold.
        """
        self.classifier = classifier or IntentClassifier()
        self.min_confidence = min_confidence
    
    def route(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> RouteDecision:
        """
        Route a user query to the appropriate handling strategy.
        
        Args:
            query: User query string.
            context: Optional context (conversation history, user info, etc.).
            
        Returns:
            RouteDecision: Routing decision with strategy and tools.
        """
        # Analyze query
        analysis = self.classifier.analyze_query(query)
        
        # Classify intent
        intent = self.classifier.classify(query)
        
        # Route based on intent
        if intent == QueryIntent.CHITCHAT:
            return self._route_chitchat(query, analysis)
        
        elif intent == QueryIntent.ADMIN_REQUEST:
            return self._route_admin_request(query, analysis)
        
        elif intent == QueryIntent.COMPARISON:
            return self._route_comparison(query, analysis)
        
        elif intent == QueryIntent.COMPLEX_REASONING:
            return self._route_complex_reasoning(query, analysis)
        
        elif intent == QueryIntent.FACTUAL_LOOKUP:
            return self._route_factual_lookup(query, analysis)
        
        elif intent == QueryIntent.KNOWLEDGE_QUERY:
            return self._route_knowledge_query(query, analysis)
        
        elif intent == QueryIntent.UNCLEAR:
            return self._route_unclear(query, analysis)
        
        else:
            # Default fallback
            return self._route_knowledge_query(query, analysis)
    
    def _route_chitchat(self, query: str, analysis) -> RouteDecision:
        """
        Route chitchat/greetings.
        
        No tools needed, simple response.
        """
        return RouteDecision(
            intent=QueryIntent.CHITCHAT,
            strategy=RoutingStrategy.SIMPLE_RESPONSE,
            tools=[],
            confidence=0.95,
            reasoning="Query is a greeting or casual conversation",
            metadata={
                "response_template": "greeting",
                "requires_knowledge_base": False,
            }
        )
    
    def _route_admin_request(self, query: str, analysis) -> RouteDecision:
        """
        Route administrative requests.
        
        Uses admin tools like get_knowledge_base_stats or search_documents.
        """
        # Determine which admin tool to use
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how many', 'count', 'total', 'statistics', 'stats']):
            tools = ['get_knowledge_base_stats']
            reasoning = "Query requests statistics about the knowledge base"
        
        elif any(word in query_lower for word in ['list', 'show', 'display', 'what documents']):
            tools = ['search_documents']
            reasoning = "Query requests document listing or search"
        
        else:
            tools = ['get_knowledge_base_stats']
            reasoning = "Default admin tool for general admin queries"
        
        return RouteDecision(
            intent=QueryIntent.ADMIN_REQUEST,
            strategy=RoutingStrategy.DIRECT_TOOL,
            tools=tools,
            confidence=0.85,
            reasoning=reasoning,
            metadata={
                "tool_category": "admin",
            }
        )
    
    def _route_comparison(self, query: str, analysis) -> RouteDecision:
        """
        Route comparison queries.
        
        Requires multiple knowledge base queries and synthesis.
        """
        # Extract entities to compare
        entities = analysis.entities if analysis.entities else []
        
        return RouteDecision(
            intent=QueryIntent.COMPARISON,
            strategy=RoutingStrategy.MULTI_STEP,
            tools=['query_knowledge_base'],  # Will be called multiple times
            confidence=0.8,
            reasoning="Query requires comparing multiple entities or concepts",
            metadata={
                "entities_to_compare": entities,
                "requires_synthesis": True,
                "estimated_steps": max(2, len(entities)),
            }
        )
    
    def _route_complex_reasoning(self, query: str, analysis) -> RouteDecision:
        """
        Route complex reasoning queries.
        
        Requires multi-step reasoning and synthesis.
        """
        return RouteDecision(
            intent=QueryIntent.COMPLEX_REASONING,
            strategy=RoutingStrategy.MULTI_STEP,
            tools=['query_knowledge_base'],
            confidence=0.75,
            reasoning="Query requires multi-step reasoning and analysis",
            metadata={
                "complexity_score": analysis.complexity_score,
                "requires_reasoning": True,
                "estimated_steps": 3,
            }
        )
    
    def _route_factual_lookup(self, query: str, analysis) -> RouteDecision:
        """
        Route factual lookup queries.
        
        Simple, direct knowledge base query.
        """
        return RouteDecision(
            intent=QueryIntent.FACTUAL_LOOKUP,
            strategy=RoutingStrategy.DIRECT_TOOL,
            tools=['query_knowledge_base'],
            confidence=0.9,
            reasoning="Query is a straightforward factual lookup",
            metadata={
                "question_type": analysis.question_type,
                "keywords": analysis.keywords,
                "top_k": 3,  # Factual queries need fewer results
            }
        )
    
    def _route_knowledge_query(self, query: str, analysis) -> RouteDecision:
        """
        Route general knowledge queries.
        
        Standard knowledge base search.
        """
        # Determine if simple or complex based on analysis
        if analysis.is_simple_query():
            strategy = RoutingStrategy.DIRECT_TOOL
            confidence = 0.85
            reasoning = "Simple knowledge query, direct search"
        else:
            strategy = RoutingStrategy.MULTI_STEP
            confidence = 0.75
            reasoning = "Complex knowledge query, may need multiple searches"
        
        return RouteDecision(
            intent=QueryIntent.KNOWLEDGE_QUERY,
            strategy=strategy,
            tools=['query_knowledge_base'],
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "keywords": analysis.keywords,
                "complexity": analysis.complexity_score,
                "top_k": 5,
            }
        )
    
    def _route_unclear(self, query: str, analysis) -> RouteDecision:
        """
        Route unclear queries.
        
        Request clarification from user.
        """
        return RouteDecision(
            intent=QueryIntent.UNCLEAR,
            strategy=RoutingStrategy.CLARIFICATION_NEEDED,
            tools=[],
            confidence=0.3,
            reasoning="Query intent is unclear, need more information",
            metadata={
                "clarification_message": (
                    "I'm not sure I understand your question. Could you please "
                    "provide more details or rephrase?"
                ),
            }
        )
    
    def explain_decision(self, decision: RouteDecision) -> str:
        """
        Generate human-readable explanation of a routing decision.
        
        Args:
            decision: Routing decision to explain.
            
        Returns:
            str: Formatted explanation.
        """
        lines = [
            f"Intent: {decision.intent.value}",
            f"Strategy: {decision.strategy.value}",
            f"Confidence: {decision.confidence:.2f}",
            f"Reasoning: {decision.reasoning}",
        ]
        
        if decision.tools:
            lines.append(f"Tools to use: {', '.join(decision.tools)}")
        
        if decision.metadata:
            lines.append(f"Metadata: {decision.metadata}")
        
        return "\n".join(lines)
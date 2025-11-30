"""
Thought generator for ReAct engine.

This module generates agent thoughts/reasoning based on the current state.
In Phase 3, we use rule-based reasoning. In Phase 4, this can be enhanced
with LLM-based reasoning.
"""

from typing import Optional

from .base import ReActStep, ReActTrace, Thought


class ThoughtGenerator:
    """
    Generates reasoning/thoughts for the agent.
    
    In Phase 3 (without LLM), this uses rule-based heuristics.
    In Phase 4, this will be enhanced with LLM-based reasoning.
    
    The thought generator analyzes:
    - Current query and context
    - Previous steps taken
    - Observations from tool calls
    - Whether goal is achieved
    
    Attributes:
        verbose: Whether to print detailed reasoning logs.
    """
    
    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize the thought generator.
        
        Args:
            verbose: Whether to print detailed logs.
        """
        self.verbose = verbose
    
    def generate_initial_thought(
        self,
        query: str,
        route_decision: Optional[dict] = None,
    ) -> Thought:
        """
        Generate the initial thought for a new query.
        
        This is the first reasoning step before any action is taken.
        
        Args:
            query: User query.
            route_decision: Optional routing decision from QueryRouter.
            
        Returns:
            Thought: Initial reasoning thought.
        """
        # Analyze the query
        considerations = [
            f"User asked: '{query}'",
        ]
        
        if route_decision:
            intent = route_decision.get("intent", "unknown")
            strategy = route_decision.get("strategy", "unknown")
            
            considerations.append(f"Query intent: {intent}")
            considerations.append(f"Routing strategy: {strategy}")
            
            # Determine reasoning based on route
            if intent == "factual_lookup":
                content = (
                    "This is a factual lookup question. I should query the "
                    "knowledge base to find the specific information requested."
                )
                reasoning_type = "planning"
            
            elif intent == "comparison":
                content = (
                    "This is a comparison question. I need to retrieve information "
                    "about multiple entities and compare them."
                )
                reasoning_type = "planning"
            
            elif intent == "complex_reasoning":
                content = (
                    "This requires complex reasoning. I should break it down into "
                    "steps and gather relevant information first."
                )
                reasoning_type = "analysis"
            
            elif intent == "admin_request":
                content = (
                    "This is an administrative request. I should use admin tools "
                    "to get the requested information."
                )
                reasoning_type = "planning"
            
            else:
                content = (
                    "I need to understand what information the user needs and "
                    "determine the best way to retrieve it from the knowledge base."
                )
                reasoning_type = "analysis"
        else:
            content = (
                "Let me analyze this query and determine the best approach "
                "to answer it."
            )
            reasoning_type = "analysis"
        
        return Thought(
            content=content,
            reasoning_type=reasoning_type,
            confidence=0.8,
            considerations=considerations,
        )
    
    def generate_next_thought(
        self,
        trace: ReActTrace,
        current_context: dict,
    ) -> Thought:
        """
        Generate the next thought based on execution history.
        
        Analyzes previous steps to decide what to do next.
        
        Args:
            trace: Current ReAct trace with previous steps.
            current_context: Current execution context.
            
        Returns:
            Thought: Next reasoning thought.
        """
        last_step = trace.steps[-1] if trace.steps else None
        
        if not last_step:
            # No previous steps, shouldn't happen
            return Thought(
                content="Starting fresh analysis of the query.",
                reasoning_type="analysis",
                confidence=0.5,
            )
        
        # Check if last step was successful
        if not last_step.is_successful():
            return self._generate_error_recovery_thought(last_step)
        
        # Analyze last observation
        observation = last_step.observation
        
        if not observation or not observation.content:
            return Thought(
                content=(
                    "The previous action didn't return useful information. "
                    "I should try a different approach."
                ),
                reasoning_type="reflection",
                confidence=0.6,
                considerations=["No useful data from last action"],
            )
        
        # Check if we have enough information
        has_answer = self._has_sufficient_answer(observation.content)
        
        if has_answer:
            return Thought(
                content=(
                    "I have gathered sufficient information to answer the "
                    "user's question. I should now synthesize the final answer."
                ),
                reasoning_type="reflection",
                confidence=0.9,
                considerations=[
                    "Retrieved relevant information",
                    "Confidence in answer quality is high",
                ],
            )
        else:
            return Thought(
                content=(
                    "The information I have is incomplete or not fully relevant. "
                    "I need to gather more details or refine my search."
                ),
                reasoning_type="analysis",
                confidence=0.7,
                considerations=[
                    "Current information may be insufficient",
                    "May need additional search",
                ],
            )
    
    def generate_reflection_thought(
        self,
        trace: ReActTrace,
        proposed_answer: str,
    ) -> Thought:
        """
        Generate a reflection thought to evaluate the proposed answer.
        
        This is used for self-reflection before returning final answer.
        
        Args:
            trace: Complete ReAct trace.
            proposed_answer: The proposed final answer.
            
        Returns:
            Thought: Reflection on answer quality.
        """
        considerations = []
        
        # Check answer length
        if len(proposed_answer) < 50:
            considerations.append("Answer seems quite brief")
            confidence = 0.6
        else:
            considerations.append("Answer has reasonable length")
            confidence = 0.8
        
        # Check if answer addresses the query
        query_words = set(trace.query.lower().split())
        answer_words = set(proposed_answer.lower().split())
        overlap = len(query_words & answer_words)
        
        if overlap >= 2:
            considerations.append("Answer addresses query terms")
            confidence = min(confidence + 0.1, 0.95)
        else:
            considerations.append("Answer may not directly address query")
            confidence = max(confidence - 0.2, 0.5)
        
        # Check number of steps taken
        step_count = trace.get_step_count()
        if step_count <= 1:
            considerations.append("Only one step taken - may be too simple")
        else:
            considerations.append(f"Took {step_count} steps to reach answer")
        
        # Generate reflection content
        if confidence >= 0.8:
            content = (
                "The proposed answer appears to be comprehensive and relevant. "
                "It addresses the user's query with appropriate detail."
            )
        elif confidence >= 0.6:
            content = (
                "The answer is acceptable but could potentially be improved "
                "with additional information or refinement."
            )
        else:
            content = (
                "I'm not fully confident in this answer. It may need "
                "additional verification or more comprehensive information."
            )
        
        return Thought(
            content=content,
            reasoning_type="reflection",
            confidence=confidence,
            considerations=considerations,
        )
    
    def _generate_error_recovery_thought(self, failed_step: ReActStep) -> Thought:
        """
        Generate thought for recovering from a failed step.
        
        Args:
            failed_step: The step that failed.
            
        Returns:
            Thought: Recovery reasoning.
        """
        error_msg = ""
        if failed_step.observation and failed_step.observation.error:
            error_msg = failed_step.observation.error
        
        return Thought(
            content=(
                f"The previous action failed: {error_msg}. "
                "I should try an alternative approach or ask for clarification."
            ),
            reasoning_type="analysis",
            confidence=0.5,
            considerations=[
                "Previous action failed",
                "Need alternative strategy",
            ],
        )
    
    def _has_sufficient_answer(self, content: str) -> bool:
        """
        Check if observation content contains sufficient information.
        
        Simple heuristic: check if content has reasonable length and
        indicates success.
        
        Args:
            content: Observation content.
            
        Returns:
            bool: True if appears sufficient.
        """
        # Basic heuristics
        if len(content) < 100:
            return False
        
        # Check for negative indicators
        negative_indicators = [
            "no relevant information",
            "couldn't find",
            "not available",
            "no results",
        ]
        
        content_lower = content.lower()
        if any(ind in content_lower for ind in negative_indicators):
            return False
        
        return True
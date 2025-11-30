"""
ReAct engine - Main orchestrator for agent execution.

This module implements the complete ReAct (Reasoning + Acting) loop,
coordinating thought generation, action execution, and observation processing.
"""

import time
from typing import Optional

from src.agent.router import QueryRouter, RouteDecision

from .action_executor import ActionExecutor
from .base import (
    Action,
    ActionType,
    ReActConfig,
    ReActStep,
    ReActTrace,
    StepStatus,
)
from .thought_generator import ThoughtGenerator


class ReActEngine:
    """
    Main ReAct engine for agent execution.
    
    This orchestrates the complete reasoning loop:
    1. Generate thought (reasoning)
    2. Decide action based on thought
    3. Execute action
    4. Observe result
    5. Repeat or return final answer
    
    Example usage:
        ```python
        engine = ReActEngine()
        result = await engine.run("What is the vacation policy?")
        print(result.final_answer)
        ```
    
    Attributes:
        thought_generator: Generates agent reasoning.
        action_executor: Executes actions and tool calls.
        query_router: Routes queries to determine strategy.
        config: ReAct configuration.
    """
    
    def __init__(
        self,
        thought_generator: Optional[ThoughtGenerator] = None,
        action_executor: Optional[ActionExecutor] = None,
        query_router: Optional[QueryRouter] = None,
        config: Optional[ReActConfig] = None,
    ) -> None:
        """
        Initialize the ReAct engine.
        
        Args:
            thought_generator: Thought generator (creates default if None).
            action_executor: Action executor (creates default if None).
            query_router: Query router (creates default if None).
            config: ReAct configuration (uses defaults if None).
        """
        self.thought_generator = thought_generator or ThoughtGenerator()
        self.action_executor = action_executor or ActionExecutor()
        self.query_router = query_router or QueryRouter()
        self.config = config or ReActConfig()
    
    async def run(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> ReActTrace:
        """
        Execute the ReAct loop for a query.
        
        Args:
            query: User query to process.
            context: Optional context (conversation history, etc.).
            
        Returns:
            ReActTrace: Complete execution trace with final answer.
        """
        start_time = time.time()
        
        # Initialize trace
        trace = ReActTrace(query=query)
        
        try:
            # Step 1: Route the query
            route_decision = self.query_router.route(query, context)
            
            if self.config.verbose:
                print(f"\n[ReActEngine] Query: {query}")
                print(f"[ReActEngine] Intent: {route_decision.intent.value}")
                print(f"[ReActEngine] Strategy: {route_decision.strategy.value}")
            
            trace.metadata["route_decision"] = route_decision.to_dict()
            
            # Check if clarification needed
            if route_decision.needs_clarification():
                return self._handle_clarification_needed(trace, route_decision)
            
            # Step 2: Execute ReAct loop
            for iteration in range(self.config.max_iterations):
                if self.config.verbose:
                    print(f"\n[ReActEngine] === Iteration {iteration + 1} ===")
                
                # Generate thought
                if iteration == 0:
                    thought = self.thought_generator.generate_initial_thought(
                        query=query,
                        route_decision=route_decision.to_dict(),
                    )
                else:
                    thought = self.thought_generator.generate_next_thought(
                        trace=trace,
                        current_context=context or {},
                    )
                
                if self.config.verbose:
                    print(f"[ReActEngine] Thought: {thought.content}")
                
                # Decide action based on thought
                action = self._decide_action(
                    thought=thought,
                    route_decision=route_decision,
                    trace=trace,
                )
                
                if self.config.verbose:
                    print(f"[ReActEngine] Action: {action}")
                
                # Create step
                step = ReActStep(
                    step_number=iteration + 1,
                    thought=thought,
                    action=action,
                )
                
                # Execute action
                step_start = time.time()
                observation = await self.action_executor.execute(action)
                step.duration_ms = (time.time() - step_start) * 1000
                
                step.observation = observation
                step.status = StepStatus.SUCCESS if observation.success else StepStatus.FAILED
                
                if self.config.verbose:
                    print(f"[ReActEngine] Observation: {observation}")
                
                # Add step to trace
                trace.add_step(step)
                
                # Check if we should stop
                if action.is_final_answer():
                    trace.final_answer = action.parameters.get("answer", "")
                    trace.success = True
                    break
                
                # Check if we hit an error
                if not observation.success:
                    # Try to recover or return error
                    if iteration == self.config.max_iterations - 1:
                        # Last iteration, return error
                        trace.final_answer = (
                            f"I encountered an error: {observation.error}. "
                            "I wasn't able to complete the task."
                        )
                        trace.success = False
                        trace.error = observation.error
                        break
            
            # If max iterations reached without final answer
            if not trace.final_answer:
                trace.final_answer = self._synthesize_answer_from_trace(trace)
                trace.success = True
                trace.metadata["reached_max_iterations"] = True
            
            # Optional: Self-reflection on final answer
            if self.config.enable_reflection and trace.success:
                await self._perform_self_reflection(trace)
        
        except Exception as e:
            trace.success = False
            trace.error = f"ReAct execution error: {str(e)}"
            trace.final_answer = (
                "I encountered an unexpected error while processing your query. "
                "Please try rephrasing or contact support if the issue persists."
            )
        
        finally:
            trace.total_duration_ms = (time.time() - start_time) * 1000
            
            if self.config.verbose:
                print(f"\n[ReActEngine] Total duration: {trace.total_duration_ms:.0f}ms")
                print(f"[ReActEngine] Final answer: {trace.final_answer[:100]}...")
        
        return trace
    
    def _decide_action(
        self,
        thought,
        route_decision: RouteDecision,
        trace: ReActTrace,
    ) -> Action:
        """
        Decide what action to take based on current thought.
        
        Args:
            thought: Current thought.
            route_decision: Routing decision from QueryRouter.
            trace: Current execution trace.
            
        Returns:
            Action: Action to take.
        """
        if route_decision.strategy == RoutingStrategy.SIMPLE_RESPONSE:
            return Action(
                action_type=ActionType.FINAL_ANSWER,
                parameters={
                    "answer": "Hello! I'm here to help you with information from our knowledge base. How can I assist you today?"
                },
                rationale="Query is chitchat/greeting, providing direct response without tool use"
            )
    
        # Check if thought suggests we have enough information
        if thought.reasoning_type == "reflection" and thought.confidence >= 0.8:
            # High confidence reflection = ready for final answer
            answer = self._synthesize_answer_from_trace(trace)
            
            return Action(
                action_type=ActionType.FINAL_ANSWER,
                parameters={"answer": answer},
                rationale="Sufficient information gathered to provide answer",
            )
        
        # Determine which tool to use based on route decision
        tools_to_use = route_decision.tools
        
        if not tools_to_use:
            # No tools specified, try to infer
            if route_decision.intent.value == "admin_request":
                tools_to_use = ["get_knowledge_base_stats"]
            else:
                tools_to_use = ["query_knowledge_base"]
        
        # Check if we've already used tools
        tools_already_used = trace.get_tool_calls()
        
        # Select tool (prefer one not yet used)
        tool_name = None
        for tool in tools_to_use:
            if tool not in tools_already_used:
                tool_name = tool
                break
        
        if not tool_name and tools_to_use:
            # All tools used, pick first one (may need refinement)
            tool_name = tools_to_use[0]
        
        if not tool_name:
            # Fallback to knowledge base query
            tool_name = "query_knowledge_base"
        
        # Determine parameters
        parameters = self._prepare_tool_parameters(
            tool_name=tool_name,
            query=trace.query,
            route_decision=route_decision,
        )
        
        return Action(
            action_type=ActionType.TOOL_CALL,
            tool_name=tool_name,
            parameters=parameters,
            rationale=f"Using {tool_name} to gather required information",
        )
    
    def _prepare_tool_parameters(
        self,
        tool_name: str,
        query: str,
        route_decision: RouteDecision,
    ) -> dict:
        """
        Prepare parameters for a tool call.
        
        Args:
            tool_name: Name of the tool.
            query: Original user query.
            route_decision: Routing decision.
            
        Returns:
            dict: Tool parameters.
        """
        if tool_name == "query_knowledge_base":
            # Use metadata from route decision if available
            top_k = route_decision.metadata.get("top_k", 5)
            
            return {
                "query": query,
                "top_k": top_k,
            }
        
        elif tool_name == "search_documents":
            # Extract filename pattern if available
            filename_pattern = route_decision.metadata.get("filename_pattern")
            
            params = {}
            if filename_pattern:
                params["filename_pattern"] = filename_pattern
            
            return params
        
        elif tool_name == "get_knowledge_base_stats":
            # No parameters needed
            return {}
        
        else:
            # Generic fallback
            return {"query": query}
    
    def _synthesize_answer_from_trace(self, trace: ReActTrace) -> str:
        """
        Synthesize final answer from execution trace.
        
        Combines observations from all successful steps.
        
        Args:
            trace: Execution trace.
            
        Returns:
            str: Synthesized answer.
        """
        successful_steps = trace.get_successful_steps()
        
        if not successful_steps:
            return (
                "I wasn't able to find relevant information to answer your "
                "question. Please try rephrasing or provide more details."
            )
        
        # Collect observations
        observations = []
        for step in successful_steps:
            if step.observation and step.observation.content:
                observations.append(step.observation.content)
        
        if not observations:
            return (
                "I searched for information but didn't find conclusive results. "
                "Please try a more specific query."
            )
        
        # For now, return the first/best observation
        # In Phase 4 with LLM, we can do proper synthesis
        return observations[0]
    
    async def _perform_self_reflection(self, trace: ReActTrace) -> None:
        """
        Perform self-reflection on the final answer.
        
        Evaluates answer quality and potentially refines it.
        
        Args:
            trace: Completed execution trace.
        """
        if not self.config.enable_reflection:
            return
        
        # Generate reflection thought
        reflection_thought = self.thought_generator.generate_reflection_thought(
            trace=trace,
            proposed_answer=trace.final_answer,
        )
        
        if self.config.verbose:
            print(f"\n[ReActEngine] Reflection: {reflection_thought.content}")
            print(f"[ReActEngine] Confidence: {reflection_thought.confidence:.2f}")
        
        # Store reflection in metadata
        trace.metadata["reflection"] = {
            "content": reflection_thought.content,
            "confidence": reflection_thought.confidence,
            "considerations": reflection_thought.considerations,
        }
        
        # If confidence is below threshold, mark it
        if reflection_thought.confidence < self.config.reflection_threshold:
            trace.metadata["low_confidence_answer"] = True
            
            if self.config.verbose:
                print(
                    f"[ReActEngine] Warning: Answer confidence "
                    f"({reflection_thought.confidence:.2f}) below threshold "
                    f"({self.config.reflection_threshold})"
                )
    
    def _handle_clarification_needed(
        self,
        trace: ReActTrace,
        route_decision: RouteDecision,
    ) -> ReActTrace:
        """
        Handle case where clarification is needed.
        
        Args:
            trace: Execution trace.
            route_decision: Route decision indicating clarification needed.
            
        Returns:
            ReActTrace: Trace with clarification message.
        """
        clarification_msg = route_decision.metadata.get(
            "clarification_message",
            "Could you please provide more details about your question?"
        )
        
        trace.final_answer = clarification_msg
        trace.success = True
        trace.metadata["needs_clarification"] = True
        
        return trace
"""
Action executor for ReAct engine.

This module executes agent actions, including tool calls and
final answer generation.
"""

import time
from typing import Any, Optional

from src.agent.tools import BaseTool, ToolResult, get_tool_registry

from .base import Action, ActionType, Observation


class ActionExecutor:
    """
    Executes agent actions including tool calls.
    
    This component:
    1. Takes an Action from the agent
    2. Executes the corresponding tool or operation
    3. Returns an Observation with the result
    
    Attributes:
        tool_registry: Registry of available tools.
        verbose: Whether to print detailed execution logs.
    """
    
    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize the action executor.
        
        Args:
            verbose: Whether to print detailed logs.
        """
        self.tool_registry = get_tool_registry()
        self.verbose = verbose
    
    async def execute(self, action: Action) -> Observation:
        """
        Execute an action and return observation.
        
        Args:
            action: Action to execute.
            
        Returns:
            Observation: Result of the action.
        """
        start_time = time.time()
        
        try:
            if action.action_type == ActionType.TOOL_CALL:
                observation = await self._execute_tool_call(action)
            
            elif action.action_type == ActionType.FINAL_ANSWER:
                observation = self._generate_final_answer(action)
            
            elif action.action_type == ActionType.CLARIFICATION:
                observation = self._generate_clarification(action)
            
            elif action.action_type == ActionType.REFLECTION:
                observation = self._perform_reflection(action)
            
            else:
                observation = Observation(
                    content="",
                    source="action_executor",
                    success=False,
                    error=f"Unknown action type: {action.action_type}",
                )
            
            # Add execution time to metadata
            duration_ms = (time.time() - start_time) * 1000
            observation.metadata["duration_ms"] = duration_ms
            
            if self.verbose:
                print(f"[ActionExecutor] Executed in {duration_ms:.0f}ms")
            
            return observation
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            if self.verbose:
                print(f"[ActionExecutor] Error: {e}")
            
            return Observation(
                content="",
                source="action_executor",
                success=False,
                error=f"Action execution failed: {str(e)}",
                metadata={"duration_ms": duration_ms},
            )
    
    async def _execute_tool_call(self, action: Action) -> Observation:
        """
        Execute a tool call action.
        
        Args:
            action: Tool call action.
            
        Returns:
            Observation: Tool execution result.
        """
        tool_name = action.tool_name
        
        if not tool_name:
            return Observation(
                content="",
                source="action_executor",
                success=False,
                error="Tool call action missing tool_name",
            )
        
        # Get tool from registry
        tool = self.tool_registry.get(tool_name)
        
        if not tool:
            return Observation(
                content="",
                source=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found in registry",
                metadata={
                    "available_tools": self.tool_registry.list_tools(),
                },
            )
        
        # Execute tool
        if self.verbose:
            print(f"[ActionExecutor] Calling tool: {tool_name}")
            print(f"[ActionExecutor] Parameters: {action.parameters}")
        
        try:
            result: ToolResult = await tool.execute(**action.parameters)
            
            # Convert ToolResult to Observation
            return self._tool_result_to_observation(result, tool_name)
        
        except Exception as e:
            return Observation(
                content="",
                source=tool_name,
                success=False,
                error=f"Tool execution error: {str(e)}",
            )
    
    def _tool_result_to_observation(
        self,
        tool_result: ToolResult,
        tool_name: str,
    ) -> Observation:
        """
        Convert ToolResult to Observation.
        
        Args:
            tool_result: Result from tool execution.
            tool_name: Name of the tool.
            
        Returns:
            Observation: Converted observation.
        """
        if tool_result.success:
            # Format successful result
            content = self._format_tool_result(tool_result.data)
            
            return Observation(
                content=content,
                source=tool_name,
                success=True,
                metadata={
                    "tool_name": tool_name,
                    "raw_data": tool_result.data,
                    "tool_metadata": tool_result.metadata,
                },
            )
        else:
            # Format error
            return Observation(
                content="",
                source=tool_name,
                success=False,
                error=tool_result.error or "Tool execution failed",
                metadata={
                    "tool_name": tool_name,
                    "tool_metadata": tool_result.metadata,
                },
            )
    
    def _format_tool_result(self, data: Any) -> str:
        """
        Format tool result data into readable string.
        
        Args:
            data: Tool result data.
            
        Returns:
            str: Formatted content.
        """
        if isinstance(data, str):
            return data
        
        elif isinstance(data, dict):
            # For dict results, format nicely
            if "answer" in data:
                # Knowledge base query result
                answer = data["answer"]
                sources = data.get("sources", [])
                
                content_parts = [answer]
                
                if sources:
                    content_parts.append("\n\nSources:")
                    for i, source in enumerate(sources, 1):
                        filename = source.get("filename", "Unknown")
                        content_parts.append(f"{i}. {filename}")
                
                return "\n".join(content_parts)
            
            elif "documents" in data:
                # Document search result
                docs = data["documents"]
                total = data.get("total_count", len(docs))
                
                content_parts = [f"Found {total} documents:"]
                
                for doc in docs[:5]:  # Show first 5
                    filename = doc.get("filename", "Unknown")
                    file_type = doc.get("file_type", "")
                    content_parts.append(f"- {filename} {file_type}")
                
                if total > 5:
                    content_parts.append(f"... and {total - 5} more")
                
                return "\n".join(content_parts)
            
            elif "total_chunks" in data:
                # Statistics result
                return (
                    f"Knowledge base statistics:\n"
                    f"- Collection: {data.get('collection_name', 'unknown')}\n"
                    f"- Total chunks: {data.get('total_chunks', 0)}\n"
                    f"- Status: {data.get('status', 'unknown')}"
                )
            
            else:
                # Generic dict formatting
                lines = []
                for key, value in data.items():
                    lines.append(f"{key}: {value}")
                return "\n".join(lines)
        
        elif isinstance(data, list):
            # Format list
            return "\n".join(f"- {item}" for item in data)
        
        else:
            # Fallback to string representation
            return str(data)
    
    def _generate_final_answer(self, action: Action) -> Observation:
        """
        Generate final answer observation.
        
        Args:
            action: Final answer action.
            
        Returns:
            Observation: Final answer observation.
        """
        answer = action.parameters.get("answer", "")
        
        if not answer:
            return Observation(
                content="",
                source="final_answer",
                success=False,
                error="Final answer action missing answer parameter",
            )
        
        return Observation(
            content=answer,
            source="final_answer",
            success=True,
            metadata={
                "is_final_answer": True,
                "reasoning": action.rationale,
            },
        )
    
    def _generate_clarification(self, action: Action) -> Observation:
        """
        Generate clarification request observation.
        
        Args:
            action: Clarification action.
            
        Returns:
            Observation: Clarification observation.
        """
        message = action.parameters.get("message", "")
        
        return Observation(
            content=message,
            source="clarification",
            success=True,
            metadata={
                "requires_user_input": True,
            },
        )
    
    def _perform_reflection(self, action: Action) -> Observation:
        """
        Perform internal reflection.
        
        Args:
            action: Reflection action.
            
        Returns:
            Observation: Reflection result.
        """
        reflection = action.parameters.get("reflection", "")
        
        return Observation(
            content=reflection,
            source="reflection",
            success=True,
            metadata={
                "is_internal": True,
            },
        )
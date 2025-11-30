"""
Agent tools module.

This module provides tools that agents can use to interact with
the knowledge base and perform various operations.
"""

from .admin_tools import GetKnowledgeBaseStatsTool, SearchDocumentsTool
from .base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    get_tool_registry,
    register_tool,
)
from .knowledge_tools import QueryKnowledgeBaseTool

__all__ = [
    # Base classes
    "BaseTool",
    "ToolParameter",
    "ToolResult",
    "ToolCategory",
    "ToolRegistry",
    "get_tool_registry",
    "register_tool",
    # Knowledge tools
    "QueryKnowledgeBaseTool",
    # Admin tools
    "GetKnowledgeBaseStatsTool",
    "SearchDocumentsTool",
]


def get_default_tools() -> list[BaseTool]:
    """
    Get the default set of tools for the agent.
    
    Returns:
        list[BaseTool]: List of initialized tools.
    """
    return [
        QueryKnowledgeBaseTool(),
        GetKnowledgeBaseStatsTool(),
        SearchDocumentsTool(),
    ]


def register_default_tools() -> None:
    """
    Register all default tools in the global registry.
    """
    registry = get_tool_registry()
    
    for tool in get_default_tools():
        try:
            registry.register(tool)
        except ValueError:
            # Tool already registered
            pass
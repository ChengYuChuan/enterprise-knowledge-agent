"""
Base tool abstraction for the agent system.

This module provides a framework-agnostic tool interface that can be
used with both LangChain and FastMCP.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional


class ToolCategory(Enum):
    """Categories for organizing tools."""
    
    KNOWLEDGE = "knowledge"  # Knowledge base operations
    ADMIN = "admin"  # Administrative operations
    SEARCH = "search"  # Search operations
    UTILITY = "utility"  # Utility functions


@dataclass
class ToolParameter:
    """
    Definition of a tool parameter.
    
    Attributes:
        name: Parameter name.
        type: Python type annotation.
        description: Human-readable description.
        required: Whether the parameter is required.
        default: Default value if not required.
    """
    
    name: str
    type: type
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolResult:
    """
    Result of a tool execution.
    
    Attributes:
        success: Whether the tool executed successfully.
        data: Result data (any type).
        error: Error message if execution failed.
        metadata: Additional metadata about the execution.
    """
    
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict = None
    
    def __post_init__(self) -> None:
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Tools must implement:
    1. name: Unique identifier
    2. description: What the tool does
    3. parameters: List of required/optional parameters
    4. execute(): Main logic
    
    This abstraction allows tools to be used with:
    - LangChain (via adapter)
    - FastMCP (via adapter)
    - Direct invocation
    """
    
    def __init__(self) -> None:
        """Initialize the tool."""
        self.category = self.get_category()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique tool identifier.
        
        Should be lowercase with underscores (e.g., 'search_knowledge_base').
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what the tool does.
        
        This is shown to the LLM to help it decide when to use the tool.
        Should be clear and specific about:
        - What the tool does
        - When to use it
        - When NOT to use it (if relevant)
        """
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> list[ToolParameter]:
        """
        List of parameters this tool accepts.
        
        Returns:
            list[ToolParameter]: Parameter definitions.
        """
        pass
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool parameters as keyword arguments.
            
        Returns:
            ToolResult: Result of the tool execution.
            
        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        pass
    
    def get_category(self) -> ToolCategory:
        """
        Get the category this tool belongs to.
        
        Override this to specify a different category.
        
        Returns:
            ToolCategory: Tool category.
        """
        return ToolCategory.UTILITY
    
    def validate_parameters(self, **kwargs: Any) -> None:
        """
        Validate that required parameters are provided.
        
        Args:
            **kwargs: Provided parameters.
            
        Raises:
            ValueError: If required parameters are missing.
        """
        # Check required parameters
        required_params = [p.name for p in self.parameters if p.required]
        missing_params = [p for p in required_params if p not in kwargs]
        
        if missing_params:
            raise ValueError(
                f"Missing required parameters for {self.name}: {missing_params}"
            )
    
    def get_parameter_schema(self) -> dict:
        """
        Get JSON schema for tool parameters.
        
        Useful for API documentation and validation.
        
        Returns:
            dict: JSON schema for parameters.
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            # Map Python types to JSON schema types
            type_mapping = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }
            
            param_schema = {
                "type": type_mapping.get(param.type, "string"),
                "description": param.description,
            }
            
            if param.default is not None:
                param_schema["default"] = param.default
            
            properties[param.name] = param_schema
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    
    async def __call__(self, **kwargs: Any) -> ToolResult:
        """
        Allow tool to be called directly.
        
        Args:
            **kwargs: Tool parameters.
            
        Returns:
            ToolResult: Execution result.
        """
        return await self.execute(**kwargs)


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Provides:
    - Tool registration and lookup
    - Category-based filtering
    - Tool listing and metadata
    """
    
    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tools: dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool instance to register.
            
        Raises:
            ValueError: If a tool with the same name already exists.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name.
            
        Returns:
            Optional[BaseTool]: Tool instance or None if not found.
        """
        return self._tools.get(name)
    
    def get_by_category(self, category: ToolCategory) -> list[BaseTool]:
        """
        Get all tools in a category.
        
        Args:
            category: Tool category.
            
        Returns:
            list[BaseTool]: List of tools in the category.
        """
        return [
            tool for tool in self._tools.values()
            if tool.category == category
        ]
    
    def list_tools(self) -> list[str]:
        """
        List all registered tool names.
        
        Returns:
            list[str]: List of tool names.
        """
        return list(self._tools.keys())
    
    def get_all_tools(self) -> list[BaseTool]:
        """
        Get all registered tools.
        
        Returns:
            list[BaseTool]: List of all tools.
        """
        return list(self._tools.values())
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name.
            
        Returns:
            bool: True if tool was removed, False if not found.
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False


# Global tool registry instance
_global_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry.
    
    Returns:
        ToolRegistry: Global registry instance.
    """
    return _global_registry


def register_tool(tool: BaseTool) -> None:
    """
    Register a tool in the global registry.
    
    Args:
        tool: Tool instance to register.
    """
    _global_registry.register(tool)
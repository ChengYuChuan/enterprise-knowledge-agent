"""
Unit tests for agent tools.

Tests the tool system including base abstractions and specific tools.
"""

import pytest

from src.agent.tools import (
    BaseTool,
    GetKnowledgeBaseStatsTool,
    QueryKnowledgeBaseTool,
    SearchDocumentsTool,
    ToolCategory,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    get_default_tools,
    register_default_tools,
)


class TestToolRegistry:
    """Test cases for ToolRegistry."""
    
    def test_register_and_get_tool(self):
        """Test registering and retrieving a tool."""
        registry = ToolRegistry()
        tool = QueryKnowledgeBaseTool()
        
        registry.register(tool)
        retrieved = registry.get("query_knowledge_base")
        
        assert retrieved is not None
        assert retrieved.name == "query_knowledge_base"
    
    def test_duplicate_registration_raises_error(self):
        """Test that registering duplicate tool raises error."""
        registry = ToolRegistry()
        tool1 = QueryKnowledgeBaseTool()
        tool2 = QueryKnowledgeBaseTool()
        
        registry.register(tool1)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool2)
    
    def test_list_tools(self):
        """Test listing all registered tools."""
        registry = ToolRegistry()
        
        for tool in get_default_tools():
            registry.register(tool)
        
        tool_names = registry.list_tools()
        
        assert "query_knowledge_base" in tool_names
        assert "get_knowledge_base_stats" in tool_names
        assert "search_documents" in tool_names
    
    def test_get_by_category(self):
        """Test filtering tools by category."""
        registry = ToolRegistry()
        
        for tool in get_default_tools():
            registry.register(tool)
        
        knowledge_tools = registry.get_by_category(ToolCategory.KNOWLEDGE)
        admin_tools = registry.get_by_category(ToolCategory.ADMIN)
        
        assert len(knowledge_tools) > 0
        assert len(admin_tools) > 0
        assert all(t.category == ToolCategory.KNOWLEDGE for t in knowledge_tools)
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = QueryKnowledgeBaseTool()
        
        registry.register(tool)
        assert registry.get("query_knowledge_base") is not None
        
        result = registry.unregister("query_knowledge_base")
        assert result is True
        assert registry.get("query_knowledge_base") is None


class TestQueryKnowledgeBaseTool:
    """Test cases for QueryKnowledgeBaseTool."""
    
    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = QueryKnowledgeBaseTool()
        
        assert tool.name == "query_knowledge_base"
        assert len(tool.description) > 0
        assert tool.category == ToolCategory.KNOWLEDGE
    
    def test_tool_parameters(self):
        """Test tool parameters are defined correctly."""
        tool = QueryKnowledgeBaseTool()
        params = tool.parameters
        
        # Should have 'query' and 'top_k' parameters
        param_names = [p.name for p in params]
        assert "query" in param_names
        assert "top_k" in param_names
        
        # 'query' should be required
        query_param = next(p for p in params if p.name == "query")
        assert query_param.required is True
        
        # 'top_k' should be optional with default
        top_k_param = next(p for p in params if p.name == "top_k")
        assert top_k_param.required is False
        assert top_k_param.default == 5
    
    def test_parameter_schema_generation(self):
        """Test JSON schema generation for parameters."""
        tool = QueryKnowledgeBaseTool()
        schema = tool.get_parameter_schema()
        
        assert "properties" in schema
        assert "required" in schema
        assert "query" in schema["properties"]
        assert "query" in schema["required"]
    
    def test_validation_requires_query(self):
        """Test that validation requires 'query' parameter."""
        tool = QueryKnowledgeBaseTool()
        
        with pytest.raises(ValueError, match="Missing required parameters"):
            tool.validate_parameters(top_k=10)  # Missing 'query'
    
    def test_validation_accepts_valid_params(self):
        """Test that validation accepts valid parameters."""
        tool = QueryKnowledgeBaseTool()
        
        # Should not raise
        tool.validate_parameters(query="test query")
        tool.validate_parameters(query="test", top_k=10)
    
    @pytest.mark.asyncio
    async def test_execute_with_no_results(self):
        """Test execution when no results are found."""
        tool = QueryKnowledgeBaseTool()
        
        # Query something unlikely to exist
        result = await tool.execute(
            query="xyzabc123 nonexistent query"
        )
        
        # Should return a result (success or failure)
        assert isinstance(result, ToolResult)


class TestGetKnowledgeBaseStatsTool:
    """Test cases for GetKnowledgeBaseStatsTool."""
    
    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = GetKnowledgeBaseStatsTool()
        
        assert tool.name == "get_knowledge_base_stats"
        assert len(tool.description) > 0
        assert tool.category == ToolCategory.ADMIN
    
    def test_no_required_parameters(self):
        """Test that tool has no required parameters."""
        tool = GetKnowledgeBaseStatsTool()
        params = tool.parameters
        
        # Should have no parameters or all optional
        assert all(not p.required for p in params)
    
    @pytest.mark.asyncio
    async def test_execute_returns_stats(self):
        """Test that execution returns statistics."""
        tool = GetKnowledgeBaseStatsTool()
        
        result = await tool.execute()
        
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "total_chunks" in result.data
        assert "collection_name" in result.data


class TestSearchDocumentsTool:
    """Test cases for SearchDocumentsTool."""
    
    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = SearchDocumentsTool()
        
        assert tool.name == "search_documents"
        assert len(tool.description) > 0
        assert tool.category == ToolCategory.SEARCH
    
    def test_optional_parameters(self):
        """Test that all parameters are optional."""
        tool = SearchDocumentsTool()
        params = tool.parameters
        
        assert all(not p.required for p in params)
    
    @pytest.mark.asyncio
    async def test_execute_without_filters(self):
        """Test execution without any filters."""
        tool = SearchDocumentsTool()
        
        result = await tool.execute()
        
        assert isinstance(result, ToolResult)
        # Should return data even if empty
        assert "documents" in result.data
    
    @pytest.mark.asyncio
    async def test_execute_with_filename_filter(self):
        """Test execution with filename filter."""
        tool = SearchDocumentsTool()
        
        result = await tool.execute(filename_pattern="vacation")
        
        assert isinstance(result, ToolResult)
        assert "documents" in result.data


class TestDefaultTools:
    """Test default tools setup."""
    
    def test_get_default_tools_returns_list(self):
        """Test that get_default_tools returns a list of tools."""
        tools = get_default_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert all(isinstance(t, BaseTool) for t in tools)
    
    def test_register_default_tools(self):
        """Test registering default tools."""
        from src.agent.tools import get_tool_registry
        
        # Clear registry (in case tests ran before)
        registry = get_tool_registry()
        for tool_name in registry.list_tools():
            registry.unregister(tool_name)
        
        # Register default tools
        register_default_tools()
        
        # Check they're registered
        assert registry.get("query_knowledge_base") is not None
        assert registry.get("get_knowledge_base_stats") is not None
        assert registry.get("search_documents") is not None
    
    def test_default_tools_have_unique_names(self):
        """Test that all default tools have unique names."""
        tools = get_default_tools()
        names = [t.name for t in tools]
        
        assert len(names) == len(set(names))  # All unique
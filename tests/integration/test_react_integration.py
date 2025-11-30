"""
Integration tests for ReAct engine.

Tests the complete ReAct loop including ThoughtGenerator,
ActionExecutor, and ReActEngine working together.
"""

import pytest

from src.agent.react import (
    Action,
    ActionExecutor,
    ActionType,
    ReActConfig,
    ReActEngine,
    ThoughtGenerator,
)



class TestThoughtGenerator:
    """Test cases for ThoughtGenerator."""
    
    def test_generate_initial_thought(self):
        """Test generating initial thought."""
        generator = ThoughtGenerator()
        
        route_decision = {
            "intent": "knowledge_query",
            "strategy": "direct_tool",
            "tools": ["query_knowledge_base"],
        }
        
        thought = generator.generate_initial_thought(
            query="What is the vacation policy?",
            route_decision=route_decision,
        )
        
        assert thought is not None
        assert len(thought.content) > 0
        assert thought.confidence > 0
    
    def test_generate_initial_thought_without_route(self):
        """Test generating thought without route decision."""
        generator = ThoughtGenerator()
        
        thought = generator.generate_initial_thought(
            query="test query",
            route_decision=None,
        )
        
        assert thought is not None
        assert thought.reasoning_type in ["analysis", "planning"]


class TestActionExecutor:
    """Test cases for ActionExecutor."""
    
    @pytest.mark.asyncio
    async def test_execute_tool_call_not_found(self):
        """Test executing tool call for non-existent tool."""
        executor = ActionExecutor()
        
        action = Action(
            action_type=ActionType.TOOL_CALL,
            tool_name="nonexistent_tool",
            parameters={},
        )
        
        observation = await executor.execute(action)
        
        assert not observation.success
        assert "not found" in observation.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_final_answer(self):
        """Test executing final answer action."""
        executor = ActionExecutor()
        
        action = Action(
            action_type=ActionType.FINAL_ANSWER,
            parameters={"answer": "This is the final answer"},
        )
        
        observation = await executor.execute(action)
        
        assert observation.success
        assert observation.content == "This is the final answer"
    
    @pytest.mark.asyncio
    async def test_execute_tool_without_name(self):
        """Test executing tool call without tool name."""
        executor = ActionExecutor()
        
        # This should fail in Action.__post_init__, but test anyway
        try:
            action = Action(
                action_type=ActionType.TOOL_CALL,
                tool_name=None,
                parameters={},
            )
            observation = await executor.execute(action)
            assert not observation.success
        except ValueError:
            # Expected to fail during Action creation
            pass


@pytest.mark.asyncio
class TestReActEngine:
    """Integration tests for ReActEngine."""
    
    async def test_engine_initialization(self):
        """Test that engine initializes properly."""
        engine = ReActEngine()
        
        assert engine.thought_generator is not None
        assert engine.action_executor is not None
        assert engine.query_router is not None
        assert engine.config is not None
    
    async def test_engine_with_custom_config(self):
        """Test engine with custom configuration."""
        config = ReActConfig(
            max_iterations=3,
            enable_reflection=False,
            verbose=True,
        )
        
        engine = ReActEngine(config=config)
        
        assert engine.config.max_iterations == 3
        assert engine.config.enable_reflection is False
    
    async def test_run_with_empty_kb(self):
        """Test running engine with empty knowledge base."""
        engine = ReActEngine(config=ReActConfig(verbose=False))
        
        trace = await engine.run("What is the vacation policy?")
        
        # Should complete even if KB is empty
        assert trace is not None
        assert trace.query == "What is the vacation policy?"
        assert trace.get_step_count() > 0
    
    async def test_run_with_chitchat(self):
        """Test running engine with chitchat query."""
        engine = ReActEngine(config=ReActConfig(verbose=False))
        
        trace = await engine.run("Hello, how are you?")
        
        assert trace is not None
        assert trace.success is True
        # Chitchat should not require many steps
        assert trace.get_step_count() <= 2
    
    async def test_run_with_admin_request(self):
        """Test running engine with admin request."""
        engine = ReActEngine(config=ReActConfig(verbose=False))
        
        trace = await engine.run("How many documents are in the knowledge base?")
        
        assert trace is not None
        assert trace.get_step_count() > 0
        # Should use admin tools
        tools_used = trace.get_unique_tools()
        assert "get_knowledge_base_stats" in tools_used or "query_knowledge_base" in tools_used
    
    async def test_run_respects_max_iterations(self):
        """Test that engine respects max_iterations."""
        config = ReActConfig(max_iterations=2, verbose=False)
        engine = ReActEngine(config=config)
        
        trace = await engine.run("test query")
        
        # Should not exceed max iterations
        assert trace.get_step_count() <= 2
    
    async def test_trace_contains_metadata(self):
        """Test that trace contains routing metadata."""
        engine = ReActEngine(config=ReActConfig(verbose=False))
        
        trace = await engine.run("What is the vacation policy?")
        
        assert "route_decision" in trace.metadata
        route = trace.metadata["route_decision"]
        assert "intent" in route
        assert "strategy" in route
    
    async def test_trace_statistics(self):
        """Test that trace includes statistics."""
        engine = ReActEngine(config=ReActConfig(verbose=False))
        
        trace = await engine.run("test query")
        
        trace_dict = trace.to_dict()
        
        assert "statistics" in trace_dict
        stats = trace_dict["statistics"]
        assert "total_steps" in stats
        assert "successful_steps" in stats
        assert "tools_used" in stats
    
    async def test_error_recovery(self):
        """Test that engine handles errors gracefully."""
        engine = ReActEngine(config=ReActConfig(verbose=False))
        
        # Even with errors, should return a trace
        trace = await engine.run("test")
        
        assert trace is not None
        assert isinstance(trace.success, bool)
        assert isinstance(trace.final_answer, str)
    
    async def test_verbose_mode(self):
        """Test that verbose mode works without errors."""
        config = ReActConfig(verbose=True, max_iterations=2)
        engine = ReActEngine(config=config)
        
        # Should not raise errors even with verbose output
        trace = await engine.run("test query")
        
        assert trace is not None


@pytest.mark.asyncio
class TestReActEngineWithKnowledgeBase:
    """
    Tests that require actual knowledge base data.
    
    These tests will be skipped if KB is empty.
    """
    
    async def test_knowledge_query_with_data(self):
        """Test knowledge query when KB has data."""
        from src.agent.tools import GetKnowledgeBaseStatsTool
        
        # Check if KB has data
        stats_tool = GetKnowledgeBaseStatsTool()
        stats_result = await stats_tool.execute()
        
        if not stats_result.success or stats_result.data["total_chunks"] == 0:
            pytest.skip("Knowledge base is empty")
        
        # Run actual query
        engine = ReActEngine(config=ReActConfig(verbose=False))
        trace = await engine.run("What is the vacation policy?")
        
        # Should successfully retrieve information
        assert trace.success
        assert len(trace.final_answer) > 50  # Should have substantial answer
        assert trace.get_step_count() >= 1
    
    async def test_comparison_query(self):
        """Test comparison query functionality."""
        from src.agent.tools import GetKnowledgeBaseStatsTool
        
        # Check if KB has data
        stats_tool = GetKnowledgeBaseStatsTool()
        stats_result = await stats_tool.execute()
        
        if not stats_result.success or stats_result.data["total_chunks"] == 0:
            pytest.skip("Knowledge base is empty")
        
        engine = ReActEngine(config=ReActConfig(verbose=False))
        trace = await engine.run(
            "Compare vacation policies for new and senior employees"
        )
        
        # Comparison might require multiple steps
        assert trace.get_step_count() >= 1
        assert trace.final_answer  # Should have an answer
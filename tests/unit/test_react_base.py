"""
Unit tests for ReAct base data structures.

Tests all core data types including Thought, Action, Observation,
ReActStep, ReActTrace, and ReActConfig.
"""

import pytest
from datetime import datetime

from src.agent.react import (
    Action,
    ActionType,
    Observation,
    ReActConfig,
    ReActStep,
    ReActTrace,
    StepStatus,
    Thought,
)


class TestThought:
    """Test cases for Thought class."""
    
    def test_create_thought(self):
        """Test creating a basic thought."""
        thought = Thought(
            content="I need to search the knowledge base",
            reasoning_type="planning",
            confidence=0.9,
        )
        
        assert thought.content == "I need to search the knowledge base"
        assert thought.reasoning_type == "planning"
        assert thought.confidence == 0.9
        assert thought.considerations == []
    
    def test_thought_with_considerations(self):
        """Test thought with considerations."""
        thought = Thought(
            content="Test",
            considerations=["User asked about vacation", "Need specific info"],
        )
        
        assert len(thought.considerations) == 2
    
    def test_thought_validation_confidence(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            Thought(content="Test", confidence=1.5)
        
        with pytest.raises(ValueError, match="Confidence must be between"):
            Thought(content="Test", confidence=-0.1)
    
    def test_thought_validation_empty_content(self):
        """Test that empty content raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Thought(content="")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            Thought(content="   ")
    
    def test_thought_to_dict(self):
        """Test converting thought to dictionary."""
        thought = Thought(
            content="Test thought",
            reasoning_type="analysis",
            confidence=0.8,
            considerations=["factor1", "factor2"],
        )
        
        result = thought.to_dict()
        
        assert result["content"] == "Test thought"
        assert result["reasoning_type"] == "analysis"
        assert result["confidence"] == 0.8
        assert result["considerations"] == ["factor1", "factor2"]
    
    def test_thought_string_representation(self):
        """Test string representation of thought."""
        thought = Thought(content="Test", reasoning_type="planning")
        
        str_repr = str(thought)
        
        assert "💭" in str_repr
        assert "planning" in str_repr
        assert "Test" in str_repr


class TestAction:
    """Test cases for Action class."""
    
    def test_create_tool_call_action(self):
        """Test creating a tool call action."""
        action = Action(
            action_type=ActionType.TOOL_CALL,
            tool_name="query_knowledge_base",
            parameters={"query": "test", "top_k": 5},
            rationale="Need to search",
        )
        
        assert action.is_tool_call()
        assert not action.is_final_answer()
        assert action.tool_name == "query_knowledge_base"
        assert action.parameters["query"] == "test"
    
    def test_create_final_answer_action(self):
        """Test creating a final answer action."""
        action = Action(
            action_type=ActionType.FINAL_ANSWER,
            parameters={"answer": "The answer is 42"},
        )
        
        assert action.is_final_answer()
        assert not action.is_tool_call()
    
    def test_action_validation_tool_call_requires_name(self):
        """Test that tool call requires tool name."""
        with pytest.raises(ValueError, match="must specify tool_name"):
            Action(action_type=ActionType.TOOL_CALL)
    
    def test_action_validation_final_answer_requires_answer(self):
        """Test that final answer requires answer parameter."""
        with pytest.raises(ValueError, match="must have 'answer' parameter"):
            Action(action_type=ActionType.FINAL_ANSWER)
    
    def test_action_to_dict(self):
        """Test converting action to dictionary."""
        action = Action(
            action_type=ActionType.TOOL_CALL,
            tool_name="test_tool",
            parameters={"key": "value"},
            rationale="test rationale",
        )
        
        result = action.to_dict()
        
        assert result["action_type"] == "tool_call"
        assert result["tool_name"] == "test_tool"
        assert result["parameters"] == {"key": "value"}
        assert result["rationale"] == "test rationale"


class TestObservation:
    """Test cases for Observation class."""
    
    def test_create_successful_observation(self):
        """Test creating a successful observation."""
        obs = Observation(
            content="Found vacation policy",
            source="query_knowledge_base",
            success=True,
        )
        
        assert obs.is_successful()
        assert obs.has_content()
    
    def test_create_failed_observation(self):
        """Test creating a failed observation."""
        obs = Observation(
            content="",
            source="test_tool",
            success=False,
            error="Tool execution failed",
        )
        
        assert not obs.is_successful()
        assert not obs.has_content()
        assert obs.error == "Tool execution failed"
    
    def test_observation_auto_error_message(self):
        """Test that failed observation gets default error."""
        obs = Observation(content="", success=False)
        
        assert obs.error is not None
        assert "Unknown error" in obs.error
    
    def test_observation_to_dict(self):
        """Test converting observation to dictionary."""
        obs = Observation(
            content="test content",
            source="test_source",
            success=True,
            metadata={"key": "value"},
        )
        
        result = obs.to_dict()
        
        assert result["content"] == "test content"
        assert result["source"] == "test_source"
        assert result["success"] is True
        assert result["metadata"]["key"] == "value"


class TestReActStep:
    """Test cases for ReActStep class."""
    
    def test_create_step(self):
        """Test creating a ReAct step."""
        thought = Thought(content="Test thought")
        action = Action(
            action_type=ActionType.TOOL_CALL,
            tool_name="test_tool",
            parameters={},
        )
        
        step = ReActStep(
            step_number=1,
            thought=thought,
            action=action,
        )
        
        assert step.step_number == 1
        assert step.thought == thought
        assert step.action == action
        assert step.observation is None
        assert step.status == StepStatus.IN_PROGRESS
    
    def test_step_validation_step_number(self):
        """Test that step number must be >= 1."""
        thought = Thought(content="Test")
        action = Action(
            action_type=ActionType.FINAL_ANSWER,
            parameters={"answer": "test"},
        )
        
        with pytest.raises(ValueError, match="must be >= 1"):
            ReActStep(step_number=0, thought=thought, action=action)
    
    def test_step_is_complete(self):
        """Test checking if step is complete."""
        thought = Thought(content="Test")
        action = Action(
            action_type=ActionType.FINAL_ANSWER,
            parameters={"answer": "test"},
        )
        
        step = ReActStep(step_number=1, thought=thought, action=action)
        
        assert not step.is_complete()
        
        step.observation = Observation(content="result", success=True)
        
        assert step.is_complete()
    
    def test_step_is_successful(self):
        """Test checking if step is successful."""
        thought = Thought(content="Test")
        action = Action(
            action_type=ActionType.FINAL_ANSWER,
            parameters={"answer": "test"},
        )
        
        step = ReActStep(step_number=1, thought=thought, action=action)
        step.observation = Observation(content="result", success=True)
        step.status = StepStatus.SUCCESS
        
        assert step.is_successful()
    
    def test_step_mark_success(self):
        """Test marking step as success."""
        thought = Thought(content="Test")
        action = Action(
            action_type=ActionType.FINAL_ANSWER,
            parameters={"answer": "test"},
        )
        
        step = ReActStep(step_number=1, thought=thought, action=action)
        step.mark_success()
        
        assert step.status == StepStatus.SUCCESS
    
    def test_step_mark_failed(self):
        """Test marking step as failed."""
        thought = Thought(content="Test")
        action = Action(
            action_type=ActionType.FINAL_ANSWER,
            parameters={"answer": "test"},
        )
        
        step = ReActStep(step_number=1, thought=thought, action=action)
        step.observation = Observation(content="", success=True)
        
        step.mark_failed("Test error")
        
        assert step.status == StepStatus.FAILED
        assert step.observation.error == "Test error"
    
    def test_step_to_dict(self):
        """Test converting step to dictionary."""
        thought = Thought(content="Test")
        action = Action(
            action_type=ActionType.FINAL_ANSWER,
            parameters={"answer": "test"},
        )
        
        step = ReActStep(step_number=1, thought=thought, action=action)
        
        result = step.to_dict()
        
        assert result["step_number"] == 1
        assert "thought" in result
        assert "action" in result


class TestReActTrace:
    """Test cases for ReActTrace class."""
    
    def test_create_trace(self):
        """Test creating a ReAct trace."""
        trace = ReActTrace(query="What is the vacation policy?")
        
        assert trace.query == "What is the vacation policy?"
        assert trace.steps == []
        assert trace.success is True
    
    def test_add_step(self):
        """Test adding steps to trace."""
        trace = ReActTrace(query="test")
        
        thought = Thought(content="Test")
        action = Action(
            action_type=ActionType.FINAL_ANSWER,
            parameters={"answer": "test"},
        )
        step = ReActStep(step_number=1, thought=thought, action=action)
        
        trace.add_step(step)
        
        assert trace.get_step_count() == 1
    
    def test_get_successful_steps(self):
        """Test getting successful steps."""
        trace = ReActTrace(query="test")
        
        # Add successful step
        thought1 = Thought(content="Test 1")
        action1 = Action(
            action_type=ActionType.FINAL_ANSWER,
            parameters={"answer": "test"},
        )
        step1 = ReActStep(step_number=1, thought=thought1, action=action1)
        step1.observation = Observation(content="success", success=True)
        step1.status = StepStatus.SUCCESS
        
        # Add failed step
        thought2 = Thought(content="Test 2")
        action2 = Action(
            action_type=ActionType.FINAL_ANSWER,
            parameters={"answer": "test"},
        )
        step2 = ReActStep(step_number=2, thought=thought2, action=action2)
        step2.status = StepStatus.FAILED
        
        trace.add_step(step1)
        trace.add_step(step2)
        
        successful = trace.get_successful_steps()
        
        assert len(successful) == 1
        assert successful[0].step_number == 1
    
    def test_get_tool_calls(self):
        """Test getting list of tool calls."""
        trace = ReActTrace(query="test")
        
        thought1 = Thought(content="Test 1")
        action1 = Action(
            action_type=ActionType.TOOL_CALL,
            tool_name="tool_a",
            parameters={},
        )
        step1 = ReActStep(step_number=1, thought=thought1, action=action1)
        
        thought2 = Thought(content="Test 2")
        action2 = Action(
            action_type=ActionType.TOOL_CALL,
            tool_name="tool_b",
            parameters={},
        )
        step2 = ReActStep(step_number=2, thought=thought2, action=action2)
        
        trace.add_step(step1)
        trace.add_step(step2)
        
        tools = trace.get_tool_calls()
        
        assert len(tools) == 2
        assert "tool_a" in tools
        assert "tool_b" in tools
    
    def test_get_unique_tools(self):
        """Test getting unique tools used."""
        trace = ReActTrace(query="test")
        
        # Use same tool twice
        thought1 = Thought(content="Test 1")
        action1 = Action(
            action_type=ActionType.TOOL_CALL,
            tool_name="tool_a",
            parameters={},
        )
        step1 = ReActStep(step_number=1, thought=thought1, action=action1)
        
        thought2 = Thought(content="Test 2")
        action2 = Action(
            action_type=ActionType.TOOL_CALL,
            tool_name="tool_a",
            parameters={},
        )
        step2 = ReActStep(step_number=2, thought=thought2, action=action2)
        
        trace.add_step(step1)
        trace.add_step(step2)
        
        unique_tools = trace.get_unique_tools()
        
        assert len(unique_tools) == 1
        assert "tool_a" in unique_tools
    
    def test_trace_to_dict_includes_statistics(self):
        """Test that to_dict includes statistics."""
        trace = ReActTrace(query="test")
        
        thought = Thought(content="Test")
        action = Action(
            action_type=ActionType.TOOL_CALL,
            tool_name="test_tool",
            parameters={},
        )
        step = ReActStep(step_number=1, thought=thought, action=action)
        step.observation = Observation(content="result", success=True)
        step.status = StepStatus.SUCCESS
        
        trace.add_step(step)
        trace.final_answer = "Test answer"
        
        result = trace.to_dict()
        
        assert "statistics" in result
        assert result["statistics"]["total_steps"] == 1
        assert result["statistics"]["successful_steps"] == 1
        assert "test_tool" in result["statistics"]["tools_used"]


class TestReActConfig:
    """Test cases for ReActConfig class."""
    
    def test_create_config_with_defaults(self):
        """Test creating config with default values."""
        config = ReActConfig()
        
        assert config.max_iterations == 5
        assert config.enable_reflection is True
        assert config.reflection_threshold == 0.7
        assert config.verbose is False
    
    def test_create_config_with_custom_values(self):
        """Test creating config with custom values."""
        config = ReActConfig(
            max_iterations=3,
            enable_reflection=False,
            verbose=True,
        )
        
        assert config.max_iterations == 3
        assert config.enable_reflection is False
        assert config.verbose is True
    
    def test_config_validation_max_iterations(self):
        """Test that max_iterations must be >= 1."""
        with pytest.raises(ValueError, match="must be >= 1"):
            ReActConfig(max_iterations=0)
    
    def test_config_validation_reflection_threshold(self):
        """Test that reflection_threshold must be between 0 and 1."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            ReActConfig(reflection_threshold=1.5)
    
    def test_config_validation_timeout(self):
        """Test that timeout must be > 0."""
        with pytest.raises(ValueError, match="must be > 0"):
            ReActConfig(timeout_seconds=0)
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = ReActConfig(max_iterations=3, verbose=True)
        
        result = config.to_dict()
        
        assert result["max_iterations"] == 3
        assert result["verbose"] is True
        assert "enable_reflection" in result
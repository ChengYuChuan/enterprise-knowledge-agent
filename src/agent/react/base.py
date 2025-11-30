"""
Base data structures for ReAct (Reasoning + Acting) engine.

This module defines the core types and data structures used in the ReAct
pattern for agent execution. The ReAct pattern combines reasoning (thinking
about what to do) with acting (executing actions) in an iterative loop.

The main components are:
- Thought: Agent's reasoning at each step
- Action: What the agent decides to do
- Observation: Result of executing an action
- ReActStep: One complete reasoning-action-observation cycle
- ReActTrace: Complete execution history
- ReActConfig: Configuration parameters

Reference:
    ReAct: Synergizing Reasoning and Acting in Language Models
    https://arxiv.org/abs/2210.03629
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ActionType(Enum):
    """
    Type of action the agent can take.
    
    Actions are categorized to enable different handling logic:
    
    - TOOL_CALL: Execute a registered tool/function
    - FINAL_ANSWER: Provide the final answer to the user
    - CLARIFICATION: Request clarification from the user
    - REFLECTION: Internal reflection (no external action)
    """
    
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    CLARIFICATION = "clarification"
    REFLECTION = "reflection"


class StepStatus(Enum):
    """
    Execution status of a ReAct step.
    
    Used to track the lifecycle of each step:
    
    - IN_PROGRESS: Step is currently being executed
    - SUCCESS: Step completed successfully
    - FAILED: Step failed with an error
    - SKIPPED: Step was skipped due to conditions
    """
    
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Thought:
    """
    Represents the agent's internal reasoning process.
    
    A Thought captures what the agent is thinking about the current
    situation and what it plans to do next. This is the "Reasoning"
    part of ReAct.
    
    Attributes:
        content: The actual thought/reasoning content.
        reasoning_type: Type of reasoning being performed.
        confidence: Agent's confidence in this reasoning (0.0-1.0).
        considerations: Key factors considered in this reasoning.
    
    Example:
        >>> thought = Thought(
        ...     content="I need to search the knowledge base for vacation policy",
        ...     reasoning_type="planning",
        ...     confidence=0.9,
        ...     considerations=["User asked about vacation", "Need specific info"]
        ... )
    """
    
    content: str
    reasoning_type: str = "analysis"  # analysis, planning, reflection, decision
    confidence: float = 0.8
    considerations: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate thought attributes."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        
        if not self.content.strip():
            raise ValueError("Thought content cannot be empty")
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return f"💭 Thought ({self.reasoning_type}): {self.content}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "reasoning_type": self.reasoning_type,
            "confidence": self.confidence,
            "considerations": self.considerations,
        }


@dataclass
class Action:
    """
    Represents an action the agent decides to take.
    
    An Action is what the agent does based on its reasoning. This is
    the "Acting" part of ReAct. Actions can be tool calls, final answers,
    or requests for clarification.
    
    Attributes:
        action_type: Type of action to perform.
        tool_name: Name of tool to call (required if action_type is TOOL_CALL).
        parameters: Parameters for the action/tool.
        rationale: Why this action was chosen.
    
    Example:
        >>> action = Action(
        ...     action_type=ActionType.TOOL_CALL,
        ...     tool_name="query_knowledge_base",
        ...     parameters={"query": "vacation policy", "top_k": 5},
        ...     rationale="Need to search for vacation policy information"
        ... )
    """
    
    action_type: ActionType
    tool_name: Optional[str] = None
    parameters: dict = field(default_factory=dict)
    rationale: str = ""
    
    def __post_init__(self) -> None:
        """Validate action attributes."""
        # Tool calls must have a tool name
        if self.action_type == ActionType.TOOL_CALL and not self.tool_name:
            raise ValueError("TOOL_CALL action must specify tool_name")
        
        # Final answers must have answer parameter
        if self.action_type == ActionType.FINAL_ANSWER and "answer" not in self.parameters:
            raise ValueError("FINAL_ANSWER action must have 'answer' parameter")
    
    def is_tool_call(self) -> bool:
        """Check if this is a tool call action."""
        return self.action_type == ActionType.TOOL_CALL
    
    def is_final_answer(self) -> bool:
        """Check if this is a final answer action."""
        return self.action_type == ActionType.FINAL_ANSWER
    
    def is_clarification(self) -> bool:
        """Check if this is a clarification request."""
        return self.action_type == ActionType.CLARIFICATION
    
    def __str__(self) -> str:
        """Human-readable representation."""
        if self.is_tool_call():
            params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            return f"🔧 Action: Call {self.tool_name}({params_str})"
        elif self.is_final_answer():
            answer_preview = self.parameters.get("answer", "")[:50]
            return f"✅ Action: Final Answer - {answer_preview}..."
        elif self.is_clarification():
            return f"❓ Action: Request Clarification"
        else:
            return f"🎯 Action: {self.action_type.value}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "action_type": self.action_type.value,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "rationale": self.rationale,
        }


@dataclass
class Observation:
    """
    Represents the observation from executing an action.
    
    An Observation is what the agent sees/receives after performing
    an action. For tool calls, this is the tool's output. For final
    answers, this might be empty.
    
    Attributes:
        content: The observation content (typically tool output).
        source: Source of this observation (tool name, system, etc.).
        success: Whether the action succeeded.
        error: Error message if action failed.
        metadata: Additional metadata about the observation.
    
    Example:
        >>> observation = Observation(
        ...     content="Vacation policy: 15 days per year for new employees",
        ...     source="query_knowledge_base",
        ...     success=True,
        ...     metadata={"confidence": 0.95, "sources": 3}
        ... )
    """
    
    content: str
    source: str = "unknown"
    success: bool = True
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate observation attributes."""
        # If not successful, should have error message
        if not self.success and not self.error:
            self.error = "Unknown error occurred"
    
    def is_successful(self) -> bool:
        """Check if observation represents success."""
        return self.success and not self.error
    
    def has_content(self) -> bool:
        """Check if observation has meaningful content."""
        return bool(self.content.strip())
    
    def __str__(self) -> str:
        """Human-readable representation."""
        if self.is_successful():
            content_preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
            return f"👀 Observation [{self.source}]: {content_preview}"
        else:
            return f"❌ Observation [{self.source}]: Error - {self.error}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "source": self.source,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ReActStep:
    """
    Represents one complete step in the ReAct loop.
    
    A ReActStep combines Thought → Action → Observation into a single
    unit, representing one iteration of the agent's reasoning-acting cycle.
    
    Attributes:
        step_number: Step number in the sequence (1-indexed).
        thought: Agent's reasoning for this step.
        action: Action decided upon.
        observation: Result of executing the action (None if not yet executed).
        status: Current execution status.
        timestamp: When this step was created.
        duration_ms: Time taken to execute this step (milliseconds).
    
    Example:
        >>> step = ReActStep(
        ...     step_number=1,
        ...     thought=Thought(content="Need to search KB"),
        ...     action=Action(
        ...         action_type=ActionType.TOOL_CALL,
        ...         tool_name="query_knowledge_base",
        ...         parameters={"query": "vacation"}
        ...     )
        ... )
    """
    
    step_number: int
    thought: Thought
    action: Action
    observation: Optional[Observation] = None
    status: StepStatus = StepStatus.IN_PROGRESS
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate step attributes."""
        if self.step_number < 1:
            raise ValueError("step_number must be >= 1")
    
    def is_complete(self) -> bool:
        """Check if step has been executed (has observation)."""
        return self.observation is not None
    
    def is_successful(self) -> bool:
        """Check if step completed successfully."""
        return (
            self.status == StepStatus.SUCCESS
            and self.observation is not None
            and self.observation.is_successful()
        )
    
    def mark_success(self) -> None:
        """Mark step as successfully completed."""
        self.status = StepStatus.SUCCESS
    
    def mark_failed(self, error: str) -> None:
        """
        Mark step as failed.
        
        Args:
            error: Error message.
        """
        self.status = StepStatus.FAILED
        if self.observation:
            self.observation.success = False
            self.observation.error = error
    
    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"\n{'='*60}",
            f"Step {self.step_number} [{self.status.value.upper()}]",
            f"{'='*60}",
            str(self.thought),
            str(self.action),
        ]
        
        if self.observation:
            lines.append(str(self.observation))
        
        if self.duration_ms > 0:
            lines.append(f"⏱️  Duration: {self.duration_ms:.0f}ms")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "step_number": self.step_number,
            "thought": self.thought.to_dict(),
            "action": self.action.to_dict(),
            "observation": self.observation.to_dict() if self.observation else None,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


@dataclass
class ReActTrace:
    """
    Complete trace of a ReAct execution.
    
    A ReActTrace records the entire execution history of the agent
    processing a query, including all steps taken, the final answer,
    and performance metadata.
    
    Attributes:
        query: Original user query.
        steps: List of all ReAct steps taken.
        final_answer: Final answer to return to user.
        total_duration_ms: Total execution time in milliseconds.
        success: Whether execution succeeded overall.
        error: Error message if execution failed.
        metadata: Additional execution metadata.
    
    Example:
        >>> trace = ReActTrace(query="What is the vacation policy?")
        >>> trace.add_step(step1)
        >>> trace.add_step(step2)
        >>> trace.final_answer = "New employees get 15 days..."
        >>> trace.success = True
    """
    
    query: str
    steps: list[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    total_duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def add_step(self, step: ReActStep) -> None:
        """
        Add a step to the trace.
        
        Args:
            step: ReAct step to add.
        """
        self.steps.append(step)
    
    def get_step_count(self) -> int:
        """Get total number of steps."""
        return len(self.steps)
    
    def get_successful_steps(self) -> list[ReActStep]:
        """Get list of successful steps only."""
        return [s for s in self.steps if s.is_successful()]
    
    def get_failed_steps(self) -> list[ReActStep]:
        """Get list of failed steps."""
        return [s for s in self.steps if s.status == StepStatus.FAILED]
    
    def get_tool_calls(self) -> list[str]:
        """Get list of all tools called during execution."""
        return [
            s.action.tool_name
            for s in self.steps
            if s.action.is_tool_call() and s.action.tool_name
        ]
    
    def get_unique_tools(self) -> set[str]:
        """Get set of unique tools used."""
        return set(self.get_tool_calls())
    
    def has_errors(self) -> bool:
        """Check if any steps failed."""
        return len(self.get_failed_steps()) > 0
    
    def to_dict(self) -> dict:
        """
        Convert trace to dictionary format.
        
        Returns:
            dict: Dictionary representation suitable for JSON serialization.
        """
        return {
            "query": self.query,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "total_duration_ms": self.total_duration_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "statistics": {
                "total_steps": self.get_step_count(),
                "successful_steps": len(self.get_successful_steps()),
                "failed_steps": len(self.get_failed_steps()),
                "tools_used": list(self.get_unique_tools()),
            },
        }
    
    def format_for_display(self, verbose: bool = True) -> str:
        """
        Format trace for human-readable display.
        
        Args:
            verbose: If True, show all step details. If False, show summary.
            
        Returns:
            str: Formatted trace string.
        """
        lines = [
            "\n" + "="*70,
            "REACT EXECUTION TRACE",
            "="*70,
            f"Query: {self.query}",
            f"Success: {'✅ Yes' if self.success else '❌ No'}",
            f"Total Duration: {self.total_duration_ms:.0f}ms",
            f"Steps Taken: {self.get_step_count()}",
            "",
        ]
        
        if verbose:
            # Show all steps
            for step in self.steps:
                lines.append(str(step))
        else:
            # Just show summary
            lines.append(f"Successful steps: {len(self.get_successful_steps())}")
            lines.append(f"Failed steps: {len(self.get_failed_steps())}")
            lines.append(f"Tools used: {', '.join(self.get_unique_tools())}")
        
        lines.extend([
            "",
            "="*70,
            "FINAL ANSWER",
            "="*70,
            self.final_answer,
            "="*70,
        ])
        
        if self.error:
            lines.extend([
                "",
                f"⚠️  Error: {self.error}",
            ])
        
        return "\n".join(lines)


@dataclass
class ReActConfig:
    """
    Configuration parameters for ReAct engine execution.
    
    These parameters control the behavior of the ReAct engine,
    including iteration limits, reflection settings, and timeouts.
    
    Attributes:
        max_iterations: Maximum number of reasoning-acting cycles.
        enable_reflection: Whether to enable self-reflection on answers.
        reflection_threshold: Minimum confidence for accepting answer.
        timeout_seconds: Maximum total execution time.
        verbose: Whether to print detailed execution logs.
        allow_empty_observations: Whether to continue if observation is empty.
    
    Example:
        >>> config = ReActConfig(
        ...     max_iterations=3,
        ...     enable_reflection=True,
        ...     verbose=True
        ... )
    """
    
    max_iterations: int = 5
    enable_reflection: bool = True
    reflection_threshold: float = 0.7
    timeout_seconds: float = 30.0
    verbose: bool = False
    allow_empty_observations: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        
        if not 0.0 <= self.reflection_threshold <= 1.0:
            raise ValueError("reflection_threshold must be between 0.0 and 1.0")
        
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "max_iterations": self.max_iterations,
            "enable_reflection": self.enable_reflection,
            "reflection_threshold": self.reflection_threshold,
            "timeout_seconds": self.timeout_seconds,
            "verbose": self.verbose,
            "allow_empty_observations": self.allow_empty_observations,
        }
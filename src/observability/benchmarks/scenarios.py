"""
Benchmark Scenarios

This module defines different types of benchmark scenarios for testing
the performance of the Knowledge Agent system.

Scenario Types:
    1. LATENCY: Measure response time distribution
    2. THROUGHPUT: Measure maximum requests per second
    3. STRESS: Find breaking points under increasing load
    4. COMPONENT: Isolate and test individual components

Each scenario defines:
    - What to test (queries, operations)
    - How to test (concurrency, duration, iterations)
    - What to measure (latency, throughput, errors)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import random


logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ScenarioType(Enum):
    """Types of benchmark scenarios."""
    
    LATENCY = "latency"           # Measure response time
    THROUGHPUT = "throughput"     # Measure requests per second
    STRESS = "stress"             # Find breaking points
    COMPONENT = "component"       # Test individual components
    ENDURANCE = "endurance"       # Long-running stability test
    SPIKE = "spike"               # Sudden load increase


class ComponentType(Enum):
    """Components that can be benchmarked."""
    
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    GENERATION = "generation"
    FULL_PIPELINE = "full_pipeline"
    API_ENDPOINT = "api_endpoint"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QuerySet:
    """
    A set of queries for benchmarking.
    
    Attributes:
        queries: List of query strings.
        weights: Optional weights for query selection probability.
        metadata: Additional metadata for the query set.
    """
    
    queries: List[str]
    weights: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize weights."""
        if self.weights:
            if len(self.weights) != len(self.queries):
                raise ValueError("Weights must match number of queries")
            # Normalize weights to sum to 1
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
    
    def sample(self) -> str:
        """Sample a query based on weights."""
        if self.weights:
            return random.choices(self.queries, weights=self.weights, k=1)[0]
        return random.choice(self.queries)
    
    def __len__(self) -> int:
        return len(self.queries)


@dataclass
class LoadProfile:
    """
    Defines the load pattern for a benchmark.
    
    Attributes:
        initial_users: Starting number of concurrent users.
        max_users: Maximum number of concurrent users.
        ramp_up_time_seconds: Time to ramp up to max users.
        steady_state_seconds: Duration at max load.
        ramp_down_time_seconds: Time to ramp down.
    """
    
    initial_users: int = 1
    max_users: int = 10
    ramp_up_time_seconds: float = 10.0
    steady_state_seconds: float = 60.0
    ramp_down_time_seconds: float = 5.0
    
    def get_users_at_time(self, elapsed_seconds: float) -> int:
        """
        Get the number of users at a given time.
        
        Args:
            elapsed_seconds: Time since benchmark start.
        
        Returns:
            Number of concurrent users.
        """
        if elapsed_seconds < self.ramp_up_time_seconds:
            # Ramp up phase
            progress = elapsed_seconds / self.ramp_up_time_seconds
            return int(self.initial_users + progress * (self.max_users - self.initial_users))
        
        elif elapsed_seconds < self.ramp_up_time_seconds + self.steady_state_seconds:
            # Steady state
            return self.max_users
        
        elif elapsed_seconds < self.total_duration:
            # Ramp down phase
            ramp_down_start = self.ramp_up_time_seconds + self.steady_state_seconds
            progress = (elapsed_seconds - ramp_down_start) / self.ramp_down_time_seconds
            return int(self.max_users - progress * (self.max_users - self.initial_users))
        
        else:
            return 0
    
    @property
    def total_duration(self) -> float:
        """Total duration of the load profile."""
        return (
            self.ramp_up_time_seconds + 
            self.steady_state_seconds + 
            self.ramp_down_time_seconds
        )


# =============================================================================
# Benchmark Scenario
# =============================================================================

@dataclass
class BenchmarkScenario:
    """
    Defines a complete benchmark scenario.
    
    Attributes:
        name: Scenario name.
        type: Type of benchmark.
        description: Human-readable description.
        query_set: Set of queries to use.
        load_profile: Load pattern configuration.
        component: Component to benchmark (for component tests).
        num_iterations: Number of iterations (for latency tests).
        duration_seconds: Test duration (for throughput tests).
        warmup_iterations: Warmup iterations before measurement.
        timeout_seconds: Request timeout.
        error_threshold: Maximum acceptable error rate (0-1).
        target_rps: Target requests per second (optional).
        custom_config: Additional configuration.
    """
    
    name: str
    type: ScenarioType
    description: str = ""
    query_set: Optional[QuerySet] = None
    load_profile: LoadProfile = field(default_factory=LoadProfile)
    component: ComponentType = ComponentType.FULL_PIPELINE
    num_iterations: int = 100
    duration_seconds: float = 60.0
    warmup_iterations: int = 10
    timeout_seconds: float = 30.0
    error_threshold: float = 0.05
    target_rps: Optional[float] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def get_query(self) -> str:
        """Get a query for this iteration."""
        if self.query_set:
            return self.query_set.sample()
        return "What is artificial intelligence?"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "component": self.component.value,
            "num_iterations": self.num_iterations,
            "duration_seconds": self.duration_seconds,
            "warmup_iterations": self.warmup_iterations,
            "timeout_seconds": self.timeout_seconds,
            "error_threshold": self.error_threshold,
            "target_rps": self.target_rps,
            "load_profile": {
                "initial_users": self.load_profile.initial_users,
                "max_users": self.load_profile.max_users,
                "ramp_up_time_seconds": self.load_profile.ramp_up_time_seconds,
                "steady_state_seconds": self.load_profile.steady_state_seconds,
            },
            "custom_config": self.custom_config,
        }


# =============================================================================
# Scenario Factory Functions
# =============================================================================

# Default test queries
DEFAULT_QUERIES = [
    "What is machine learning?",
    "How does natural language processing work?",
    "Explain the concept of neural networks.",
    "What are the applications of artificial intelligence?",
    "How does deep learning differ from traditional ML?",
    "What is transfer learning?",
    "Explain reinforcement learning with examples.",
    "What are transformer models?",
    "How does BERT work?",
    "What is GPT and how is it used?",
]


def create_latency_scenario(
    queries: Optional[List[str]] = None,
    num_runs: int = 100,
    warmup: int = 10,
    timeout: float = 30.0,
    name: str = "latency_benchmark",
) -> BenchmarkScenario:
    """
    Create a latency measurement scenario.
    
    This scenario runs queries sequentially and measures response time
    distribution (P50, P95, P99, etc.).
    
    Args:
        queries: List of queries to test.
        num_runs: Number of test iterations.
        warmup: Number of warmup iterations.
        timeout: Request timeout in seconds.
        name: Scenario name.
    
    Returns:
        Configured BenchmarkScenario.
    
    Example:
        ```python
        scenario = create_latency_scenario(
            queries=["What is AI?", "How does ML work?"],
            num_runs=100,
        )
        result = await runner.run(scenario)
        print(f"P99 Latency: {result.latency_p99_ms}ms")
        ```
    """
    query_set = QuerySet(queries=queries or DEFAULT_QUERIES)
    
    return BenchmarkScenario(
        name=name,
        type=ScenarioType.LATENCY,
        description=f"Measure response latency over {num_runs} requests",
        query_set=query_set,
        load_profile=LoadProfile(
            initial_users=1,
            max_users=1,
            ramp_up_time_seconds=0,
            steady_state_seconds=0,
        ),
        num_iterations=num_runs,
        warmup_iterations=warmup,
        timeout_seconds=timeout,
    )


def create_throughput_scenario(
    queries: Optional[List[str]] = None,
    duration_seconds: float = 60.0,
    concurrency: int = 10,
    ramp_up_seconds: float = 10.0,
    target_rps: Optional[float] = None,
    name: str = "throughput_benchmark",
) -> BenchmarkScenario:
    """
    Create a throughput measurement scenario.
    
    This scenario runs concurrent requests and measures maximum
    sustainable throughput (requests per second).
    
    Args:
        queries: List of queries to test.
        duration_seconds: Test duration.
        concurrency: Number of concurrent users.
        ramp_up_seconds: Ramp up time.
        target_rps: Target requests per second (None for max).
        name: Scenario name.
    
    Returns:
        Configured BenchmarkScenario.
    """
    query_set = QuerySet(queries=queries or DEFAULT_QUERIES)
    
    return BenchmarkScenario(
        name=name,
        type=ScenarioType.THROUGHPUT,
        description=f"Measure throughput with {concurrency} concurrent users",
        query_set=query_set,
        load_profile=LoadProfile(
            initial_users=1,
            max_users=concurrency,
            ramp_up_time_seconds=ramp_up_seconds,
            steady_state_seconds=duration_seconds,
            ramp_down_time_seconds=5.0,
        ),
        duration_seconds=duration_seconds,
        target_rps=target_rps,
    )


def create_stress_scenario(
    queries: Optional[List[str]] = None,
    start_users: int = 1,
    max_users: int = 100,
    step_users: int = 10,
    step_duration_seconds: float = 30.0,
    error_threshold: float = 0.10,
    name: str = "stress_test",
) -> BenchmarkScenario:
    """
    Create a stress test scenario.
    
    This scenario gradually increases load until the system fails
    or error rate exceeds threshold, finding the breaking point.
    
    Args:
        queries: List of queries to test.
        start_users: Starting number of users.
        max_users: Maximum users to scale to.
        step_users: Users to add per step.
        step_duration_seconds: Duration of each step.
        error_threshold: Error rate that indicates failure.
        name: Scenario name.
    
    Returns:
        Configured BenchmarkScenario.
    """
    query_set = QuerySet(queries=queries or DEFAULT_QUERIES)
    
    num_steps = (max_users - start_users) // step_users + 1
    total_duration = num_steps * step_duration_seconds
    
    return BenchmarkScenario(
        name=name,
        type=ScenarioType.STRESS,
        description=f"Stress test from {start_users} to {max_users} users",
        query_set=query_set,
        load_profile=LoadProfile(
            initial_users=start_users,
            max_users=max_users,
            ramp_up_time_seconds=total_duration * 0.8,
            steady_state_seconds=total_duration * 0.2,
            ramp_down_time_seconds=10.0,
        ),
        duration_seconds=total_duration,
        error_threshold=error_threshold,
        custom_config={
            "step_users": step_users,
            "step_duration_seconds": step_duration_seconds,
        },
    )


def create_component_scenario(
    component: ComponentType,
    queries: Optional[List[str]] = None,
    num_runs: int = 50,
    name: Optional[str] = None,
) -> BenchmarkScenario:
    """
    Create a component-level benchmark scenario.
    
    This scenario isolates a specific component (retrieval, embedding,
    etc.) for focused performance measurement.
    
    Args:
        component: Component to benchmark.
        queries: List of queries to test.
        num_runs: Number of test iterations.
        name: Scenario name.
    
    Returns:
        Configured BenchmarkScenario.
    """
    query_set = QuerySet(queries=queries or DEFAULT_QUERIES)
    scenario_name = name or f"{component.value}_benchmark"
    
    return BenchmarkScenario(
        name=scenario_name,
        type=ScenarioType.COMPONENT,
        description=f"Benchmark {component.value} component",
        query_set=query_set,
        component=component,
        load_profile=LoadProfile(
            initial_users=1,
            max_users=1,
            ramp_up_time_seconds=0,
            steady_state_seconds=0,
        ),
        num_iterations=num_runs,
        warmup_iterations=5,
    )


def create_spike_scenario(
    queries: Optional[List[str]] = None,
    baseline_users: int = 5,
    spike_users: int = 50,
    baseline_duration: float = 30.0,
    spike_duration: float = 10.0,
    recovery_duration: float = 30.0,
    name: str = "spike_test",
) -> BenchmarkScenario:
    """
    Create a spike test scenario.
    
    This scenario simulates sudden traffic spikes to test system
    resilience and recovery.
    
    Args:
        queries: List of queries to test.
        baseline_users: Normal load level.
        spike_users: Peak load during spike.
        baseline_duration: Duration at baseline before spike.
        spike_duration: Duration of the spike.
        recovery_duration: Duration to measure recovery.
        name: Scenario name.
    
    Returns:
        Configured BenchmarkScenario.
    """
    query_set = QuerySet(queries=queries or DEFAULT_QUERIES)
    
    total_duration = baseline_duration + spike_duration + recovery_duration
    
    return BenchmarkScenario(
        name=name,
        type=ScenarioType.SPIKE,
        description=f"Spike from {baseline_users} to {spike_users} users",
        query_set=query_set,
        load_profile=LoadProfile(
            initial_users=baseline_users,
            max_users=spike_users,
            ramp_up_time_seconds=1.0,  # Sudden spike
            steady_state_seconds=spike_duration,
            ramp_down_time_seconds=1.0,  # Sudden drop
        ),
        duration_seconds=total_duration,
        custom_config={
            "baseline_users": baseline_users,
            "spike_users": spike_users,
            "baseline_duration": baseline_duration,
            "spike_duration": spike_duration,
            "recovery_duration": recovery_duration,
        },
    )


def create_endurance_scenario(
    queries: Optional[List[str]] = None,
    users: int = 10,
    duration_hours: float = 1.0,
    name: str = "endurance_test",
) -> BenchmarkScenario:
    """
    Create an endurance test scenario.
    
    This scenario runs for an extended period to detect memory leaks,
    resource exhaustion, and degradation over time.
    
    Args:
        queries: List of queries to test.
        users: Number of concurrent users.
        duration_hours: Test duration in hours.
        name: Scenario name.
    
    Returns:
        Configured BenchmarkScenario.
    """
    query_set = QuerySet(queries=queries or DEFAULT_QUERIES)
    duration_seconds = duration_hours * 3600
    
    return BenchmarkScenario(
        name=name,
        type=ScenarioType.ENDURANCE,
        description=f"Endurance test for {duration_hours} hours",
        query_set=query_set,
        load_profile=LoadProfile(
            initial_users=users,
            max_users=users,
            ramp_up_time_seconds=60.0,
            steady_state_seconds=duration_seconds - 120.0,
            ramp_down_time_seconds=60.0,
        ),
        duration_seconds=duration_seconds,
        custom_config={
            "sample_interval_seconds": 60.0,  # Collect metrics every minute
        },
    )
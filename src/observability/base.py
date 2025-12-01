"""
Base Observability Classes

This module provides abstract base classes and interfaces for the observability
layer. It follows the Ports & Adapters pattern, allowing different implementations
(Phoenix, OpenTelemetry, Prometheus, etc.) to be swapped without changing the
core application code.

Design Principles:
    1. Interface Segregation: Separate interfaces for tracing and metrics
    2. Dependency Injection: Implementations are injected at runtime
    3. Context Propagation: Trace context flows through async boundaries
    4. Low Overhead: Observability should not significantly impact performance

Example Usage:
    ```python
    from src.observability import get_tracer, get_metrics

    tracer = get_tracer()
    metrics = get_metrics()

    with tracer.span("my_operation") as span:
        span.set_attribute("key", "value")
        metrics.increment("operation_count")
        # ... do work ...
    ```
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union
import functools
import time


# =============================================================================
# Enums and Constants
# =============================================================================

class SpanKind(Enum):
    """Types of spans in the trace hierarchy."""
    
    INTERNAL = "internal"      # Internal operation
    CLIENT = "client"          # Outgoing request (e.g., LLM call)
    SERVER = "server"          # Incoming request (e.g., API endpoint)
    PRODUCER = "producer"      # Message producer
    CONSUMER = "consumer"      # Message consumer
    LLM = "llm"               # LLM inference call
    RETRIEVER = "retriever"    # Retrieval operation
    EMBEDDING = "embedding"    # Embedding generation
    RERANKER = "reranker"      # Reranking operation
    CHAIN = "chain"            # Chain/pipeline operation
    TOOL = "tool"              # Tool execution
    AGENT = "agent"            # Agent operation


class SpanStatus(Enum):
    """Status of a span."""
    
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class MetricType(Enum):
    """Types of metrics."""
    
    COUNTER = "counter"          # Monotonically increasing value
    GAUGE = "gauge"              # Value that can go up and down
    HISTOGRAM = "histogram"      # Distribution of values
    SUMMARY = "summary"          # Similar to histogram with quantiles


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SpanContext:
    """
    Context information for a span.
    
    This allows trace context to be propagated across service boundaries
    and async operations.
    
    Attributes:
        trace_id: Unique identifier for the trace.
        span_id: Unique identifier for this span.
        parent_span_id: ID of the parent span, if any.
        trace_flags: Flags for trace sampling, etc.
        trace_state: Additional vendor-specific trace state.
    """
    
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 0
    trace_state: Dict[str, str] = field(default_factory=dict)


@dataclass
class SpanAttributes:
    """
    Standard attributes for different span types.
    
    These follow OpenTelemetry semantic conventions where applicable,
    with extensions for LLM-specific attributes.
    """
    
    # Common attributes
    service_name: Optional[str] = None
    service_version: Optional[str] = None
    environment: Optional[str] = None
    
    # LLM attributes
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None
    llm_input_tokens: Optional[int] = None
    llm_output_tokens: Optional[int] = None
    llm_total_tokens: Optional[int] = None
    llm_prompt: Optional[str] = None
    llm_completion: Optional[str] = None
    
    # Retrieval attributes
    retrieval_query: Optional[str] = None
    retrieval_top_k: Optional[int] = None
    retrieval_num_results: Optional[int] = None
    retrieval_strategy: Optional[str] = None  # vector, bm25, hybrid
    
    # Embedding attributes
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    embedding_num_texts: Optional[int] = None
    
    # Reranker attributes
    reranker_model: Optional[str] = None
    reranker_top_k: Optional[int] = None
    
    # Error attributes
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_stacktrace: Optional[str] = None
    
    # Custom attributes
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if key == "custom":
                    result.update(value)
                else:
                    result[key] = value
        return result


@dataclass
class MetricLabels:
    """
    Labels for metrics.
    
    Labels allow metrics to be filtered and aggregated by different dimensions.
    
    Attributes:
        service: Service name.
        operation: Operation name (e.g., "chat", "search").
        provider: Provider name (e.g., "openai", "anthropic").
        model: Model name.
        status: Status (e.g., "success", "error").
        error_type: Type of error, if any.
    """
    
    service: str = "knowledge_agent"
    operation: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    status: Optional[str] = None
    error_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# =============================================================================
# Abstract Base Classes
# =============================================================================

class Span(ABC):
    """
    Abstract base class for a trace span.
    
    A span represents a unit of work or operation. It tracks the time
    spent in the operation, along with metadata and any errors.
    
    Spans can be nested to form a trace tree, showing the hierarchical
    relationship between operations.
    """
    
    @property
    @abstractmethod
    def context(self) -> SpanContext:
        """Get the span context."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the span name."""
        pass
    
    @property
    @abstractmethod
    def kind(self) -> SpanKind:
        """Get the span kind."""
        pass
    
    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> "Span":
        """
        Set a single attribute on the span.
        
        Args:
            key: Attribute name.
            value: Attribute value (must be serializable).
            
        Returns:
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def set_attributes(self, attributes: Union[SpanAttributes, Dict[str, Any]]) -> "Span":
        """
        Set multiple attributes on the span.
        
        Args:
            attributes: SpanAttributes object or dictionary.
            
        Returns:
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def set_status(self, status: SpanStatus, description: Optional[str] = None) -> "Span":
        """
        Set the span status.
        
        Args:
            status: Status code.
            description: Optional description for error status.
            
        Returns:
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def record_exception(self, exception: Exception) -> "Span":
        """
        Record an exception that occurred during the span.
        
        Args:
            exception: The exception to record.
            
        Returns:
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        """
        Add an event to the span.
        
        Events are time-stamped annotations that can be used to record
        interesting happenings within the span.
        
        Args:
            name: Event name.
            attributes: Optional event attributes.
            
        Returns:
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def end(self) -> None:
        """End the span, recording its duration."""
        pass


class Tracer(ABC):
    """
    Abstract base class for a tracer.
    
    A tracer is responsible for creating spans and managing trace context.
    It provides methods for starting spans and propagating context.
    """
    
    @abstractmethod
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """
        Start a new span.
        
        Args:
            name: Span name.
            kind: Type of span.
            parent: Optional parent span context.
            attributes: Optional initial attributes.
            
        Returns:
            A new Span instance.
        """
        pass
    
    @contextmanager
    @abstractmethod
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """
        Context manager for creating spans.
        
        This is the preferred way to create spans as it ensures
        proper cleanup even if exceptions occur.
        
        Args:
            name: Span name.
            kind: Type of span.
            attributes: Optional initial attributes.
            
        Yields:
            The created Span instance.
            
        Example:
            ```python
            with tracer.span("my_operation") as span:
                span.set_attribute("key", "value")
                # ... do work ...
            ```
        """
        pass
    
    @abstractmethod
    def get_current_span(self) -> Optional[Span]:
        """
        Get the currently active span.
        
        Returns:
            The current span, or None if no span is active.
        """
        pass
    
    @abstractmethod
    def inject_context(self, carrier: Dict[str, str]) -> None:
        """
        Inject trace context into a carrier for propagation.
        
        Args:
            carrier: Dictionary to inject context into.
        """
        pass
    
    @abstractmethod
    def extract_context(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """
        Extract trace context from a carrier.
        
        Args:
            carrier: Dictionary containing trace context.
            
        Returns:
            Extracted SpanContext, or None if not present.
        """
        pass


class MetricsCollector(ABC):
    """
    Abstract base class for metrics collection.
    
    Provides methods for recording different types of metrics.
    """
    
    @abstractmethod
    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[MetricLabels] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name.
            value: Value to add (default 1).
            labels: Optional metric labels.
            description: Optional metric description.
        """
        pass
    
    @abstractmethod
    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[MetricLabels] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Set a gauge metric.
        
        Args:
            name: Metric name.
            value: Current value.
            labels: Optional metric labels.
            description: Optional metric description.
        """
        pass
    
    @abstractmethod
    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[MetricLabels] = None,
        description: Optional[str] = None,
        buckets: Optional[List[float]] = None,
    ) -> None:
        """
        Record a histogram observation.
        
        Args:
            name: Metric name.
            value: Observed value.
            labels: Optional metric labels.
            description: Optional metric description.
            buckets: Optional histogram buckets.
        """
        pass
    
    @contextmanager
    @abstractmethod
    def timer(
        self,
        name: str,
        labels: Optional[MetricLabels] = None,
        description: Optional[str] = None,
    ) -> Generator[None, None, None]:
        """
        Context manager for timing operations.
        
        Records the duration as a histogram metric.
        
        Args:
            name: Metric name.
            labels: Optional metric labels.
            description: Optional metric description.
            
        Example:
            ```python
            with metrics.timer("operation_duration"):
                # ... do work ...
            ```
        """
        pass


# =============================================================================
# Decorator Utilities
# =============================================================================

T = TypeVar("T", bound=Callable[..., Any])


def traced(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[T], T]:
    """
    Decorator for tracing function calls.
    
    Creates a span for the decorated function, recording its duration
    and any exceptions that occur.
    
    Args:
        name: Span name (defaults to function name).
        kind: Type of span.
        attributes: Optional attributes to add to the span.
        
    Returns:
        Decorated function.
        
    Example:
        ```python
        @traced(kind=SpanKind.LLM)
        async def generate_response(prompt: str) -> str:
            # ... call LLM ...
        ```
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            from src.observability import get_tracer
            tracer = get_tracer()
            span_name = name or func.__name__
            
            with tracer.span(span_name, kind=kind, attributes=attributes) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(SpanStatus.OK)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(SpanStatus.ERROR, str(e))
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            from src.observability import get_tracer
            tracer = get_tracer()
            span_name = name or func.__name__
            
            with tracer.span(span_name, kind=kind, attributes=attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(SpanStatus.OK)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(SpanStatus.ERROR, str(e))
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


def timed(
    name: Optional[str] = None,
    labels: Optional[MetricLabels] = None,
) -> Callable[[T], T]:
    """
    Decorator for timing function calls.
    
    Records the function duration as a histogram metric.
    
    Args:
        name: Metric name (defaults to function name + "_duration_seconds").
        labels: Optional metric labels.
        
    Returns:
        Decorated function.
        
    Example:
        ```python
        @timed()
        async def process_request(request: Request) -> Response:
            # ... process ...
        ```
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            from src.observability import get_metrics
            metrics = get_metrics()
            metric_name = name or f"{func.__name__}_duration_seconds"
            
            with metrics.timer(metric_name, labels=labels):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            from src.observability import get_metrics
            metrics = get_metrics()
            metric_name = name or f"{func.__name__}_duration_seconds"
            
            with metrics.timer(metric_name, labels=labels):
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


# =============================================================================
# No-op Implementations (for testing or when observability is disabled)
# =============================================================================

class NoOpSpan(Span):
    """No-operation span implementation."""
    
    def __init__(self, name: str = "", kind: SpanKind = SpanKind.INTERNAL):
        self._name = name
        self._kind = kind
        self._context = SpanContext(
            trace_id="00000000000000000000000000000000",
            span_id="0000000000000000",
        )
    
    @property
    def context(self) -> SpanContext:
        return self._context
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def kind(self) -> SpanKind:
        return self._kind
    
    def set_attribute(self, key: str, value: Any) -> "NoOpSpan":
        return self
    
    def set_attributes(self, attributes: Union[SpanAttributes, Dict[str, Any]]) -> "NoOpSpan":
        return self
    
    def set_status(self, status: SpanStatus, description: Optional[str] = None) -> "NoOpSpan":
        return self
    
    def record_exception(self, exception: Exception) -> "NoOpSpan":
        return self
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "NoOpSpan":
        return self
    
    def end(self) -> None:
        pass


class NoOpTracer(Tracer):
    """No-operation tracer implementation."""
    
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        return NoOpSpan(name, kind)
    
    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        span = NoOpSpan(name, kind)
        try:
            yield span
        finally:
            span.end()
    
    def get_current_span(self) -> Optional[Span]:
        return None
    
    def inject_context(self, carrier: Dict[str, str]) -> None:
        pass
    
    def extract_context(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        return None


class NoOpMetricsCollector(MetricsCollector):
    """No-operation metrics collector implementation."""
    
    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[MetricLabels] = None,
        description: Optional[str] = None,
    ) -> None:
        pass
    
    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[MetricLabels] = None,
        description: Optional[str] = None,
    ) -> None:
        pass
    
    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[MetricLabels] = None,
        description: Optional[str] = None,
        buckets: Optional[List[float]] = None,
    ) -> None:
        pass
    
    @contextmanager
    def timer(
        self,
        name: str,
        labels: Optional[MetricLabels] = None,
        description: Optional[str] = None,
    ) -> Generator[None, None, None]:
        yield
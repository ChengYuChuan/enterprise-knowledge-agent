"""
Arize Phoenix Tracer Implementation

This module provides integration with Arize Phoenix for LLM observability.
Phoenix is an open-source tool designed specifically for tracing and debugging
LLM applications.

Key Features:
    - Automatic LLM call tracing with token counting
    - Retrieval span tracking with relevance scores
    - Embedding generation monitoring
    - Reranking operation tracing
    - Conversation/session grouping

Architecture:
    Phoenix uses OpenTelemetry under the hood, but provides LLM-specific
    semantic conventions and a specialized UI for debugging RAG pipelines.

    ┌─────────────────────────────────────────────────────────────┐
    │                    Your Application                          │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
    │  │  LLM    │  │Retrieval│  │Embedding│  │    Reranker     │ │
    │  │  Calls  │  │  Calls  │  │  Calls  │  │     Calls       │ │
    │  └────┬────┘  └────┬────┘  └────┬────┘  └───────┬─────────┘ │
    │       │            │            │               │           │
    │       └────────────┴────────────┴───────────────┘           │
    │                          │                                   │
    │                   PhoenixTracer                              │
    │                          │                                   │
    └──────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Arize Phoenix     │
                    │   (localhost:6006)  │
                    │                     │
                    │  - Trace Viewer     │
                    │  - Span Analysis    │
                    │  - Token Metrics    │
                    └─────────────────────┘

Usage:
    ```python
    from src.observability.tracing import PhoenixTracer

    # Initialize tracer
    tracer = PhoenixTracer(
        endpoint="http://localhost:6006",
        service_name="knowledge-agent"
    )

    # Trace an LLM call
    with tracer.span("chat_completion", kind=SpanKind.LLM) as span:
        span.set_attributes({
            "llm.provider": "openai",
            "llm.model": "gpt-4",
            "llm.prompt": prompt,
        })
        
        response = await openai_client.chat.completions.create(...)
        
        span.set_attributes({
            "llm.completion": response.choices[0].message.content,
            "llm.input_tokens": response.usage.prompt_tokens,
            "llm.output_tokens": response.usage.completion_tokens,
        })
    ```

Requirements:
    pip install arize-phoenix opentelemetry-api opentelemetry-sdk

References:
    - Phoenix Documentation: https://docs.arize.com/phoenix
    - OpenTelemetry Python: https://opentelemetry.io/docs/instrumentation/python/
"""

import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Union

from src.observability.base import (
    Span,
    SpanContext,
    SpanKind,
    SpanStatus,
    SpanAttributes,
    Tracer,
)


logger = logging.getLogger(__name__)

# Context variable for tracking current span
_current_span: ContextVar[Optional["PhoenixSpan"]] = ContextVar(
    "current_span", default=None
)


# =============================================================================
# Phoenix Span Implementation
# =============================================================================

class PhoenixSpan(Span):
    """
    Phoenix-compatible span implementation.
    
    This span implementation integrates with Arize Phoenix for LLM-specific
    tracing and debugging capabilities.
    
    Attributes:
        _name: Span name.
        _kind: Type of span.
        _context: Span context for trace propagation.
        _attributes: Span attributes.
        _events: Recorded events.
        _status: Current span status.
        _start_time: Span start timestamp.
        _end_time: Span end timestamp.
        _otel_span: Underlying OpenTelemetry span (if available).
    """
    
    def __init__(
        self,
        name: str,
        kind: SpanKind,
        context: SpanContext,
        parent_span: Optional["PhoenixSpan"] = None,
        otel_span: Optional[Any] = None,
    ):
        """
        Initialize a Phoenix span.
        
        Args:
            name: Span name.
            kind: Type of span.
            context: Span context.
            parent_span: Optional parent span.
            otel_span: Optional underlying OpenTelemetry span.
        """
        self._name = name
        self._kind = kind
        self._context = context
        self._parent_span = parent_span
        self._otel_span = otel_span
        
        self._attributes: Dict[str, Any] = {}
        self._events: List[Dict[str, Any]] = []
        self._status = SpanStatus.UNSET
        self._status_description: Optional[str] = None
        self._exception: Optional[Exception] = None
        
        self._start_time = time.time()
        self._end_time: Optional[float] = None
    
    @property
    def context(self) -> SpanContext:
        """Get the span context."""
        return self._context
    
    @property
    def name(self) -> str:
        """Get the span name."""
        return self._name
    
    @property
    def kind(self) -> SpanKind:
        """Get the span kind."""
        return self._kind
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self._end_time is None:
            return None
        return (self._end_time - self._start_time) * 1000
    
    def set_attribute(self, key: str, value: Any) -> "PhoenixSpan":
        """
        Set a single attribute on the span.
        
        Args:
            key: Attribute name.
            value: Attribute value.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[key] = value
        
        # Also set on OpenTelemetry span if available
        if self._otel_span is not None:
            try:
                self._otel_span.set_attribute(key, self._serialize_value(value))
            except Exception as e:
                logger.debug(f"Failed to set OTel attribute: {e}")
        
        return self
    
    def set_attributes(
        self, 
        attributes: Union[SpanAttributes, Dict[str, Any]]
    ) -> "PhoenixSpan":
        """
        Set multiple attributes on the span.
        
        Args:
            attributes: SpanAttributes object or dictionary.
            
        Returns:
            Self for method chaining.
        """
        if isinstance(attributes, SpanAttributes):
            attr_dict = attributes.to_dict()
        else:
            attr_dict = attributes
        
        for key, value in attr_dict.items():
            self.set_attribute(key, value)
        
        return self
    
    def set_status(
        self, 
        status: SpanStatus, 
        description: Optional[str] = None
    ) -> "PhoenixSpan":
        """
        Set the span status.
        
        Args:
            status: Status code.
            description: Optional description.
            
        Returns:
            Self for method chaining.
        """
        self._status = status
        self._status_description = description
        
        # Set on OpenTelemetry span if available
        if self._otel_span is not None:
            try:
                from opentelemetry.trace import StatusCode
                
                otel_status = {
                    SpanStatus.UNSET: StatusCode.UNSET,
                    SpanStatus.OK: StatusCode.OK,
                    SpanStatus.ERROR: StatusCode.ERROR,
                }.get(status, StatusCode.UNSET)
                
                self._otel_span.set_status(otel_status, description)
            except Exception as e:
                logger.debug(f"Failed to set OTel status: {e}")
        
        return self
    
    def record_exception(self, exception: Exception) -> "PhoenixSpan":
        """
        Record an exception that occurred during the span.
        
        Args:
            exception: The exception to record.
            
        Returns:
            Self for method chaining.
        """
        self._exception = exception
        
        # Add exception attributes
        import traceback
        self.set_attribute("error.type", type(exception).__name__)
        self.set_attribute("error.message", str(exception))
        self.set_attribute("error.stacktrace", traceback.format_exc())
        
        # Add event
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            }
        )
        
        # Record on OpenTelemetry span if available
        if self._otel_span is not None:
            try:
                self._otel_span.record_exception(exception)
            except Exception as e:
                logger.debug(f"Failed to record OTel exception: {e}")
        
        return self
    
    def add_event(
        self, 
        name: str, 
        attributes: Optional[Dict[str, Any]] = None
    ) -> "PhoenixSpan":
        """
        Add an event to the span.
        
        Args:
            name: Event name.
            attributes: Optional event attributes.
            
        Returns:
            Self for method chaining.
        """
        event = {
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        }
        self._events.append(event)
        
        # Add to OpenTelemetry span if available
        if self._otel_span is not None:
            try:
                self._otel_span.add_event(name, attributes or {})
            except Exception as e:
                logger.debug(f"Failed to add OTel event: {e}")
        
        return self
    
    def end(self) -> None:
        """End the span, recording its duration."""
        self._end_time = time.time()
        
        # Add final timing attribute
        self.set_attribute("duration_ms", self.duration_ms)
        
        # End OpenTelemetry span if available
        if self._otel_span is not None:
            try:
                self._otel_span.end()
            except Exception as e:
                logger.debug(f"Failed to end OTel span: {e}")
        
        # Log span completion for debugging
        logger.debug(
            f"Span ended: {self._name} "
            f"(kind={self._kind.value}, "
            f"duration={self.duration_ms:.2f}ms, "
            f"status={self._status.value})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert span to dictionary representation.
        
        Returns:
            Dictionary with span data.
        """
        return {
            "name": self._name,
            "kind": self._kind.value,
            "trace_id": self._context.trace_id,
            "span_id": self._context.span_id,
            "parent_span_id": self._context.parent_span_id,
            "start_time": datetime.fromtimestamp(self._start_time).isoformat(),
            "end_time": (
                datetime.fromtimestamp(self._end_time).isoformat()
                if self._end_time else None
            ),
            "duration_ms": self.duration_ms,
            "status": self._status.value,
            "status_description": self._status_description,
            "attributes": self._attributes,
            "events": self._events,
        }
    
    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """
        Serialize a value for OpenTelemetry.
        
        OpenTelemetry only accepts certain types for attributes.
        """
        if isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return [PhoenixSpan._serialize_value(v) for v in value]
        else:
            return str(value)


# =============================================================================
# Phoenix Tracer Implementation
# =============================================================================

class PhoenixTracer(Tracer):
    """
    Arize Phoenix tracer implementation.
    
    This tracer integrates with Phoenix for LLM-specific observability.
    It supports both local span collection and export to a Phoenix server.
    
    Attributes:
        endpoint: Phoenix collector endpoint.
        service_name: Name of the service for identification.
        _tracer: Underlying OpenTelemetry tracer (if available).
        _spans: Local span storage for debugging.
    """
    
    def __init__(
        self,
        endpoint: str = "http://localhost:6006",
        service_name: str = "enterprise-knowledge-agent",
        project_name: Optional[str] = None,
    ):
        """
        Initialize the Phoenix tracer.
        
        Args:
            endpoint: Phoenix collector endpoint URL.
            service_name: Service name for trace identification.
            project_name: Optional project name for Phoenix grouping.
        """
        self.endpoint = endpoint
        self.service_name = service_name
        self.project_name = project_name or service_name
        
        self._tracer: Optional[Any] = None
        self._spans: List[PhoenixSpan] = []
        self._initialized = False
        
        # Try to initialize Phoenix
        self._initialize_phoenix()
    
    def _initialize_phoenix(self) -> None:
        """
        Initialize Phoenix and OpenTelemetry instrumentation.
        
        This sets up the tracing pipeline to export spans to Phoenix.
        """
        try:
            # Import Phoenix
            import phoenix as px
            from phoenix.otel import register
            
            # Register with Phoenix
            tracer_provider = register(
                project_name=self.project_name,
                endpoint=f"{self.endpoint}/v1/traces",
            )
            
            # Get tracer from provider
            from opentelemetry import trace
            self._tracer = trace.get_tracer(
                self.service_name,
                tracer_provider=tracer_provider,
            )
            
            self._initialized = True
            logger.info(
                f"Phoenix tracer initialized: endpoint={self.endpoint}, "
                f"project={self.project_name}"
            )
            
        except ImportError as e:
            logger.warning(
                f"Phoenix or OpenTelemetry not available: {e}. "
                "Spans will be collected locally only."
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize Phoenix: {e}. "
                "Spans will be collected locally only."
            )
    
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> PhoenixSpan:
        """
        Start a new span.
        
        Args:
            name: Span name.
            kind: Type of span.
            parent: Optional parent span context.
            attributes: Optional initial attributes.
            
        Returns:
            A new PhoenixSpan instance.
        """
        # Determine parent span
        parent_span = _current_span.get()
        
        if parent is not None:
            parent_span_id = parent.span_id
            trace_id = parent.trace_id
        elif parent_span is not None:
            parent_span_id = parent_span.context.span_id
            trace_id = parent_span.context.trace_id
        else:
            parent_span_id = None
            trace_id = self._generate_trace_id()
        
        # Create span context
        context = SpanContext(
            trace_id=trace_id,
            span_id=self._generate_span_id(),
            parent_span_id=parent_span_id,
        )
        
        # Create OpenTelemetry span if tracer is available
        otel_span = None
        if self._tracer is not None:
            try:
                from opentelemetry.trace import SpanKind as OTelSpanKind
                
                # Map our SpanKind to OpenTelemetry SpanKind
                otel_kind_map = {
                    SpanKind.INTERNAL: OTelSpanKind.INTERNAL,
                    SpanKind.CLIENT: OTelSpanKind.CLIENT,
                    SpanKind.SERVER: OTelSpanKind.SERVER,
                    SpanKind.PRODUCER: OTelSpanKind.PRODUCER,
                    SpanKind.CONSUMER: OTelSpanKind.CONSUMER,
                    SpanKind.LLM: OTelSpanKind.CLIENT,
                    SpanKind.RETRIEVER: OTelSpanKind.INTERNAL,
                    SpanKind.EMBEDDING: OTelSpanKind.CLIENT,
                    SpanKind.RERANKER: OTelSpanKind.CLIENT,
                    SpanKind.CHAIN: OTelSpanKind.INTERNAL,
                    SpanKind.TOOL: OTelSpanKind.INTERNAL,
                    SpanKind.AGENT: OTelSpanKind.INTERNAL,
                }
                otel_kind = otel_kind_map.get(kind, OTelSpanKind.INTERNAL)
                
                otel_span = self._tracer.start_span(
                    name,
                    kind=otel_kind,
                    attributes=self._prepare_otel_attributes(kind, attributes),
                )
            except Exception as e:
                logger.debug(f"Failed to create OTel span: {e}")
        
        # Create our span wrapper
        span = PhoenixSpan(
            name=name,
            kind=kind,
            context=context,
            parent_span=parent_span,
            otel_span=otel_span,
        )
        
        # Set initial attributes
        span.set_attribute("service.name", self.service_name)
        span.set_attribute("span.kind", kind.value)
        
        if attributes:
            span.set_attributes(attributes)
        
        # Store span
        self._spans.append(span)
        
        return span
    
    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[PhoenixSpan, None, None]:
        """
        Context manager for creating spans.
        
        Args:
            name: Span name.
            kind: Type of span.
            attributes: Optional initial attributes.
            
        Yields:
            The created PhoenixSpan instance.
        """
        span = self.start_span(name, kind=kind, attributes=attributes)
        token = _current_span.set(span)
        
        try:
            yield span
            if span._status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.record_exception(e)
            span.set_status(SpanStatus.ERROR, str(e))
            raise
        finally:
            span.end()
            _current_span.reset(token)
    
    def get_current_span(self) -> Optional[PhoenixSpan]:
        """
        Get the currently active span.
        
        Returns:
            The current span, or None if no span is active.
        """
        return _current_span.get()
    
    def inject_context(self, carrier: Dict[str, str]) -> None:
        """
        Inject trace context into a carrier for propagation.
        
        Args:
            carrier: Dictionary to inject context into.
        """
        span = self.get_current_span()
        if span is not None:
            carrier["traceparent"] = (
                f"00-{span.context.trace_id}-{span.context.span_id}-01"
            )
    
    def extract_context(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """
        Extract trace context from a carrier.
        
        Args:
            carrier: Dictionary containing trace context.
            
        Returns:
            Extracted SpanContext, or None if not present.
        """
        traceparent = carrier.get("traceparent")
        if traceparent is None:
            return None
        
        try:
            parts = traceparent.split("-")
            if len(parts) >= 3:
                return SpanContext(
                    trace_id=parts[1],
                    span_id=parts[2],
                )
        except Exception:
            pass
        
        return None
    
    def get_spans(self) -> List[PhoenixSpan]:
        """
        Get all recorded spans.
        
        Returns:
            List of recorded spans.
        """
        return self._spans.copy()
    
    def clear_spans(self) -> None:
        """Clear all recorded spans."""
        self._spans.clear()
    
    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return uuid.uuid4().hex
    
    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return uuid.uuid4().hex[:16]
    
    def _prepare_otel_attributes(
        self,
        kind: SpanKind,
        attributes: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Prepare attributes for OpenTelemetry, adding semantic conventions.
        
        Args:
            kind: Span kind.
            attributes: User-provided attributes.
            
        Returns:
            Prepared attributes dictionary.
        """
        result: Dict[str, Any] = {
            "openinference.span.kind": kind.value.upper(),
        }
        
        # Add LLM-specific semantic attributes
        if kind == SpanKind.LLM:
            result["openinference.span.kind"] = "LLM"
        elif kind == SpanKind.RETRIEVER:
            result["openinference.span.kind"] = "RETRIEVER"
        elif kind == SpanKind.EMBEDDING:
            result["openinference.span.kind"] = "EMBEDDING"
        elif kind == SpanKind.RERANKER:
            result["openinference.span.kind"] = "RERANKER"
        elif kind == SpanKind.CHAIN:
            result["openinference.span.kind"] = "CHAIN"
        elif kind == SpanKind.TOOL:
            result["openinference.span.kind"] = "TOOL"
        elif kind == SpanKind.AGENT:
            result["openinference.span.kind"] = "AGENT"
        
        # Merge user attributes
        if attributes:
            for key, value in attributes.items():
                result[key] = PhoenixSpan._serialize_value(value)
        
        return result


# =============================================================================
# Convenience Functions for LLM Tracing
# =============================================================================

def trace_llm_call(
    tracer: PhoenixTracer,
    name: str = "llm_call",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> contextmanager:
    """
    Create a context manager for tracing LLM calls.
    
    This is a convenience function that sets up common LLM attributes.
    
    Args:
        tracer: The tracer to use.
        name: Span name.
        provider: LLM provider name.
        model: Model name.
        
    Returns:
        Context manager that yields a span.
        
    Example:
        ```python
        with trace_llm_call(tracer, provider="openai", model="gpt-4") as span:
            response = await client.chat.completions.create(...)
            span.set_attribute("llm.input_tokens", response.usage.prompt_tokens)
        ```
    """
    attributes = {}
    if provider:
        attributes["llm.provider"] = provider
    if model:
        attributes["llm.model"] = model
    
    return tracer.span(name, kind=SpanKind.LLM, attributes=attributes)


def trace_retrieval(
    tracer: PhoenixTracer,
    name: str = "retrieval",
    query: Optional[str] = None,
    top_k: Optional[int] = None,
    strategy: Optional[str] = None,
) -> contextmanager:
    """
    Create a context manager for tracing retrieval operations.
    
    Args:
        tracer: The tracer to use.
        name: Span name.
        query: Search query.
        top_k: Number of results requested.
        strategy: Retrieval strategy (vector, bm25, hybrid).
        
    Returns:
        Context manager that yields a span.
    """
    attributes = {}
    if query:
        attributes["retrieval.query"] = query
    if top_k:
        attributes["retrieval.top_k"] = top_k
    if strategy:
        attributes["retrieval.strategy"] = strategy
    
    return tracer.span(name, kind=SpanKind.RETRIEVER, attributes=attributes)


def trace_embedding(
    tracer: PhoenixTracer,
    name: str = "embedding",
    model: Optional[str] = None,
    num_texts: Optional[int] = None,
) -> contextmanager:
    """
    Create a context manager for tracing embedding operations.
    
    Args:
        tracer: The tracer to use.
        name: Span name.
        model: Embedding model name.
        num_texts: Number of texts being embedded.
        
    Returns:
        Context manager that yields a span.
    """
    attributes = {}
    if model:
        attributes["embedding.model"] = model
    if num_texts:
        attributes["embedding.num_texts"] = num_texts
    
    return tracer.span(name, kind=SpanKind.EMBEDDING, attributes=attributes)
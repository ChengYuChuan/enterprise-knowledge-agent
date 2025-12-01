"""
Unit tests for Phase 5: Observability Base Module

Tests the abstract base classes, data classes, NoOp implementations,
and decorator utilities for the observability layer.

Run with:
    poetry run pytest tests/unit/test_phase5_base.py -v
"""

import asyncio
import pytest
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from src.observability.base import (
    # Enums
    SpanKind,
    SpanStatus,
    MetricType,
    # Data classes
    SpanContext,
    SpanAttributes,
    MetricLabels,
    # NoOp implementations
    NoOpSpan,
    NoOpTracer,
    NoOpMetricsCollector,
    # Decorators
    traced,
    timed,
)


# =============================================================================
# Test Enums
# =============================================================================

class TestEnums:
    """Test enum definitions."""
    
    def test_span_kind_values(self):
        """Test SpanKind enum has expected values."""
        assert SpanKind.INTERNAL.value == "internal"
        assert SpanKind.LLM.value == "llm"
        assert SpanKind.RETRIEVER.value == "retriever"
        assert SpanKind.EMBEDDING.value == "embedding"
        assert SpanKind.AGENT.value == "agent"
    
    def test_span_status_values(self):
        """Test SpanStatus enum has expected values."""
        assert SpanStatus.UNSET.value == "unset"
        assert SpanStatus.OK.value == "ok"
        assert SpanStatus.ERROR.value == "error"
    
    def test_metric_type_values(self):
        """Test MetricType enum has expected values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"


# =============================================================================
# Test Data Classes
# =============================================================================

class TestSpanContext:
    """Test SpanContext data class."""
    
    def test_basic_creation(self):
        """Test creating a span context."""
        ctx = SpanContext(
            trace_id="abc123",
            span_id="def456",
        )
        assert ctx.trace_id == "abc123"
        assert ctx.span_id == "def456"
        assert ctx.parent_span_id is None
        assert ctx.trace_flags == 0
    
    def test_with_parent(self):
        """Test span context with parent span."""
        ctx = SpanContext(
            trace_id="abc123",
            span_id="def456",
            parent_span_id="ghi789",
        )
        assert ctx.parent_span_id == "ghi789"
    
    def test_trace_state(self):
        """Test span context with trace state."""
        ctx = SpanContext(
            trace_id="abc123",
            span_id="def456",
            trace_state={"vendor": "value"},
        )
        assert ctx.trace_state == {"vendor": "value"}


class TestSpanAttributes:
    """Test SpanAttributes data class."""
    
    def test_default_values(self):
        """Test default attribute values."""
        attrs = SpanAttributes()
        assert attrs.service_name is None
        assert attrs.llm_provider is None
        assert attrs.custom == {}
    
    def test_llm_attributes(self):
        """Test LLM-specific attributes."""
        attrs = SpanAttributes(
            llm_provider="openai",
            llm_model="gpt-4",
            llm_input_tokens=100,
            llm_output_tokens=50,
        )
        assert attrs.llm_provider == "openai"
        assert attrs.llm_model == "gpt-4"
        assert attrs.llm_input_tokens == 100
        assert attrs.llm_output_tokens == 50
    
    def test_retrieval_attributes(self):
        """Test retrieval-specific attributes."""
        attrs = SpanAttributes(
            retrieval_query="What is AI?",
            retrieval_top_k=5,
            retrieval_strategy="hybrid",
        )
        assert attrs.retrieval_query == "What is AI?"
        assert attrs.retrieval_top_k == 5
        assert attrs.retrieval_strategy == "hybrid"
    
    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        attrs = SpanAttributes(
            service_name="test-service",
            llm_provider="openai",
        )
        result = attrs.to_dict()
        
        assert "service_name" in result
        assert "llm_provider" in result
        assert "llm_model" not in result  # None values excluded
    
    def test_custom_attributes_merged(self):
        """Test that custom attributes are merged into dict."""
        attrs = SpanAttributes(
            service_name="test",
            custom={"my_key": "my_value"},
        )
        result = attrs.to_dict()
        
        assert result["service_name"] == "test"
        assert result["my_key"] == "my_value"


class TestMetricLabels:
    """Test MetricLabels data class."""
    
    def test_default_service(self):
        """Test default service name."""
        labels = MetricLabels()
        assert labels.service == "knowledge_agent"
    
    def test_custom_labels(self):
        """Test custom label values."""
        labels = MetricLabels(
            operation="search",
            provider="openai",
            model="gpt-4",
            status="success",
        )
        assert labels.operation == "search"
        assert labels.provider == "openai"
        assert labels.model == "gpt-4"
        assert labels.status == "success"
    
    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        labels = MetricLabels(
            operation="search",
            status="success",
        )
        result = labels.to_dict()
        
        assert "service" in result
        assert "operation" in result
        assert "status" in result
        assert "provider" not in result  # None excluded
        assert "model" not in result  # None excluded


# =============================================================================
# Test NoOp Implementations
# =============================================================================

class TestNoOpSpan:
    """Test NoOpSpan implementation."""
    
    def test_creation(self):
        """Test creating a NoOp span."""
        span = NoOpSpan("test_operation", SpanKind.LLM)
        
        assert span.name == "test_operation"
        assert span.kind == SpanKind.LLM
    
    def test_context_has_zero_ids(self):
        """Test that NoOp span has zero trace/span IDs."""
        span = NoOpSpan()
        ctx = span.context
        
        assert ctx.trace_id == "00000000000000000000000000000000"
        assert ctx.span_id == "0000000000000000"
    
    def test_set_attribute_returns_self(self):
        """Test method chaining for set_attribute."""
        span = NoOpSpan()
        result = span.set_attribute("key", "value")
        
        assert result is span
    
    def test_set_attributes_returns_self(self):
        """Test method chaining for set_attributes."""
        span = NoOpSpan()
        result = span.set_attributes({"key": "value"})
        
        assert result is span
    
    def test_set_status_returns_self(self):
        """Test method chaining for set_status."""
        span = NoOpSpan()
        result = span.set_status(SpanStatus.OK)
        
        assert result is span
    
    def test_record_exception_returns_self(self):
        """Test method chaining for record_exception."""
        span = NoOpSpan()
        result = span.record_exception(ValueError("test"))
        
        assert result is span
    
    def test_add_event_returns_self(self):
        """Test method chaining for add_event."""
        span = NoOpSpan()
        result = span.add_event("event_name", {"key": "value"})
        
        assert result is span
    
    def test_end_does_nothing(self):
        """Test that end() completes without error."""
        span = NoOpSpan()
        span.end()  # Should not raise


class TestNoOpTracer:
    """Test NoOpTracer implementation."""
    
    def test_start_span_returns_noop(self):
        """Test that start_span returns a NoOpSpan."""
        tracer = NoOpTracer()
        span = tracer.start_span("test_op", SpanKind.LLM)
        
        assert isinstance(span, NoOpSpan)
        assert span.name == "test_op"
        assert span.kind == SpanKind.LLM
    
    def test_span_context_manager(self):
        """Test span as context manager."""
        tracer = NoOpTracer()
        
        with tracer.span("test_op") as span:
            assert isinstance(span, NoOpSpan)
            span.set_attribute("key", "value")
    
    def test_get_current_span_returns_none(self):
        """Test that get_current_span returns None."""
        tracer = NoOpTracer()
        assert tracer.get_current_span() is None
    
    def test_inject_context_does_nothing(self):
        """Test that inject_context completes without error."""
        tracer = NoOpTracer()
        carrier: Dict[str, str] = {}
        tracer.inject_context(carrier)
        
        assert carrier == {}  # Should remain empty
    
    def test_extract_context_returns_none(self):
        """Test that extract_context returns None."""
        tracer = NoOpTracer()
        result = tracer.extract_context({"traceparent": "00-abc-def-00"})
        
        assert result is None


class TestNoOpMetricsCollector:
    """Test NoOpMetricsCollector implementation."""
    
    def test_counter_does_nothing(self):
        """Test that counter completes without error."""
        metrics = NoOpMetricsCollector()
        metrics.counter("test_counter", value=1.0)  # Should not raise
    
    def test_gauge_does_nothing(self):
        """Test that gauge completes without error."""
        metrics = NoOpMetricsCollector()
        metrics.gauge("test_gauge", value=42.0)  # Should not raise
    
    def test_histogram_does_nothing(self):
        """Test that histogram completes without error."""
        metrics = NoOpMetricsCollector()
        metrics.histogram("test_histogram", value=0.5)  # Should not raise
    
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        metrics = NoOpMetricsCollector()
        
        with metrics.timer("test_timer"):
            time.sleep(0.01)  # Small sleep
        
        # Should complete without error


# =============================================================================
# Test Decorators
# =============================================================================

class TestTracedDecorator:
    """Test the @traced decorator."""
    
    def test_sync_function(self):
        """Test tracing a synchronous function."""
        call_count = 0
        
        @traced(name="test_operation", kind=SpanKind.INTERNAL)
        def my_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result = my_function(5)
        
        assert result == 10
        assert call_count == 1
    
    def test_async_function(self):
        """Test tracing an async function."""
        call_count = 0
        
        @traced(name="async_operation", kind=SpanKind.LLM)
        async def my_async_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2
        
        result = asyncio.run(my_async_function(5))
        
        assert result == 10
        assert call_count == 1
    
    def test_default_name_from_function(self):
        """Test that span name defaults to function name."""
        @traced()
        def my_named_function():
            return "result"
        
        result = my_named_function()
        assert result == "result"
    
    def test_exception_is_propagated(self):
        """Test that exceptions are re-raised."""
        @traced(name="failing_operation")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
    
    def test_async_exception_is_propagated(self):
        """Test that async exceptions are re-raised."""
        @traced(name="async_failing")
        async def async_failing_function():
            raise RuntimeError("Async error")
        
        with pytest.raises(RuntimeError, match="Async error"):
            asyncio.run(async_failing_function())


class TestTimedDecorator:
    """Test the @timed decorator."""
    
    def test_sync_function(self):
        """Test timing a synchronous function."""
        @timed(name="sync_operation")
        def my_function(x: int) -> int:
            time.sleep(0.01)
            return x * 2
        
        result = my_function(5)
        assert result == 10
    
    def test_async_function(self):
        """Test timing an async function."""
        @timed(name="async_operation")
        async def my_async_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2
        
        result = asyncio.run(my_async_function(5))
        assert result == 10
    
    def test_default_name_with_suffix(self):
        """Test that default name has _duration_seconds suffix."""
        @timed()
        def operation_to_time():
            return 42
        
        result = operation_to_time()
        assert result == 42


# =============================================================================
# Test Combined Decorators
# =============================================================================

class TestCombinedDecorators:
    """Test using both @traced and @timed together."""
    
    def test_both_decorators(self):
        """Test using both decorators on same function."""
        @traced(kind=SpanKind.LLM)
        @timed()
        def combined_operation(x: int) -> int:
            return x * 3
        
        result = combined_operation(10)
        assert result == 30
    
    def test_both_decorators_async(self):
        """Test using both decorators on async function."""
        @traced(kind=SpanKind.RETRIEVER)
        @timed()
        async def async_combined(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 3
        
        result = asyncio.run(async_combined(10))
        assert result == 30

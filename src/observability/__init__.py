"""
Observability Module

This module provides unified observability capabilities for the Knowledge Agent:

1. **Tracing**: Distributed tracing with Arize Phoenix for LLM observability
2. **Metrics**: Prometheus-compatible metrics for system monitoring
3. **Evaluation**: RAG evaluation with Ragas framework
4. **Benchmarks**: Performance benchmarking tools

Architecture:
    The module uses a plugin-based architecture where implementations can be
    swapped without changing application code. By default, it uses:
    - Arize Phoenix for tracing (with fallback to NoOp)
    - Prometheus client for metrics (with fallback to NoOp)

Usage:
    ```python
    from src.observability import get_tracer, get_metrics, traced, timed

    # Get singleton instances
    tracer = get_tracer()
    metrics = get_metrics()

    # Use decorators for automatic instrumentation
    @traced(kind=SpanKind.LLM)
    @timed()
    async def my_llm_call():
        ...

    # Or use context managers
    with tracer.span("operation") as span:
        span.set_attribute("key", "value")
        metrics.counter("operation_count")
    ```

Configuration:
    Observability is configured via environment variables or settings:

    - PHOENIX_ENABLED: Enable/disable Phoenix tracing (default: true)
    - PHOENIX_ENDPOINT: Phoenix collector endpoint
    - PROMETHEUS_ENABLED: Enable/disable Prometheus metrics (default: true)
    - OBSERVABILITY_SERVICE_NAME: Service name for traces/metrics
"""

from typing import Optional

from src.observability.base import (
    # Enums
    SpanKind,
    SpanStatus,
    MetricType,
    # Data classes
    SpanContext,
    SpanAttributes,
    MetricLabels,
    # Abstract classes
    Span,
    Tracer,
    MetricsCollector,
    # Decorators
    traced,
    timed,
    # NoOp implementations
    NoOpSpan,
    NoOpTracer,
    NoOpMetricsCollector,
)

# =============================================================================
# Module-level singletons
# =============================================================================

_tracer: Optional[Tracer] = None
_metrics: Optional[MetricsCollector] = None


# =============================================================================
# Factory Functions
# =============================================================================

def get_tracer(force_reinit: bool = False) -> Tracer:
    """
    Get the global tracer instance.
    
    This function returns a singleton tracer. On first call, it initializes
    the appropriate tracer based on configuration. If Phoenix is enabled and
    available, it uses PhoenixTracer; otherwise, it falls back to NoOpTracer.
    
    Args:
        force_reinit: Force reinitialization of the tracer.
        
    Returns:
        Global Tracer instance.
        
    Example:
        ```python
        tracer = get_tracer()
        with tracer.span("my_operation") as span:
            span.set_attribute("key", "value")
        ```
    """
    global _tracer
    
    if _tracer is None or force_reinit:
        _tracer = _initialize_tracer()
    
    return _tracer


def get_metrics(force_reinit: bool = False) -> MetricsCollector:
    """
    Get the global metrics collector instance.
    
    This function returns a singleton metrics collector. On first call, it
    initializes the appropriate collector based on configuration. If Prometheus
    is enabled, it uses PrometheusMetrics; otherwise, it falls back to
    NoOpMetricsCollector.
    
    Args:
        force_reinit: Force reinitialization of the collector.
        
    Returns:
        Global MetricsCollector instance.
        
    Example:
        ```python
        metrics = get_metrics()
        metrics.counter("requests_total")
        metrics.histogram("request_duration_seconds", 0.5)
        ```
    """
    global _metrics
    
    if _metrics is None or force_reinit:
        _metrics = _initialize_metrics()
    
    return _metrics


def _initialize_tracer() -> Tracer:
    """
    Initialize the tracer based on configuration.
    
    Returns:
        Configured Tracer instance.
        
    Configuration:
        Phoenix tracing is disabled by default for local development.
        To enable, set environment variable: OBSERVABILITY_PHOENIX_ENABLED=true
    """
    try:
        from src.config import get_settings
        settings = get_settings()
        
        # Check if Phoenix is enabled via nested observability settings
        # Default: False for local development (no Phoenix server required)
        obs_settings = getattr(settings, "observability", None)
        if obs_settings:
            phoenix_enabled = getattr(obs_settings, "phoenix_enabled", False)
            phoenix_endpoint = getattr(obs_settings, "phoenix_endpoint", "http://localhost:6006")
            service_name = getattr(obs_settings, "service_name", "enterprise-knowledge-agent")
        else:
            # Fallback for older config without observability section
            phoenix_enabled = getattr(settings, "phoenix_enabled", False)
            phoenix_endpoint = getattr(settings, "phoenix_endpoint", "http://localhost:6006")
            service_name = getattr(settings, "service_name", "enterprise-knowledge-agent")
        
        if phoenix_enabled:
            try:
                from src.observability.tracing.phoenix import PhoenixTracer
                
                return PhoenixTracer(
                    endpoint=phoenix_endpoint,
                    service_name=service_name,
                )
            except ImportError:
                # Phoenix not installed, fall back to NoOp
                import logging
                logging.getLogger(__name__).warning(
                    "Phoenix tracing enabled but arize-phoenix not installed. "
                    "Install with: pip install arize-phoenix"
                )
                return NoOpTracer()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to initialize Phoenix tracer: {e}. Using NoOp tracer."
                )
                return NoOpTracer()
        else:
            return NoOpTracer()
            
    except Exception:
        # If settings aren't available, use NoOp
        return NoOpTracer()


def _initialize_metrics() -> MetricsCollector:
    """
    Initialize the metrics collector based on configuration.
    
    Returns:
        Configured MetricsCollector instance.
        
    Configuration:
        Prometheus metrics are enabled by default.
        To disable, set environment variable: OBSERVABILITY_PROMETHEUS_ENABLED=false
    """
    try:
        from src.config import get_settings
        settings = get_settings()
        
        # Check if Prometheus is enabled via nested observability settings
        obs_settings = getattr(settings, "observability", None)
        if obs_settings:
            prometheus_enabled = getattr(obs_settings, "prometheus_enabled", True)
            service_name = getattr(obs_settings, "service_name", "enterprise_knowledge_agent")
        else:
            # Fallback for older config without observability section
            prometheus_enabled = getattr(settings, "prometheus_enabled", True)
            service_name = getattr(settings, "service_name", "enterprise_knowledge_agent")
        
        if prometheus_enabled:
            try:
                from src.observability.metrics.prometheus import PrometheusMetrics
                
                return PrometheusMetrics(
                    namespace=service_name.replace("-", "_"),
                )
            except ImportError:
                # prometheus_client not installed, fall back to NoOp
                import logging
                logging.getLogger(__name__).warning(
                    "Prometheus metrics enabled but prometheus_client not installed. "
                    "Install with: pip install prometheus-client"
                )
                return NoOpMetricsCollector()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to initialize Prometheus metrics: {e}. Using NoOp collector."
                )
                return NoOpMetricsCollector()
        else:
            return NoOpMetricsCollector()
            
    except Exception:
        # If settings aren't available, use NoOp
        return NoOpMetricsCollector()


# =============================================================================
# Convenience Functions
# =============================================================================

def set_tracer(tracer: Tracer) -> None:
    """
    Set the global tracer instance.
    
    This is useful for testing or when you want to use a custom tracer.
    
    Args:
        tracer: Tracer instance to use globally.
    """
    global _tracer
    _tracer = tracer


def set_metrics(metrics: MetricsCollector) -> None:
    """
    Set the global metrics collector instance.
    
    This is useful for testing or when you want to use a custom collector.
    
    Args:
        metrics: MetricsCollector instance to use globally.
    """
    global _metrics
    _metrics = metrics


def reset() -> None:
    """
    Reset all global observability instances.
    
    This is primarily useful for testing.
    """
    global _tracer, _metrics
    _tracer = None
    _metrics = None


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enums
    "SpanKind",
    "SpanStatus",
    "MetricType",
    # Data classes
    "SpanContext",
    "SpanAttributes",
    "MetricLabels",
    # Abstract classes
    "Span",
    "Tracer",
    "MetricsCollector",
    # Decorators
    "traced",
    "timed",
    # NoOp implementations
    "NoOpSpan",
    "NoOpTracer",
    "NoOpMetricsCollector",
    # Factory functions
    "get_tracer",
    "get_metrics",
    # Utility functions
    "set_tracer",
    "set_metrics",
    "reset",
]
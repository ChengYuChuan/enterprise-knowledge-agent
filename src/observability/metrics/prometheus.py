"""
Prometheus Metrics Implementation

This module provides integration with Prometheus for system metrics collection.
Prometheus is a widely-adopted monitoring system that provides a multi-dimensional
data model, flexible query language, and robust alerting.

Architecture:
    ┌────────────────────────────────────────────────────────────────────┐
    │                      Your Application                              │
    │  ┌──────────────────────────────────────────────────────────────┐ │
    │  │                   PrometheusMetrics                           │ │
    │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │ │
    │  │  │ Counters │  │  Gauges  │  │Histograms│  │  Summaries   │  │ │
    │  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │ │
    │  └──────────────────────────────────────────────────────────────┘ │
    │                              │                                     │
    │                    GET /metrics                                    │
    └──────────────────────────────┼─────────────────────────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │     Prometheus      │
                        │     Server          │
                        │                     │
                        │  - Time series DB   │
                        │  - PromQL queries   │
                        │  - Alerting rules   │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │      Grafana        │
                        │                     │
                        │  - Dashboards       │
                        │  - Visualization    │
                        └─────────────────────┘

Metric Types:
    1. Counter: Cumulative metric that only increases (e.g., request count)
    2. Gauge: Metric that can increase or decrease (e.g., active connections)
    3. Histogram: Samples observations into buckets (e.g., request duration)
    4. Summary: Similar to histogram but calculates quantiles (less common)

Naming Conventions:
    - Use snake_case for metric names
    - Include unit in name suffix (e.g., _seconds, _bytes, _total)
    - Use consistent label names across metrics

Usage:
    ```python
    from src.observability.metrics import PrometheusMetrics, MetricLabels

    metrics = PrometheusMetrics(namespace="knowledge_agent")

    # Counter example
    metrics.counter(
        "requests_total",
        labels=MetricLabels(operation="chat", status="success"),
        description="Total number of requests"
    )

    # Gauge example
    metrics.gauge(
        "active_connections",
        value=42,
        description="Number of active connections"
    )

    # Histogram example (for latency)
    metrics.histogram(
        "request_duration_seconds",
        value=0.5,
        labels=MetricLabels(operation="search"),
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )

    # Timer context manager
    with metrics.timer("operation_duration_seconds"):
        # ... do work ...
    ```

Requirements:
    pip install prometheus-client

References:
    - Prometheus: https://prometheus.io/
    - prometheus_client: https://github.com/prometheus/client_python
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

from src.observability.base import MetricLabels, MetricType, MetricsCollector


logger = logging.getLogger(__name__)

# Singleton instance
_prometheus_metrics: Optional["PrometheusMetrics"] = None


def get_prometheus_metrics() -> "PrometheusMetrics":
    """
    Get the global PrometheusMetrics instance.
    
    Returns:
        Singleton PrometheusMetrics instance.
    """
    global _prometheus_metrics
    if _prometheus_metrics is None:
        _prometheus_metrics = PrometheusMetrics()
    return _prometheus_metrics


# =============================================================================
# Default Bucket Configurations
# =============================================================================

# Latency buckets (in seconds)
LATENCY_BUCKETS = (
    0.005,   # 5ms
    0.01,    # 10ms
    0.025,   # 25ms
    0.05,    # 50ms
    0.075,   # 75ms
    0.1,     # 100ms
    0.25,    # 250ms
    0.5,     # 500ms
    0.75,    # 750ms
    1.0,     # 1s
    2.5,     # 2.5s
    5.0,     # 5s
    7.5,     # 7.5s
    10.0,    # 10s
    float("inf"),
)

# Token count buckets
TOKEN_BUCKETS = (
    10, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, float("inf")
)

# Document count buckets
DOCUMENT_BUCKETS = (1, 2, 5, 10, 20, 50, 100, float("inf"))

# Score buckets (0-1 range)
SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


# =============================================================================
# Prometheus Metrics Implementation
# =============================================================================

class PrometheusMetrics(MetricsCollector):
    """
    Prometheus metrics collector implementation.
    
    This class provides a wrapper around the prometheus_client library,
    implementing the MetricsCollector interface with automatic metric
    registration and label management.
    
    Attributes:
        namespace: Metric namespace prefix.
        _counters: Registered counter metrics.
        _gauges: Registered gauge metrics.
        _histograms: Registered histogram metrics.
        _summaries: Registered summary metrics.
    """
    
    def __init__(
        self,
        namespace: str = "knowledge_agent",
        subsystem: str = "",
    ):
        """
        Initialize the Prometheus metrics collector.
        
        Args:
            namespace: Metric namespace (prefix for all metrics).
            subsystem: Optional subsystem name.
        """
        self.namespace = namespace
        self.subsystem = subsystem
        
        self._counters: Dict[str, Any] = {}
        self._gauges: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}
        self._summaries: Dict[str, Any] = {}
        
        self._prometheus_available = False
        self._initialize_prometheus()
    
    def _initialize_prometheus(self) -> None:
        """Initialize prometheus_client if available."""
        try:
            import prometheus_client
            self._prometheus_available = True
            logger.info("Prometheus metrics initialized")
        except ImportError:
            logger.warning(
                "prometheus_client not installed. "
                "Install with: pip install prometheus-client"
            )
    
    def _get_metric_name(self, name: str) -> str:
        """
        Get the full metric name with namespace.
        
        Args:
            name: Base metric name.
            
        Returns:
            Full metric name with namespace prefix.
        """
        parts = [self.namespace]
        if self.subsystem:
            parts.append(self.subsystem)
        parts.append(name)
        return "_".join(parts)
    
    def _get_label_names(self, labels: Optional[MetricLabels]) -> List[str]:
        """
        Get label names from MetricLabels.
        
        Args:
            labels: MetricLabels instance.
            
        Returns:
            List of label names.
        """
        if labels is None:
            return []
        return list(labels.to_dict().keys())
    
    def _get_label_values(self, labels: Optional[MetricLabels]) -> Dict[str, str]:
        """
        Get label values from MetricLabels.
        
        Args:
            labels: MetricLabels instance.
            
        Returns:
            Dictionary of label values.
        """
        if labels is None:
            return {}
        return labels.to_dict()
    
    def _get_or_create_counter(
        self,
        name: str,
        description: str,
        label_names: List[str],
    ) -> Any:
        """
        Get or create a counter metric.
        
        Args:
            name: Metric name.
            description: Metric description.
            label_names: List of label names.
            
        Returns:
            Counter metric object.
        """
        if not self._prometheus_available:
            return None
        
        full_name = self._get_metric_name(name)
        key = (full_name, tuple(label_names))
        
        if key not in self._counters:
            from prometheus_client import Counter
            
            self._counters[key] = Counter(
                full_name,
                description or f"Counter for {name}",
                label_names,
            )
        
        return self._counters[key]
    
    def _get_or_create_gauge(
        self,
        name: str,
        description: str,
        label_names: List[str],
    ) -> Any:
        """
        Get or create a gauge metric.
        
        Args:
            name: Metric name.
            description: Metric description.
            label_names: List of label names.
            
        Returns:
            Gauge metric object.
        """
        if not self._prometheus_available:
            return None
        
        full_name = self._get_metric_name(name)
        key = (full_name, tuple(label_names))
        
        if key not in self._gauges:
            from prometheus_client import Gauge
            
            self._gauges[key] = Gauge(
                full_name,
                description or f"Gauge for {name}",
                label_names,
            )
        
        return self._gauges[key]
    
    def _get_or_create_histogram(
        self,
        name: str,
        description: str,
        label_names: List[str],
        buckets: Optional[List[float]] = None,
    ) -> Any:
        """
        Get or create a histogram metric.
        
        Args:
            name: Metric name.
            description: Metric description.
            label_names: List of label names.
            buckets: Optional histogram buckets.
            
        Returns:
            Histogram metric object.
        """
        if not self._prometheus_available:
            return None
        
        full_name = self._get_metric_name(name)
        key = (full_name, tuple(label_names))
        
        if key not in self._histograms:
            from prometheus_client import Histogram
            
            self._histograms[key] = Histogram(
                full_name,
                description or f"Histogram for {name}",
                label_names,
                buckets=buckets or LATENCY_BUCKETS,
            )
        
        return self._histograms[key]
    
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
            value: Value to add (must be positive).
            labels: Optional metric labels.
            description: Optional metric description.
        """
        if not self._prometheus_available:
            return
        
        try:
            label_names = self._get_label_names(labels)
            label_values = self._get_label_values(labels)
            
            counter = self._get_or_create_counter(
                name, description or "", label_names
            )
            
            if counter is not None:
                if label_values:
                    counter.labels(**label_values).inc(value)
                else:
                    counter.inc(value)
        except Exception as e:
            logger.debug(f"Failed to increment counter {name}: {e}")
    
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
        if not self._prometheus_available:
            return
        
        try:
            label_names = self._get_label_names(labels)
            label_values = self._get_label_values(labels)
            
            gauge = self._get_or_create_gauge(
                name, description or "", label_names
            )
            
            if gauge is not None:
                if label_values:
                    gauge.labels(**label_values).set(value)
                else:
                    gauge.set(value)
        except Exception as e:
            logger.debug(f"Failed to set gauge {name}: {e}")
    
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
        if not self._prometheus_available:
            return
        
        try:
            label_names = self._get_label_names(labels)
            label_values = self._get_label_values(labels)
            
            histogram = self._get_or_create_histogram(
                name, description or "", label_names, buckets
            )
            
            if histogram is not None:
                if label_values:
                    histogram.labels(**label_values).observe(value)
                else:
                    histogram.observe(value)
        except Exception as e:
            logger.debug(f"Failed to observe histogram {name}: {e}")
    
    @contextmanager
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
            name: Metric name (should end with _seconds).
            labels: Optional metric labels.
            description: Optional metric description.
            
        Yields:
            None
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.histogram(name, duration, labels, description)
    
    def generate_metrics(self) -> str:
        """
        Generate metrics in Prometheus text format.
        
        Returns:
            Metrics in Prometheus exposition format.
        """
        if not self._prometheus_available:
            return ""
        
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            return generate_latest().decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return ""
    
    def get_content_type(self) -> str:
        """
        Get the content type for Prometheus metrics.
        
        Returns:
            Prometheus content type string.
        """
        if not self._prometheus_available:
            return "text/plain"
        
        try:
            from prometheus_client import CONTENT_TYPE_LATEST
            return CONTENT_TYPE_LATEST
        except Exception:
            return "text/plain; version=0.0.4; charset=utf-8"


# =============================================================================
# Pre-defined Metric Names
# =============================================================================

class MetricNames:
    """
    Standard metric names for the Knowledge Agent.
    
    These follow Prometheus naming conventions and include the appropriate
    suffixes for the metric type.
    """
    
    # Request metrics
    REQUESTS_TOTAL = "requests_total"
    REQUEST_DURATION_SECONDS = "request_duration_seconds"
    REQUEST_SIZE_BYTES = "request_size_bytes"
    RESPONSE_SIZE_BYTES = "response_size_bytes"
    
    # LLM metrics
    LLM_CALLS_TOTAL = "llm_calls_total"
    LLM_CALL_DURATION_SECONDS = "llm_call_duration_seconds"
    LLM_INPUT_TOKENS_TOTAL = "llm_input_tokens_total"
    LLM_OUTPUT_TOKENS_TOTAL = "llm_output_tokens_total"
    LLM_ERRORS_TOTAL = "llm_errors_total"
    
    # Retrieval metrics
    RETRIEVAL_CALLS_TOTAL = "retrieval_calls_total"
    RETRIEVAL_DURATION_SECONDS = "retrieval_duration_seconds"
    RETRIEVAL_DOCUMENTS_RETURNED = "retrieval_documents_returned"
    RETRIEVAL_SCORE_HISTOGRAM = "retrieval_score"
    
    # Embedding metrics
    EMBEDDING_CALLS_TOTAL = "embedding_calls_total"
    EMBEDDING_DURATION_SECONDS = "embedding_duration_seconds"
    EMBEDDING_TEXTS_TOTAL = "embedding_texts_total"
    
    # Reranking metrics
    RERANKING_CALLS_TOTAL = "reranking_calls_total"
    RERANKING_DURATION_SECONDS = "reranking_duration_seconds"
    
    # Cache metrics
    CACHE_HITS_TOTAL = "cache_hits_total"
    CACHE_MISSES_TOTAL = "cache_misses_total"
    CACHE_SIZE_BYTES = "cache_size_bytes"
    
    # System metrics
    ACTIVE_REQUESTS = "active_requests"
    QUEUE_SIZE = "queue_size"
    MEMORY_USAGE_BYTES = "memory_usage_bytes"
    
    # Error metrics
    ERRORS_TOTAL = "errors_total"
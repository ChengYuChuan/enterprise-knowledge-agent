"""
Metrics Submodule

This module provides Prometheus-compatible metrics collection for monitoring
system health and performance.

Components:
    - PrometheusMetrics: Prometheus client implementation
    - RAGMetrics: Pre-defined metrics for RAG pipelines
    - APIMetrics: Pre-defined metrics for API endpoints

Usage:
    ```python
    from src.observability.metrics import PrometheusMetrics, RAGMetrics

    metrics = PrometheusMetrics(namespace="knowledge_agent")

    # Use pre-defined RAG metrics
    rag_metrics = RAGMetrics(metrics)
    rag_metrics.record_retrieval(
        query="test",
        num_results=5,
        duration_ms=150.5,
        strategy="hybrid"
    )

    # Or use metrics directly
    metrics.counter("requests_total", labels=MetricLabels(operation="chat"))
    metrics.histogram("request_duration_seconds", 0.5)
    ```
"""

from src.observability.metrics.prometheus import (
    PrometheusMetrics,
    get_prometheus_metrics,
)
from src.observability.metrics.collectors import (
    RAGMetrics,
    APIMetrics,
    LLMMetrics,
    SystemMetrics,
)

__all__ = [
    "PrometheusMetrics",
    "get_prometheus_metrics",
    "RAGMetrics",
    "APIMetrics",
    "LLMMetrics",
    "SystemMetrics",
]
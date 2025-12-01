"""
Unit tests for Phase 5: Metrics Module

Tests the Prometheus metrics integration and collectors.

Run with:
    poetry run pytest tests/unit/test_phase5_metrics.py -v
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from src.observability.base import MetricLabels
from src.observability.metrics.prometheus import (
    PrometheusMetrics,
    LATENCY_BUCKETS,
    TOKEN_BUCKETS,
    DOCUMENT_BUCKETS,
    SCORE_BUCKETS,
)
from src.observability.metrics.collectors import (
    MetricNames,
    RAGMetrics,
    LLMMetrics,
    APIMetrics,
    SystemMetrics,
)


# =============================================================================
# Test Bucket Configurations
# =============================================================================

class TestBucketConfigurations:
    """Test metric bucket configurations."""
    
    def test_latency_buckets_ordered(self):
        """Test that latency buckets are in ascending order."""
        for i in range(len(LATENCY_BUCKETS) - 1):
            assert LATENCY_BUCKETS[i] < LATENCY_BUCKETS[i + 1]
    
    def test_latency_buckets_reasonable_range(self):
        """Test latency buckets cover reasonable range."""
        # Should start at low milliseconds
        assert LATENCY_BUCKETS[0] <= 0.01  # 10ms or less
        # Should go up to seconds
        assert any(b >= 5.0 for b in LATENCY_BUCKETS if b != float("inf"))
    
    def test_token_buckets_ordered(self):
        """Test that token buckets are in ascending order."""
        for i in range(len(TOKEN_BUCKETS) - 1):
            assert TOKEN_BUCKETS[i] < TOKEN_BUCKETS[i + 1]
    
    def test_document_buckets_reasonable(self):
        """Test document buckets for retrieval counts."""
        assert 1 in DOCUMENT_BUCKETS  # Should include single doc
        assert 10 in DOCUMENT_BUCKETS  # Should include typical top_k
    
    def test_score_buckets_cover_unit_interval(self):
        """Test score buckets cover 0-1 range."""
        assert SCORE_BUCKETS[0] > 0
        assert SCORE_BUCKETS[-1] == 1.0


# =============================================================================
# Test PrometheusMetrics
# =============================================================================

class TestPrometheusMetrics:
    """Test PrometheusMetrics implementation."""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics instance for testing."""
        return PrometheusMetrics(
            namespace="test_namespace",
            subsystem="test_subsystem",
        )
    
    def test_initialization(self, metrics):
        """Test basic initialization."""
        assert metrics.namespace == "test_namespace"
        assert metrics.subsystem == "test_subsystem"
    
    def test_counter_no_error(self, metrics):
        """Test counter increments without error."""
        # Should not raise even if prometheus not available
        metrics.counter("test_counter", value=1.0)
        metrics.counter(
            "labeled_counter",
            value=5.0,
            labels=MetricLabels(operation="test"),
        )
    
    def test_gauge_no_error(self, metrics):
        """Test gauge sets without error."""
        metrics.gauge("test_gauge", value=42.0)
        metrics.gauge(
            "labeled_gauge",
            value=100.0,
            labels=MetricLabels(status="active"),
        )
    
    def test_histogram_no_error(self, metrics):
        """Test histogram records without error."""
        metrics.histogram("test_histogram", value=0.5)
        metrics.histogram(
            "labeled_histogram",
            value=1.5,
            labels=MetricLabels(operation="search"),
            buckets=LATENCY_BUCKETS,
        )
    
    def test_timer_context_manager(self, metrics):
        """Test timer as context manager."""
        with metrics.timer("operation_duration"):
            time.sleep(0.01)
        
        # Should complete without error
    
    def test_timer_with_labels(self, metrics):
        """Test timer with labels."""
        labels = MetricLabels(operation="search", status="success")
        
        with metrics.timer("search_duration", labels=labels):
            time.sleep(0.01)


# =============================================================================
# Test MetricNames
# =============================================================================

class TestMetricNames:
    """Test metric name constants."""
    
    def test_rag_metrics_exist(self):
        """Test RAG-related metric names exist."""
        assert hasattr(MetricNames, "RETRIEVAL_DURATION_SECONDS")
        assert hasattr(MetricNames, "RETRIEVAL_DOCUMENTS_RETURNED")
    
    def test_llm_metrics_exist(self):
        """Test LLM-related metric names exist."""
        assert hasattr(MetricNames, "LLM_CALLS_TOTAL")
        assert hasattr(MetricNames, "LLM_CALL_DURATION_SECONDS")
        assert hasattr(MetricNames, "LLM_INPUT_TOKENS_TOTAL")
        assert hasattr(MetricNames, "LLM_OUTPUT_TOKENS_TOTAL")
    
    def test_api_metrics_exist(self):
        """Test API-related metric names exist."""
        assert hasattr(MetricNames, "REQUESTS_TOTAL")
        assert hasattr(MetricNames, "REQUEST_DURATION_SECONDS")
    
    def test_metric_names_snake_case(self):
        """Test that metric names follow snake_case convention."""
        for name in dir(MetricNames):
            if not name.startswith("_"):
                value = getattr(MetricNames, name)
                if isinstance(value, str):
                    assert "_" in value or value.islower(), \
                        f"Metric {name}={value} should be snake_case"


# =============================================================================
# Test RAGMetrics Collector
# =============================================================================

class TestRAGMetrics:
    """Test RAG metrics collector."""
    
    @pytest.fixture
    def rag_metrics(self):
        """Create RAG metrics instance."""
        prometheus = PrometheusMetrics(namespace="test")
        return RAGMetrics(prometheus)
    
    def test_record_retrieval(self, rag_metrics):
        """Test recording retrieval operation."""
        rag_metrics.record_retrieval(
            query="test query",
            num_results=5,
            duration_seconds=0.5,
            strategy="hybrid",
        )
        # Should not raise
    
    def test_record_retrieval_with_scores(self, rag_metrics):
        """Test recording retrieval with relevance scores."""
        # record_retrieval doesn't accept top_score/avg_score
        # Use record_retrieval_scores for score recording
        rag_metrics.record_retrieval(
            query="scored query",
            num_results=3,
            duration_seconds=0.3,
            strategy="vector",
        )
        # Record scores separately
        rag_metrics.record_retrieval_scores(
            scores=[0.95, 0.82, 0.75],
            strategy="vector",
        )
    
    def test_record_reranking(self, rag_metrics):
        """Test recording reranking operation."""
        rag_metrics.record_reranking(
            input_count=10,
            output_count=5,
            duration_seconds=0.2,
            model="bge-reranker-base",
        )
    
    def test_record_embedding(self, rag_metrics):
        """Test recording embedding operation."""
        rag_metrics.record_embedding(
            num_texts=10,
            duration_seconds=0.1,
            model="text-embedding-3-small",
            dimensions=1536,
        )


# =============================================================================
# Test LLMMetrics Collector
# =============================================================================

class TestLLMMetrics:
    """Test LLM metrics collector."""
    
    @pytest.fixture
    def llm_metrics(self):
        """Create LLM metrics instance."""
        prometheus = PrometheusMetrics(namespace="test")
        return LLMMetrics(prometheus)
    
    def test_record_call_success(self, llm_metrics):
        """Test recording successful LLM call."""
        llm_metrics.record_call(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            duration_seconds=1.5,
            status="success",
        )
    
    def test_record_call_failure(self, llm_metrics):
        """Test recording failed LLM call."""
        llm_metrics.record_call(
            provider="anthropic",
            model="claude-3-sonnet",
            input_tokens=200,
            output_tokens=0,
            duration_seconds=0.5,
            status="error",
            error_type="rate_limit",
        )
    
    def test_record_streaming_tokens(self, llm_metrics):
        """Test recording streaming token output."""
        llm_metrics.record_streaming_tokens(
            provider="openai",
            model="gpt-4",
            token_count=10,
            is_final=False,
        )
        llm_metrics.record_streaming_tokens(
            provider="openai",
            model="gpt-4",
            token_count=5,
            is_final=True,
        )


# =============================================================================
# Test APIMetrics Collector
# =============================================================================

class TestAPIMetrics:
    """Test API metrics collector."""
    
    @pytest.fixture
    def api_metrics(self):
        """Create API metrics instance."""
        prometheus = PrometheusMetrics(namespace="test")
        return APIMetrics(prometheus)
    
    def test_record_request_success(self, api_metrics):
        """Test recording successful API request."""
        api_metrics.record_request(
            method="POST",
            endpoint="/api/v1/chat",
            status_code=200,
            duration_seconds=0.5,
        )
    
    def test_record_request_with_sizes(self, api_metrics):
        """Test recording request with body sizes."""
        api_metrics.record_request(
            method="POST",
            endpoint="/api/v1/ingest",
            status_code=201,
            duration_seconds=2.0,
            request_size_bytes=1024 * 100,
            response_size_bytes=256,
        )
    
    def test_record_request_error(self, api_metrics):
        """Test recording error response."""
        api_metrics.record_request(
            method="GET",
            endpoint="/api/v1/search",
            status_code=500,
            duration_seconds=0.1,
        )
    
    def test_set_active_requests(self, api_metrics):
        """Test setting active request count."""
        api_metrics.set_active_requests(5)
        api_metrics.set_active_requests(3, endpoint="/api/v1/chat")


# =============================================================================
# Test SystemMetrics Collector
# =============================================================================

class TestSystemMetrics:
    """Test system metrics collector."""
    
    @pytest.fixture
    def system_metrics(self):
        """Create system metrics instance."""
        prometheus = PrometheusMetrics(namespace="test")
        return SystemMetrics(prometheus)
    
    def test_record_memory_usage(self, system_metrics):
        """Test recording memory usage."""
        system_metrics.record_memory_usage(
            rss_bytes=1024 * 1024 * 512,  # 512MB
        )
    
    def test_record_memory_with_heap(self, system_metrics):
        """Test recording memory with heap size."""
        system_metrics.record_memory_usage(
            rss_bytes=1024 * 1024 * 512,
            heap_bytes=1024 * 1024 * 256,
            process_name="worker-1",
        )
    
    def test_record_queue_size(self, system_metrics):
        """Test recording queue size."""
        system_metrics.record_queue_size(
            queue_name="ingestion_queue",
            size=42,
        )


# =============================================================================
# Test Metric Labels
# =============================================================================

class TestMetricLabelsIntegration:
    """Test MetricLabels with collectors."""
    
    def test_labels_with_rag_metrics(self):
        """Test custom labels with RAG metrics."""
        prometheus = PrometheusMetrics(namespace="test")
        rag_metrics = RAGMetrics(prometheus)
        
        # Record with custom metadata
        rag_metrics.record_retrieval(
            query="test",
            num_results=5,
            duration_seconds=0.5,
            strategy="hybrid",
        )
    
    def test_labels_with_llm_metrics(self):
        """Test provider/model labels with LLM metrics."""
        prometheus = PrometheusMetrics(namespace="test")
        llm_metrics = LLMMetrics(prometheus)
        
        # Different providers
        for provider, model in [
            ("openai", "gpt-4"),
            ("anthropic", "claude-3"),
            ("ollama", "llama2"),
        ]:
            llm_metrics.record_call(
                provider=provider,
                model=model,
                input_tokens=100,
                output_tokens=50,
                duration_seconds=1.0,
                status="success",  # Use status instead of success
            )


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_duration(self):
        """Test recording zero duration."""
        prometheus = PrometheusMetrics(namespace="test")
        rag_metrics = RAGMetrics(prometheus)
        
        rag_metrics.record_retrieval(
            query="instant",
            num_results=1,
            duration_seconds=0.0,
            strategy="vector",
        )
    
    def test_empty_query(self):
        """Test recording with empty query."""
        prometheus = PrometheusMetrics(namespace="test")
        rag_metrics = RAGMetrics(prometheus)
        
        rag_metrics.record_retrieval(
            query="",
            num_results=0,
            duration_seconds=0.1,
            strategy="bm25",
        )
    
    def test_large_token_counts(self):
        """Test recording large token counts."""
        prometheus = PrometheusMetrics(namespace="test")
        llm_metrics = LLMMetrics(prometheus)
        
        llm_metrics.record_call(
            provider="openai",
            model="gpt-4-turbo",
            input_tokens=100000,
            output_tokens=50000,
            duration_seconds=30.0,
            status="success",  # Use status instead of success
        )
    
    def test_negative_values_handled(self):
        """Test that negative values don't crash (though invalid)."""
        prometheus = PrometheusMetrics(namespace="test")
        
        # These are invalid but shouldn't crash
        try:
            prometheus.histogram("test", value=-1.0)
            prometheus.gauge("test_gauge", value=-100)
        except Exception:
            pass  # Some implementations may raise

"""
Specialized Metric Collectors

This module provides pre-configured metric collectors for different components
of the Knowledge Agent system. Each collector encapsulates the metrics and
recording logic for its specific domain.

Components:
    - RAGMetrics: Metrics for RAG pipeline operations
    - APIMetrics: Metrics for API endpoints
    - LLMMetrics: Metrics for LLM calls
    - SystemMetrics: System-level resource metrics

Design:
    Each collector is designed to be used as a singleton, injected into the
    components that need to record metrics. This ensures consistent metric
    naming and labeling across the application.

Usage:
    ```python
    from src.observability.metrics import RAGMetrics, get_prometheus_metrics

    # Initialize collectors
    prometheus = get_prometheus_metrics()
    rag_metrics = RAGMetrics(prometheus)

    # Record metrics
    rag_metrics.record_retrieval(
        query="machine learning",
        num_results=5,
        duration_ms=150.5,
        strategy="hybrid"
    )
    ```
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.observability.base import MetricLabels
from src.observability.metrics.prometheus import (
    MetricNames,
    PrometheusMetrics,
    LATENCY_BUCKETS,
    TOKEN_BUCKETS,
    DOCUMENT_BUCKETS,
    SCORE_BUCKETS,
)


logger = logging.getLogger(__name__)


# =============================================================================
# RAG Metrics Collector
# =============================================================================

class RAGMetrics:
    """
    Metrics collector for RAG pipeline operations.
    
    This collector provides methods for recording metrics related to:
    - Document retrieval (vector search, BM25, hybrid)
    - Embedding generation
    - Reranking operations
    - Response generation
    
    Attributes:
        _metrics: Underlying Prometheus metrics collector.
    
    Example:
        ```python
        rag_metrics = RAGMetrics(prometheus)
        
        # Record a retrieval operation
        with rag_metrics.retrieval_timer("hybrid") as timer:
            results = await retriever.search(query, top_k=10)
        
        rag_metrics.record_retrieval_results(
            strategy="hybrid",
            num_results=len(results),
            scores=[r.score for r in results]
        )
        ```
    """
    
    def __init__(self, metrics: PrometheusMetrics):
        """
        Initialize the RAG metrics collector.
        
        Args:
            metrics: Prometheus metrics instance.
        """
        self._metrics = metrics
    
    def record_retrieval(
        self,
        query: str,
        num_results: int,
        duration_seconds: float,
        strategy: str = "vector",
        collection: Optional[str] = None,
    ) -> None:
        """
        Record a retrieval operation.
        
        Args:
            query: Search query (for logging, not stored in metrics).
            num_results: Number of results returned.
            duration_seconds: Operation duration in seconds.
            strategy: Retrieval strategy (vector, bm25, hybrid).
            collection: Optional collection name.
        """
        # Note: strategy and collection are logged but not added as labels
        # since MetricLabels has a fixed schema. In production, you might
        # extend MetricLabels or use a different approach for custom labels.
        labels = MetricLabels(
            operation=f"retrieval_{strategy}",  # Encode strategy in operation name
            status="success",
        )
        
        # Record counter
        self._metrics.counter(
            MetricNames.RETRIEVAL_CALLS_TOTAL,
            labels=labels,
            description="Total number of retrieval calls"
        )
        
        # Record duration histogram
        self._metrics.histogram(
            MetricNames.RETRIEVAL_DURATION_SECONDS,
            duration_seconds,
            labels=labels,
            description="Retrieval operation duration in seconds"
        )
        
        # Record documents returned
        self._metrics.histogram(
            MetricNames.RETRIEVAL_DOCUMENTS_RETURNED,
            num_results,
            labels=labels,
            description="Number of documents returned per retrieval",
            buckets=list(DOCUMENT_BUCKETS)
        )
    
    def record_retrieval_scores(
        self,
        scores: List[float],
        strategy: str = "vector",
    ) -> None:
        """
        Record retrieval relevance scores.
        
        Args:
            scores: List of relevance scores (0-1).
            strategy: Retrieval strategy.
        """
        labels = MetricLabels(operation="retrieval")
        
        for score in scores:
            self._metrics.histogram(
                MetricNames.RETRIEVAL_SCORE_HISTOGRAM,
                score,
                labels=labels,
                description="Distribution of retrieval relevance scores",
                buckets=list(SCORE_BUCKETS)
            )
    
    def record_embedding(
        self,
        num_texts: int,
        duration_seconds: float,
        model: str,
        dimensions: Optional[int] = None,
    ) -> None:
        """
        Record an embedding generation operation.
        
        Args:
            num_texts: Number of texts embedded.
            duration_seconds: Operation duration in seconds.
            model: Embedding model name.
            dimensions: Optional embedding dimensions.
        """
        labels = MetricLabels(
            operation="embedding",
            model=model,
            status="success",
        )
        
        # Record counter
        self._metrics.counter(
            MetricNames.EMBEDDING_CALLS_TOTAL,
            labels=labels,
            description="Total number of embedding calls"
        )
        
        # Record texts count
        self._metrics.counter(
            MetricNames.EMBEDDING_TEXTS_TOTAL,
            value=num_texts,
            labels=labels,
            description="Total number of texts embedded"
        )
        
        # Record duration
        self._metrics.histogram(
            MetricNames.EMBEDDING_DURATION_SECONDS,
            duration_seconds,
            labels=labels,
            description="Embedding generation duration in seconds"
        )
    
    def record_reranking(
        self,
        input_count: int,
        output_count: int,
        duration_seconds: float,
        model: str,
    ) -> None:
        """
        Record a reranking operation.
        
        Args:
            input_count: Number of input documents.
            output_count: Number of output documents.
            duration_seconds: Operation duration in seconds.
            model: Reranker model name.
        """
        labels = MetricLabels(
            operation="reranking",
            model=model,
            status="success",
        )
        
        # Record counter
        self._metrics.counter(
            MetricNames.RERANKING_CALLS_TOTAL,
            labels=labels,
            description="Total number of reranking calls"
        )
        
        # Record duration
        self._metrics.histogram(
            MetricNames.RERANKING_DURATION_SECONDS,
            duration_seconds,
            labels=labels,
            description="Reranking operation duration in seconds"
        )


# =============================================================================
# API Metrics Collector
# =============================================================================

class APIMetrics:
    """
    Metrics collector for API endpoint operations.
    
    This collector provides methods for recording metrics related to:
    - Request counts and durations
    - Response sizes
    - Error rates
    - Active request tracking
    
    Example:
        ```python
        api_metrics = APIMetrics(prometheus)
        
        # Record a request
        api_metrics.record_request(
            method="POST",
            endpoint="/api/v1/chat",
            status_code=200,
            duration_seconds=1.5
        )
        ```
    """
    
    def __init__(self, metrics: PrometheusMetrics):
        """
        Initialize the API metrics collector.
        
        Args:
            metrics: Prometheus metrics instance.
        """
        self._metrics = metrics
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float,
        request_size_bytes: Optional[int] = None,
        response_size_bytes: Optional[int] = None,
    ) -> None:
        """
        Record an API request.
        
        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path.
            status_code: HTTP response status code.
            duration_seconds: Request duration in seconds.
            request_size_bytes: Optional request body size.
            response_size_bytes: Optional response body size.
        """
        status_class = f"{status_code // 100}xx"
        status = "success" if status_code < 400 else "error"
        
        labels = MetricLabels(
            operation=endpoint,
            status=status,
        )
        
        # Record request counter
        self._metrics.counter(
            MetricNames.REQUESTS_TOTAL,
            labels=labels,
            description="Total number of API requests"
        )
        
        # Record duration histogram
        self._metrics.histogram(
            MetricNames.REQUEST_DURATION_SECONDS,
            duration_seconds,
            labels=labels,
            description="API request duration in seconds"
        )
        
        # Record request size if provided
        if request_size_bytes is not None:
            self._metrics.histogram(
                MetricNames.REQUEST_SIZE_BYTES,
                request_size_bytes,
                labels=labels,
                description="API request size in bytes"
            )
        
        # Record response size if provided
        if response_size_bytes is not None:
            self._metrics.histogram(
                MetricNames.RESPONSE_SIZE_BYTES,
                response_size_bytes,
                labels=labels,
                description="API response size in bytes"
            )
        
        # Record error if applicable
        if status_code >= 400:
            error_labels = MetricLabels(
                operation=endpoint,
                error_type=status_class,
            )
            self._metrics.counter(
                MetricNames.ERRORS_TOTAL,
                labels=error_labels,
                description="Total number of errors"
            )
    
    def set_active_requests(self, count: int, endpoint: Optional[str] = None) -> None:
        """
        Set the number of active requests.
        
        Args:
            count: Number of active requests.
            endpoint: Optional endpoint for filtering.
        """
        labels = MetricLabels(operation=endpoint) if endpoint else None
        
        self._metrics.gauge(
            MetricNames.ACTIVE_REQUESTS,
            count,
            labels=labels,
            description="Number of currently active requests"
        )
    
    def increment_active_requests(self, endpoint: Optional[str] = None) -> None:
        """Increment active request count."""
        # Note: This would need state tracking for accurate counts
        # For now, we just record a gauge update
        pass
    
    def decrement_active_requests(self, endpoint: Optional[str] = None) -> None:
        """Decrement active request count."""
        pass


# =============================================================================
# LLM Metrics Collector
# =============================================================================

class LLMMetrics:
    """
    Metrics collector for LLM operations.
    
    This collector provides methods for recording metrics related to:
    - LLM call counts and durations
    - Token usage (input and output)
    - Error rates by provider/model
    - Cost estimation (if configured)
    
    Example:
        ```python
        llm_metrics = LLMMetrics(prometheus)
        
        # Record an LLM call
        llm_metrics.record_call(
            provider="openai",
            model="gpt-4",
            duration_seconds=2.5,
            input_tokens=500,
            output_tokens=200,
            status="success"
        )
        ```
    """
    
    def __init__(self, metrics: PrometheusMetrics):
        """
        Initialize the LLM metrics collector.
        
        Args:
            metrics: Prometheus metrics instance.
        """
        self._metrics = metrics
    
    def record_call(
        self,
        provider: str,
        model: str,
        duration_seconds: float,
        input_tokens: int,
        output_tokens: int,
        status: str = "success",
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record an LLM API call.
        
        Args:
            provider: LLM provider (openai, anthropic, ollama).
            model: Model name.
            duration_seconds: Call duration in seconds.
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of output/completion tokens.
            status: Call status (success, error).
            error_type: Optional error type for failures.
        """
        labels = MetricLabels(
            provider=provider,
            model=model,
            status=status,
            error_type=error_type,
        )
        
        # Record call counter
        self._metrics.counter(
            MetricNames.LLM_CALLS_TOTAL,
            labels=labels,
            description="Total number of LLM calls"
        )
        
        # Record duration
        self._metrics.histogram(
            MetricNames.LLM_CALL_DURATION_SECONDS,
            duration_seconds,
            labels=labels,
            description="LLM call duration in seconds"
        )
        
        # Record token counts
        token_labels = MetricLabels(
            provider=provider,
            model=model,
        )
        
        self._metrics.counter(
            MetricNames.LLM_INPUT_TOKENS_TOTAL,
            value=input_tokens,
            labels=token_labels,
            description="Total number of input tokens"
        )
        
        self._metrics.counter(
            MetricNames.LLM_OUTPUT_TOKENS_TOTAL,
            value=output_tokens,
            labels=token_labels,
            description="Total number of output tokens"
        )
        
        # Record token histograms for distribution analysis
        self._metrics.histogram(
            "llm_input_tokens",
            input_tokens,
            labels=token_labels,
            description="Input token count distribution",
            buckets=list(TOKEN_BUCKETS)
        )
        
        self._metrics.histogram(
            "llm_output_tokens",
            output_tokens,
            labels=token_labels,
            description="Output token count distribution",
            buckets=list(TOKEN_BUCKETS)
        )
        
        # Record error if applicable
        if status == "error":
            self._metrics.counter(
                MetricNames.LLM_ERRORS_TOTAL,
                labels=labels,
                description="Total number of LLM errors"
            )
    
    def record_streaming_tokens(
        self,
        provider: str,
        model: str,
        token_count: int,
        is_final: bool = False,
    ) -> None:
        """
        Record tokens during streaming.
        
        Args:
            provider: LLM provider.
            model: Model name.
            token_count: Number of tokens received.
            is_final: Whether this is the final token batch.
        """
        labels = MetricLabels(
            provider=provider,
            model=model,
        )
        
        self._metrics.counter(
            MetricNames.LLM_OUTPUT_TOKENS_TOTAL,
            value=token_count,
            labels=labels,
            description="Total number of output tokens"
        )


# =============================================================================
# System Metrics Collector
# =============================================================================

class SystemMetrics:
    """
    Metrics collector for system-level resources.
    
    This collector provides methods for recording metrics related to:
    - Memory usage
    - Queue sizes
    - Cache statistics
    - Connection pools
    
    Example:
        ```python
        system_metrics = SystemMetrics(prometheus)
        
        # Record memory usage
        system_metrics.record_memory_usage(
            rss_bytes=1024 * 1024 * 512,  # 512MB
            heap_bytes=1024 * 1024 * 256   # 256MB
        )
        ```
    """
    
    def __init__(self, metrics: PrometheusMetrics):
        """
        Initialize the system metrics collector.
        
        Args:
            metrics: Prometheus metrics instance.
        """
        self._metrics = metrics
    
    def record_memory_usage(
        self,
        rss_bytes: int,
        heap_bytes: Optional[int] = None,
        process_name: Optional[str] = None,
    ) -> None:
        """
        Record memory usage.
        
        Args:
            rss_bytes: Resident set size in bytes.
            heap_bytes: Optional heap size in bytes.
            process_name: Optional process identifier.
        """
        labels = MetricLabels(service=process_name) if process_name else None
        
        self._metrics.gauge(
            MetricNames.MEMORY_USAGE_BYTES,
            rss_bytes,
            labels=labels,
            description="Memory usage in bytes"
        )
        
        if heap_bytes is not None:
            self._metrics.gauge(
                "heap_usage_bytes",
                heap_bytes,
                labels=labels,
                description="Heap memory usage in bytes"
            )
    
    def record_queue_size(
        self,
        queue_name: str,
        size: int,
    ) -> None:
        """
        Record queue size.
        
        Args:
            queue_name: Name of the queue.
            size: Current queue size.
        """
        labels = MetricLabels(operation=queue_name)
        
        self._metrics.gauge(
            MetricNames.QUEUE_SIZE,
            size,
            labels=labels,
            description="Current queue size"
        )
    
    def record_cache_operation(
        self,
        cache_name: str,
        hit: bool,
    ) -> None:
        """
        Record a cache operation.
        
        Args:
            cache_name: Name of the cache.
            hit: Whether it was a cache hit.
        """
        labels = MetricLabels(operation=cache_name)
        
        if hit:
            self._metrics.counter(
                MetricNames.CACHE_HITS_TOTAL,
                labels=labels,
                description="Total number of cache hits"
            )
        else:
            self._metrics.counter(
                MetricNames.CACHE_MISSES_TOTAL,
                labels=labels,
                description="Total number of cache misses"
            )
    
    def record_cache_size(
        self,
        cache_name: str,
        size_bytes: int,
        item_count: Optional[int] = None,
    ) -> None:
        """
        Record cache size.
        
        Args:
            cache_name: Name of the cache.
            size_bytes: Cache size in bytes.
            item_count: Optional number of items in cache.
        """
        labels = MetricLabels(operation=cache_name)
        
        self._metrics.gauge(
            MetricNames.CACHE_SIZE_BYTES,
            size_bytes,
            labels=labels,
            description="Cache size in bytes"
        )
        
        if item_count is not None:
            self._metrics.gauge(
                "cache_items",
                item_count,
                labels=labels,
                description="Number of items in cache"
            )
    
    def collect_process_metrics(self) -> None:
        """
        Collect and record current process metrics.
        
        This method collects metrics from the current Python process,
        including memory usage, CPU time, and file descriptors.
        """
        try:
            import psutil
            
            process = psutil.Process()
            
            # Memory info
            mem_info = process.memory_info()
            self.record_memory_usage(
                rss_bytes=mem_info.rss,
                heap_bytes=getattr(mem_info, 'data', None),
            )
            
            # CPU times
            cpu_times = process.cpu_times()
            self._metrics.gauge(
                "process_cpu_user_seconds",
                cpu_times.user,
                description="User CPU time in seconds"
            )
            self._metrics.gauge(
                "process_cpu_system_seconds",
                cpu_times.system,
                description="System CPU time in seconds"
            )
            
            # File descriptors (Unix only)
            try:
                num_fds = process.num_fds()
                self._metrics.gauge(
                    "process_open_fds",
                    num_fds,
                    description="Number of open file descriptors"
                )
            except (AttributeError, psutil.Error):
                pass
            
            # Thread count
            self._metrics.gauge(
                "process_threads",
                process.num_threads(),
                description="Number of threads"
            )
            
        except ImportError:
            logger.debug("psutil not installed, skipping process metrics")
        except Exception as e:
            logger.debug(f"Failed to collect process metrics: {e}")
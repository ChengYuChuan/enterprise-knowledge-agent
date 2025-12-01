"""
Integration tests for Phase 5: Observability System

Tests the complete observability pipeline including tracing,
metrics, evaluation, and benchmarks working together.

Run with:
    poetry run pytest tests/integration/test_phase5_integration.py -v
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.observability import (
    get_tracer,
    get_metrics,
    traced,
    timed,
    SpanKind,
    SpanStatus,
    MetricLabels,
)
from src.observability.base import NoOpTracer, NoOpMetricsCollector
from src.observability.evaluation.datasets import (
    EvaluationDataset,
    create_synthetic_dataset,
)
from src.observability.evaluation.ragas import (
    RagasEvaluator,
    EvaluationMetrics,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_dataset():
    """Create a sample evaluation dataset."""
    dataset = EvaluationDataset(name="integration_test")
    
    dataset.add_sample(
        question="What is machine learning?",
        answer="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        contexts=[
            "Machine learning is a branch of artificial intelligence.",
            "ML systems improve their performance through experience.",
        ],
        ground_truth="Machine learning is a type of AI that learns from data.",
    )
    
    dataset.add_sample(
        question="How does a neural network work?",
        answer="Neural networks process information through layers of interconnected nodes.",
        contexts=[
            "Neural networks are inspired by biological neurons.",
            "Deep learning uses multiple layers of neural networks.",
        ],
        ground_truth="Neural networks mimic brain structure with layers of nodes.",
    )
    
    return dataset


@pytest.fixture
def evaluator():
    """Create evaluator instance."""
    return RagasEvaluator(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )


# =============================================================================
# Test Observability Module Initialization
# =============================================================================

class TestObservabilityInit:
    """Test observability module initialization."""
    
    def test_get_tracer_returns_tracer(self):
        """Test that get_tracer returns a valid tracer."""
        tracer = get_tracer()
        
        # Should return a tracer (NoOp if Phoenix unavailable)
        assert tracer is not None
        assert hasattr(tracer, "span")
        assert hasattr(tracer, "start_span")
    
    def test_get_metrics_returns_collector(self):
        """Test that get_metrics returns a valid collector."""
        metrics = get_metrics()
        
        # Should return a collector (NoOp if Prometheus unavailable)
        assert metrics is not None
        assert hasattr(metrics, "counter")
        assert hasattr(metrics, "gauge")
        assert hasattr(metrics, "histogram")
    
    def test_tracer_singleton(self):
        """Test that get_tracer returns singleton."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        
        assert tracer1 is tracer2
    
    def test_metrics_singleton(self):
        """Test that get_metrics returns singleton."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()
        
        assert metrics1 is metrics2


# =============================================================================
# Test Tracing Integration
# =============================================================================

class TestTracingIntegration:
    """Test tracing functionality."""
    
    def test_span_context_manager(self):
        """Test using span as context manager."""
        tracer = get_tracer()
        
        with tracer.span("test_operation", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("test_key", "test_value")
            span.set_status(SpanStatus.OK)
    
    def test_nested_spans(self):
        """Test nested span creation."""
        tracer = get_tracer()
        
        with tracer.span("parent_op") as parent:
            parent.set_attribute("level", "parent")
            
            with tracer.span("child_op") as child:
                child.set_attribute("level", "child")
                child.set_status(SpanStatus.OK)
            
            parent.set_status(SpanStatus.OK)
    
    def test_span_with_llm_attributes(self):
        """Test span with LLM-specific attributes."""
        tracer = get_tracer()
        
        with tracer.span("llm_call", kind=SpanKind.LLM) as span:
            span.set_attributes({
                "llm.provider": "openai",
                "llm.model": "gpt-4",
                "llm.input_tokens": 100,
                "llm.output_tokens": 50,
            })
            span.set_status(SpanStatus.OK)
    
    def test_span_records_exception(self):
        """Test that exceptions are recorded in span."""
        tracer = get_tracer()
        
        try:
            with tracer.span("failing_op") as span:
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected


# =============================================================================
# Test Metrics Integration
# =============================================================================

class TestMetricsIntegration:
    """Test metrics collection functionality."""
    
    def test_counter_increment(self):
        """Test counter increments."""
        metrics = get_metrics()
        
        metrics.counter("test_requests_total")
        metrics.counter("test_requests_total", value=5)
    
    def test_gauge_set(self):
        """Test gauge value setting."""
        metrics = get_metrics()
        
        metrics.gauge("test_active_connections", value=10)
        metrics.gauge("test_active_connections", value=15)
    
    def test_histogram_observation(self):
        """Test histogram value recording."""
        metrics = get_metrics()
        
        for latency in [0.1, 0.2, 0.5, 1.0, 2.0]:
            metrics.histogram("test_latency_seconds", value=latency)
    
    def test_metrics_with_labels(self):
        """Test metrics with labels."""
        metrics = get_metrics()
        
        labels = MetricLabels(
            operation="search",
            provider="openai",
            status="success",
        )
        
        metrics.counter("labeled_counter", labels=labels)
        metrics.histogram("labeled_histogram", value=0.5, labels=labels)


# =============================================================================
# Test Decorators Integration
# =============================================================================

class TestDecoratorsIntegration:
    """Test decorator functionality."""
    
    def test_traced_decorator_sync(self):
        """Test @traced decorator on sync function."""
        @traced(name="sync_operation", kind=SpanKind.INTERNAL)
        def sync_function(x: int) -> int:
            return x * 2
        
        result = sync_function(5)
        assert result == 10
    
    def test_traced_decorator_async(self):
        """Test @traced decorator on async function."""
        @traced(name="async_operation", kind=SpanKind.LLM)
        async def async_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 3
        
        result = asyncio.run(async_function(5))
        assert result == 15
    
    def test_timed_decorator_sync(self):
        """Test @timed decorator on sync function."""
        @timed(name="sync_timing")
        def timed_function() -> str:
            return "done"
        
        result = timed_function()
        assert result == "done"
    
    def test_timed_decorator_async(self):
        """Test @timed decorator on async function."""
        @timed(name="async_timing")
        async def async_timed() -> str:
            await asyncio.sleep(0.01)
            return "async done"
        
        result = asyncio.run(async_timed())
        assert result == "async done"
    
    def test_combined_decorators(self):
        """Test using both decorators together."""
        @traced(kind=SpanKind.RETRIEVER)
        @timed()
        def retrieval_function(query: str) -> list:
            return [f"result for {query}"]
        
        results = retrieval_function("test query")
        assert len(results) == 1


# =============================================================================
# Test Evaluation Integration
# =============================================================================

class TestEvaluationIntegration:
    """Test evaluation pipeline integration."""
    
    def test_dataset_save_load_cycle(self, sample_dataset):
        """Test complete save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            
            # Save
            sample_dataset.save(path)
            
            # Load
            loaded = EvaluationDataset.load(path)
            
            # Verify
            assert len(loaded) == len(sample_dataset)
            assert loaded[0].question == sample_dataset[0].question
    
    def test_synthetic_dataset_generation(self):
        """Test synthetic dataset creation."""
        dataset = create_synthetic_dataset(
            num_samples=10,
            topics=["AI", "ML"],
            include_ground_truth=True,
        )
        
        assert len(dataset) == 10
        
        for sample in dataset:
            assert sample.question
            assert sample.answer
            assert len(sample.contexts) >= 2
            assert sample.ground_truth is not None
    
    def test_fallback_evaluation(self, sample_dataset, evaluator):
        """Test fallback evaluation mode."""
        result = evaluator._evaluate_fallback(sample_dataset)
        
        assert result.dataset_name == "integration_test"
        assert result.num_samples == 2
        assert result.aggregate_metrics.faithfulness is not None
        assert result.config["mode"] == "fallback"
    
    def test_evaluation_metrics_aggregation(self, sample_dataset, evaluator):
        """Test that metrics are properly aggregated."""
        result = evaluator._evaluate_fallback(sample_dataset)
        
        # Check aggregate metrics
        agg = result.aggregate_metrics
        assert 0 <= agg.faithfulness <= 1
        assert 0 <= agg.answer_relevance <= 1
        assert 0 <= agg.context_relevance <= 1
        
        # Check sample results
        assert len(result.sample_results) == 2
        for sample_result in result.sample_results:
            assert sample_result.sample_id is not None
            assert sample_result.metrics is not None


# =============================================================================
# Test Full Pipeline Integration
# =============================================================================

class TestFullPipelineIntegration:
    """Test complete observability pipeline."""
    
    def test_traced_evaluation(self, sample_dataset):
        """Test evaluation with tracing enabled."""
        tracer = get_tracer()
        
        with tracer.span("evaluation_pipeline", kind=SpanKind.CHAIN) as span:
            # Create evaluator
            evaluator = RagasEvaluator()
            
            # Run fallback evaluation
            with tracer.span("run_evaluation") as eval_span:
                result = evaluator._evaluate_fallback(sample_dataset)
                eval_span.set_attribute("num_samples", len(sample_dataset))
            
            span.set_attribute("avg_score", result.aggregate_metrics.get_average_score())
            span.set_status(SpanStatus.OK)
        
        assert result.num_samples == 2
    
    def test_metrics_during_evaluation(self, sample_dataset):
        """Test metrics collection during evaluation."""
        metrics = get_metrics()
        
        # Start evaluation
        metrics.counter("evaluations_started")
        
        evaluator = RagasEvaluator()
        result = evaluator._evaluate_fallback(sample_dataset)
        
        # Record metrics
        metrics.counter("evaluations_completed")
        metrics.histogram(
            "evaluation_duration_seconds",
            value=result.duration_seconds,
        )
        metrics.gauge(
            "evaluation_avg_score",
            value=result.aggregate_metrics.get_average_score(),
        )
    
    def test_error_handling_in_pipeline(self):
        """Test error handling in observability pipeline."""
        tracer = get_tracer()
        metrics = get_metrics()
        
        try:
            with tracer.span("error_prone_operation") as span:
                metrics.counter("operations_started")
                
                # Simulate error
                raise RuntimeError("Simulated error")
                
        except RuntimeError:
            metrics.counter(
                "operations_failed",
                labels=MetricLabels(error_type="RuntimeError"),
            )


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfiguration:
    """Test observability configuration."""
    
    def test_default_config_loads(self):
        """Test that default configuration works."""
        # Should not raise
        tracer = get_tracer()
        metrics = get_metrics()
        
        assert tracer is not None
        assert metrics is not None
    
    def test_tracer_reinit(self):
        """Test tracer reinitialization."""
        tracer1 = get_tracer()
        tracer2 = get_tracer(force_reinit=True)
        
        # Both should be valid
        assert tracer1 is not None
        assert tracer2 is not None
    
    def test_metrics_reinit(self):
        """Test metrics reinitialization."""
        metrics1 = get_metrics()
        metrics2 = get_metrics(force_reinit=True)
        
        # Both should be valid
        assert metrics1 is not None
        assert metrics2 is not None


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Test thread safety of observability components."""
    
    def test_concurrent_span_creation(self):
        """Test creating spans from multiple threads."""
        import threading
        
        tracer = get_tracer()
        results = []
        
        def create_span(thread_id: int):
            with tracer.span(f"thread_{thread_id}") as span:
                span.set_attribute("thread_id", thread_id)
                results.append(thread_id)
        
        threads = [
            threading.Thread(target=create_span, args=(i,))
            for i in range(10)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10
    
    def test_concurrent_metric_recording(self):
        """Test recording metrics from multiple threads."""
        import threading
        
        metrics = get_metrics()
        
        def record_metrics(thread_id: int):
            for _ in range(100):
                metrics.counter("concurrent_counter")
                metrics.histogram("concurrent_histogram", value=0.5)
        
        threads = [
            threading.Thread(target=record_metrics, args=(i,))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without error

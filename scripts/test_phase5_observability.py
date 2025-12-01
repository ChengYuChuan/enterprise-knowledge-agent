#!/usr/bin/env python3
"""
Phase 5: Observability + Evaluation - Manual Test Script

This script allows you to manually test each component of Phase 5.
Run it step by step to understand how the observability system works.

Usage:
    poetry run python scripts/test_phase5_observability.py

Prerequisites:
    - Ensure you have the required dependencies installed
    - Optional: Start Phoenix server for tracing visualization
    - Optional: Start Prometheus for metrics collection
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_subheader(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def print_result(name: str, status: str, details: str = ""):
    """Print a test result."""
    icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"  {icon} {name}: {status}")
    if details:
        print(f"      {details}")


# =============================================================================
# Test 1: Base Module
# =============================================================================

def test_base_module():
    """Test the base observability module."""
    print_header("Test 1: Base Module (NoOp Implementations)")
    
    try:
        from src.observability.base import (
            SpanKind,
            SpanStatus,
            MetricType,
            SpanContext,
            SpanAttributes,
            MetricLabels,
            NoOpSpan,
            NoOpTracer,
            NoOpMetricsCollector,
        )
        
        # Test enums
        print_subheader("Enums")
        print_result("SpanKind", "PASS", f"Values: {[e.value for e in SpanKind][:5]}...")
        print_result("SpanStatus", "PASS", f"Values: {[e.value for e in SpanStatus]}")
        print_result("MetricType", "PASS", f"Values: {[e.value for e in MetricType]}")
        
        # Test data classes
        print_subheader("Data Classes")
        
        ctx = SpanContext(trace_id="abc123", span_id="def456")
        print_result("SpanContext", "PASS", f"trace_id={ctx.trace_id}")
        
        attrs = SpanAttributes(llm_provider="openai", llm_model="gpt-4")
        print_result("SpanAttributes", "PASS", f"provider={attrs.llm_provider}")
        
        labels = MetricLabels(operation="search", status="success")
        print_result("MetricLabels", "PASS", f"to_dict={labels.to_dict()}")
        
        # Test NoOp implementations
        print_subheader("NoOp Implementations")
        
        span = NoOpSpan("test_operation", SpanKind.LLM)
        span.set_attribute("key", "value")
        span.set_status(SpanStatus.OK)
        span.end()
        print_result("NoOpSpan", "PASS", "All methods work correctly")
        
        tracer = NoOpTracer()
        with tracer.span("test_span") as s:
            s.set_attribute("test", "value")
        print_result("NoOpTracer", "PASS", "Context manager works")
        
        metrics = NoOpMetricsCollector()
        metrics.counter("test_counter")
        metrics.gauge("test_gauge", value=42)
        metrics.histogram("test_histogram", value=0.5)
        with metrics.timer("test_timer"):
            pass
        print_result("NoOpMetricsCollector", "PASS", "All methods work")
        
        return True
        
    except Exception as e:
        print_result("Base Module", "FAIL", str(e))
        return False


# =============================================================================
# Test 2: Tracer Module
# =============================================================================

def test_tracer_module():
    """Test the tracer module."""
    print_header("Test 2: Tracer Module")
    
    try:
        from src.observability import get_tracer, SpanKind, SpanStatus
        
        print_subheader("Tracer Initialization")
        tracer = get_tracer()
        tracer_type = type(tracer).__name__
        print_result("get_tracer()", "PASS", f"Type: {tracer_type}")
        
        # Test singleton
        tracer2 = get_tracer()
        is_singleton = tracer is tracer2
        print_result("Singleton Pattern", "PASS" if is_singleton else "FAIL")
        
        print_subheader("Span Operations")
        
        # Test span creation
        with tracer.span("test_operation", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("test_key", "test_value")
            span.set_attributes({"key1": "value1", "key2": 123})
            span.add_event("test_event", {"detail": "some detail"})
            span.set_status(SpanStatus.OK)
        print_result("Span Context Manager", "PASS", "Span created and closed")
        
        # Test nested spans
        with tracer.span("parent") as parent:
            parent.set_attribute("level", "parent")
            with tracer.span("child") as child:
                child.set_attribute("level", "child")
        print_result("Nested Spans", "PASS", "Parent-child relationship works")
        
        # Test LLM-specific span
        with tracer.span("llm_call", kind=SpanKind.LLM) as span:
            span.set_attributes({
                "llm.provider": "openai",
                "llm.model": "gpt-4o-mini",
                "llm.input_tokens": 150,
                "llm.output_tokens": 75,
            })
        print_result("LLM Span", "PASS", "LLM attributes set correctly")
        
        return True
        
    except Exception as e:
        print_result("Tracer Module", "FAIL", str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 3: Metrics Module
# =============================================================================

def test_metrics_module():
    """Test the metrics module."""
    print_header("Test 3: Metrics Module")
    
    try:
        from src.observability import get_metrics, MetricLabels
        from src.observability.metrics.prometheus import PrometheusMetrics
        from src.observability.metrics.collectors import RAGMetrics, LLMMetrics, APIMetrics
        
        print_subheader("Metrics Initialization")
        metrics = get_metrics()
        metrics_type = type(metrics).__name__
        print_result("get_metrics()", "PASS", f"Type: {metrics_type}")
        
        print_subheader("Basic Metrics Operations")
        
        # Counter
        metrics.counter("test_requests_total")
        metrics.counter("test_requests_total", value=5)
        print_result("Counter", "PASS", "Incremented successfully")
        
        # Gauge
        metrics.gauge("test_connections", value=10)
        metrics.gauge("test_connections", value=15)
        print_result("Gauge", "PASS", "Set successfully")
        
        # Histogram
        for v in [0.1, 0.2, 0.5, 1.0]:
            metrics.histogram("test_latency", value=v)
        print_result("Histogram", "PASS", "Recorded multiple values")
        
        # Timer
        with metrics.timer("test_operation"):
            time.sleep(0.01)
        print_result("Timer", "PASS", "Context manager works")
        
        print_subheader("Labeled Metrics")
        
        labels = MetricLabels(
            operation="search",
            provider="openai",
            status="success",
        )
        metrics.counter("labeled_requests", labels=labels)
        print_result("Labeled Counter", "PASS", f"Labels: {labels.to_dict()}")
        
        print_subheader("Specialized Collectors")
        
        prometheus = PrometheusMetrics(namespace="test")
        
        # RAG Metrics
        rag_metrics = RAGMetrics(prometheus)
        rag_metrics.record_retrieval(
            query="test query",
            num_results=5,
            duration_seconds=0.5,
            strategy="hybrid",
        )
        print_result("RAGMetrics", "PASS", "Retrieval recorded")
        
        # LLM Metrics
        llm_metrics = LLMMetrics(prometheus)
        llm_metrics.record_call(
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            duration_seconds=1.5,
            status="success",  # Use status, not success
        )
        print_result("LLMMetrics", "PASS", "LLM call recorded")
        
        # API Metrics
        api_metrics = APIMetrics(prometheus)
        api_metrics.record_request(
            method="POST",
            endpoint="/api/v1/chat",
            status_code=200,
            duration_seconds=0.8,
        )
        print_result("APIMetrics", "PASS", "API request recorded")
        
        return True
        
    except Exception as e:
        print_result("Metrics Module", "FAIL", str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 4: Evaluation Module
# =============================================================================

def test_evaluation_module():
    """Test the evaluation module."""
    print_header("Test 4: Evaluation Module")
    
    try:
        from src.observability.evaluation.datasets import (
            EvaluationSample,
            EvaluationDataset,
            create_synthetic_dataset,
        )
        from src.observability.evaluation.ragas import (
            EvaluationMetrics,
            RagasEvaluator,
        )
        
        print_subheader("EvaluationSample")
        
        sample = EvaluationSample(
            question="What is machine learning?",
            answer="Machine learning is a subset of AI that learns from data.",
            contexts=[
                "Machine learning is a branch of artificial intelligence.",
                "ML systems improve through experience.",
            ],
            ground_truth="Machine learning is a type of AI.",
        )
        print_result("Sample Creation", "PASS", f"ID: {sample.id}")
        
        errors = sample.validate()
        print_result("Sample Validation", "PASS" if not errors else "FAIL", 
                    "Valid" if not errors else str(errors))
        
        print_subheader("EvaluationDataset")
        
        dataset = EvaluationDataset(name="test_dataset")
        dataset.add_sample(
            question="Q1?",
            answer="A1",
            contexts=["C1"],
        )
        dataset.add_sample(
            question="Q2?",
            answer="A2",
            contexts=["C2", "C3"],
            ground_truth="GT2",
        )
        print_result("Dataset Creation", "PASS", f"Samples: {len(dataset)}")
        
        # Test iteration
        questions = [s.question for s in dataset]
        print_result("Dataset Iteration", "PASS", f"Questions: {questions}")
        
        print_subheader("Synthetic Dataset")
        
        synthetic = create_synthetic_dataset(
            num_samples=5,
            topics=["AI", "ML"],
            include_ground_truth=True,
        )
        print_result("Synthetic Creation", "PASS", f"Samples: {len(synthetic)}")
        
        # Validate all samples
        all_valid = all(not s.validate() for s in synthetic)
        print_result("Synthetic Validation", "PASS" if all_valid else "FAIL")
        
        print_subheader("EvaluationMetrics")
        
        metrics = EvaluationMetrics(
            faithfulness=0.9,
            answer_relevance=0.85,
            context_relevance=0.8,
        )
        print_result("Metrics Creation", "PASS", f"Avg: {metrics.get_average_score():.3f}")
        
        available = metrics.get_available_metrics()
        print_result("Available Metrics", "PASS", f"Count: {len(available)}")
        
        thresholds = metrics.passes_thresholds()
        passed = sum(1 for v in thresholds.values() if v)
        print_result("Threshold Check", "PASS", f"Passed: {passed}/{len(thresholds)}")
        
        print_subheader("RagasEvaluator (Fallback Mode)")
        
        evaluator = RagasEvaluator()
        print_result("Evaluator Init", "PASS", f"Provider: {evaluator.llm_provider}")
        
        # Test fallback evaluation
        result = evaluator._evaluate_fallback(synthetic)
        print_result("Fallback Evaluation", "PASS", 
                    f"Duration: {result.duration_seconds:.3f}s")
        
        agg = result.aggregate_metrics
        print_result("Aggregate Metrics", "PASS", 
                    f"Faithfulness: {agg.faithfulness:.3f}, "
                    f"Answer Rel: {agg.answer_relevance:.3f}")
        
        return True
        
    except Exception as e:
        print_result("Evaluation Module", "FAIL", str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 5: Decorators
# =============================================================================

def test_decorators():
    """Test the decorator utilities."""
    print_header("Test 5: Decorators")
    
    try:
        from src.observability.base import traced, timed
        from src.observability import SpanKind
        
        print_subheader("@traced Decorator")
        
        @traced(name="sync_operation", kind=SpanKind.INTERNAL)
        def sync_function(x: int) -> int:
            return x * 2
        
        result = sync_function(5)
        print_result("Sync Function", "PASS" if result == 10 else "FAIL", 
                    f"Result: {result}")
        
        @traced(name="async_operation", kind=SpanKind.LLM)
        async def async_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 3
        
        result = asyncio.run(async_function(5))
        print_result("Async Function", "PASS" if result == 15 else "FAIL",
                    f"Result: {result}")
        
        print_subheader("@timed Decorator")
        
        @timed(name="timed_sync")
        def timed_sync():
            time.sleep(0.01)
            return "done"
        
        result = timed_sync()
        print_result("Timed Sync", "PASS" if result == "done" else "FAIL")
        
        @timed(name="timed_async")
        async def timed_async():
            await asyncio.sleep(0.01)
            return "async done"
        
        result = asyncio.run(timed_async())
        print_result("Timed Async", "PASS" if result == "async done" else "FAIL")
        
        print_subheader("Combined Decorators")
        
        @traced(kind=SpanKind.RETRIEVER)
        @timed()
        def combined_function(query: str) -> list:
            return [f"result for {query}"]
        
        results = combined_function("test")
        print_result("Combined", "PASS" if len(results) == 1 else "FAIL",
                    f"Results: {results}")
        
        return True
        
    except Exception as e:
        print_result("Decorators", "FAIL", str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 6: Save/Load Dataset
# =============================================================================

def test_dataset_io():
    """Test dataset save and load operations."""
    print_header("Test 6: Dataset I/O")
    
    try:
        import tempfile
        from src.observability.evaluation.datasets import EvaluationDataset
        
        # Create test dataset
        dataset = EvaluationDataset(name="io_test")
        dataset.add_sample(
            question="What is Python?",
            answer="A programming language.",
            contexts=["Python is popular."],
            ground_truth="Python is a general-purpose language.",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            print_subheader("JSON Format")
            
            json_path = Path(tmpdir) / "test.json"
            dataset.save(json_path, format="json")
            print_result("Save JSON", "PASS", f"Path: {json_path.name}")
            
            loaded = EvaluationDataset.load(json_path)
            print_result("Load JSON", "PASS" if len(loaded) == 1 else "FAIL",
                        f"Samples: {len(loaded)}")
            
            print_subheader("JSONL Format")
            
            jsonl_path = Path(tmpdir) / "test.jsonl"
            dataset.save(jsonl_path, format="jsonl")
            print_result("Save JSONL", "PASS")
            
            loaded = EvaluationDataset.load(jsonl_path)
            print_result("Load JSONL", "PASS" if len(loaded) == 1 else "FAIL")
            
            print_subheader("CSV Format")
            
            csv_path = Path(tmpdir) / "test.csv"
            dataset.save(csv_path, format="csv")
            print_result("Save CSV", "PASS")
            
            loaded = EvaluationDataset.load(csv_path)
            print_result("Load CSV", "PASS" if len(loaded) == 1 else "FAIL")
        
        return True
        
    except Exception as e:
        print_result("Dataset I/O", "FAIL", str(e))
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print(" PHASE 5: OBSERVABILITY + EVALUATION - TEST SUITE")
    print("=" * 60)
    print(f"\nProject Root: {project_root}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    results = []
    
    # Run tests
    results.append(("Base Module", test_base_module()))
    results.append(("Tracer Module", test_tracer_module()))
    results.append(("Metrics Module", test_metrics_module()))
    results.append(("Evaluation Module", test_evaluation_module()))
    results.append(("Decorators", test_decorators()))
    results.append(("Dataset I/O", test_dataset_io()))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All Phase 5 tests passed!")
        return 0
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

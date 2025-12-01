"""
Benchmarks Submodule

This module provides performance benchmarking tools for the Knowledge Agent,
measuring latency, throughput, and resource utilization.

Components:
    - BenchmarkRunner: Orchestrates benchmark execution
    - BenchmarkScenario: Defines test scenarios
    - BenchmarkResult: Contains benchmark results and analysis

Benchmark Types:
    1. Latency Benchmarks: Measure response time percentiles (P50, P95, P99)
    2. Throughput Benchmarks: Measure requests per second
    3. Stress Tests: Find system limits under load
    4. Component Benchmarks: Isolate individual component performance

Usage:
    ```python
    from src.observability.benchmarks import (
        BenchmarkRunner,
        create_latency_scenario,
        create_throughput_scenario,
    )

    # Create runner
    runner = BenchmarkRunner()

    # Run latency benchmark
    scenario = create_latency_scenario(
        queries=["What is AI?", "How does ML work?"],
        num_runs=100,
    )
    result = await runner.run(scenario)

    # Analyze results
    print(f"P50 Latency: {result.latency_p50_ms:.2f}ms")
    print(f"P99 Latency: {result.latency_p99_ms:.2f}ms")
    print(f"Throughput: {result.requests_per_second:.2f} req/s")
    ```
"""

from src.observability.benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkConfig,
)
from src.observability.benchmarks.scenarios import (
    BenchmarkScenario,
    ScenarioType,
    create_latency_scenario,
    create_throughput_scenario,
    create_stress_scenario,
    create_component_scenario,
)
from src.observability.benchmarks.results import (
    BenchmarkResult,
    LatencyStats,
    ThroughputStats,
    ResourceStats,
    generate_benchmark_report,
)

__all__ = [
    # Runner
    "BenchmarkRunner",
    "BenchmarkConfig",
    # Scenarios
    "BenchmarkScenario",
    "ScenarioType",
    "create_latency_scenario",
    "create_throughput_scenario",
    "create_stress_scenario",
    "create_component_scenario",
    # Results
    "BenchmarkResult",
    "LatencyStats",
    "ThroughputStats",
    "ResourceStats",
    "generate_benchmark_report",
]
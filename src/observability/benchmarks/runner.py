"""
Benchmark Runner

This module provides the BenchmarkRunner class that orchestrates
the execution of benchmark scenarios and collects performance metrics.

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                     BenchmarkRunner                           │
    │  ┌──────────────────────────────────────────────────────────┐│
    │  │                    Scenario                               ││
    │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐  ││
    │  │  │ Query 1 │  │ Query 2 │  │ Query 3 │  │    ...      │  ││
    │  │  └────┬────┘  └────┬────┘  └────┬────┘  └──────┬──────┘  ││
    │  └───────┼────────────┼────────────┼──────────────┼─────────┘│
    │          │            │            │              │          │
    │          ▼            ▼            ▼              ▼          │
    │  ┌───────────────────────────────────────────────────────┐   │
    │  │              Concurrent Workers                        │   │
    │  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐       │   │
    │  │  │Worker 1│  │Worker 2│  │Worker 3│  │Worker N│       │   │
    │  │  └────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘       │   │
    │  └───────┼───────────┼───────────┼───────────┼───────────┘   │
    │          │           │           │           │               │
    │          └───────────┴─────┬─────┴───────────┘               │
    │                            ▼                                  │
    │  ┌──────────────────────────────────────────────────────────┐│
    │  │                  Metrics Collector                        ││
    │  │  • Latency measurements                                   ││
    │  │  • Throughput calculation                                 ││
    │  │  • Error tracking                                         ││
    │  │  • Resource monitoring                                    ││
    │  └──────────────────────────────────────────────────────────┘│
    │                            │                                  │
    │                            ▼                                  │
    │                    BenchmarkResult                            │
    └──────────────────────────────────────────────────────────────┘

Usage:
    ```python
    from src.observability.benchmarks import BenchmarkRunner, create_latency_scenario

    runner = BenchmarkRunner()
    scenario = create_latency_scenario(num_runs=100)
    
    result = await runner.run(scenario)
    
    print(f"P50: {result.latency_stats.p50_ms}ms")
    print(f"P99: {result.latency_stats.p99_ms}ms")
    ```
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar

from src.observability.benchmarks.scenarios import (
    BenchmarkScenario,
    ScenarioType,
    ComponentType,
)
from src.observability.benchmarks.results import (
    BenchmarkResult,
    LatencyStats,
    ThroughputStats,
    ResourceStats,
    RequestResult,
)


logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """
    Configuration for the benchmark runner.
    
    Attributes:
        collect_resource_metrics: Whether to collect system resource metrics.
        resource_sample_interval: How often to sample resources (seconds).
        log_progress: Whether to log progress during execution.
        progress_interval: How often to log progress (seconds).
        save_raw_results: Whether to save individual request results.
        max_concurrent_requests: Maximum concurrent requests allowed.
    """
    
    collect_resource_metrics: bool = True
    resource_sample_interval: float = 1.0
    log_progress: bool = True
    progress_interval: float = 5.0
    save_raw_results: bool = True
    max_concurrent_requests: int = 100


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """
    Executes benchmark scenarios and collects performance metrics.
    
    This class handles the orchestration of benchmark runs, including:
    - Managing concurrent workers
    - Collecting timing metrics
    - Monitoring resource usage
    - Calculating statistics
    
    Attributes:
        config: Runner configuration.
        _target_func: Function to benchmark.
    
    Example:
        ```python
        runner = BenchmarkRunner()
        
        # Set the function to benchmark
        runner.set_target(my_async_function)
        
        # Run scenario
        scenario = create_latency_scenario(num_runs=100)
        result = await runner.run(scenario)
        ```
    """
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        target_func: Optional[Callable[..., Coroutine[Any, Any, Any]]] = None,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            config: Runner configuration.
            target_func: Optional async function to benchmark.
        """
        self.config = config or BenchmarkConfig()
        self._target_func = target_func
        self._request_results: List[RequestResult] = []
        self._resource_samples: List[Dict[str, Any]] = []
        self._running = False
        self._start_time: Optional[float] = None
    
    def set_target(
        self, 
        func: Callable[..., Coroutine[Any, Any, Any]]
    ) -> None:
        """
        Set the target function to benchmark.
        
        Args:
            func: Async function that takes a query string and returns a response.
        """
        self._target_func = func
    
    async def run(
        self,
        scenario: BenchmarkScenario,
        target_func: Optional[Callable[..., Coroutine[Any, Any, Any]]] = None,
    ) -> BenchmarkResult:
        """
        Run a benchmark scenario.
        
        Args:
            scenario: The scenario to execute.
            target_func: Optional override for the target function.
        
        Returns:
            BenchmarkResult with collected metrics.
        
        Raises:
            ValueError: If no target function is set.
        """
        func = target_func or self._target_func
        if func is None:
            # Use a mock function for demonstration
            func = self._mock_target_function
            logger.warning("No target function set, using mock function")
        
        logger.info(f"Starting benchmark: {scenario.name}")
        logger.info(f"Type: {scenario.type.value}")
        logger.info(f"Iterations: {scenario.num_iterations}")
        
        self._request_results = []
        self._resource_samples = []
        self._running = True
        self._start_time = time.time()
        
        try:
            # Run based on scenario type
            if scenario.type == ScenarioType.LATENCY:
                await self._run_latency_benchmark(scenario, func)
            elif scenario.type == ScenarioType.THROUGHPUT:
                await self._run_throughput_benchmark(scenario, func)
            elif scenario.type == ScenarioType.STRESS:
                await self._run_stress_benchmark(scenario, func)
            elif scenario.type == ScenarioType.COMPONENT:
                await self._run_component_benchmark(scenario, func)
            else:
                await self._run_latency_benchmark(scenario, func)
            
        finally:
            self._running = False
        
        # Calculate results
        return self._calculate_results(scenario)
    
    async def _run_latency_benchmark(
        self,
        scenario: BenchmarkScenario,
        func: Callable,
    ) -> None:
        """Run a sequential latency benchmark."""
        # Warmup
        if scenario.warmup_iterations > 0:
            logger.info(f"Running {scenario.warmup_iterations} warmup iterations...")
            for _ in range(scenario.warmup_iterations):
                query = scenario.get_query()
                try:
                    await asyncio.wait_for(
                        func(query),
                        timeout=scenario.timeout_seconds
                    )
                except Exception:
                    pass
        
        # Main benchmark
        logger.info(f"Running {scenario.num_iterations} benchmark iterations...")
        
        for i in range(scenario.num_iterations):
            query = scenario.get_query()
            result = await self._execute_request(func, query, scenario.timeout_seconds)
            self._request_results.append(result)
            
            # Log progress
            if self.config.log_progress and (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{scenario.num_iterations}")
    
    async def _run_throughput_benchmark(
        self,
        scenario: BenchmarkScenario,
        func: Callable,
    ) -> None:
        """Run a concurrent throughput benchmark."""
        end_time = time.time() + scenario.duration_seconds
        semaphore = asyncio.Semaphore(scenario.load_profile.max_users)
        
        async def worker():
            while time.time() < end_time and self._running:
                async with semaphore:
                    query = scenario.get_query()
                    result = await self._execute_request(
                        func, query, scenario.timeout_seconds
                    )
                    self._request_results.append(result)
        
        # Start workers
        workers = []
        for _ in range(scenario.load_profile.max_users):
            workers.append(asyncio.create_task(worker()))
        
        # Start resource monitoring
        if self.config.collect_resource_metrics:
            monitor_task = asyncio.create_task(
                self._monitor_resources(end_time - time.time())
            )
            workers.append(monitor_task)
        
        # Wait for completion
        await asyncio.gather(*workers, return_exceptions=True)
    
    async def _run_stress_benchmark(
        self,
        scenario: BenchmarkScenario,
        func: Callable,
    ) -> None:
        """Run a stress test with increasing load."""
        step_users = scenario.custom_config.get("step_users", 10)
        step_duration = scenario.custom_config.get("step_duration_seconds", 30.0)
        
        current_users = scenario.load_profile.initial_users
        
        while current_users <= scenario.load_profile.max_users:
            logger.info(f"Stress test: {current_users} concurrent users")
            
            # Run for step duration
            step_end = time.time() + step_duration
            semaphore = asyncio.Semaphore(current_users)
            step_results: List[RequestResult] = []
            
            async def worker():
                while time.time() < step_end and self._running:
                    async with semaphore:
                        query = scenario.get_query()
                        result = await self._execute_request(
                            func, query, scenario.timeout_seconds
                        )
                        step_results.append(result)
                        self._request_results.append(result)
            
            workers = [
                asyncio.create_task(worker()) 
                for _ in range(current_users)
            ]
            await asyncio.gather(*workers, return_exceptions=True)
            
            # Check error rate
            errors = sum(1 for r in step_results if r.error is not None)
            error_rate = errors / max(len(step_results), 1)
            
            if error_rate > scenario.error_threshold:
                logger.warning(
                    f"Error threshold exceeded at {current_users} users "
                    f"(rate: {error_rate:.2%})"
                )
                break
            
            current_users += step_users
    
    async def _run_component_benchmark(
        self,
        scenario: BenchmarkScenario,
        func: Callable,
    ) -> None:
        """Run a component-specific benchmark."""
        # Component benchmarks are similar to latency benchmarks
        # but may target specific functions
        await self._run_latency_benchmark(scenario, func)
    
    async def _execute_request(
        self,
        func: Callable,
        query: str,
        timeout: float,
    ) -> RequestResult:
        """
        Execute a single request and measure timing.
        
        Args:
            func: Function to call.
            query: Query string.
            timeout: Request timeout.
        
        Returns:
            RequestResult with timing and status.
        """
        start_time = time.perf_counter()
        error = None
        response = None
        
        try:
            response = await asyncio.wait_for(
                func(query),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            error = "Timeout"
        except Exception as e:
            error = str(e)
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        return RequestResult(
            query=query,
            latency_ms=latency_ms,
            success=error is None,
            error=error,
            response_size=len(str(response)) if response else 0,
            timestamp=datetime.utcnow().isoformat(),
        )
    
    async def _monitor_resources(self, duration: float) -> None:
        """Monitor system resources during benchmark."""
        end_time = time.time() + duration
        
        while time.time() < end_time and self._running:
            try:
                sample = self._collect_resource_sample()
                self._resource_samples.append(sample)
            except Exception as e:
                logger.debug(f"Failed to collect resource sample: {e}")
            
            await asyncio.sleep(self.config.resource_sample_interval)
    
    def _collect_resource_sample(self) -> Dict[str, Any]:
        """Collect current resource usage."""
        sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_seconds": time.time() - (self._start_time or time.time()),
        }
        
        try:
            import psutil
            
            process = psutil.Process()
            sample["memory_rss_mb"] = process.memory_info().rss / (1024 * 1024)
            sample["memory_percent"] = process.memory_percent()
            sample["cpu_percent"] = process.cpu_percent()
            sample["num_threads"] = process.num_threads()
            
            # System-wide
            sample["system_cpu_percent"] = psutil.cpu_percent()
            sample["system_memory_percent"] = psutil.virtual_memory().percent
            
        except ImportError:
            pass
        
        return sample
    
    def _calculate_results(self, scenario: BenchmarkScenario) -> BenchmarkResult:
        """Calculate benchmark results from collected data."""
        total_duration = time.time() - (self._start_time or time.time())
        
        # Filter successful results for latency calculation
        successful = [r for r in self._request_results if r.success]
        failed = [r for r in self._request_results if not r.success]
        
        # Calculate latency statistics
        latency_stats = self._calculate_latency_stats(successful)
        
        # Calculate throughput statistics
        throughput_stats = self._calculate_throughput_stats(total_duration)
        
        # Calculate resource statistics
        resource_stats = self._calculate_resource_stats()
        
        return BenchmarkResult(
            scenario_name=scenario.name,
            scenario_type=scenario.type.value,
            started_at=datetime.fromtimestamp(self._start_time or time.time()).isoformat(),
            duration_seconds=total_duration,
            total_requests=len(self._request_results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            error_rate=len(failed) / max(len(self._request_results), 1),
            latency_stats=latency_stats,
            throughput_stats=throughput_stats,
            resource_stats=resource_stats,
            request_results=self._request_results if self.config.save_raw_results else [],
            scenario_config=scenario.to_dict(),
        )
    
    def _calculate_latency_stats(
        self, 
        successful_results: List[RequestResult]
    ) -> LatencyStats:
        """Calculate latency statistics."""
        if not successful_results:
            return LatencyStats()
        
        latencies = sorted([r.latency_ms for r in successful_results])
        n = len(latencies)
        
        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return latencies[min(idx, n - 1)]
        
        return LatencyStats(
            min_ms=latencies[0],
            max_ms=latencies[-1],
            mean_ms=sum(latencies) / n,
            median_ms=latencies[n // 2],
            p50_ms=percentile(50),
            p75_ms=percentile(75),
            p90_ms=percentile(90),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
            std_dev_ms=self._calculate_std_dev(latencies),
        )
    
    def _calculate_throughput_stats(
        self, 
        duration_seconds: float
    ) -> ThroughputStats:
        """Calculate throughput statistics."""
        if duration_seconds <= 0:
            return ThroughputStats()
        
        successful = sum(1 for r in self._request_results if r.success)
        
        return ThroughputStats(
            requests_per_second=len(self._request_results) / duration_seconds,
            successful_per_second=successful / duration_seconds,
            total_requests=len(self._request_results),
            duration_seconds=duration_seconds,
        )
    
    def _calculate_resource_stats(self) -> ResourceStats:
        """Calculate resource usage statistics."""
        if not self._resource_samples:
            return ResourceStats()
        
        memory_samples = [s.get("memory_rss_mb", 0) for s in self._resource_samples]
        cpu_samples = [s.get("cpu_percent", 0) for s in self._resource_samples]
        
        return ResourceStats(
            avg_memory_mb=sum(memory_samples) / len(memory_samples) if memory_samples else 0,
            max_memory_mb=max(memory_samples) if memory_samples else 0,
            avg_cpu_percent=sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
            max_cpu_percent=max(cpu_samples) if cpu_samples else 0,
            samples=self._resource_samples,
        )
    
    @staticmethod
    def _calculate_std_dev(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    @staticmethod
    async def _mock_target_function(query: str) -> str:
        """Mock function for testing the runner."""
        # Simulate variable latency
        import random
        await asyncio.sleep(random.uniform(0.05, 0.2))
        return f"Mock response for: {query}"
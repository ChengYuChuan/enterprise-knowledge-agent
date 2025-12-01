"""
Benchmark Results

This module provides data structures for storing benchmark results
and utilities for generating reports.

Result Structure:
    BenchmarkResult
    ├── LatencyStats (min, max, percentiles, etc.)
    ├── ThroughputStats (RPS, total requests, etc.)
    ├── ResourceStats (memory, CPU, etc.)
    └── RequestResults (individual request data)

Usage:
    ```python
    from src.observability.benchmarks.results import (
        BenchmarkResult,
        generate_benchmark_report,
    )

    # After running benchmark
    result = await runner.run(scenario)

    # Access statistics
    print(f"P99 Latency: {result.latency_stats.p99_ms}ms")
    print(f"Throughput: {result.throughput_stats.requests_per_second} RPS")

    # Generate report
    generate_benchmark_report(result, "benchmark_report.html")
    ```
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RequestResult:
    """
    Result of a single benchmark request.
    
    Attributes:
        query: The query that was sent.
        latency_ms: Response time in milliseconds.
        success: Whether the request succeeded.
        error: Error message if failed.
        response_size: Size of response in bytes.
        timestamp: When the request was made.
    """
    
    query: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    response_size: int = 0
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
            "response_size": self.response_size,
            "timestamp": self.timestamp,
        }


@dataclass
class LatencyStats:
    """
    Latency statistics.
    
    Attributes:
        min_ms: Minimum latency in milliseconds.
        max_ms: Maximum latency in milliseconds.
        mean_ms: Mean latency in milliseconds.
        median_ms: Median latency in milliseconds.
        p50_ms: 50th percentile latency.
        p75_ms: 75th percentile latency.
        p90_ms: 90th percentile latency.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        std_dev_ms: Standard deviation.
    """
    
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p50_ms: float = 0.0
    p75_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    std_dev_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p50_ms": self.p50_ms,
            "p75_ms": self.p75_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "std_dev_ms": self.std_dev_ms,
        }
    
    def meets_sla(
        self,
        p50_threshold_ms: float = 500,
        p95_threshold_ms: float = 1000,
        p99_threshold_ms: float = 2000,
    ) -> Dict[str, bool]:
        """
        Check if latencies meet SLA thresholds.
        
        Args:
            p50_threshold_ms: P50 latency threshold.
            p95_threshold_ms: P95 latency threshold.
            p99_threshold_ms: P99 latency threshold.
        
        Returns:
            Dictionary of SLA check results.
        """
        return {
            "p50_ok": self.p50_ms <= p50_threshold_ms,
            "p95_ok": self.p95_ms <= p95_threshold_ms,
            "p99_ok": self.p99_ms <= p99_threshold_ms,
            "all_ok": (
                self.p50_ms <= p50_threshold_ms and
                self.p95_ms <= p95_threshold_ms and
                self.p99_ms <= p99_threshold_ms
            ),
        }


@dataclass
class ThroughputStats:
    """
    Throughput statistics.
    
    Attributes:
        requests_per_second: Average RPS.
        successful_per_second: Successful requests per second.
        total_requests: Total number of requests.
        duration_seconds: Test duration.
    """
    
    requests_per_second: float = 0.0
    successful_per_second: float = 0.0
    total_requests: int = 0
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requests_per_second": self.requests_per_second,
            "successful_per_second": self.successful_per_second,
            "total_requests": self.total_requests,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class ResourceStats:
    """
    Resource usage statistics.
    
    Attributes:
        avg_memory_mb: Average memory usage in MB.
        max_memory_mb: Peak memory usage in MB.
        avg_cpu_percent: Average CPU usage percentage.
        max_cpu_percent: Peak CPU usage percentage.
        samples: Raw resource samples.
    """
    
    avg_memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    samples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avg_memory_mb": self.avg_memory_mb,
            "max_memory_mb": self.max_memory_mb,
            "avg_cpu_percent": self.avg_cpu_percent,
            "max_cpu_percent": self.max_cpu_percent,
            "num_samples": len(self.samples),
        }


@dataclass
class BenchmarkResult:
    """
    Complete benchmark result.
    
    Attributes:
        scenario_name: Name of the benchmark scenario.
        scenario_type: Type of benchmark (latency, throughput, etc.).
        started_at: When the benchmark started.
        duration_seconds: Total benchmark duration.
        total_requests: Total number of requests made.
        successful_requests: Number of successful requests.
        failed_requests: Number of failed requests.
        error_rate: Percentage of failed requests.
        latency_stats: Latency statistics.
        throughput_stats: Throughput statistics.
        resource_stats: Resource usage statistics.
        request_results: Individual request results.
        scenario_config: Configuration used for the scenario.
    """
    
    scenario_name: str
    scenario_type: str
    started_at: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    latency_stats: LatencyStats
    throughput_stats: ThroughputStats
    resource_stats: ResourceStats
    request_results: List[RequestResult] = field(default_factory=list)
    scenario_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "scenario_type": self.scenario_type,
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": self.error_rate,
            "latency_stats": self.latency_stats.to_dict(),
            "throughput_stats": self.throughput_stats.to_dict(),
            "resource_stats": self.resource_stats.to_dict(),
            "scenario_config": self.scenario_config,
        }
    
    def save_json(self, path: str) -> None:
        """Save result to JSON file."""
        data = self.to_dict()
        # Include raw results if present
        if self.request_results:
            data["request_results"] = [r.to_dict() for r in self.request_results]
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {path}")
    
    @classmethod
    def load_json(cls, path: str) -> "BenchmarkResult":
        """Load result from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(
            scenario_name=data["scenario_name"],
            scenario_type=data["scenario_type"],
            started_at=data["started_at"],
            duration_seconds=data["duration_seconds"],
            total_requests=data["total_requests"],
            successful_requests=data["successful_requests"],
            failed_requests=data["failed_requests"],
            error_rate=data["error_rate"],
            latency_stats=LatencyStats(**data["latency_stats"]),
            throughput_stats=ThroughputStats(**data["throughput_stats"]),
            resource_stats=ResourceStats(**{
                k: v for k, v in data["resource_stats"].items()
                if k != "num_samples"
            }),
            request_results=[
                RequestResult(**r) 
                for r in data.get("request_results", [])
            ],
            scenario_config=data.get("scenario_config", {}),
        )
    
    def get_summary(self) -> str:
        """Get a text summary of results."""
        return f"""
Benchmark Summary: {self.scenario_name}
{'=' * 50}
Type: {self.scenario_type}
Duration: {self.duration_seconds:.2f}s
Started: {self.started_at}

Requests:
  Total: {self.total_requests}
  Successful: {self.successful_requests}
  Failed: {self.failed_requests}
  Error Rate: {self.error_rate:.2%}

Latency (ms):
  Min: {self.latency_stats.min_ms:.2f}
  Max: {self.latency_stats.max_ms:.2f}
  Mean: {self.latency_stats.mean_ms:.2f}
  P50: {self.latency_stats.p50_ms:.2f}
  P95: {self.latency_stats.p95_ms:.2f}
  P99: {self.latency_stats.p99_ms:.2f}

Throughput:
  RPS: {self.throughput_stats.requests_per_second:.2f}

Resources:
  Avg Memory: {self.resource_stats.avg_memory_mb:.2f} MB
  Max Memory: {self.resource_stats.max_memory_mb:.2f} MB
  Avg CPU: {self.resource_stats.avg_cpu_percent:.1f}%
"""


# =============================================================================
# Report Generation
# =============================================================================

def generate_benchmark_report(
    result: BenchmarkResult,
    output_path: str,
    format: str = "html",
) -> None:
    """
    Generate a benchmark report.
    
    Args:
        result: Benchmark result to report.
        output_path: Output file path.
        format: Report format (html, json, markdown).
    """
    if format == "html":
        html = _generate_html_report(result)
        Path(output_path).write_text(html, encoding="utf-8")
    elif format == "json":
        result.save_json(output_path)
    elif format == "markdown":
        md = _generate_markdown_report(result)
        Path(output_path).write_text(md, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Benchmark report saved to {output_path}")


def _generate_html_report(result: BenchmarkResult) -> str:
    """Generate HTML benchmark report."""
    sla_check = result.latency_stats.meets_sla()
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Report - {result.scenario_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 2rem;
        }}
        h1 {{ color: #2c3e50; margin-bottom: 0.5rem; }}
        h2 {{
            color: #34495e;
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #eee;
        }}
        .timestamp {{ color: #666; font-size: 0.9rem; margin-bottom: 2rem; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
        }}
        .card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
        }}
        .card .label {{
            color: #666;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }}
        .card.success {{ background: #e8f5e9; }}
        .card.warning {{ background: #fff3e0; }}
        .card.error {{ background: #ffebee; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .sla-pass {{ color: #27ae60; }}
        .sla-fail {{ color: #e74c3c; }}
        .footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 0.85rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Benchmark Report: {result.scenario_name}</h1>
        <p class="timestamp">
            Type: {result.scenario_type} |
            Started: {result.started_at} |
            Duration: {result.duration_seconds:.2f}s
        </p>
        
        <h2>Summary</h2>
        <div class="grid">
            <div class="card">
                <div class="value">{result.total_requests:,}</div>
                <div class="label">Total Requests</div>
            </div>
            <div class="card {'success' if result.error_rate < 0.01 else 'error' if result.error_rate > 0.05 else 'warning'}">
                <div class="value">{result.error_rate:.2%}</div>
                <div class="label">Error Rate</div>
            </div>
            <div class="card">
                <div class="value">{result.throughput_stats.requests_per_second:.1f}</div>
                <div class="label">Requests/Second</div>
            </div>
            <div class="card {'success' if sla_check['all_ok'] else 'warning'}">
                <div class="value">{'PASS' if sla_check['all_ok'] else 'FAIL'}</div>
                <div class="label">SLA Status</div>
            </div>
        </div>
        
        <h2>Latency Distribution</h2>
        <table>
            <thead>
                <tr>
                    <th>Percentile</th>
                    <th>Latency (ms)</th>
                    <th>SLA Threshold</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Minimum</td>
                    <td>{result.latency_stats.min_ms:.2f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>P50 (Median)</td>
                    <td>{result.latency_stats.p50_ms:.2f}</td>
                    <td>500ms</td>
                    <td class="{'sla-pass' if sla_check['p50_ok'] else 'sla-fail'}">
                        {'✓' if sla_check['p50_ok'] else '✗'}
                    </td>
                </tr>
                <tr>
                    <td>P75</td>
                    <td>{result.latency_stats.p75_ms:.2f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>P90</td>
                    <td>{result.latency_stats.p90_ms:.2f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>P95</td>
                    <td>{result.latency_stats.p95_ms:.2f}</td>
                    <td>1000ms</td>
                    <td class="{'sla-pass' if sla_check['p95_ok'] else 'sla-fail'}">
                        {'✓' if sla_check['p95_ok'] else '✗'}
                    </td>
                </tr>
                <tr>
                    <td>P99</td>
                    <td>{result.latency_stats.p99_ms:.2f}</td>
                    <td>2000ms</td>
                    <td class="{'sla-pass' if sla_check['p99_ok'] else 'sla-fail'}">
                        {'✓' if sla_check['p99_ok'] else '✗'}
                    </td>
                </tr>
                <tr>
                    <td>Maximum</td>
                    <td>{result.latency_stats.max_ms:.2f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            </tbody>
        </table>
        
        <h2>Resource Usage</h2>
        <div class="grid">
            <div class="card">
                <div class="value">{result.resource_stats.avg_memory_mb:.1f}</div>
                <div class="label">Avg Memory (MB)</div>
            </div>
            <div class="card">
                <div class="value">{result.resource_stats.max_memory_mb:.1f}</div>
                <div class="label">Max Memory (MB)</div>
            </div>
            <div class="card">
                <div class="value">{result.resource_stats.avg_cpu_percent:.1f}%</div>
                <div class="label">Avg CPU</div>
            </div>
            <div class="card">
                <div class="value">{result.resource_stats.max_cpu_percent:.1f}%</div>
                <div class="label">Max CPU</div>
            </div>
        </div>
        
        <div class="footer">
            Generated by Enterprise Knowledge Agent Benchmark System
        </div>
    </div>
</body>
</html>
"""
    return html


def _generate_markdown_report(result: BenchmarkResult) -> str:
    """Generate Markdown benchmark report."""
    sla_check = result.latency_stats.meets_sla()
    
    return f"""# Benchmark Report: {result.scenario_name}

**Type:** {result.scenario_type}  
**Started:** {result.started_at}  
**Duration:** {result.duration_seconds:.2f}s

## Summary

| Metric | Value |
|--------|-------|
| Total Requests | {result.total_requests:,} |
| Successful | {result.successful_requests:,} |
| Failed | {result.failed_requests:,} |
| Error Rate | {result.error_rate:.2%} |
| Throughput | {result.throughput_stats.requests_per_second:.1f} RPS |
| SLA Status | {'✅ PASS' if sla_check['all_ok'] else '❌ FAIL'} |

## Latency Distribution

| Percentile | Latency (ms) | SLA | Status |
|------------|--------------|-----|--------|
| Min | {result.latency_stats.min_ms:.2f} | - | - |
| P50 | {result.latency_stats.p50_ms:.2f} | 500ms | {'✅' if sla_check['p50_ok'] else '❌'} |
| P75 | {result.latency_stats.p75_ms:.2f} | - | - |
| P90 | {result.latency_stats.p90_ms:.2f} | - | - |
| P95 | {result.latency_stats.p95_ms:.2f} | 1000ms | {'✅' if sla_check['p95_ok'] else '❌'} |
| P99 | {result.latency_stats.p99_ms:.2f} | 2000ms | {'✅' if sla_check['p99_ok'] else '❌'} |
| Max | {result.latency_stats.max_ms:.2f} | - | - |
| Mean | {result.latency_stats.mean_ms:.2f} | - | - |
| Std Dev | {result.latency_stats.std_dev_ms:.2f} | - | - |

## Resource Usage

| Resource | Average | Maximum |
|----------|---------|---------|
| Memory | {result.resource_stats.avg_memory_mb:.1f} MB | {result.resource_stats.max_memory_mb:.1f} MB |
| CPU | {result.resource_stats.avg_cpu_percent:.1f}% | {result.resource_stats.max_cpu_percent:.1f}% |

---
*Generated by Enterprise Knowledge Agent Benchmark System*
"""


def compare_results(
    results: List[BenchmarkResult],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare multiple benchmark results.
    
    Args:
        results: List of benchmark results to compare.
        output_path: Optional path to save comparison report.
    
    Returns:
        Comparison data.
    """
    if len(results) < 2:
        raise ValueError("Need at least 2 results to compare")
    
    comparison = {
        "scenarios": [r.scenario_name for r in results],
        "latency_comparison": {
            "p50": [r.latency_stats.p50_ms for r in results],
            "p95": [r.latency_stats.p95_ms for r in results],
            "p99": [r.latency_stats.p99_ms for r in results],
        },
        "throughput_comparison": [
            r.throughput_stats.requests_per_second for r in results
        ],
        "error_rate_comparison": [r.error_rate for r in results],
    }
    
    # Find best performer
    p99_values = [r.latency_stats.p99_ms for r in results]
    best_idx = p99_values.index(min(p99_values))
    comparison["best_p99"] = results[best_idx].scenario_name
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
    
    return comparison
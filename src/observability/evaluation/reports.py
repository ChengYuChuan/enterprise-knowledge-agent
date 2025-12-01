"""
Evaluation Report Generation

This module provides utilities for generating evaluation reports in various
formats (HTML, JSON, Markdown). Reports include visualizations, summary
statistics, and detailed per-sample analysis.

Report Contents:
    - Executive summary with key metrics
    - Metric distributions and histograms
    - Pass/fail analysis against thresholds
    - Per-sample breakdown with problem identification
    - Recommendations for improvement

Usage:
    ```python
    from src.observability.evaluation.reports import (
        EvaluationReport,
        generate_html_report,
    )

    # Generate report from evaluation result
    report = EvaluationReport(result)
    
    # Save as HTML
    report.save_html("report.html")
    
    # Or use convenience function
    generate_html_report(result, "report.html")
    ```
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.observability.evaluation.ragas import EvaluationResult


logger = logging.getLogger(__name__)


# =============================================================================
# Report Data Classes
# =============================================================================

@dataclass
class MetricSummary:
    """Summary for a single metric."""
    
    name: str
    value: float
    threshold: float
    passed: bool
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "interpretation": self.interpretation,
        }


@dataclass
class ReportSection:
    """A section of the report."""
    
    title: str
    content: str
    subsections: List["ReportSection"] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections],
        }


# =============================================================================
# Evaluation Report
# =============================================================================

class EvaluationReport:
    """
    Generates comprehensive evaluation reports.
    
    This class takes evaluation results and produces formatted reports
    with visualizations, analysis, and recommendations.
    
    Attributes:
        result: The evaluation result to report on.
        thresholds: Metric thresholds for pass/fail determination.
    
    Example:
        ```python
        report = EvaluationReport(result)
        
        # Get summary
        summary = report.get_summary()
        
        # Generate HTML
        html = report.to_html()
        
        # Save to file
        report.save_html("evaluation_report.html")
        ```
    """
    
    # Default thresholds for metric assessment
    DEFAULT_THRESHOLDS = {
        "faithfulness": 0.85,
        "answer_relevance": 0.80,
        "context_relevance": 0.75,
        "context_precision": 0.75,
        "context_recall": 0.70,
    }
    
    # Metric interpretations
    METRIC_INTERPRETATIONS = {
        "faithfulness": {
            "high": "The answers are well-grounded in the provided context with minimal hallucination.",
            "medium": "Some answers contain information not fully supported by the context.",
            "low": "Many answers contain hallucinated or unsupported information. Review retrieval quality.",
        },
        "answer_relevance": {
            "high": "The answers directly address the questions asked.",
            "medium": "Some answers are partially relevant or include unnecessary information.",
            "low": "Many answers fail to address the questions. Review generation prompts.",
        },
        "context_relevance": {
            "high": "The retrieved context is highly relevant to the questions.",
            "medium": "Some retrieved context is only partially relevant.",
            "low": "Retrieved context often misses relevant information. Improve retrieval strategy.",
        },
        "context_precision": {
            "high": "Top-ranked retrieved documents are consistently relevant.",
            "medium": "Ranking quality is moderate; some irrelevant documents appear early.",
            "low": "Poor ranking quality. Consider adding or improving reranking.",
        },
        "context_recall": {
            "high": "Retrieved context captures most of the information needed.",
            "medium": "Some relevant information is missing from retrieved context.",
            "low": "Significant information gaps in retrieval. Expand retrieval scope.",
        },
    }
    
    def __init__(
        self,
        result: "EvaluationResult",
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the report generator.
        
        Args:
            result: Evaluation result to report on.
            thresholds: Custom thresholds for pass/fail determination.
        """
        self.result = result
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self._generated_at = datetime.utcnow().isoformat()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get executive summary of evaluation results.
        
        Returns:
            Dictionary with summary information.
        """
        metrics = self.result.aggregate_metrics
        available = metrics.get_available_metrics()
        
        # Calculate pass/fail for each metric
        metric_summaries = []
        passed_count = 0
        
        for name, value in available.items():
            threshold = self.thresholds.get(name, 0.75)
            passed = value >= threshold
            if passed:
                passed_count += 1
            
            # Get interpretation
            level = "high" if value >= 0.85 else "medium" if value >= 0.7 else "low"
            interpretation = self.METRIC_INTERPRETATIONS.get(
                name, {}
            ).get(level, "No interpretation available.")
            
            metric_summaries.append(MetricSummary(
                name=name,
                value=value,
                threshold=threshold,
                passed=passed,
                interpretation=interpretation,
            ))
        
        # Calculate overall health
        if not available:
            overall_health = "unknown"
        elif passed_count == len(available):
            overall_health = "excellent"
        elif passed_count >= len(available) * 0.7:
            overall_health = "good"
        elif passed_count >= len(available) * 0.5:
            overall_health = "needs_improvement"
        else:
            overall_health = "poor"
        
        return {
            "generated_at": self._generated_at,
            "dataset_name": self.result.dataset_name,
            "num_samples": self.result.num_samples,
            "duration_seconds": self.result.duration_seconds,
            "overall_health": overall_health,
            "average_score": metrics.get_average_score(),
            "passed_metrics": passed_count,
            "total_metrics": len(available),
            "metric_summaries": [m.to_dict() for m in metric_summaries],
        }
    
    def get_problem_samples(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get samples with the lowest scores.
        
        Args:
            limit: Maximum number of samples to return.
        
        Returns:
            List of problem sample details.
        """
        # Sort samples by average score
        scored_samples = []
        for sample in self.result.sample_results:
            avg_score = sample.metrics.get_average_score()
            scored_samples.append((avg_score, sample))
        
        scored_samples.sort(key=lambda x: x[0])
        
        problems = []
        for score, sample in scored_samples[:limit]:
            problems.append({
                "sample_id": sample.sample_id,
                "question": sample.question[:100] + "..." if len(sample.question) > 100 else sample.question,
                "average_score": score,
                "metrics": sample.metrics.get_available_metrics(),
                "issues": self._identify_issues(sample),
            })
        
        return problems
    
    def _identify_issues(self, sample) -> List[str]:
        """Identify specific issues with a sample."""
        issues = []
        metrics = sample.metrics
        
        if metrics.faithfulness is not None and metrics.faithfulness < 0.7:
            issues.append("Low faithfulness - possible hallucination")
        
        if metrics.answer_relevance is not None and metrics.answer_relevance < 0.7:
            issues.append("Low answer relevance - answer may not address the question")
        
        if metrics.context_relevance is not None and metrics.context_relevance < 0.7:
            issues.append("Low context relevance - retrieval may need improvement")
        
        if metrics.context_precision is not None and metrics.context_precision < 0.7:
            issues.append("Low context precision - ranking needs improvement")
        
        if metrics.context_recall is not None and metrics.context_recall < 0.7:
            issues.append("Low context recall - missing relevant information")
        
        return issues
    
    def get_recommendations(self) -> List[str]:
        """
        Generate recommendations based on evaluation results.
        
        Returns:
            List of actionable recommendations.
        """
        recommendations = []
        metrics = self.result.aggregate_metrics
        
        # Check faithfulness
        if metrics.faithfulness is not None and metrics.faithfulness < 0.85:
            recommendations.append(
                "Improve faithfulness by: (1) Adding explicit instructions to only use "
                "provided context, (2) Implementing citation requirements, (3) Adding "
                "a verification step before outputting answers."
            )
        
        # Check answer relevance
        if metrics.answer_relevance is not None and metrics.answer_relevance < 0.80:
            recommendations.append(
                "Improve answer relevance by: (1) Refining the system prompt to focus "
                "on directly answering the question, (2) Adding examples of good answers, "
                "(3) Implementing query reformulation."
            )
        
        # Check context relevance
        if metrics.context_relevance is not None and metrics.context_relevance < 0.75:
            recommendations.append(
                "Improve context relevance by: (1) Tuning embedding model selection, "
                "(2) Adjusting chunk size and overlap, (3) Implementing hybrid search, "
                "(4) Adding metadata filtering."
            )
        
        # Check context precision
        if metrics.context_precision is not None and metrics.context_precision < 0.75:
            recommendations.append(
                "Improve context precision by: (1) Adding a reranking step, "
                "(2) Reducing the number of retrieved documents, (3) Implementing "
                "diversity-aware retrieval."
            )
        
        # Check context recall
        if metrics.context_recall is not None and metrics.context_recall < 0.70:
            recommendations.append(
                "Improve context recall by: (1) Increasing top_k for retrieval, "
                "(2) Using query expansion techniques, (3) Implementing multi-query "
                "retrieval, (4) Adding BM25 for keyword matching."
            )
        
        if not recommendations:
            recommendations.append(
                "All metrics are above thresholds. Consider: (1) Expanding test coverage, "
                "(2) Testing edge cases, (3) Running adversarial evaluations."
            )
        
        return recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to dictionary.
        
        Returns:
            Complete report as dictionary.
        """
        return {
            "summary": self.get_summary(),
            "aggregate_metrics": self.result.aggregate_metrics.to_dict(),
            "problem_samples": self.get_problem_samples(),
            "recommendations": self.get_recommendations(),
            "config": self.result.config,
            "sample_results": [
                {
                    "sample_id": s.sample_id,
                    "question": s.question,
                    "metrics": s.metrics.to_dict(),
                    "error": s.error,
                }
                for s in self.result.sample_results
            ],
        }
    
    def to_html(self) -> str:
        """
        Generate HTML report.
        
        Returns:
            HTML string.
        """
        summary = self.get_summary()
        problems = self.get_problem_samples()
        recommendations = self.get_recommendations()
        
        # Generate metric cards HTML
        metric_cards = ""
        for metric in summary["metric_summaries"]:
            status_class = "metric-pass" if metric["passed"] else "metric-fail"
            metric_cards += f"""
            <div class="metric-card {status_class}">
                <h3>{metric["name"].replace("_", " ").title()}</h3>
                <div class="metric-value">{metric["value"]:.2%}</div>
                <div class="metric-threshold">Threshold: {metric["threshold"]:.0%}</div>
                <p class="metric-interpretation">{metric["interpretation"]}</p>
            </div>
            """
        
        # Generate problem samples HTML
        problem_rows = ""
        for p in problems:
            issues_html = "<br>".join(f"• {issue}" for issue in p["issues"])
            problem_rows += f"""
            <tr>
                <td>{p["sample_id"]}</td>
                <td>{p["question"]}</td>
                <td>{p["average_score"]:.2%}</td>
                <td class="issues">{issues_html or "None identified"}</td>
            </tr>
            """
        
        # Generate recommendations HTML
        recommendations_html = "\n".join(
            f"<li>{rec}</li>" for rec in recommendations
        )
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report - {summary["dataset_name"]}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
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
        h1 {{
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }}
        h2 {{
            color: #34495e;
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #eee;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
        }}
        .summary-card .label {{
            color: #666;
            font-size: 0.9rem;
        }}
        .health-excellent {{ color: #27ae60; }}
        .health-good {{ color: #3498db; }}
        .health-needs_improvement {{ color: #f39c12; }}
        .health-poor {{ color: #e74c3c; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid;
        }}
        .metric-pass {{
            background: #e8f5e9;
            border-color: #27ae60;
        }}
        .metric-fail {{
            background: #ffebee;
            border-color: #e74c3c;
        }}
        .metric-card h3 {{
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }}
        .metric-value {{
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 0.25rem;
        }}
        .metric-threshold {{
            color: #666;
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
        }}
        .metric-interpretation {{
            font-size: 0.9rem;
            color: #555;
        }}
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
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .issues {{
            font-size: 0.85rem;
            color: #e74c3c;
        }}
        .recommendations {{
            background: #fff3cd;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }}
        .recommendations ul {{
            margin-left: 1.5rem;
        }}
        .recommendations li {{
            margin-bottom: 0.75rem;
        }}
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
        <h1>RAG Evaluation Report</h1>
        <p class="timestamp">
            Dataset: {summary["dataset_name"]} | 
            Generated: {summary["generated_at"]} |
            Duration: {summary["duration_seconds"]:.2f}s
        </p>
        
        <h2>Executive Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{summary["num_samples"]}</div>
                <div class="label">Samples Evaluated</div>
            </div>
            <div class="summary-card">
                <div class="value health-{summary["overall_health"]}">{summary["overall_health"].replace("_", " ").title()}</div>
                <div class="label">Overall Health</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary["average_score"]:.1%}</div>
                <div class="label">Average Score</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary["passed_metrics"]}/{summary["total_metrics"]}</div>
                <div class="label">Metrics Passed</div>
            </div>
        </div>
        
        <h2>Metric Details</h2>
        <div class="metrics-grid">
            {metric_cards}
        </div>
        
        <h2>Problem Samples</h2>
        <p>The following samples had the lowest scores and may need attention:</p>
        <table>
            <thead>
                <tr>
                    <th>Sample ID</th>
                    <th>Question</th>
                    <th>Avg Score</th>
                    <th>Issues</th>
                </tr>
            </thead>
            <tbody>
                {problem_rows or "<tr><td colspan='4'>No problem samples identified</td></tr>"}
            </tbody>
        </table>
        
        <h2>Recommendations</h2>
        <div class="recommendations">
            <ul>
                {recommendations_html}
            </ul>
        </div>
        
        <div class="footer">
            Generated by Enterprise Knowledge Agent Evaluation System
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def save_html(self, path: str) -> None:
        """Save report as HTML file."""
        Path(path).write_text(self.to_html(), encoding="utf-8")
        logger.info(f"HTML report saved to {path}")
    
    def save_json(self, path: str) -> None:
        """Save report as JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"JSON report saved to {path}")
    
    def to_markdown(self) -> str:
        """
        Generate Markdown report.
        
        Returns:
            Markdown string.
        """
        summary = self.get_summary()
        problems = self.get_problem_samples(limit=5)
        recommendations = self.get_recommendations()
        
        md = f"""# RAG Evaluation Report

**Dataset:** {summary["dataset_name"]}  
**Generated:** {summary["generated_at"]}  
**Duration:** {summary["duration_seconds"]:.2f}s  

## Executive Summary

| Metric | Value |
|--------|-------|
| Samples Evaluated | {summary["num_samples"]} |
| Overall Health | {summary["overall_health"].replace("_", " ").title()} |
| Average Score | {summary["average_score"]:.1%} |
| Metrics Passed | {summary["passed_metrics"]}/{summary["total_metrics"]} |

## Metric Details

"""
        for metric in summary["metric_summaries"]:
            status = "✅" if metric["passed"] else "❌"
            md += f"""### {metric["name"].replace("_", " ").title()} {status}

- **Score:** {metric["value"]:.2%}
- **Threshold:** {metric["threshold"]:.0%}
- **Status:** {"Passed" if metric["passed"] else "Failed"}

{metric["interpretation"]}

"""
        
        md += """## Problem Samples

| Sample ID | Question | Avg Score | Issues |
|-----------|----------|-----------|--------|
"""
        for p in problems:
            issues = "; ".join(p["issues"]) or "None"
            md += f"| {p['sample_id']} | {p['question'][:50]}... | {p['average_score']:.1%} | {issues} |\n"
        
        md += """
## Recommendations

"""
        for i, rec in enumerate(recommendations, 1):
            md += f"{i}. {rec}\n\n"
        
        return md
    
    def save_markdown(self, path: str) -> None:
        """Save report as Markdown file."""
        Path(path).write_text(self.to_markdown(), encoding="utf-8")
        logger.info(f"Markdown report saved to {path}")


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_html_report(
    result: "EvaluationResult",
    output_path: str,
    thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """
    Generate and save an HTML report.
    
    Args:
        result: Evaluation result.
        output_path: Output file path.
        thresholds: Optional custom thresholds.
    """
    report = EvaluationReport(result, thresholds)
    report.save_html(output_path)


def generate_json_report(
    result: "EvaluationResult",
    output_path: str,
    thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """
    Generate and save a JSON report.
    
    Args:
        result: Evaluation result.
        output_path: Output file path.
        thresholds: Optional custom thresholds.
    """
    report = EvaluationReport(result, thresholds)
    report.save_json(output_path)
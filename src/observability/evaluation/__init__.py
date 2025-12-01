"""
Evaluation Submodule

This module provides RAG evaluation capabilities using the Ragas framework
and custom evaluation metrics.

Components:
    - RagasEvaluator: Ragas framework integration
    - EvaluationDataset: Dataset management for evaluation
    - EvaluationReport: Report generation and export

Evaluation Metrics:
    1. Context Relevance: How relevant is the retrieved context to the question?
    2. Faithfulness: Is the answer grounded in the provided context?
    3. Answer Relevance: How relevant is the answer to the question?
    4. Groundedness: Are all claims in the answer supported by the context?

Usage:
    ```python
    from src.observability.evaluation import RagasEvaluator, EvaluationDataset

    # Create dataset
    dataset = EvaluationDataset()
    dataset.add_sample(
        question="What is machine learning?",
        answer="Machine learning is a subset of AI...",
        contexts=["Machine learning is a branch of..."],
        ground_truth="Machine learning is..."  # Optional
    )

    # Run evaluation
    evaluator = RagasEvaluator()
    results = await evaluator.evaluate(dataset)

    # Generate report
    report = evaluator.generate_report(results)
    report.save("evaluation_report.html")
    ```
"""

from src.observability.evaluation.ragas import (
    RagasEvaluator,
    EvaluationMetrics,
    EvaluationResult,
)
from src.observability.evaluation.datasets import (
    EvaluationDataset,
    EvaluationSample,
    load_dataset_from_file,
    create_synthetic_dataset,
)
from src.observability.evaluation.reports import (
    EvaluationReport,
    generate_html_report,
    generate_json_report,
)

__all__ = [
    # Evaluator
    "RagasEvaluator",
    "EvaluationMetrics",
    "EvaluationResult",
    # Datasets
    "EvaluationDataset",
    "EvaluationSample",
    "load_dataset_from_file",
    "create_synthetic_dataset",
    # Reports
    "EvaluationReport",
    "generate_html_report",
    "generate_json_report",
]
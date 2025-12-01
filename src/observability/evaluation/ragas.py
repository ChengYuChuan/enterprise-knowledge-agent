"""
Ragas Evaluation Framework Integration

This module provides integration with the Ragas framework for evaluating
RAG (Retrieval-Augmented Generation) system quality.

Ragas (RAG Assessment) is an open-source framework that provides metrics
specifically designed for evaluating RAG pipelines:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                     RAG Evaluation Pipeline                          │
    │                                                                      │
    │   Question ──────────────────────────────────────────────┐          │
    │       │                                                   │          │
    │       ▼                                                   │          │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │          │
    │  │  Retriever  │───▶│  Generator  │───▶│   Answer    │   │          │
    │  └─────────────┘    └─────────────┘    └─────────────┘   │          │
    │       │                                      │            │          │
    │       ▼                                      ▼            ▼          │
    │   Contexts                              Generated    Ground Truth    │
    │       │                                  Answer       (optional)     │
    │       │                                      │            │          │
    │       └──────────────────┬───────────────────┴────────────┘          │
    │                          │                                           │
    │                          ▼                                           │
    │                  ┌──────────────┐                                    │
    │                  │    Ragas     │                                    │
    │                  │  Evaluation  │                                    │
    │                  └──────────────┘                                    │
    │                          │                                           │
    │                          ▼                                           │
    │    ┌─────────────────────────────────────────────────────────────┐  │
    │    │  Metrics:                                                    │  │
    │    │  • Context Relevance  • Faithfulness  • Answer Relevance    │  │
    │    │  • Context Precision  • Context Recall • Answer Similarity  │  │
    │    └─────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────┘

Core Metrics:
    1. **Context Relevance**: Measures how relevant the retrieved context is
       to the given question. Low scores indicate retrieval issues.
    
    2. **Faithfulness**: Measures whether the generated answer is faithful to
       the provided context (i.e., no hallucinations). This is critical for
       ensuring the RAG system doesn't make up information.
    
    3. **Answer Relevance**: Measures how relevant the generated answer is to
       the question. Low scores indicate generation issues.
    
    4. **Context Precision**: Measures the proportion of relevant items in the
       retrieved context (ranking quality at the top).
    
    5. **Context Recall**: Measures how much of the relevant information from
       ground truth is present in the retrieved context.

Usage:
    ```python
    from src.observability.evaluation.ragas import RagasEvaluator
    from src.observability.evaluation.datasets import EvaluationDataset

    # Prepare dataset
    dataset = EvaluationDataset()
    dataset.add_sample(
        question="What is the capital of France?",
        answer="The capital of France is Paris.",
        contexts=["Paris is the capital and largest city of France."],
        ground_truth="Paris"
    )

    # Initialize evaluator
    evaluator = RagasEvaluator(
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )

    # Run evaluation
    results = await evaluator.evaluate(dataset)

    # Access scores
    print(f"Faithfulness: {results.faithfulness:.3f}")
    print(f"Answer Relevance: {results.answer_relevance:.3f}")
    print(f"Context Relevance: {results.context_relevance:.3f}")
    ```

Requirements:
    pip install ragas datasets

References:
    - Ragas Documentation: https://docs.ragas.io/
    - Ragas Paper: https://arxiv.org/abs/2309.15217
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.observability.evaluation.datasets import EvaluationDataset


logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class EvaluationMetricType(Enum):
    """Types of evaluation metrics."""
    
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_RELEVANCE = "context_relevance"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_ENTITY_RECALL = "context_entity_recall"
    ANSWER_SIMILARITY = "answer_similarity"
    ANSWER_CORRECTNESS = "answer_correctness"


# Default thresholds for metric quality assessment
DEFAULT_THRESHOLDS = {
    EvaluationMetricType.FAITHFULNESS: 0.85,
    EvaluationMetricType.ANSWER_RELEVANCE: 0.80,
    EvaluationMetricType.CONTEXT_RELEVANCE: 0.75,
    EvaluationMetricType.CONTEXT_PRECISION: 0.75,
    EvaluationMetricType.CONTEXT_RECALL: 0.70,
    EvaluationMetricType.ANSWER_SIMILARITY: 0.80,
    EvaluationMetricType.ANSWER_CORRECTNESS: 0.75,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvaluationMetrics:
    """
    Container for evaluation metric scores.
    
    Attributes:
        faithfulness: Score for answer faithfulness to context (0-1).
        answer_relevance: Score for answer relevance to question (0-1).
        context_relevance: Score for context relevance to question (0-1).
        context_precision: Score for context precision (0-1).
        context_recall: Score for context recall (0-1).
        context_entity_recall: Score for entity recall (0-1).
        answer_similarity: Semantic similarity to ground truth (0-1).
        answer_correctness: Factual correctness score (0-1).
    """
    
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_relevance: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    context_entity_recall: Optional[float] = None
    answer_similarity: Optional[float] = None
    answer_correctness: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary."""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_relevance": self.context_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "context_entity_recall": self.context_entity_recall,
            "answer_similarity": self.answer_similarity,
            "answer_correctness": self.answer_correctness,
        }
    
    def get_available_metrics(self) -> Dict[str, float]:
        """Get only metrics that have values."""
        return {k: v for k, v in self.to_dict().items() if v is not None}
    
    def get_average_score(self) -> float:
        """Calculate average of all available metrics."""
        available = self.get_available_metrics()
        if not available:
            return 0.0
        return sum(available.values()) / len(available)
    
    def passes_thresholds(
        self, 
        thresholds: Optional[Dict[EvaluationMetricType, float]] = None
    ) -> Dict[str, bool]:
        """
        Check if metrics pass the given thresholds.
        
        Args:
            thresholds: Dictionary of metric type to threshold value.
                       Uses defaults if not provided.
        
        Returns:
            Dictionary of metric name to pass/fail status.
        """
        thresholds = thresholds or DEFAULT_THRESHOLDS
        results = {}
        
        metric_mapping = {
            "faithfulness": EvaluationMetricType.FAITHFULNESS,
            "answer_relevance": EvaluationMetricType.ANSWER_RELEVANCE,
            "context_relevance": EvaluationMetricType.CONTEXT_RELEVANCE,
            "context_precision": EvaluationMetricType.CONTEXT_PRECISION,
            "context_recall": EvaluationMetricType.CONTEXT_RECALL,
            "answer_similarity": EvaluationMetricType.ANSWER_SIMILARITY,
            "answer_correctness": EvaluationMetricType.ANSWER_CORRECTNESS,
        }
        
        for metric_name, metric_type in metric_mapping.items():
            value = getattr(self, metric_name)
            if value is not None and metric_type in thresholds:
                results[metric_name] = value >= thresholds[metric_type]
        
        return results


@dataclass
class SampleEvaluationResult:
    """
    Evaluation result for a single sample.
    
    Attributes:
        sample_id: Identifier for the sample.
        question: The question being evaluated.
        metrics: Evaluation metrics for this sample.
        raw_scores: Raw score data from Ragas.
        error: Any error that occurred during evaluation.
    """
    
    sample_id: str
    question: str
    metrics: EvaluationMetrics
    raw_scores: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "metrics": self.metrics.to_dict(),
            "raw_scores": self.raw_scores,
            "error": self.error,
        }


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a dataset.
    
    Attributes:
        dataset_name: Name of the evaluated dataset.
        num_samples: Number of samples evaluated.
        aggregate_metrics: Aggregated metrics across all samples.
        sample_results: Individual results for each sample.
        config: Configuration used for evaluation.
        duration_seconds: Time taken for evaluation.
    """
    
    dataset_name: str
    num_samples: int
    aggregate_metrics: EvaluationMetrics
    sample_results: List[SampleEvaluationResult] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    
    def get_failed_samples(
        self, 
        thresholds: Optional[Dict[EvaluationMetricType, float]] = None
    ) -> List[SampleEvaluationResult]:
        """
        Get samples that failed to meet thresholds.
        
        Args:
            thresholds: Threshold values for each metric.
        
        Returns:
            List of failed sample results.
        """
        failed = []
        for sample in self.sample_results:
            if sample.error:
                failed.append(sample)
            elif not all(sample.metrics.passes_thresholds(thresholds).values()):
                failed.append(sample)
        return failed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dataset_name": self.dataset_name,
            "num_samples": self.num_samples,
            "aggregate_metrics": self.aggregate_metrics.to_dict(),
            "sample_results": [
                {
                    "sample_id": s.sample_id,
                    "question": s.question,
                    "metrics": s.metrics.to_dict(),
                    "error": s.error,
                }
                for s in self.sample_results
            ],
            "config": self.config,
            "duration_seconds": self.duration_seconds,
        }


# =============================================================================
# Ragas Evaluator
# =============================================================================

class RagasEvaluator:
    """
    Evaluator using the Ragas framework.
    
    This class provides methods for evaluating RAG system quality using
    the Ragas metrics. It supports both synchronous and asynchronous
    evaluation modes.
    
    Attributes:
        llm_provider: LLM provider for evaluation (e.g., "openai").
        llm_model: Model to use for evaluation.
        embedding_model: Embedding model for similarity metrics.
        metrics: List of metrics to compute.
    
    Example:
        ```python
        evaluator = RagasEvaluator(
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            metrics=[
                EvaluationMetricType.FAITHFULNESS,
                EvaluationMetricType.ANSWER_RELEVANCE,
                EvaluationMetricType.CONTEXT_RELEVANCE,
            ]
        )
        
        results = await evaluator.evaluate(dataset)
        ```
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        metrics: Optional[List[EvaluationMetricType]] = None,
    ):
        """
        Initialize the Ragas evaluator.
        
        Args:
            llm_provider: LLM provider name.
            llm_model: Model to use for evaluation.
            embedding_model: Embedding model for similarity metrics.
            metrics: List of metrics to compute. Defaults to core metrics.
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # Default metrics if not specified
        self.metrics = metrics or [
            EvaluationMetricType.FAITHFULNESS,
            EvaluationMetricType.ANSWER_RELEVANCE,
            EvaluationMetricType.CONTEXT_RELEVANCE,
        ]
        
        self._ragas_available = False
        self._ragas_metrics = []
        self._initialize_ragas()
    
    def _initialize_ragas(self) -> None:
        """Initialize Ragas and configure metrics."""
        try:
            # Try to import Ragas
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            
            self._ragas_available = True
            
            # Map our metric types to Ragas metrics
            metric_mapping = {
                EvaluationMetricType.FAITHFULNESS: faithfulness,
                EvaluationMetricType.ANSWER_RELEVANCE: answer_relevancy,
                EvaluationMetricType.CONTEXT_PRECISION: context_precision,
                EvaluationMetricType.CONTEXT_RECALL: context_recall,
            }
            
            # Build list of Ragas metrics to use
            for metric_type in self.metrics:
                if metric_type in metric_mapping:
                    self._ragas_metrics.append(metric_mapping[metric_type])
            
            logger.info(
                f"Ragas evaluator initialized with {len(self._ragas_metrics)} metrics"
            )
            
        except ImportError as e:
            logger.warning(
                f"Ragas not available: {e}. "
                "Install with: pip install ragas"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Ragas: {e}")
    
    async def evaluate(
        self,
        dataset: "EvaluationDataset",
        batch_size: int = 10,
    ) -> EvaluationResult:
        """
        Evaluate a dataset using Ragas metrics.
        
        Args:
            dataset: Dataset to evaluate.
            batch_size: Number of samples to process at once.
        
        Returns:
            EvaluationResult with aggregate and per-sample metrics.
        """
        import time
        start_time = time.time()
        
        if not self._ragas_available:
            return self._evaluate_fallback(dataset)
        
        try:
            from ragas import evaluate
            from datasets import Dataset
            
            # Convert to Ragas dataset format
            ragas_data = self._convert_to_ragas_format(dataset)
            hf_dataset = Dataset.from_dict(ragas_data)
            
            # Run evaluation
            result = evaluate(
                dataset=hf_dataset,
                metrics=self._ragas_metrics,
            )
            
            # Parse results
            duration = time.time() - start_time
            return self._parse_ragas_result(dataset, result, duration)
            
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            return self._evaluate_fallback(dataset)
    
    def evaluate_sync(
        self,
        dataset: "EvaluationDataset",
    ) -> EvaluationResult:
        """
        Synchronous version of evaluate.
        
        Args:
            dataset: Dataset to evaluate.
        
        Returns:
            EvaluationResult with aggregate and per-sample metrics.
        """
        import asyncio
        return asyncio.run(self.evaluate(dataset))
    
    def _convert_to_ragas_format(
        self, 
        dataset: "EvaluationDataset"
    ) -> Dict[str, List]:
        """
        Convert our dataset format to Ragas format.
        
        Ragas expects:
        - question: List of questions
        - answer: List of generated answers
        - contexts: List of lists of context strings
        - ground_truth: List of ground truth answers (optional)
        """
        return {
            "question": [s.question for s in dataset.samples],
            "answer": [s.answer for s in dataset.samples],
            "contexts": [s.contexts for s in dataset.samples],
            "ground_truth": [
                s.ground_truth or "" for s in dataset.samples
            ],
        }
    
    def _parse_ragas_result(
        self,
        dataset: "EvaluationDataset",
        ragas_result: Any,
        duration: float,
    ) -> EvaluationResult:
        """Parse Ragas evaluation result into our format."""
        # Extract aggregate scores
        aggregate_metrics = EvaluationMetrics(
            faithfulness=ragas_result.get("faithfulness"),
            answer_relevance=ragas_result.get("answer_relevancy"),
            context_precision=ragas_result.get("context_precision"),
            context_recall=ragas_result.get("context_recall"),
        )
        
        # Build sample results
        sample_results = []
        df = ragas_result.to_pandas()
        
        for i, sample in enumerate(dataset.samples):
            row = df.iloc[i] if i < len(df) else {}
            
            sample_metrics = EvaluationMetrics(
                faithfulness=row.get("faithfulness"),
                answer_relevance=row.get("answer_relevancy"),
                context_precision=row.get("context_precision"),
                context_recall=row.get("context_recall"),
            )
            
            sample_results.append(SampleEvaluationResult(
                sample_id=sample.id,
                question=sample.question,
                metrics=sample_metrics,
                raw_scores=row.to_dict() if hasattr(row, 'to_dict') else {},
            ))
        
        return EvaluationResult(
            dataset_name=dataset.name,
            num_samples=len(dataset.samples),
            aggregate_metrics=aggregate_metrics,
            sample_results=sample_results,
            config={
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "metrics": [m.value for m in self.metrics],
            },
            duration_seconds=duration,
        )
    
    def _evaluate_fallback(
        self, 
        dataset: "EvaluationDataset"
    ) -> EvaluationResult:
        """
        Fallback evaluation when Ragas is not available.
        
        Uses simple heuristics for basic quality assessment.
        """
        import time
        start_time = time.time()
        
        sample_results = []
        
        for sample in dataset.samples:
            # Simple heuristic metrics
            # Context relevance: Check keyword overlap
            context_text = " ".join(sample.contexts).lower()
            question_words = set(sample.question.lower().split())
            context_words = set(context_text.split())
            context_relevance = len(question_words & context_words) / max(len(question_words), 1)
            
            # Answer relevance: Check if answer mentions key question terms
            answer_words = set(sample.answer.lower().split())
            answer_relevance = len(question_words & answer_words) / max(len(question_words), 1)
            
            # Faithfulness: Check if answer terms appear in context
            answer_unique = answer_words - question_words
            faithfulness = len(answer_unique & context_words) / max(len(answer_unique), 1)
            
            sample_metrics = EvaluationMetrics(
                faithfulness=min(faithfulness, 1.0),
                answer_relevance=min(answer_relevance, 1.0),
                context_relevance=min(context_relevance, 1.0),
            )
            
            sample_results.append(SampleEvaluationResult(
                sample_id=sample.id,
                question=sample.question,
                metrics=sample_metrics,
            ))
        
        # Compute aggregates
        if sample_results:
            aggregate_metrics = EvaluationMetrics(
                faithfulness=sum(
                    s.metrics.faithfulness or 0 for s in sample_results
                ) / len(sample_results),
                answer_relevance=sum(
                    s.metrics.answer_relevance or 0 for s in sample_results
                ) / len(sample_results),
                context_relevance=sum(
                    s.metrics.context_relevance or 0 for s in sample_results
                ) / len(sample_results),
            )
        else:
            aggregate_metrics = EvaluationMetrics()
        
        duration = time.time() - start_time
        
        return EvaluationResult(
            dataset_name=dataset.name,
            num_samples=len(dataset.samples),
            aggregate_metrics=aggregate_metrics,
            sample_results=sample_results,
            config={
                "mode": "fallback",
                "note": "Ragas not available, using simple heuristics",
            },
            duration_seconds=duration,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

async def quick_evaluate(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: Optional[List[str]] = None,
) -> EvaluationMetrics:
    """
    Quick evaluation without creating a full dataset.
    
    Args:
        questions: List of questions.
        answers: List of generated answers.
        contexts: List of context lists.
        ground_truths: Optional list of ground truth answers.
    
    Returns:
        Aggregate EvaluationMetrics.
    """
    from src.observability.evaluation.datasets import EvaluationDataset
    
    dataset = EvaluationDataset(name="quick_eval")
    
    for i, (q, a, c) in enumerate(zip(questions, answers, contexts)):
        gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
        dataset.add_sample(
            question=q,
            answer=a,
            contexts=c,
            ground_truth=gt,
        )
    
    evaluator = RagasEvaluator()
    result = await evaluator.evaluate(dataset)
    
    return result.aggregate_metrics
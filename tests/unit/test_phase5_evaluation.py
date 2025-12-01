"""
Unit tests for Phase 5: Evaluation Module

Tests the Ragas evaluation framework integration, datasets management,
and report generation.

Run with:
    poetry run pytest tests/unit/test_phase5_evaluation.py -v
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import List

from src.observability.evaluation.datasets import (
    EvaluationSample,
    EvaluationDataset,
    DatasetMetadata,
    create_synthetic_dataset,
    merge_datasets,
    load_dataset_from_file,
)
from src.observability.evaluation.ragas import (
    EvaluationMetricType,
    EvaluationMetrics,
    SampleEvaluationResult,
    EvaluationResult,
    RagasEvaluator,
    DEFAULT_THRESHOLDS,
)


# =============================================================================
# Test EvaluationSample
# =============================================================================

class TestEvaluationSample:
    """Test EvaluationSample data class."""
    
    def test_basic_creation(self):
        """Test creating a sample with required fields."""
        sample = EvaluationSample(
            question="What is AI?",
            answer="AI stands for Artificial Intelligence.",
            contexts=["AI is a field of computer science."],
        )
        
        assert sample.question == "What is AI?"
        assert sample.answer == "AI stands for Artificial Intelligence."
        assert len(sample.contexts) == 1
        assert sample.ground_truth is None
        assert sample.id is not None  # Auto-generated
    
    def test_with_ground_truth(self):
        """Test sample with ground truth."""
        sample = EvaluationSample(
            question="What is ML?",
            answer="ML is Machine Learning.",
            contexts=["ML is a subset of AI."],
            ground_truth="Machine Learning is a type of AI.",
        )
        
        assert sample.ground_truth == "Machine Learning is a type of AI."
    
    def test_with_metadata(self):
        """Test sample with metadata."""
        sample = EvaluationSample(
            question="Test?",
            answer="Answer",
            contexts=["Context"],
            metadata={"source": "test", "difficulty": "easy"},
        )
        
        assert sample.metadata["source"] == "test"
        assert sample.metadata["difficulty"] == "easy"
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        sample = EvaluationSample(
            id="test-id",
            question="Q?",
            answer="A",
            contexts=["C1", "C2"],
            ground_truth="GT",
        )
        
        result = sample.to_dict()
        
        assert result["id"] == "test-id"
        assert result["question"] == "Q?"
        assert result["answer"] == "A"
        assert result["contexts"] == ["C1", "C2"]
        assert result["ground_truth"] == "GT"
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "sample-123",
            "question": "What is Python?",
            "answer": "A programming language.",
            "contexts": ["Python is popular."],
            "ground_truth": "Python is a language.",
        }
        
        sample = EvaluationSample.from_dict(data)
        
        assert sample.id == "sample-123"
        assert sample.question == "What is Python?"
        assert sample.answer == "A programming language."
    
    def test_validate_valid_sample(self):
        """Test validation of valid sample."""
        sample = EvaluationSample(
            question="Valid question?",
            answer="Valid answer.",
            contexts=["Valid context."],
        )
        
        errors = sample.validate()
        assert errors == []
    
    def test_validate_empty_question(self):
        """Test validation catches empty question."""
        sample = EvaluationSample(
            question="",
            answer="Answer",
            contexts=["Context"],
        )
        
        errors = sample.validate()
        assert "Question is empty" in errors
    
    def test_validate_empty_answer(self):
        """Test validation catches empty answer."""
        sample = EvaluationSample(
            question="Question?",
            answer="   ",  # Whitespace only
            contexts=["Context"],
        )
        
        errors = sample.validate()
        assert "Answer is empty" in errors
    
    def test_validate_no_contexts(self):
        """Test validation catches missing contexts."""
        sample = EvaluationSample(
            question="Question?",
            answer="Answer",
            contexts=[],
        )
        
        errors = sample.validate()
        assert "No contexts provided" in errors


# =============================================================================
# Test EvaluationDataset
# =============================================================================

class TestEvaluationDataset:
    """Test EvaluationDataset class."""
    
    def test_empty_dataset(self):
        """Test creating an empty dataset."""
        dataset = EvaluationDataset(name="test_dataset")
        
        assert dataset.name == "test_dataset"
        assert len(dataset) == 0
        assert len(dataset.samples) == 0
    
    def test_add_sample(self):
        """Test adding samples to dataset."""
        dataset = EvaluationDataset(name="test")
        
        dataset.add_sample(
            question="Q1?",
            answer="A1",
            contexts=["C1"],
        )
        dataset.add_sample(
            question="Q2?",
            answer="A2",
            contexts=["C2"],
            ground_truth="GT2",
        )
        
        assert len(dataset) == 2
        assert dataset[0].question == "Q1?"
        assert dataset[1].ground_truth == "GT2"
    
    def test_iteration(self):
        """Test iterating over dataset."""
        dataset = EvaluationDataset(name="test")
        dataset.add_sample(question="Q1", answer="A1", contexts=["C1"])
        dataset.add_sample(question="Q2", answer="A2", contexts=["C2"])
        
        questions = [s.question for s in dataset]
        
        assert questions == ["Q1", "Q2"]
    
    def test_indexing(self):
        """Test indexing dataset samples."""
        dataset = EvaluationDataset(name="test")
        dataset.add_sample(question="Q1", answer="A1", contexts=["C1"])
        dataset.add_sample(question="Q2", answer="A2", contexts=["C2"])
        
        assert dataset[0].question == "Q1"
        assert dataset[1].question == "Q2"
        assert dataset[-1].question == "Q2"
    
    def test_slicing(self):
        """Test slicing dataset."""
        dataset = EvaluationDataset(name="test")
        for i in range(5):
            dataset.add_sample(
                question=f"Q{i}",
                answer=f"A{i}",
                contexts=[f"C{i}"],
            )
        
        sliced = dataset[:3]
        assert len(sliced) == 3
    
    def test_save_and_load_json(self):
        """Test saving and loading JSON format."""
        dataset = EvaluationDataset(name="json_test")
        dataset.add_sample(
            question="What is Python?",
            answer="A programming language.",
            contexts=["Python is popular."],
            ground_truth="Python is a language.",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dataset.json"
            dataset.save(path, format="json")
            
            # Verify file exists
            assert path.exists()
            
            # Load and verify
            loaded = EvaluationDataset.load(path)
            assert loaded.name == "json_test"
            assert len(loaded) == 1
            assert loaded[0].question == "What is Python?"
    
    def test_save_and_load_jsonl(self):
        """Test saving and loading JSONL format."""
        dataset = EvaluationDataset(name="jsonl_test")
        dataset.add_sample(question="Q1", answer="A1", contexts=["C1"])
        dataset.add_sample(question="Q2", answer="A2", contexts=["C2"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dataset.jsonl"
            dataset.save(path, format="jsonl")
            
            # Verify file content
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            
            # Load and verify
            loaded = EvaluationDataset.load(path)
            assert len(loaded) == 2
    
    def test_save_and_load_csv(self):
        """Test saving and loading CSV format."""
        dataset = EvaluationDataset(name="csv_test")
        dataset.add_sample(
            question="Q1?",
            answer="A1",
            contexts=["C1", "C2"],
            ground_truth="GT1",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dataset.csv"
            dataset.save(path, format="csv")
            
            # Load and verify
            loaded = EvaluationDataset.load(path)
            assert len(loaded) == 1
            assert loaded[0].question == "Q1?"
            assert loaded[0].contexts == ["C1", "C2"]
    
    def test_to_dict(self):
        """Test converting dataset to dictionary."""
        dataset = EvaluationDataset(
            name="dict_test",
            description="Test description",
        )
        dataset.add_sample(question="Q", answer="A", contexts=["C"])
        
        result = dataset.to_dict()
        
        assert "metadata" in result
        assert "samples" in result
        assert result["metadata"]["name"] == "dict_test"
        assert len(result["samples"]) == 1
    
    def test_from_dict(self):
        """Test creating dataset from dictionary."""
        data = {
            "metadata": {
                "name": "from_dict_test",
                "description": "Created from dict",
            },
            "samples": [
                {
                    "id": "s1",
                    "question": "Q?",
                    "answer": "A",
                    "contexts": ["C"],
                }
            ],
        }
        
        dataset = EvaluationDataset.from_dict(data)
        
        assert dataset.name == "from_dict_test"
        assert len(dataset) == 1


# =============================================================================
# Test Synthetic Dataset Creation
# =============================================================================

class TestSyntheticDataset:
    """Test synthetic dataset generation."""
    
    def test_create_default(self):
        """Test creating synthetic dataset with defaults."""
        dataset = create_synthetic_dataset(num_samples=10)
        
        assert len(dataset) == 10
        assert dataset.name == "synthetic_eval_dataset"
        assert dataset.metadata.source == "synthetic"
    
    def test_create_with_custom_topics(self):
        """Test creating with custom topics."""
        topics = ["quantum computing", "blockchain"]
        dataset = create_synthetic_dataset(
            num_samples=5,
            topics=topics,
        )
        
        assert len(dataset) == 5
        # Check that topics are used
        for sample in dataset:
            assert sample.metadata.get("topic") in topics
    
    def test_create_without_ground_truth(self):
        """Test creating without ground truth."""
        dataset = create_synthetic_dataset(
            num_samples=3,
            include_ground_truth=False,
        )
        
        for sample in dataset:
            assert sample.ground_truth is None
    
    def test_all_samples_valid(self):
        """Test that all generated samples are valid."""
        dataset = create_synthetic_dataset(num_samples=20)
        
        for sample in dataset:
            errors = sample.validate()
            assert errors == [], f"Sample {sample.id} invalid: {errors}"


# =============================================================================
# Test Dataset Merging
# =============================================================================

class TestMergeDatasets:
    """Test dataset merging functionality."""
    
    def test_merge_two_datasets(self):
        """Test merging two datasets."""
        ds1 = EvaluationDataset(name="dataset1")
        ds1.add_sample(question="Q1", answer="A1", contexts=["C1"])
        
        ds2 = EvaluationDataset(name="dataset2")
        ds2.add_sample(question="Q2", answer="A2", contexts=["C2"])
        ds2.add_sample(question="Q3", answer="A3", contexts=["C3"])
        
        merged = merge_datasets([ds1, ds2], name="merged")
        
        assert len(merged) == 3
        assert merged.name == "merged"
        assert merged.metadata.source == "merged"
    
    def test_merge_preserves_samples(self):
        """Test that merging preserves all sample data."""
        ds1 = EvaluationDataset(name="ds1")
        ds1.add_sample(
            question="Specific Q?",
            answer="Specific A",
            contexts=["Specific C"],
            ground_truth="Specific GT",
        )
        
        ds2 = EvaluationDataset(name="ds2")
        ds2.add_sample(question="Q2", answer="A2", contexts=["C2"])
        
        merged = merge_datasets([ds1, ds2])
        
        # Find the specific sample
        found = [s for s in merged if s.question == "Specific Q?"]
        assert len(found) == 1
        assert found[0].ground_truth == "Specific GT"


# =============================================================================
# Test EvaluationMetrics
# =============================================================================

class TestEvaluationMetrics:
    """Test EvaluationMetrics data class."""
    
    def test_default_values(self):
        """Test default metric values are None."""
        metrics = EvaluationMetrics()
        
        assert metrics.faithfulness is None
        assert metrics.answer_relevance is None
        assert metrics.context_relevance is None
    
    def test_with_values(self):
        """Test metrics with values."""
        metrics = EvaluationMetrics(
            faithfulness=0.95,
            answer_relevance=0.88,
            context_relevance=0.75,
        )
        
        assert metrics.faithfulness == 0.95
        assert metrics.answer_relevance == 0.88
        assert metrics.context_relevance == 0.75
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = EvaluationMetrics(
            faithfulness=0.9,
            answer_relevance=0.8,
        )
        
        result = metrics.to_dict()
        
        assert result["faithfulness"] == 0.9
        assert result["answer_relevance"] == 0.8
        assert result["context_relevance"] is None
    
    def test_get_available_metrics(self):
        """Test getting only available metrics."""
        metrics = EvaluationMetrics(
            faithfulness=0.9,
            context_precision=0.85,
        )
        
        available = metrics.get_available_metrics()
        
        assert "faithfulness" in available
        assert "context_precision" in available
        assert "answer_relevance" not in available
        assert len(available) == 2
    
    def test_get_average_score(self):
        """Test calculating average score."""
        metrics = EvaluationMetrics(
            faithfulness=0.8,
            answer_relevance=0.9,
            context_relevance=0.7,
        )
        
        avg = metrics.get_average_score()
        
        assert abs(avg - 0.8) < 0.001  # (0.8 + 0.9 + 0.7) / 3 = 0.8
    
    def test_average_score_empty(self):
        """Test average score with no metrics."""
        metrics = EvaluationMetrics()
        
        assert metrics.get_average_score() == 0.0
    
    def test_passes_thresholds(self):
        """Test threshold checking."""
        metrics = EvaluationMetrics(
            faithfulness=0.9,  # Passes (threshold: 0.85)
            answer_relevance=0.7,  # Fails (threshold: 0.80)
            context_relevance=0.8,  # Passes (threshold: 0.75)
        )
        
        results = metrics.passes_thresholds()
        
        assert results["faithfulness"] is True
        assert results["answer_relevance"] is False
        assert results["context_relevance"] is True


# =============================================================================
# Test RagasEvaluator
# =============================================================================

class TestRagasEvaluator:
    """Test RagasEvaluator class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        evaluator = RagasEvaluator()
        
        assert evaluator.llm_provider == "openai"
        assert evaluator.llm_model == "gpt-4o-mini"
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        evaluator = RagasEvaluator(
            llm_provider="anthropic",
            llm_model="claude-3-sonnet",
        )
        
        assert evaluator.llm_provider == "anthropic"
        assert evaluator.llm_model == "claude-3-sonnet"
    
    def test_fallback_evaluation(self):
        """Test fallback evaluation when Ragas unavailable."""
        dataset = EvaluationDataset(name="fallback_test")
        dataset.add_sample(
            question="What is machine learning?",
            answer="Machine learning is a subset of artificial intelligence.",
            contexts=["Machine learning is a branch of AI that enables systems to learn."],
        )
        
        evaluator = RagasEvaluator()
        
        # Force fallback by calling internal method
        result = evaluator._evaluate_fallback(dataset)
        
        assert result.dataset_name == "fallback_test"
        assert result.num_samples == 1
        assert result.aggregate_metrics.faithfulness is not None
        assert result.aggregate_metrics.answer_relevance is not None
        assert result.aggregate_metrics.context_relevance is not None
        assert result.config.get("mode") == "fallback"
    
    def test_fallback_metrics_bounded(self):
        """Test that fallback metrics are between 0 and 1."""
        dataset = create_synthetic_dataset(num_samples=5)
        
        evaluator = RagasEvaluator()
        result = evaluator._evaluate_fallback(dataset)
        
        for sample_result in result.sample_results:
            m = sample_result.metrics
            if m.faithfulness is not None:
                assert 0 <= m.faithfulness <= 1
            if m.answer_relevance is not None:
                assert 0 <= m.answer_relevance <= 1
            if m.context_relevance is not None:
                assert 0 <= m.context_relevance <= 1


# =============================================================================
# Test EvaluationResult
# =============================================================================

class TestEvaluationResult:
    """Test EvaluationResult class."""
    
    def test_basic_creation(self):
        """Test creating evaluation result."""
        result = EvaluationResult(
            dataset_name="test",
            num_samples=10,
            aggregate_metrics=EvaluationMetrics(faithfulness=0.9),
            sample_results=[],
            config={"mode": "test"},
            duration_seconds=5.0,
        )
        
        assert result.dataset_name == "test"
        assert result.num_samples == 10
        assert result.duration_seconds == 5.0
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = EvaluationResult(
            dataset_name="dict_test",
            num_samples=5,
            aggregate_metrics=EvaluationMetrics(faithfulness=0.85),
            sample_results=[],
            config={},
            duration_seconds=2.5,
        )
        
        data = result.to_dict()
        
        assert data["dataset_name"] == "dict_test"
        assert data["num_samples"] == 5
        assert data["aggregate_metrics"]["faithfulness"] == 0.85


# =============================================================================
# Test SampleEvaluationResult
# =============================================================================

class TestSampleEvaluationResult:
    """Test SampleEvaluationResult class."""
    
    def test_basic_creation(self):
        """Test creating sample result."""
        sample_result = SampleEvaluationResult(
            sample_id="s1",
            question="Test question?",
            metrics=EvaluationMetrics(faithfulness=0.9),
        )
        
        assert sample_result.sample_id == "s1"
        assert sample_result.question == "Test question?"
        assert sample_result.metrics.faithfulness == 0.9
    
    def test_with_raw_scores(self):
        """Test sample result with raw scores."""
        sample_result = SampleEvaluationResult(
            sample_id="s2",
            question="Q?",
            metrics=EvaluationMetrics(),
            raw_scores={"custom_metric": 0.77},
        )
        
        assert sample_result.raw_scores["custom_metric"] == 0.77
    
    def test_to_dict(self):
        """Test serialization."""
        sample_result = SampleEvaluationResult(
            sample_id="s3",
            question="Serialize me?",
            metrics=EvaluationMetrics(answer_relevance=0.8),
        )
        
        data = sample_result.to_dict()
        
        assert data["sample_id"] == "s3"
        assert data["question"] == "Serialize me?"
        assert data["metrics"]["answer_relevance"] == 0.8


# =============================================================================
# Test Default Thresholds
# =============================================================================

class TestDefaultThresholds:
    """Test default threshold configuration."""
    
    def test_thresholds_exist(self):
        """Test that default thresholds are defined."""
        assert EvaluationMetricType.FAITHFULNESS in DEFAULT_THRESHOLDS
        assert EvaluationMetricType.ANSWER_RELEVANCE in DEFAULT_THRESHOLDS
        assert EvaluationMetricType.CONTEXT_RELEVANCE in DEFAULT_THRESHOLDS
    
    def test_thresholds_reasonable(self):
        """Test that thresholds are in reasonable range."""
        for metric, threshold in DEFAULT_THRESHOLDS.items():
            assert 0.5 <= threshold <= 1.0, f"{metric} threshold {threshold} out of range"
    
    def test_faithfulness_highest(self):
        """Test that faithfulness has high threshold (critical metric)."""
        assert DEFAULT_THRESHOLDS[EvaluationMetricType.FAITHFULNESS] >= 0.85

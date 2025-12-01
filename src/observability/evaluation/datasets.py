"""
Evaluation Dataset Management

This module provides classes and utilities for managing evaluation datasets
for RAG system quality assessment.

Dataset Structure:
    An evaluation dataset consists of samples, where each sample contains:
    - question: The input query
    - answer: The generated answer from the RAG system
    - contexts: The retrieved context documents
    - ground_truth: (Optional) The expected correct answer

Dataset Formats:
    The module supports multiple formats for loading and saving datasets:
    - JSON: Native format with full metadata support
    - CSV: Simple tabular format
    - JSONL: Line-delimited JSON for streaming
    - HuggingFace Datasets: Integration with datasets library

Usage:
    ```python
    from src.observability.evaluation.datasets import (
        EvaluationDataset,
        EvaluationSample,
        load_dataset_from_file,
    )

    # Create dataset manually
    dataset = EvaluationDataset(name="my_eval")
    dataset.add_sample(
        question="What is machine learning?",
        answer="Machine learning is a subset of AI...",
        contexts=["Machine learning is a branch of AI..."],
        ground_truth="Machine learning is a type of AI..."
    )

    # Save dataset
    dataset.save("eval_dataset.json")

    # Load dataset
    loaded = load_dataset_from_file("eval_dataset.json")

    # Create synthetic dataset for testing
    synthetic = create_synthetic_dataset(
        num_samples=100,
        topics=["AI", "ML", "NLP"]
    )
    ```
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvaluationSample:
    """
    A single evaluation sample.
    
    Attributes:
        id: Unique identifier for the sample.
        question: The input question/query.
        answer: The generated answer from the RAG system.
        contexts: List of retrieved context documents.
        ground_truth: Optional expected correct answer.
        metadata: Additional metadata (e.g., source, difficulty).
        created_at: Timestamp when the sample was created.
    """
    
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationSample":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            question=data["question"],
            answer=data["answer"],
            contexts=data.get("contexts", []),
            ground_truth=data.get("ground_truth"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )
    
    def validate(self) -> List[str]:
        """
        Validate the sample.
        
        Returns:
            List of validation error messages.
        """
        errors = []
        
        if not self.question or not self.question.strip():
            errors.append("Question is empty")
        
        if not self.answer or not self.answer.strip():
            errors.append("Answer is empty")
        
        if not self.contexts:
            errors.append("No contexts provided")
        
        return errors


@dataclass
class DatasetMetadata:
    """
    Metadata for an evaluation dataset.
    
    Attributes:
        name: Dataset name.
        description: Dataset description.
        version: Dataset version.
        created_at: Creation timestamp.
        source: Data source (e.g., "synthetic", "production", "manual").
        tags: List of tags for categorization.
        config: Configuration used to create the dataset.
    """
    
    name: str
    description: str = ""
    version: str = "1.0.0"
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    source: str = "manual"
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at,
            "source": self.source,
            "tags": self.tags,
            "config": self.config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            source=data.get("source", "manual"),
            tags=data.get("tags", []),
            config=data.get("config", {}),
        )


# =============================================================================
# Evaluation Dataset
# =============================================================================

class EvaluationDataset:
    """
    A collection of evaluation samples.
    
    This class provides methods for managing, filtering, and transforming
    evaluation datasets. It supports various I/O formats and provides
    iteration and indexing capabilities.
    
    Attributes:
        name: Dataset name.
        samples: List of evaluation samples.
        metadata: Dataset metadata.
    
    Example:
        ```python
        dataset = EvaluationDataset(name="rag_eval_v1")
        
        # Add samples
        dataset.add_sample(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language..."],
            ground_truth="Python is a general-purpose programming language."
        )
        
        # Filter samples
        filtered = dataset.filter(lambda s: len(s.contexts) >= 3)
        
        # Save and load
        dataset.save("dataset.json")
        loaded = EvaluationDataset.load("dataset.json")
        ```
    """
    
    def __init__(
        self,
        name: str = "evaluation_dataset",
        description: str = "",
        samples: Optional[List[EvaluationSample]] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            name: Dataset name.
            description: Dataset description.
            samples: Optional initial list of samples.
        """
        self.name = name
        self.samples: List[EvaluationSample] = samples or []
        self.metadata = DatasetMetadata(
            name=name,
            description=description,
        )
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __iter__(self) -> Iterator[EvaluationSample]:
        """Iterate over samples."""
        return iter(self.samples)
    
    def __getitem__(self, index: int) -> EvaluationSample:
        """Get sample by index."""
        return self.samples[index]
    
    def add_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sample_id: Optional[str] = None,
    ) -> EvaluationSample:
        """
        Add a new sample to the dataset.
        
        Args:
            question: The input question.
            answer: The generated answer.
            contexts: Retrieved context documents.
            ground_truth: Optional expected answer.
            metadata: Optional sample metadata.
            sample_id: Optional custom sample ID.
        
        Returns:
            The created EvaluationSample.
        """
        sample = EvaluationSample(
            id=sample_id or str(uuid.uuid4())[:8],
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            metadata=metadata or {},
        )
        self.samples.append(sample)
        return sample
    
    def add_samples(self, samples: List[EvaluationSample]) -> None:
        """
        Add multiple samples.
        
        Args:
            samples: List of samples to add.
        """
        self.samples.extend(samples)
    
    def remove_sample(self, sample_id: str) -> bool:
        """
        Remove a sample by ID.
        
        Args:
            sample_id: ID of the sample to remove.
        
        Returns:
            True if sample was removed, False if not found.
        """
        for i, sample in enumerate(self.samples):
            if sample.id == sample_id:
                del self.samples[i]
                return True
        return False
    
    def get_sample(self, sample_id: str) -> Optional[EvaluationSample]:
        """
        Get a sample by ID.
        
        Args:
            sample_id: ID of the sample to get.
        
        Returns:
            The sample, or None if not found.
        """
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        return None
    
    def filter(
        self, 
        predicate: callable
    ) -> "EvaluationDataset":
        """
        Filter samples based on a predicate function.
        
        Args:
            predicate: Function that takes a sample and returns bool.
        
        Returns:
            New dataset with filtered samples.
        """
        filtered_samples = [s for s in self.samples if predicate(s)]
        new_dataset = EvaluationDataset(
            name=f"{self.name}_filtered",
            samples=filtered_samples,
        )
        new_dataset.metadata = DatasetMetadata(
            name=new_dataset.name,
            description=f"Filtered from {self.name}",
            source="filtered",
            config={"original_dataset": self.name},
        )
        return new_dataset
    
    def split(
        self, 
        ratio: float = 0.8
    ) -> tuple["EvaluationDataset", "EvaluationDataset"]:
        """
        Split dataset into train and test sets.
        
        Args:
            ratio: Ratio of samples for the first set (0-1).
        
        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        import random
        
        shuffled = self.samples.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * ratio)
        train_samples = shuffled[:split_idx]
        test_samples = shuffled[split_idx:]
        
        train_dataset = EvaluationDataset(
            name=f"{self.name}_train",
            samples=train_samples,
        )
        test_dataset = EvaluationDataset(
            name=f"{self.name}_test",
            samples=test_samples,
        )
        
        return train_dataset, test_dataset
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate all samples in the dataset.
        
        Returns:
            Validation report with errors and statistics.
        """
        errors = []
        valid_count = 0
        
        for sample in self.samples:
            sample_errors = sample.validate()
            if sample_errors:
                errors.append({
                    "sample_id": sample.id,
                    "errors": sample_errors,
                })
            else:
                valid_count += 1
        
        return {
            "total_samples": len(self.samples),
            "valid_samples": valid_count,
            "invalid_samples": len(errors),
            "errors": errors,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics.
        """
        if not self.samples:
            return {"num_samples": 0}
        
        question_lengths = [len(s.question) for s in self.samples]
        answer_lengths = [len(s.answer) for s in self.samples]
        context_counts = [len(s.contexts) for s in self.samples]
        has_ground_truth = sum(1 for s in self.samples if s.ground_truth)
        
        return {
            "num_samples": len(self.samples),
            "has_ground_truth": has_ground_truth,
            "question_length": {
                "min": min(question_lengths),
                "max": max(question_lengths),
                "avg": sum(question_lengths) / len(question_lengths),
            },
            "answer_length": {
                "min": min(answer_lengths),
                "max": max(answer_lengths),
                "avg": sum(answer_lengths) / len(answer_lengths),
            },
            "context_count": {
                "min": min(context_counts),
                "max": max(context_counts),
                "avg": sum(context_counts) / len(context_counts),
            },
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "samples": [s.to_dict() for s in self.samples],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationDataset":
        """Create dataset from dictionary."""
        metadata = DatasetMetadata.from_dict(data.get("metadata", {}))
        samples = [
            EvaluationSample.from_dict(s) 
            for s in data.get("samples", [])
        ]
        
        dataset = cls(
            name=metadata.name,
            description=metadata.description,
            samples=samples,
        )
        dataset.metadata = metadata
        return dataset
    
    def save(self, path: Union[str, Path], format: str = "json") -> None:
        """
        Save dataset to file.
        
        Args:
            path: Output file path.
            format: File format (json, jsonl, csv).
        """
        path = Path(path)
        
        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        elif format == "jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for sample in self.samples:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False))
                    f.write("\n")
        
        elif format == "csv":
            import csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, 
                    fieldnames=["id", "question", "answer", "contexts", "ground_truth"]
                )
                writer.writeheader()
                for sample in self.samples:
                    writer.writerow({
                        "id": sample.id,
                        "question": sample.question,
                        "answer": sample.answer,
                        "contexts": json.dumps(sample.contexts),
                        "ground_truth": sample.ground_truth or "",
                    })
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Dataset saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvaluationDataset":
        """
        Load dataset from file.
        
        Args:
            path: Input file path.
        
        Returns:
            Loaded EvaluationDataset.
        """
        path = Path(path)
        
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        
        elif path.suffix == ".jsonl":
            samples = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(EvaluationSample.from_dict(json.loads(line)))
            return cls(name=path.stem, samples=samples)
        
        elif path.suffix == ".csv":
            import csv
            samples = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(EvaluationSample(
                        id=row.get("id", str(uuid.uuid4())[:8]),
                        question=row["question"],
                        answer=row["answer"],
                        contexts=json.loads(row.get("contexts", "[]")),
                        ground_truth=row.get("ground_truth") or None,
                    ))
            return cls(name=path.stem, samples=samples)
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


# =============================================================================
# Utility Functions
# =============================================================================

def load_dataset_from_file(path: Union[str, Path]) -> EvaluationDataset:
    """
    Load a dataset from file.
    
    Args:
        path: Path to the dataset file.
    
    Returns:
        Loaded EvaluationDataset.
    """
    return EvaluationDataset.load(path)


def create_synthetic_dataset(
    num_samples: int = 50,
    topics: Optional[List[str]] = None,
    include_ground_truth: bool = True,
) -> EvaluationDataset:
    """
    Create a synthetic dataset for testing.
    
    This generates fake samples with realistic-looking questions and answers
    for testing the evaluation pipeline.
    
    Args:
        num_samples: Number of samples to generate.
        topics: List of topics to generate questions about.
        include_ground_truth: Whether to include ground truth answers.
    
    Returns:
        Synthetic EvaluationDataset.
    """
    import random
    
    topics = topics or [
        "machine learning",
        "artificial intelligence", 
        "natural language processing",
        "computer vision",
        "deep learning",
        "neural networks",
        "data science",
        "python programming",
    ]
    
    question_templates = [
        "What is {topic}?",
        "How does {topic} work?",
        "What are the main applications of {topic}?",
        "What are the benefits of {topic}?",
        "How is {topic} used in industry?",
        "What are the challenges in {topic}?",
        "Explain {topic} in simple terms.",
        "What is the history of {topic}?",
    ]
    
    answer_templates = [
        "{topic} is a field of study that focuses on developing systems and algorithms.",
        "{topic} involves the use of computational methods to solve complex problems.",
        "The main concept behind {topic} is to enable machines to perform tasks.",
        "{topic} has numerous applications in various industries including healthcare, finance, and technology.",
    ]
    
    context_templates = [
        "{topic} is a rapidly evolving field with significant impact on modern technology.",
        "Research in {topic} has led to breakthroughs in automation and decision-making.",
        "The foundations of {topic} were established in the mid-20th century.",
        "{topic} combines theoretical concepts with practical applications.",
    ]
    
    dataset = EvaluationDataset(
        name="synthetic_eval_dataset",
        description="Synthetic dataset for testing evaluation pipeline",
    )
    dataset.metadata.source = "synthetic"
    dataset.metadata.config = {
        "num_samples": num_samples,
        "topics": topics,
        "include_ground_truth": include_ground_truth,
    }
    
    for i in range(num_samples):
        topic = random.choice(topics)
        question = random.choice(question_templates).format(topic=topic)
        answer = random.choice(answer_templates).format(topic=topic.title())
        
        # Generate 2-5 context strings
        num_contexts = random.randint(2, 5)
        contexts = [
            random.choice(context_templates).format(topic=topic.title())
            for _ in range(num_contexts)
        ]
        
        ground_truth = None
        if include_ground_truth:
            ground_truth = f"{topic.title()} is a comprehensive field encompassing various techniques and methodologies."
        
        dataset.add_sample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            metadata={
                "topic": topic,
                "synthetic": True,
            },
        )
    
    return dataset


def merge_datasets(
    datasets: List[EvaluationDataset],
    name: str = "merged_dataset",
) -> EvaluationDataset:
    """
    Merge multiple datasets into one.
    
    Args:
        datasets: List of datasets to merge.
        name: Name for the merged dataset.
    
    Returns:
        Merged EvaluationDataset.
    """
    all_samples = []
    source_names = []
    
    for dataset in datasets:
        all_samples.extend(dataset.samples)
        source_names.append(dataset.name)
    
    merged = EvaluationDataset(
        name=name,
        description=f"Merged from: {', '.join(source_names)}",
        samples=all_samples,
    )
    merged.metadata.source = "merged"
    merged.metadata.config = {"source_datasets": source_names}
    
    return merged
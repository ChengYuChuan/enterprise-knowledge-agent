"""
Mock embedder for testing purposes.

This module provides a simple mock embedder that generates embeddings
based on word frequency for testing. For production, use OpenAI or
other real embedding models.
"""

import re
from collections import Counter
from typing import Union


class MockEmbedder:
    """
    Mock embedder that generates simple word-frequency based embeddings.

    This creates embeddings based on word presence, allowing similar
    texts to have similar embeddings (unlike pure hash-based approach).

    Attributes:
        dimension: Embedding vector dimension.
    """

    def __init__(self, dimension: int = 1536) -> None:
        """
        Initialize the mock embedder.

        Args:
            dimension: Dimension of embedding vectors.
        """
        self.dimension = dimension
        # Build a simple vocabulary from common words
        self.vocab = self._build_vocab()

    def _build_vocab(self) -> dict[str, int]:
        """
        Build a simple vocabulary mapping.

        Returns:
            dict: Mapping from words to indices.
        """
        # Common English words that might appear in documents
        common_words = [
            "the", "is", "at", "which", "on", "a", "an", "as", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "should", "could", "may", "might", "must", "can",
            "shall", "employee", "employees", "work", "working", "company", "policy",
            "vacation", "days", "time", "year", "years", "remote", "office", "hours",
            "request", "approval", "manager", "benefits", "paid", "unpaid", "sick",
            "leave", "equipment", "laptop", "computer", "internet", "home", "available",
            "required", "please", "contact", "email", "phone", "new", "senior",
            "experienced", "full", "part", "time", "accrual", "accrue", "carry", "over",
        ]
        return {word: i for i, word in enumerate(common_words)}

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenization.

        Args:
            text: Text to tokenize.

        Returns:
            list[str]: List of tokens.
        """
        # Lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words

    def embed_text(self, text: str) -> list[float]:
        """
        Generate a simple embedding for text.

        Creates embedding based on word frequency and vocabulary.

        Args:
            text: Text to embed.

        Returns:
            list[float]: Embedding vector.
        """
        # Tokenize text
        words = self._tokenize(text)
        word_counts = Counter(words)

        # Initialize embedding with zeros
        embedding = [0.0] * self.dimension

        # Fill embedding based on word presence
        for word, count in word_counts.items():
            if word in self.vocab:
                idx = self.vocab[word] % self.dimension
                # Use normalized count
                embedding[idx] = min(count / len(words), 1.0)

        # Add some variance based on text length
        text_hash = hash(text) % 100
        for i in range(min(10, self.dimension)):
            embedding[i] += (text_hash / 1000.0)

        # Normalize the vector
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        return [self.embed_text(text) for text in texts]

    def embed(self, text: Union[str, list[str]]) -> Union[list[float], list[list[float]]]:
        """
        Generate embeddings (unified interface).

        Args:
            text: Single text or list of texts.

        Returns:
            Embedding vector(s).
        """
        if isinstance(text, str):
            return self.embed_text(text)
        else:
            return self.embed_batch(text)
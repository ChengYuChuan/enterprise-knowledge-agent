"""
Chunker factory for automatic strategy selection.

This module provides a factory to select and configure
the appropriate chunking strategy.
"""

from typing import Optional, Type

from .base_chunker import BaseChunker, Chunk
from .fixed_size_chunker import FixedSizeChunker
from .semantic_chunker import SemanticChunker
from .sentence_chunker import SentenceChunker


class ChunkerFactory:
    """
    Factory for creating chunkers based on strategy name.

    Supported strategies:
    - 'fixed': Fixed-size chunking
    - 'sentence': Sentence-based chunking
    - 'semantic': Semantic chunking (heuristic-based)
    """

    CHUNKER_MAP: dict[str, Type[BaseChunker]] = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "semantic": SemanticChunker,
    }

    @classmethod
    def create_chunker(
        cls,
        strategy: str = "semantic",
        chunk_size: Optional[int] = None,
        **kwargs,
    ) -> BaseChunker:
        """
        Create a chunker with the specified strategy.

        Args:
            strategy: Chunking strategy name ('fixed', 'sentence', 'semantic').
            chunk_size: Override default chunk size.
            **kwargs: Additional arguments for the specific chunker.

        Returns:
            BaseChunker: Configured chunker instance.

        Raises:
            ValueError: If strategy is not supported.
        """
        if strategy not in cls.CHUNKER_MAP:
            supported = ", ".join(cls.CHUNKER_MAP.keys())
            raise ValueError(
                f"Unsupported chunking strategy: {strategy}. "
                f"Supported strategies: {supported}"
            )

        chunker_class = cls.CHUNKER_MAP[strategy]

        # Configure based on strategy
        if strategy == "fixed":
            config = {
                "chunk_size": chunk_size or 512,
                "chunk_overlap": kwargs.get("chunk_overlap", 50),
            }
        elif strategy == "sentence":
            config = {
                "sentences_per_chunk": kwargs.get("sentences_per_chunk", 5),
                "max_chunk_size": chunk_size or 1000,
            }
        elif strategy == "semantic":
            config = {
                "target_chunk_size": chunk_size or 512,
                "max_chunk_size": kwargs.get("max_chunk_size", 1000),
            }
        else:
            config = {}

        return chunker_class(**config)

    @classmethod
    def chunk_text(
        cls,
        text: str,
        strategy: str = "semantic",
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> list[Chunk]:
        """
        Convenience method to chunk text in one call.

        Args:
            text: The text to chunk.
            strategy: Chunking strategy to use.
            metadata: Optional metadata to attach to chunks.
            **kwargs: Additional arguments for the chunker.

        Returns:
            list[Chunk]: List of text chunks.
        """
        chunker = cls.create_chunker(strategy=strategy, **kwargs)
        return chunker.chunk(text, metadata=metadata)

    @classmethod
    def register_chunker(cls, name: str, chunker_class: Type[BaseChunker]) -> None:
        """
        Register a custom chunker strategy.

        Args:
            name: Strategy name.
            chunker_class: Chunker class to register.
        """
        cls.CHUNKER_MAP[name] = chunker_class

    @classmethod
    def available_strategies(cls) -> list[str]:
        """
        Get list of available chunking strategies.

        Returns:
            list[str]: List of strategy names.
        """
        return list(cls.CHUNKER_MAP.keys())

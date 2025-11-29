"""
Base chunker interface and data structures.

This module defines the abstract base class for text chunkers.
The Chunk dataclass has been moved to src.rag.types to avoid circular imports.
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.rag.types import Chunk


class BaseChunker(ABC):
    """
    Abstract base class for text chunkers.

    All chunking strategies must inherit from this class and implement
    the chunk() method.
    """

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: The text to chunk.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            list[Chunk]: List of text chunks.
        """
        pass

    def _create_chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
    ) -> Chunk:
        """
        Helper method to create a chunk with metadata.

        Args:
            text: The chunk text.
            metadata: Optional metadata dictionary.
            start_char: Starting character position.
            end_char: Ending character position.

        Returns:
            Chunk: Created chunk object.
        """
        chunk_metadata = metadata.copy() if metadata else {}
        return Chunk(
            text=text.strip(),
            metadata=chunk_metadata,
            start_char=start_char,
            end_char=end_char,
        )
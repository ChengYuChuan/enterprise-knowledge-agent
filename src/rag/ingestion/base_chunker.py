"""
Base chunker interface and data structures.

This module defines the abstract base class for text chunkers
and the Chunk data structure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


@dataclass
class Chunk:
    """
    Represents a chunk of text with metadata.

    Attributes:
        text: The chunk's text content.
        metadata: Dictionary containing chunk metadata.
        chunk_id: Unique identifier for the chunk.
        start_char: Starting character position in the original document.
        end_char: Ending character position in the original document.
    """

    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    def __post_init__(self) -> None:
        """Generate chunk_id if not provided."""
        if self.chunk_id is None:
            self.chunk_id = str(uuid4())

    def __len__(self) -> int:
        """Return the length of the chunk text."""
        return len(self.text)

    def to_dict(self) -> dict:
        """
        Convert chunk to dictionary format.

        Returns:
            dict: Dictionary representation of the chunk.
        """
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "length": len(self.text),
        }


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

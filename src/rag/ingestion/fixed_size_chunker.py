"""
Fixed-size text chunker implementation.

This module provides a simple chunking strategy that splits text
into fixed-size chunks with configurable overlap.
"""

from typing import Optional

from .base_chunker import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """
    Chunks text into fixed-size segments with optional overlap.

    This is the simplest chunking strategy, useful for:
    - Code files
    - Log files
    - When consistent chunk sizes are required

    Attributes:
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """
        Initialize the fixed-size chunker.

        Args:
            chunk_size: Target size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.

        Raises:
            ValueError: If chunk_overlap >= chunk_size.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Split text into fixed-size chunks.

        Args:
            text: The text to chunk.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            list[Chunk]: List of text chunks.
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)

            # Extract chunk text
            chunk_text = text[start:end]

            # Create chunk with position tracking
            chunk = self._create_chunk(
                text=chunk_text,
                metadata=metadata,
                start_char=start,
                end_char=end,
            )

            # Add chunk index to metadata
            chunk.metadata["chunk_index"] = len(chunks)

            chunks.append(chunk)

            # Move to next position with overlap
            start += self.chunk_size - self.chunk_overlap

        return chunks

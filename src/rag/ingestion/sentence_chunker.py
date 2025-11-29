"""
Sentence-based text chunker implementation.

This module provides chunking that preserves sentence boundaries,
ensuring chunks don't break in the middle of sentences.
"""

import re
from typing import Optional

from .base_chunker import BaseChunker, Chunk


class SentenceChunker(BaseChunker):
    """
    Chunks text by grouping complete sentences.

    This strategy ensures semantic integrity by never splitting
    sentences across chunks. Best for:
    - Natural language documents
    - FAQs
    - Conversational content

    Attributes:
        sentences_per_chunk: Target number of sentences per chunk.
        max_chunk_size: Maximum characters per chunk (safety limit).
    """

    # Sentence boundary patterns
    SENTENCE_ENDINGS = re.compile(r"([.!?]+[\s\n]+)|([.!?]+$)")

    def __init__(
        self, sentences_per_chunk: int = 5, max_chunk_size: int = 1000
    ) -> None:
        """
        Initialize the sentence-based chunker.

        Args:
            sentences_per_chunk: Target number of sentences per chunk.
            max_chunk_size: Maximum characters per chunk.
        """
        self.sentences_per_chunk = sentences_per_chunk
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Split text into chunks at sentence boundaries.

        Args:
            text: The text to chunk.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            list[Chunk]: List of text chunks.
        """
        if not text.strip():
            return []

        # Split into sentences
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk_sentences = []
        current_chunk_size = 0
        start_char = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Check if adding this sentence would exceed limits
            would_exceed_count = len(current_chunk_sentences) >= self.sentences_per_chunk
            would_exceed_size = current_chunk_size + sentence_length > self.max_chunk_size

            # If we have sentences and would exceed limits, create a chunk
            if current_chunk_sentences and (would_exceed_count or would_exceed_size):
                chunk_text = "".join(current_chunk_sentences)
                end_char = start_char + len(chunk_text)

                chunk = self._create_chunk(
                    text=chunk_text,
                    metadata=metadata,
                    start_char=start_char,
                    end_char=end_char,
                )
                chunk.metadata["chunk_index"] = len(chunks)
                chunk.metadata["sentence_count"] = len(current_chunk_sentences)

                chunks.append(chunk)

                # Reset for next chunk
                current_chunk_sentences = []
                current_chunk_size = 0
                start_char = end_char

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_size += sentence_length

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = "".join(current_chunk_sentences)
            end_char = start_char + len(chunk_text)

            chunk = self._create_chunk(
                text=chunk_text,
                metadata=metadata,
                start_char=start_char,
                end_char=end_char,
            )
            chunk.metadata["chunk_index"] = len(chunks)
            chunk.metadata["sentence_count"] = len(current_chunk_sentences)

            chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: The text to split.

        Returns:
            list[str]: List of sentences.
        """
        # Split on sentence endings
        parts = self.SENTENCE_ENDINGS.split(text)

        sentences = []
        i = 0
        while i < len(parts):
            sentence = parts[i]
            
            # Skip None values
            if sentence is None:
                i += 1
                continue

            # Combine with the delimiter if it exists
            if i + 1 < len(parts) and parts[i + 1] is not None:
                sentence += parts[i + 1]
                i += 2
            else:
                i += 1

            # Only add non-empty sentences
            if sentence and sentence.strip():
                sentences.append(sentence)

        return sentences
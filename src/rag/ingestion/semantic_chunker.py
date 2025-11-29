"""
Semantic text chunker implementation.

This module provides semantic chunking that attempts to group
related content together by detecting topic boundaries.

Note: This is a simplified implementation. For production use with
embeddings-based semantic chunking, consider using LlamaIndex's
SemanticSplitterNodeParser.
"""

import re
from typing import Optional

from .base_chunker import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    """
    Chunks text based on semantic boundaries.

    This implementation uses heuristics to detect topic changes:
    - Paragraph boundaries (double newlines)
    - Section headers (markdown style)
    - Significant whitespace changes

    For more advanced semantic chunking with embeddings, use LlamaIndex.

    Attributes:
        target_chunk_size: Target size for chunks in characters.
        max_chunk_size: Maximum chunk size before forcing a split.
    """

    # Patterns for detecting section breaks
    SECTION_HEADER = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
    PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

    def __init__(self, target_chunk_size: int = 512, max_chunk_size: int = 1000) -> None:
        """
        Initialize the semantic chunker.

        Args:
            target_chunk_size: Target size for chunks.
            max_chunk_size: Maximum chunk size.
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: The text to chunk.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            list[Chunk]: List of text chunks.
        """
        if not text.strip():
            return []

        # Split into semantic sections
        sections = self._split_into_sections(text)

        chunks = []
        current_chunk_text = ""
        start_char = 0

        for section in sections:
            section_length = len(section)

            # If section alone is too large, split it further
            if section_length > self.max_chunk_size:
                # First, save current chunk if it exists
                if current_chunk_text:
                    end_char = start_char + len(current_chunk_text)
                    chunk = self._create_chunk(
                        text=current_chunk_text,
                        metadata=metadata,
                        start_char=start_char,
                        end_char=end_char,
                    )
                    chunk.metadata["chunk_index"] = len(chunks)
                    chunks.append(chunk)

                    current_chunk_text = ""
                    start_char = end_char

                # Split large section using paragraph breaks
                subsections = self._split_large_section(section)
                for subsection in subsections:
                    end_char = start_char + len(subsection)
                    chunk = self._create_chunk(
                        text=subsection,
                        metadata=metadata,
                        start_char=start_char,
                        end_char=end_char,
                    )
                    chunk.metadata["chunk_index"] = len(chunks)
                    chunks.append(chunk)
                    start_char = end_char

                continue

            # Check if adding this section would exceed target size
            would_exceed = len(current_chunk_text) + section_length > self.target_chunk_size

            # If we have content and would exceed, create chunk
            if current_chunk_text and would_exceed:
                end_char = start_char + len(current_chunk_text)
                chunk = self._create_chunk(
                    text=current_chunk_text,
                    metadata=metadata,
                    start_char=start_char,
                    end_char=end_char,
                )
                chunk.metadata["chunk_index"] = len(chunks)
                chunks.append(chunk)

                current_chunk_text = section
                start_char = end_char
            else:
                # Add section to current chunk
                current_chunk_text += section

        # Don't forget the last chunk
        if current_chunk_text.strip():
            end_char = start_char + len(current_chunk_text)
            chunk = self._create_chunk(
                text=current_chunk_text,
                metadata=metadata,
                start_char=start_char,
                end_char=end_char,
            )
            chunk.metadata["chunk_index"] = len(chunks)
            chunks.append(chunk)

        return chunks

    def _split_into_sections(self, text: str) -> list[str]:
        """
        Split text into semantic sections.

        Uses markdown headers and paragraph breaks as boundaries.

        Args:
            text: The text to split.

        Returns:
            list[str]: List of text sections.
        """
        # First, try to split by headers
        sections = []
        current_section = ""
        found_headers = False

        for line in text.split("\n"):
            # Check if this is a header
            if self.SECTION_HEADER.match(line):
                found_headers = True
                # Save previous section
                if current_section.strip():
                    sections.append(current_section)
                # Start new section with header
                current_section = line + "\n"
            else:
                current_section += line + "\n"

        # Add the last section
        if current_section.strip():
            sections.append(current_section)

        # If no headers found, split by paragraph breaks
        if not found_headers or len(sections) == 1:
            sections = self.PARAGRAPH_BREAK.split(text)
            # Re-add paragraph breaks and filter empty
            sections = [s.strip() + "\n\n" if i < len(sections) - 1 else s.strip() 
                       for i, s in enumerate(sections) if s.strip()]

        return sections

    def _split_large_section(self, section: str) -> list[str]:
        """
        Split a large section into smaller chunks.

        Args:
            section: The section to split.

        Returns:
            list[str]: List of smaller chunks.
        """
        # Split by paragraph breaks
        paragraphs = self.PARAGRAPH_BREAK.split(section)

        chunks = []
        current = ""

        for para in paragraphs:
            if not para.strip():
                continue

            if len(current) + len(para) > self.max_chunk_size:
                if current:
                    chunks.append(current)
                current = para
            else:
                current += para + "\n\n"

        if current.strip():
            chunks.append(current)

        return chunks
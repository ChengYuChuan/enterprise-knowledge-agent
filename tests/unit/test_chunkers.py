"""
Unit tests for chunking strategies.
"""

import pytest

from src.rag.ingestion import (
    ChunkerFactory,
    FixedSizeChunker,
    SemanticChunker,
    SentenceChunker,
)


class TestFixedSizeChunker:
    """Test cases for FixedSizeChunker."""

    def test_basic_chunking(self):
        """Test basic fixed-size chunking."""
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
        text = "This is a test text for chunking."

        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        assert all(len(chunk.text) <= 10 for chunk in chunks)

    def test_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=3)
        text = "0123456789" * 3  # 30 characters

        chunks = chunker.chunk(text)

        # Verify overlap exists
        assert len(chunks) >= 2
        # Check that consecutive chunks share some content
        if len(chunks) >= 2:
            # Last 3 chars of first chunk should appear in second chunk
            overlap_text = chunks[0].text[-3:]
            assert overlap_text in chunks[1].text

    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("")

        assert len(chunks) == 0

    def test_invalid_overlap(self):
        """Test that invalid overlap raises error."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, chunk_overlap=10)


class TestSentenceChunker:
    """Test cases for SentenceChunker."""

    def test_sentence_boundaries(self):
        """Test that sentences are not split."""
        chunker = SentenceChunker(sentences_per_chunk=2)
        text = "First sentence. Second sentence. Third sentence."

        chunks = chunker.chunk(text)

        # Each chunk should contain complete sentences
        for chunk in chunks:
            assert chunk.text.count(".") >= 1  # At least one complete sentence

    def test_sentence_count(self):
        """Test that chunks respect sentence count limit."""
        chunker = SentenceChunker(sentences_per_chunk=2)
        text = "One. Two. Three. Four. Five."

        chunks = chunker.chunk(text)

        # Should have multiple chunks
        assert len(chunks) >= 2

    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = SentenceChunker()
        chunks = chunker.chunk("")

        assert len(chunks) == 0


class TestSemanticChunker:
    """Test cases for SemanticChunker."""

    def test_paragraph_boundaries(self):
        """Test that paragraphs influence chunk boundaries."""
        chunker = SemanticChunker(target_chunk_size=50)
        text = """First paragraph here.

Second paragraph here.

Third paragraph here."""

        chunks = chunker.chunk(text)

        assert len(chunks) > 0

    def test_markdown_headers(self):
        """Test that markdown headers create boundaries."""
        chunker = SemanticChunker(target_chunk_size=30)  # Small size to force splitting
        text = """# Header 1
Content for section 1.

# Header 2
Content for section 2."""

        chunks = chunker.chunk(text)

        # Should have at least one chunk per section
        assert len(chunks) >= 2

    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("")

        assert len(chunks) == 0


class TestChunkerFactory:
    """Test cases for ChunkerFactory."""

    def test_create_fixed_chunker(self):
        """Test creating fixed-size chunker."""
        chunker = ChunkerFactory.create_chunker("fixed", chunk_size=100)

        assert isinstance(chunker, FixedSizeChunker)
        assert chunker.chunk_size == 100

    def test_create_sentence_chunker(self):
        """Test creating sentence chunker."""
        chunker = ChunkerFactory.create_chunker("sentence")

        assert isinstance(chunker, SentenceChunker)

    def test_create_semantic_chunker(self):
        """Test creating semantic chunker."""
        chunker = ChunkerFactory.create_chunker("semantic", chunk_size=200)

        assert isinstance(chunker, SemanticChunker)
        assert chunker.target_chunk_size == 200

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError):
            ChunkerFactory.create_chunker("invalid_strategy")

    def test_chunk_text_convenience(self):
        """Test the convenience chunk_text method."""
        text = "This is a test. Another sentence here."
        chunks = ChunkerFactory.chunk_text(text, strategy="sentence")

        assert len(chunks) > 0
        assert all(hasattr(chunk, "text") for chunk in chunks)

    def test_available_strategies(self):
        """Test getting list of available strategies."""
        strategies = ChunkerFactory.available_strategies()

        assert "fixed" in strategies
        assert "sentence" in strategies
        assert "semantic" in strategies
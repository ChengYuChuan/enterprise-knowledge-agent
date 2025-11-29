"""
RAG ingestion module.

This module handles document loading and chunking.
"""

from .base_chunker import BaseChunker, Chunk
from .base_loader import BaseLoader, Document
from .chunkers import ChunkerFactory
from .fixed_size_chunker import FixedSizeChunker
from .loaders import LoaderFactory
from .pdf_loader import PDFLoader
from .semantic_chunker import SemanticChunker
from .sentence_chunker import SentenceChunker
from .text_loader import TextLoader

__all__ = [
    # Loaders
    "BaseLoader",
    "Document",
    "PDFLoader",
    "TextLoader",
    "LoaderFactory",
    # Chunkers
    "BaseChunker",
    "Chunk",
    "FixedSizeChunker",
    "SentenceChunker",
    "SemanticChunker",
    "ChunkerFactory",
    # Pipeline
    "IngestionPipeline",
]
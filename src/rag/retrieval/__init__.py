"""
RAG retrieval module.

This module handles document retrieval and search.
"""

from .bm25_search import BM25Search
# from .embedder import MockEmbedder
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker
from .retriever import Retriever
from .vector_store import QdrantVectorStore
from .openai_embedder import OpenAIEmbedder, get_embedder

__all__ = [
    # Core components
    "QdrantVectorStore",
    "MockEmbedder",
    "Retriever",
    # Phase 2: Advanced retrieval
    "BM25Search",
    "HybridRetriever",
    "Reranker",
    "OpenAIEmbedder",
    "get_embedder",
]
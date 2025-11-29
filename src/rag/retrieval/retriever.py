"""
Document retrieval module.

This module provides functionality to query the vector store
and retrieve relevant document chunks.
"""

from typing import Optional

from src.config import get_settings
from src.rag.retrieval.embedder import MockEmbedder
from src.rag.retrieval.vector_store import QdrantVectorStore


class Retriever:
    """
    Document retriever for querying the vector store.

    Attributes:
        vector_store: Vector store instance.
        embedder: Embedder instance.
    """

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        embedder: Optional[MockEmbedder] = None,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            vector_store: Vector store instance (creates default if None).
            embedder: Embedder instance (creates mock if None).
        """
        settings = get_settings()

        # Initialize vector store
        if vector_store is None:
            self.vector_store = QdrantVectorStore(
                url=settings.qdrant.url,
                api_key=settings.qdrant.api_key,
                collection_name=settings.qdrant.collection_name,
                vector_size=settings.qdrant.vector_size,
            )
        else:
            self.vector_store = vector_store

        # Initialize embedder
        if embedder is None:
            self.embedder = MockEmbedder(dimension=settings.qdrant.vector_size)
        else:
            self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Search for relevant documents.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score.

        Returns:
            list[dict]: Search results with text, metadata, and scores.
        """
        settings = get_settings()

        # Use default top_k from settings if not specified
        if top_k is None:
            top_k = settings.retrieval.top_k

        # Use default threshold from settings if not specified
        if score_threshold is None:
            score_threshold = settings.retrieval.similarity_threshold

        # If threshold is 0, pass None to disable it
        if score_threshold == 0:
            score_threshold = None

        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        return results

    def format_results(self, results: list[dict]) -> str:
        """
        Format search results as readable text.

        Args:
            results: Search results from retrieve().

        Returns:
            str: Formatted results.
        """
        if not results:
            return "No results found."

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"Result {i} (score: {result['score']:.3f}):\n"
                f"{result['text']}\n"
                f"Source: {result['metadata'].get('filename', 'unknown')}\n"
            )

        return "\n".join(formatted)
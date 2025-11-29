"""
Qdrant vector store implementation.

This module provides a wrapper around Qdrant for storing and retrieving
document chunks with vector embeddings.
"""

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.rag.ingestion import Chunk


class QdrantVectorStore:
    """
    Wrapper for Qdrant vector database operations.

    Handles collection management, document ingestion, and vector search.

    Attributes:
        client: Qdrant client instance.
        collection_name: Name of the collection to use.
        vector_size: Dimension of the embedding vectors.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "knowledge_base",
        vector_size: int = 1536,  # OpenAI text-embedding-3-small dimension
    ) -> None:
        """
        Initialize the Qdrant vector store.

        Args:
            url: Qdrant server URL.
            api_key: Optional API key for authentication.
            collection_name: Name of the collection.
            vector_size: Dimension of embedding vectors.
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """
        Ensure the collection exists, create if not.
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Insert or update chunks with their embeddings.

        Args:
            chunks: List of text chunks.
            embeddings: List of embedding vectors (same length as chunks).

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        # Create points for Qdrant
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=chunk.chunk_id,  # Use chunk_id as point ID
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                },
            )
            points.append(point)

        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score (0-1).
            filters: Optional metadata filters (not implemented yet).

        Returns:
            list[dict]: List of search results with text, metadata, and score.
        """
        from qdrant_client.models import SearchRequest

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
        ).points

        # Format results
        results = []
        for result in search_results:
            results.append(
                {
                    "chunk_id": result.id,
                    "text": result.payload["text"],
                    "metadata": result.payload["metadata"],
                    "score": result.score,
                    "start_char": result.payload.get("start_char"),
                    "end_char": result.payload.get("end_char"),
                }
            )

        return results

    def delete_collection(self) -> None:
        """
        Delete the entire collection.

        Warning: This will permanently delete all data.
        """
        self.client.delete_collection(collection_name=self.collection_name)

    def count(self) -> int:
        """
        Get the number of points in the collection.

        Returns:
            int: Number of points.
        """
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        return collection_info.points_count

    def get_collection_info(self) -> dict:
        """
        Get information about the collection.

        Returns:
            dict: Collection metadata.
        """
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": collection_info.points_count,
            "vector_size": collection_info.config.params.vectors.size,
            "distance": collection_info.config.params.vectors.distance,
        }
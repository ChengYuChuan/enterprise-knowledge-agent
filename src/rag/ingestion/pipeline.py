"""
Document ingestion pipeline.

This module provides an end-to-end pipeline for ingesting documents
into the vector store.
"""

from pathlib import Path
from typing import Optional

from src.config import get_settings
from src.rag.ingestion import ChunkerFactory, LoaderFactory
# from src.rag.retrieval.embedder import MockEmbedder
from src.rag.retrieval import get_embedder
from src.rag.retrieval.vector_store import QdrantVectorStore


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Handles:
    1. Loading documents
    2. Chunking text
    3. Generating embeddings
    4. Storing in vector database

    Attributes:
        vector_store: Vector store instance.
        embedder: Embedding model instance.
        chunking_strategy: Chunking strategy to use.
    """

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        embedder: Optional = None,
        chunking_strategy: str = "semantic",
    ) -> None:
        """
        Initialize the ingestion pipeline.

        Args:
            vector_store: Vector store instance (creates default if None).
            embedder: Embedder instance (creates mock if None).
            chunking_strategy: Chunking strategy ('fixed', 'sentence', 'semantic').
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

        # Initialize embedder (mock for now)
        # if embedder is None:
        #     self.embedder = MockEmbedder(dimension=settings.qdrant.vector_size)
        # else:
        #     self.embedder = embedder
        if embedder is None:
            self.embedder = get_embedder(dimension=settings.qdrant.vector_size)
        else:
            self.embedder = embedder

        self.chunking_strategy = chunking_strategy

    def ingest_file(self, file_path: Path) -> dict:
        """
        Ingest a single document file.

        Args:
            file_path: Path to the document file.

        Returns:
            dict: Ingestion statistics.
        """
        # Step 1: Load document
        document = LoaderFactory.load_document(file_path)

        # Step 2: Chunk the document
        settings = get_settings()
        chunks = ChunkerFactory.chunk_text(
            text=document.content,
            strategy=self.chunking_strategy,
            metadata=document.metadata,
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
        )

        # Step 3: Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)

        # Step 4: Store in vector database
        self.vector_store.upsert_chunks(chunks, embeddings)

        # Return statistics
        return {
            "file_path": str(file_path),
            "chunks_created": len(chunks),
            "total_characters": len(document.content),
            "collection_name": self.vector_store.collection_name,
        }

    def ingest_directory(self, directory: Path) -> dict:
        """
        Ingest all supported documents in a directory.

        Args:
            directory: Path to the directory.

        Returns:
            dict: Aggregated ingestion statistics.
        """
        supported_extensions = LoaderFactory.supported_extensions()

        results = []
        total_chunks = 0

        for ext in supported_extensions:
            for file_path in directory.glob(f"**/*{ext}"):
                try:
                    result = self.ingest_file(file_path)
                    results.append(result)
                    total_chunks += result["chunks_created"]
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")

        return {
            "files_processed": len(results),
            "total_chunks": total_chunks,
            "results": results,
        }

    def get_stats(self) -> dict:
        """
        Get vector store statistics.

        Returns:
            dict: Statistics about the indexed documents.
        """
        return self.vector_store.get_collection_info()

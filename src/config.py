"""
Configuration management for the Enterprise Knowledge Agent Platform.

This module provides centralized configuration using Pydantic Settings,
loading from environment variables and .env files.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""

    url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    api_key: Optional[str] = Field(default=None, description="Qdrant API key (optional)")
    collection_name: str = Field(
        default="knowledge_base", description="Default collection name"
    )
    vector_size: int = Field(default=1536, description="Vector embedding dimension")

    model_config = SettingsConfigDict(env_prefix="QDRANT_")


class ChunkingSettings(BaseSettings):
    """Document chunking configuration."""

    strategy: str = Field(default="semantic", description="Chunking strategy to use")
    chunk_size: int = Field(default=512, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")

    model_config = SettingsConfigDict(env_prefix="CHUNKING_")


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""

    top_k: int = Field(default=10, description="Number of results to retrieve")
    similarity_threshold: float = Field(
        default=0.0, description="Minimum similarity score threshold (0 = no threshold)"
    )

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")


class AppSettings(BaseSettings):
    """Main application settings."""

    # Application metadata
    app_name: str = Field(default="Enterprise Knowledge Agent", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Environment (dev/staging/prod)")
    log_level: str = Field(default="INFO", description="Logging level")

    # Component settings
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Global settings instance
settings = AppSettings()


def get_settings() -> AppSettings:
    """
    Get the global settings instance.

    Returns:
        AppSettings: The application settings object.
    """
    return settings
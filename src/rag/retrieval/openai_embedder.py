"""
OpenAI embeddings provider.

This module provides real embedding functionality using OpenAI's API.
Falls back to MockEmbedder if API key is not available.
"""

import os
from pathlib import Path
from typing import Optional, Union

from .embedder import MockEmbedder

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    # python-dotenv not installed, rely on system environment
    pass


class OpenAIEmbedder:
    """
    OpenAI embeddings provider using text-embedding-3-small.
    
    This provides high-quality semantic embeddings for production use.
    Automatically falls back to MockEmbedder if API key is not configured.
    
    Attributes:
        model: OpenAI model name.
        dimension: Embedding dimension (1536 for text-embedding-3-small).
        client: OpenAI client instance.
    """
    
    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSION = 1536
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
    ) -> None:
        """
        Initialize OpenAI embedder.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: OpenAI embedding model name.
            dimension: Embedding dimension.
        """
        self.model = model
        self.dimension = dimension
        
        # Get API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("⚠️  Warning: OPENAI_API_KEY not found. Using MockEmbedder fallback.")
            print("   Set OPENAI_API_KEY in .env to use real OpenAI embeddings.")
            self._use_mock = True
            self._mock_embedder = MockEmbedder(dimension=dimension)
            self.client = None
        else:
            try:
                from openai import OpenAI
                
                self.client = OpenAI(api_key=api_key)
                self._use_mock = False
                self._mock_embedder = None
                print(f"✓ OpenAI embedder initialized with model: {model}")
            except ImportError:
                print("⚠️  Warning: openai package not installed. Using MockEmbedder fallback.")
                print("   Install with: poetry add openai")
                self._use_mock = True
                self._mock_embedder = MockEmbedder(dimension=dimension)
                self.client = None
    
    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            list[float]: Embedding vector.
        """
        if self._use_mock:
            return self._mock_embedder.embed_text(text)
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        
        except Exception as e:
            print(f"⚠️  OpenAI API error: {e}. Falling back to MockEmbedder.")
            if self._mock_embedder is None:
                self._mock_embedder = MockEmbedder(dimension=self.dimension)
            return self._mock_embedder.embed_text(text)
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            list[list[float]]: List of embedding vectors.
        """
        if self._use_mock:
            return self._mock_embedder.embed_batch(texts)
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        
        except Exception as e:
            print(f"⚠️  OpenAI API error: {e}. Falling back to MockEmbedder.")
            if self._mock_embedder is None:
                self._mock_embedder = MockEmbedder(dimension=self.dimension)
            return self._mock_embedder.embed_batch(texts)
    
    def embed(self, text: Union[str, list[str]]) -> Union[list[float], list[list[float]]]:
        """
        Generate embeddings (unified interface).
        
        Args:
            text: Single text or list of texts.
            
        Returns:
            Embedding vector(s).
        """
        if isinstance(text, str):
            return self.embed_text(text)
        else:
            return self.embed_batch(text)
    
    def is_using_mock(self) -> bool:
        """
        Check if currently using MockEmbedder fallback.
        
        Returns:
            bool: True if using mock, False if using real OpenAI.
        """
        return self._use_mock


def get_embedder(
    prefer_openai: bool = True,
    dimension: int = 1536,
) -> Union[OpenAIEmbedder, MockEmbedder]:
    """
    Get the appropriate embedder based on configuration.
    
    Args:
        prefer_openai: Whether to prefer OpenAI if API key is available.
        dimension: Embedding dimension.
        
    Returns:
        Embedder instance (OpenAI or Mock).
    """
    if prefer_openai and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbedder(dimension=dimension)
    else:
        return MockEmbedder(dimension=dimension)
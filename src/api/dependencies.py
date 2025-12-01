"""
FastAPI Dependencies

Dependency injection for the API layer.
Provides shared resources to route handlers.

Design Pattern: Dependency Injection
    - Resources are created once and shared
    - Easy to mock for testing
    - Configuration is centralized

Usage:
    ```python
    from fastapi import Depends
    from src.api.dependencies import get_llm_provider, get_settings
    
    @router.post("/chat")
    async def chat(
        request: ChatRequest,
        llm: BaseLLMProvider = Depends(get_llm_provider),
        settings: Settings = Depends(get_settings),
    ):
        response = await llm.generate(messages)
        return response
    ```
"""

import os
from typing import Optional, AsyncGenerator
from functools import lru_cache
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request

# Import LLM components
from src.llm import (
    BaseLLMProvider,
    LLMConfig,
    get_provider,
    get_provider_from_env,
)
from src.llm.exceptions import LLMError, LLMAuthenticationError

# Import auth components
from src.api.middleware.auth import User, get_current_user, get_optional_user


# =============================================================================
# Settings
# =============================================================================

@dataclass
class Settings:
    """Application settings loaded from environment."""
    
    # API settings
    api_version: str = "1.0.0"
    api_title: str = "Enterprise Knowledge Agent API"
    debug: bool = False
    
    # LLM settings
    default_llm_provider: str = "openai"
    default_llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    
    # RAG settings
    default_top_k: int = 5
    default_collection: str = "default"
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    
    # Auth settings
    auth_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            api_version=os.getenv("API_VERSION", "1.0.0"),
            api_title=os.getenv("API_TITLE", "Enterprise Knowledge Agent API"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            default_llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            default_llm_model=os.getenv("LLM_MODEL", "gpt-4"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
            default_collection=os.getenv("DEFAULT_COLLECTION", "default"),
            rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
            auth_enabled=os.getenv("AUTH_ENABLED", "true").lower() == "true",
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings.
    
    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings.from_env()


# =============================================================================
# LLM Provider Dependencies
# =============================================================================

# Cache for provider instances
_provider_cache: dict[str, BaseLLMProvider] = {}


async def get_llm_provider(
    settings: Settings = Depends(get_settings),
) -> BaseLLMProvider:
    """Get the default LLM provider.
    
    Creates and caches a provider instance based on settings.
    
    Returns:
        Configured LLM provider.
    
    Raises:
        HTTPException: If provider cannot be initialized.
    """
    cache_key = f"{settings.default_llm_provider}:{settings.default_llm_model}"
    
    if cache_key not in _provider_cache:
        try:
            config = LLMConfig(
                model=settings.default_llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            provider = get_provider(settings.default_llm_provider, config)
            await provider.initialize()
            _provider_cache[cache_key] = provider
            
        except LLMAuthenticationError as e:
            raise HTTPException(
                status_code=500,
                detail=f"LLM authentication failed: {e.message}"
            )
        except LLMError as e:
            raise HTTPException(
                status_code=500,
                detail=f"LLM initialization failed: {e.message}"
            )
    
    return _provider_cache[cache_key]


async def get_llm_provider_for_request(
    request: Request,
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    settings: Settings = Depends(get_settings),
) -> BaseLLMProvider:
    """Get an LLM provider based on request parameters.
    
    Allows overriding the default provider/model per request.
    
    Args:
        request: The FastAPI request.
        provider_name: Override provider name.
        model: Override model name.
        settings: Application settings.
    
    Returns:
        Configured LLM provider.
    """
    # Use defaults if not specified
    provider_name = provider_name or settings.default_llm_provider
    model = model or settings.default_llm_model
    
    cache_key = f"{provider_name}:{model}"
    
    if cache_key not in _provider_cache:
        try:
            config = LLMConfig(
                model=model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            provider = get_provider(provider_name, config)
            await provider.initialize()
            _provider_cache[cache_key] = provider
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize {provider_name} provider: {str(e)}"
            )
    
    return _provider_cache[cache_key]


# =============================================================================
# RAG Pipeline Dependencies
# =============================================================================

from src.rag.ingestion.pipeline import IngestionPipeline
from src.rag.retrieval.vector_store import QdrantVectorStore
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.retrieval.openai_embedder import OpenAIEmbedder
from src.config import get_settings as get_app_settings


class RAGPipeline:
    """RAG pipeline for search and ingestion."""
    
    def __init__(self):
        self._initialized = False
        self._vector_store: Optional[QdrantVectorStore] = None
        self._embedder: Optional[OpenAIEmbedder] = None
        self._retriever: Optional[HybridRetriever] = None
        self._ingestion: Optional[IngestionPipeline] = None
    
    def _ensure_initialized(self):
        """Lazy initialization of RAG components."""
        if self._initialized:
            return
        
        app_settings = get_app_settings()
        
        # Initialize vector store
        self._vector_store = QdrantVectorStore(
            url=app_settings.qdrant.url,
            api_key=app_settings.qdrant.api_key,
            collection_name=app_settings.qdrant.collection_name,
            vector_size=app_settings.qdrant.vector_size,
        )
        
        # Initialize embedder
        self._embedder = OpenAIEmbedder(
            dimension=app_settings.qdrant.vector_size,
        )
        
        # Initialize hybrid retriever
        self._retriever = HybridRetriever(
            vector_store=self._vector_store,
            embedder=self._embedder,
            alpha=0.5,  # Balanced hybrid search
            use_reranking=False,
        )
        
        # Initialize ingestion pipeline
        self._ingestion = IngestionPipeline(
            vector_store=self._vector_store,
            embedder=self._embedder,
        )
        
        self._initialized = True
    
    async def search(self, query: str, top_k: int = 10) -> list:
        """Search documents using hybrid retrieval."""
        self._ensure_initialized()
        
        # HybridRetriever.search is synchronous
        results = self._retriever.search(query=query, top_k=top_k)
        return results
    
    async def ingest_text(self, content: str, filename: str, metadata: dict = None) -> dict:
        """Ingest text content."""
        self._ensure_initialized()
        
        from src.rag.types import Chunk
        from uuid import uuid4
        
        # Create a chunk from the text
        chunk = Chunk(
            chunk_id=str(uuid4()),  # UUID
            text=content,
            metadata=metadata or {"filename": filename},
            start_char=0,
            end_char=len(content),
        )
        
        # Generate embedding
        embedding = self._embedder.embed_text(content)
        
        # Store in vector database
        self._vector_store.upsert_chunks([chunk], [embedding])
        
        # Index for BM25
        self._retriever.index_for_bm25([{
            "chunk_id": chunk.chunk_id,
            "text": content,
            "metadata": metadata or {"filename": filename},
        }])
        
        return {
            "document_id": chunk.chunk_id,
            "chunks_created": 1,
            "filename": filename,
        }
    
    async def query(self, query: str, top_k: int = 5) -> dict:
        """Query the knowledge base and return formatted results for chat."""
        self._ensure_initialized()
        
        # Use existing search method
        results = self._retriever.search(query=query, top_k=top_k)
        
        # Format results for chat
        sources = []
        for i, result in enumerate(results):
            sources.append({
                "document_id": f"doc_{i}",
                "chunk_id": result.get("chunk_id", ""),
                "filename": result.get("metadata", {}).get("filename", "unknown"),
                "content": result.get("text", ""),
                "score": result.get("score", 0.0),
                "metadata": result.get("metadata", {}),
            })
        
        return {"sources": sources} 
    
    def get_stats(self) -> dict:
        """Get RAG pipeline statistics."""
        self._ensure_initialized()
        return self._vector_store.get_collection_info()
    

_rag_pipeline: Optional[RAGPipeline] = None

async def get_rag_pipeline() -> RAGPipeline:
    """Get the RAG pipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


# =============================================================================
# Conversation Store Dependencies (Placeholder)
# =============================================================================

class ConversationStore:
    """Placeholder for conversation storage."""
    
    def __init__(self):
        self._conversations: dict[str, list] = {}
    
    async def get(self, conversation_id: str) -> Optional[list]:
        """Get conversation history."""
        return self._conversations.get(conversation_id)
    
    async def save(self, conversation_id: str, messages: list) -> None:
        """Save conversation history."""
        self._conversations[conversation_id] = messages
    
    async def create(self) -> str:
        """Create a new conversation."""
        import uuid
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        self._conversations[conv_id] = []
        return conv_id


_conversation_store: Optional[ConversationStore] = None


async def get_conversation_store() -> ConversationStore:
    """Get the conversation store instance."""
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore()
    return _conversation_store


# =============================================================================
# Job Queue Dependencies (Placeholder)
# =============================================================================

class JobQueue:
    """Placeholder for async job queue (for document ingestion)."""
    
    def __init__(self):
        self._jobs: dict[str, dict] = {}
    
    async def enqueue(self, job_type: str, data: dict) -> str:
        """Enqueue a job."""
        import uuid
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        self._jobs[job_id] = {
            "id": job_id,
            "type": job_type,
            "status": "pending",
            "data": data,
        }
        return job_id
    
    async def get_status(self, job_id: str) -> Optional[dict]:
        """Get job status."""
        return self._jobs.get(job_id)


_job_queue: Optional[JobQueue] = None


async def get_job_queue() -> JobQueue:
    """Get the job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue


# =============================================================================
# Cleanup
# =============================================================================

async def cleanup_providers():
    """Clean up provider resources on shutdown."""
    for provider in _provider_cache.values():
        await provider.close()
    _provider_cache.clear()


# =============================================================================
# Request Context Dependencies
# =============================================================================

async def get_request_context(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Get context information for the current request.
    
    Useful for logging and debugging.
    """
    return {
        "request_id": getattr(request.state, "request_id", None),
        "user_id": user.id if user else None,
        "path": request.url.path,
        "method": request.method,
        "client_ip": request.client.host if request.client else None,
    }
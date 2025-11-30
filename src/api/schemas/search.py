"""
Search Schemas

Pydantic models for search-related endpoints.
Supports semantic search, keyword search, and hybrid search.

Endpoints:
    POST /api/v1/search - Search documents
"""

from typing import Optional, Any
from enum import Enum
from pydantic import Field, ConfigDict, field_validator

from src.api.schemas.common import (
    BaseSchema,
    SourceDocument,
    PaginatedResponse,
)


# =============================================================================
# Enums
# =============================================================================

class SearchMode(str, Enum):
    """Search mode selection."""
    SEMANTIC = "semantic"    # Vector similarity search
    KEYWORD = "keyword"      # BM25 keyword search
    HYBRID = "hybrid"        # Combined vector + keyword with RRF


class RerankerType(str, Enum):
    """Reranker model selection."""
    NONE = "none"            # No reranking
    CROSS_ENCODER = "cross_encoder"  # Cross-encoder reranking
    COHERE = "cohere"        # Cohere reranker API


# =============================================================================
# Request Models
# =============================================================================

class SearchFilters(BaseSchema):
    """Filters for search queries."""
    file_types: Optional[list[str]] = Field(
        default=None,
        description="Filter by file types (e.g., ['pdf', 'md'])"
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="Filter by tags (AND logic)"
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Filter documents created after this date (ISO format)"
    )
    date_to: Optional[str] = Field(
        default=None,
        description="Filter documents created before this date (ISO format)"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Filter by custom metadata fields"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_types": ["pdf", "docx"],
                "tags": ["finance", "2024"],
                "date_from": "2024-01-01",
                "metadata": {"department": "Finance"}
            }
        }
    )


class SearchRequest(BaseSchema):
    """Request body for search endpoint."""
    query: str = Field(
        min_length=1,
        max_length=1000,
        description="Search query text"
    )
    
    # Search configuration
    mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description="Search mode: semantic, keyword, or hybrid"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return"
    )
    
    # Reranking
    rerank: bool = Field(
        default=True,
        description="Whether to apply reranking"
    )
    reranker: RerankerType = Field(
        default=RerankerType.CROSS_ENCODER,
        description="Reranker to use"
    )
    
    # Hybrid search weights (only for hybrid mode)
    semantic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for semantic search in hybrid mode"
    )
    
    # Filtering
    filters: Optional[SearchFilters] = Field(
        default=None,
        description="Optional filters to narrow results"
    )
    
    # Collection
    collection: Optional[str] = Field(
        default=None,
        description="Specific collection to search (searches all if not specified)"
    )
    
    # Include options
    include_content: bool = Field(
        default=True,
        description="Include document content in results"
    )
    include_embeddings: bool = Field(
        default=False,
        description="Include embedding vectors in results"
    )
    
    @field_validator("semantic_weight")
    @classmethod
    def validate_weights(cls, v: float) -> float:
        """Validate semantic weight is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("semantic_weight must be between 0.0 and 1.0")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "quarterly revenue growth",
                "mode": "hybrid",
                "top_k": 10,
                "rerank": True,
                "semantic_weight": 0.7,
                "filters": {
                    "file_types": ["pdf"],
                    "tags": ["finance"]
                }
            }
        }
    )


# =============================================================================
# Response Models
# =============================================================================

class SearchResult(BaseSchema):
    """A single search result."""
    document_id: str = Field(
        description="Unique document identifier"
    )
    chunk_id: str = Field(
        description="Chunk identifier within the document"
    )
    filename: str = Field(
        description="Original filename"
    )
    content: str = Field(
        description="Matching content snippet"
    )
    
    # Scores
    score: float = Field(
        ge=0.0,
        description="Final relevance score"
    )
    semantic_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Semantic similarity score"
    )
    keyword_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="BM25 keyword score"
    )
    rerank_score: Optional[float] = Field(
        default=None,
        description="Reranker score"
    )
    
    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata"
    )
    
    # Optional embedding
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Embedding vector (if requested)"
    )
    
    # Highlights (optional)
    highlights: Optional[list[str]] = Field(
        default=None,
        description="Highlighted matching passages"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_123",
                "chunk_id": "chunk_456",
                "filename": "q3_report.pdf",
                "content": "Revenue grew by 25% compared to Q2...",
                "score": 0.92,
                "semantic_score": 0.89,
                "keyword_score": 0.95,
                "rerank_score": 0.92,
                "metadata": {
                    "page": 5,
                    "section": "Financial Summary"
                },
                "highlights": [
                    "Revenue <mark>grew</mark> by 25%"
                ]
            }
        }
    )


class SearchResponse(BaseSchema):
    """Response from search endpoint."""
    results: list[SearchResult] = Field(
        description="Search results"
    )
    total: int = Field(
        ge=0,
        description="Total number of matching documents"
    )
    query: str = Field(
        description="Original search query"
    )
    mode: SearchMode = Field(
        description="Search mode used"
    )
    latency_ms: float = Field(
        ge=0,
        description="Search latency in milliseconds"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "document_id": "doc_123",
                        "chunk_id": "chunk_456",
                        "filename": "q3_report.pdf",
                        "content": "Revenue grew by 25%...",
                        "score": 0.92,
                        "metadata": {}
                    }
                ],
                "total": 15,
                "query": "quarterly revenue growth",
                "mode": "hybrid",
                "latency_ms": 150.5
            }
        }
    )


# =============================================================================
# Similar Documents
# =============================================================================

class SimilarDocumentsRequest(BaseSchema):
    """Request for finding similar documents."""
    document_id: str = Field(
        description="Document ID to find similar documents for"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of similar documents to return"
    )
    exclude_same_source: bool = Field(
        default=True,
        description="Exclude documents from the same source file"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_123",
                "top_k": 5,
                "exclude_same_source": True
            }
        }
    )
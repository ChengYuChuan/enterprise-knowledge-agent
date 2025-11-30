"""
API Schemas

Pydantic models for request validation and response serialization.
All schemas are designed with OpenAPI documentation in mind.

Organization:
    - common.py: Shared base classes and utility schemas
    - chat.py: Chat endpoint schemas
    - search.py: Search endpoint schemas
    - ingest.py: Document ingestion schemas
"""

# Common schemas
from src.api.schemas.common import (
    # Base
    BaseSchema,
    StatusEnum,
    ProviderEnum,
    # Responses
    HealthResponse,
    ErrorResponse,
    PaginationParams,
    PaginatedResponse,
    # Documents
    SourceDocument,
    TokenUsage,
    DocumentMetadata,
)

# Chat schemas
from src.api.schemas.chat import (
    ChatMessage,
    ChatRequest,
    StreamChatRequest,
    ChatResponse,
    StreamChunk,
    ConversationInfo,
)

# Search schemas
from src.api.schemas.search import (
    SearchMode,
    RerankerType,
    SearchFilters,
    SearchRequest,
    SearchResult,
    SearchResponse,
    SimilarDocumentsRequest,
)

# Ingest schemas
from src.api.schemas.ingest import (
    ChunkingStrategy,
    IngestSource,
    IngestConfig,
    IngestRequest,
    IngestURLRequest,
    IngestTextRequest,
    IngestResponse,
    IngestJobStatus,
    DocumentInfo,
    DocumentListResponse,
    DeleteDocumentResponse,
)

__all__ = [
    # Common
    "BaseSchema",
    "StatusEnum",
    "ProviderEnum",
    "HealthResponse",
    "ErrorResponse",
    "PaginationParams",
    "PaginatedResponse",
    "SourceDocument",
    "TokenUsage",
    "DocumentMetadata",
    # Chat
    "ChatMessage",
    "ChatRequest",
    "StreamChatRequest",
    "ChatResponse",
    "StreamChunk",
    "ConversationInfo",
    # Search
    "SearchMode",
    "RerankerType",
    "SearchFilters",
    "SearchRequest",
    "SearchResult",
    "SearchResponse",
    "SimilarDocumentsRequest",
    # Ingest
    "ChunkingStrategy",
    "IngestSource",
    "IngestConfig",
    "IngestRequest",
    "IngestURLRequest",
    "IngestTextRequest",
    "IngestResponse",
    "IngestJobStatus",
    "DocumentInfo",
    "DocumentListResponse",
    "DeleteDocumentResponse",
]
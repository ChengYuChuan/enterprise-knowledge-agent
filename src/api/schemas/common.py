"""
Common Schemas

Shared Pydantic models used across multiple endpoints.
These provide consistent data structures for API requests and responses.

Design Principles:
    1. Use Pydantic v2 syntax (model_validator, field_validator)
    2. Include examples for OpenAPI documentation
    3. Provide sensible defaults where appropriate
    4. Use strict types to catch errors early
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Enums
# =============================================================================

class StatusEnum(str, Enum):
    """Standard status values for async operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProviderEnum(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


# =============================================================================
# Base Models
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = ConfigDict(
        # Use enum values in serialization
        use_enum_values=True,
        # Validate on assignment (not just initialization)
        validate_assignment=True,
        # Include field descriptions in JSON schema
        json_schema_extra={
            "example": {}  # Override in subclasses
        }
    )


class TimestampMixin(BaseModel):
    """Mixin for adding timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# =============================================================================
# Response Models
# =============================================================================

class HealthResponse(BaseSchema):
    """Health check response."""
    status: str = Field(
        default="healthy",
        description="Service health status"
    )
    version: str = Field(
        description="API version"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Current server time"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    )


class ErrorResponse(BaseSchema):
    """Standard error response format."""
    error: str = Field(
        description="Error type/code"
    )
    message: str = Field(
        description="Human-readable error message"
    )
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "validation_error",
                "message": "Invalid request parameters",
                "details": {"field": "query", "issue": "cannot be empty"},
                "request_id": "req_abc123"
            }
        }
    )


class PaginationParams(BaseSchema):
    """Pagination parameters for list endpoints."""
    page: int = Field(
        default=1,
        ge=1,
        description="Page number (1-indexed)"
    )
    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Items per page"
    )
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseSchema):
    """Base model for paginated responses."""
    total: int = Field(
        description="Total number of items"
    )
    page: int = Field(
        description="Current page number"
    )
    page_size: int = Field(
        description="Items per page"
    )
    pages: int = Field(
        description="Total number of pages"
    )
    
    @classmethod
    def calculate_pages(cls, total: int, page_size: int) -> int:
        """Calculate total pages from total items and page size."""
        return (total + page_size - 1) // page_size


# =============================================================================
# Source/Citation Models
# =============================================================================

class SourceDocument(BaseSchema):
    """A source document referenced in a response."""
    document_id: str = Field(
        description="Unique document identifier"
    )
    filename: str = Field(
        description="Original filename"
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Specific chunk identifier"
    )
    content: str = Field(
        description="Relevant content snippet"
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score (0-1)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional document metadata"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_123",
                "filename": "quarterly_report.pdf",
                "chunk_id": "chunk_456",
                "content": "Revenue increased by 25% in Q3...",
                "score": 0.92,
                "metadata": {"page": 5, "section": "Financial Summary"}
            }
        }
    )


# =============================================================================
# Token Usage Models
# =============================================================================

class TokenUsage(BaseSchema):
    """Token usage statistics from LLM calls."""
    prompt_tokens: int = Field(
        ge=0,
        description="Tokens in the prompt"
    )
    completion_tokens: int = Field(
        ge=0,
        description="Tokens in the completion"
    )
    total_tokens: int = Field(
        ge=0,
        description="Total tokens used"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt_tokens": 150,
                "completion_tokens": 200,
                "total_tokens": 350
            }
        }
    )


# =============================================================================
# Metadata Models
# =============================================================================

class DocumentMetadata(BaseSchema):
    """Metadata for document ingestion."""
    title: Optional[str] = Field(
        default=None,
        description="Document title"
    )
    author: Optional[str] = Field(
        default=None,
        description="Document author"
    )
    source: Optional[str] = Field(
        default=None,
        description="Source system or URL"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization"
    )
    custom: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata fields"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Q3 Financial Report",
                "author": "Finance Team",
                "source": "internal",
                "tags": ["finance", "quarterly", "2024"],
                "custom": {"department": "Finance", "confidential": True}
            }
        }
    )
"""
Ingest Schemas

Pydantic models for document ingestion endpoints.
Supports file uploads, URL ingestion, and batch processing.

Endpoints:
    POST /api/v1/ingest - Upload and ingest documents
    GET /api/v1/ingest/{job_id}/status - Check ingestion status
    DELETE /api/v1/documents/{doc_id} - Delete a document
"""

from typing import Optional, Any
from enum import Enum
from pydantic import Field, ConfigDict, field_validator

from src.api.schemas.common import (
    BaseSchema,
    StatusEnum,
    DocumentMetadata,
)


# =============================================================================
# Enums
# =============================================================================

class ChunkingStrategy(str, Enum):
    """Document chunking strategy."""
    FIXED = "fixed"           # Fixed size chunks
    SENTENCE = "sentence"     # Sentence-based splitting
    SEMANTIC = "semantic"     # Semantic-aware splitting
    PARAGRAPH = "paragraph"   # Paragraph-based splitting


class IngestSource(str, Enum):
    """Source type for ingestion."""
    FILE = "file"            # Uploaded file
    URL = "url"              # Web URL
    TEXT = "text"            # Raw text


# =============================================================================
# Request Models
# =============================================================================

class IngestConfig(BaseSchema):
    """Configuration for document ingestion."""
    
    # Chunking settings
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.SEMANTIC,
        description="Strategy for splitting documents into chunks"
    )
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=4000,
        description="Target chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks in tokens"
    )
    
    # Processing settings
    extract_tables: bool = Field(
        default=True,
        description="Extract and process tables separately"
    )
    extract_images: bool = Field(
        default=False,
        description="Extract and process images (OCR)"
    )
    
    # Collection settings
    collection: Optional[str] = Field(
        default=None,
        description="Target collection (uses default if not specified)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunking_strategy": "semantic",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "extract_tables": True,
                "collection": "default"
            }
        }
    )


class IngestRequest(BaseSchema):
    """Request body for document ingestion (JSON part of multipart request).
    
    Note: The actual file is sent as multipart form data.
    This schema defines the metadata and configuration.
    """
    metadata: Optional[DocumentMetadata] = Field(
        default=None,
        description="Document metadata"
    )
    config: Optional[IngestConfig] = Field(
        default=None,
        description="Ingestion configuration"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "metadata": {
                    "title": "Q3 Financial Report",
                    "tags": ["finance", "quarterly"]
                },
                "config": {
                    "chunking_strategy": "semantic",
                    "chunk_size": 512
                }
            }
        }
    )


class IngestURLRequest(BaseSchema):
    """Request for ingesting a document from URL."""
    url: str = Field(
        description="URL of the document to ingest"
    )
    metadata: Optional[DocumentMetadata] = Field(
        default=None,
        description="Document metadata"
    )
    config: Optional[IngestConfig] = Field(
        default=None,
        description="Ingestion configuration"
    )
    
    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Basic URL validation."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://example.com/report.pdf",
                "metadata": {
                    "title": "External Report",
                    "source": "example.com"
                }
            }
        }
    )


class IngestTextRequest(BaseSchema):
    """Request for ingesting raw text."""
    content: str = Field(
        min_length=1,
        max_length=1000000,  # ~1MB of text
        description="Text content to ingest"
    )
    filename: str = Field(
        description="Virtual filename for the text"
    )
    metadata: Optional[DocumentMetadata] = Field(
        default=None,
        description="Document metadata"
    )
    config: Optional[IngestConfig] = Field(
        default=None,
        description="Ingestion configuration"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "This is the document content...",
                "filename": "meeting_notes.txt",
                "metadata": {
                    "title": "Team Meeting Notes",
                    "tags": ["meeting", "notes"]
                }
            }
        }
    )


# =============================================================================
# Response Models
# =============================================================================

class IngestResponse(BaseSchema):
    """Response from ingestion endpoint."""
    job_id: str = Field(
        description="Unique job identifier for tracking"
    )
    status: StatusEnum = Field(
        description="Current job status"
    )
    message: str = Field(
        description="Status message"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "job_abc123",
                "status": "processing",
                "message": "Document queued for processing"
            }
        }
    )


class IngestJobStatus(BaseSchema):
    """Detailed status of an ingestion job."""
    job_id: str = Field(
        description="Job identifier"
    )
    status: StatusEnum = Field(
        description="Current status"
    )
    progress: float = Field(
        ge=0.0,
        le=1.0,
        description="Progress percentage (0.0 to 1.0)"
    )
    
    # Document info
    filename: str = Field(
        description="Name of the file being processed"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Document ID (available after completion)"
    )
    
    # Processing details
    chunks_created: int = Field(
        default=0,
        ge=0,
        description="Number of chunks created"
    )
    
    # Timing
    started_at: Optional[str] = Field(
        default=None,
        description="When processing started"
    )
    completed_at: Optional[str] = Field(
        default=None,
        description="When processing completed"
    )
    
    # Error info
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "job_abc123",
                "status": "completed",
                "progress": 1.0,
                "filename": "report.pdf",
                "document_id": "doc_xyz789",
                "chunks_created": 45,
                "started_at": "2024-01-15T10:00:00Z",
                "completed_at": "2024-01-15T10:00:30Z",
                "error": None
            }
        }
    )


# =============================================================================
# Document Management
# =============================================================================

class DocumentInfo(BaseSchema):
    """Information about an ingested document."""
    document_id: str = Field(
        description="Unique document identifier"
    )
    filename: str = Field(
        description="Original filename"
    )
    file_type: str = Field(
        description="File type (pdf, md, txt, etc.)"
    )
    file_size: int = Field(
        ge=0,
        description="File size in bytes"
    )
    chunk_count: int = Field(
        ge=0,
        description="Number of chunks"
    )
    metadata: DocumentMetadata = Field(
        description="Document metadata"
    )
    created_at: str = Field(
        description="When the document was ingested"
    )
    collection: str = Field(
        description="Collection the document belongs to"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_xyz789",
                "filename": "report.pdf",
                "file_type": "pdf",
                "file_size": 1024000,
                "chunk_count": 45,
                "metadata": {
                    "title": "Q3 Report",
                    "tags": ["finance"]
                },
                "created_at": "2024-01-15T10:00:00Z",
                "collection": "default"
            }
        }
    )


class DocumentListResponse(BaseSchema):
    """Response for listing documents."""
    documents: list[DocumentInfo] = Field(
        description="List of documents"
    )
    total: int = Field(
        ge=0,
        description="Total number of documents"
    )
    page: int = Field(
        ge=1,
        description="Current page"
    )
    page_size: int = Field(
        ge=1,
        description="Items per page"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": [],
                "total": 100,
                "page": 1,
                "page_size": 20
            }
        }
    )


class DeleteDocumentResponse(BaseSchema):
    """Response from document deletion."""
    document_id: str = Field(
        description="ID of deleted document"
    )
    deleted: bool = Field(
        description="Whether deletion was successful"
    )
    chunks_deleted: int = Field(
        ge=0,
        description="Number of chunks deleted"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_xyz789",
                "deleted": True,
                "chunks_deleted": 45
            }
        }
    )
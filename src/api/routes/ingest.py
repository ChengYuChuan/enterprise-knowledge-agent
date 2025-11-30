"""
Ingest Routes

Endpoints for document ingestion and management.

Endpoints:
    POST /ingest - Upload and ingest a document
    POST /ingest/url - Ingest from URL
    POST /ingest/text - Ingest raw text
    GET /ingest/{job_id}/status - Check ingestion status
    GET /documents - List all documents
    GET /documents/{doc_id} - Get document info
    DELETE /documents/{doc_id} - Delete a document
"""

import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from src.api.schemas import (
    IngestConfig,
    IngestResponse,
    IngestJobStatus,
    IngestURLRequest,
    IngestTextRequest,
    DocumentInfo,
    DocumentListResponse,
    DeleteDocumentResponse,
    StatusEnum,
    DocumentMetadata,
    PaginationParams,
)
from src.api.middleware import User, get_current_user, rate_limit
from src.api.dependencies import (
    get_settings,
    Settings,
    get_job_queue,
    JobQueue,
)

router = APIRouter(tags=["Ingest"])


# =============================================================================
# File Upload Routes
# =============================================================================

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload and ingest a document",
    description="Upload a file and add it to the knowledge base.",
)
async def ingest_file(
    file: UploadFile = File(..., description="File to ingest"),
    metadata_json: Optional[str] = Form(None, description="JSON string of metadata"),
    config_json: Optional[str] = Form(None, description="JSON string of config"),
    user: User = Depends(get_current_user),
    job_queue: JobQueue = Depends(get_job_queue),
    settings: Settings = Depends(get_settings),
    _: None = Depends(rate_limit(requests=10, window=60)),  # 10 uploads/min
) -> IngestResponse:
    """Upload and ingest a document.
    
    Supports:
    - PDF files
    - Markdown files
    - Plain text files
    - Word documents (docx)
    
    The document is processed asynchronously. Use the returned job_id
    to check the status.
    """
    import json
    
    # Validate file type
    allowed_types = {".pdf", ".md", ".txt", ".docx", ".doc"}
    filename = file.filename or "unknown"
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    
    if extension not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Allowed: {allowed_types}"
        )
    
    # Parse metadata and config
    metadata = None
    config = None
    
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    
    if config_json:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid config JSON")
    
    # Read file content
    content = await file.read()
    
    # Queue the ingestion job
    job_id = await job_queue.enqueue(
        job_type="ingest_file",
        data={
            "filename": filename,
            "content_size": len(content),
            "metadata": metadata,
            "config": config,
            "user_id": user.id,
        }
    )
    
    return IngestResponse(
        job_id=job_id,
        status=StatusEnum.PENDING,
        message=f"Document '{filename}' queued for processing",
    )


@router.post(
    "/ingest/url",
    response_model=IngestResponse,
    summary="Ingest from URL",
    description="Download and ingest a document from a URL.",
)
async def ingest_url(
    request: IngestURLRequest,
    user: User = Depends(get_current_user),
    job_queue: JobQueue = Depends(get_job_queue),
    settings: Settings = Depends(get_settings),
    _: None = Depends(rate_limit(requests=10, window=60)),
) -> IngestResponse:
    """Ingest a document from URL.
    
    Downloads the document and processes it the same as file uploads.
    """
    # Queue the ingestion job
    job_id = await job_queue.enqueue(
        job_type="ingest_url",
        data={
            "url": request.url,
            "metadata": request.metadata.model_dump() if request.metadata else None,
            "config": request.config.model_dump() if request.config else None,
            "user_id": user.id,
        }
    )
    
    return IngestResponse(
        job_id=job_id,
        status=StatusEnum.PENDING,
        message=f"URL '{request.url}' queued for processing",
    )


@router.post(
    "/ingest/text",
    response_model=IngestResponse,
    summary="Ingest raw text",
    description="Ingest raw text content directly.",
)
async def ingest_text(
    request: IngestTextRequest,
    user: User = Depends(get_current_user),
    job_queue: JobQueue = Depends(get_job_queue),
    settings: Settings = Depends(get_settings),
    _: None = Depends(rate_limit(requests=20, window=60)),
) -> IngestResponse:
    """Ingest raw text content.
    
    Useful for:
    - Pasting content directly
    - API integrations
    - Text that doesn't come from files
    """
    # Queue the ingestion job
    job_id = await job_queue.enqueue(
        job_type="ingest_text",
        data={
            "content": request.content,
            "filename": request.filename,
            "metadata": request.metadata.model_dump() if request.metadata else None,
            "config": request.config.model_dump() if request.config else None,
            "user_id": user.id,
        }
    )
    
    return IngestResponse(
        job_id=job_id,
        status=StatusEnum.PENDING,
        message=f"Text '{request.filename}' queued for processing",
    )


# =============================================================================
# Job Status Routes
# =============================================================================

@router.get(
    "/ingest/{job_id}/status",
    response_model=IngestJobStatus,
    summary="Check ingestion status",
    description="Get the status of an ingestion job.",
)
async def get_ingest_status(
    job_id: str,
    user: User = Depends(get_current_user),
    job_queue: JobQueue = Depends(get_job_queue),
) -> IngestJobStatus:
    """Get ingestion job status.
    
    Poll this endpoint to track document processing progress.
    """
    job = await job_queue.get_status(job_id)
    
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    return IngestJobStatus(
        job_id=job_id,
        status=StatusEnum(job.get("status", "pending")),
        progress=job.get("progress", 0.0),
        filename=job.get("data", {}).get("filename", "unknown"),
        document_id=job.get("document_id"),
        chunks_created=job.get("chunks_created", 0),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        error=job.get("error"),
    )


# =============================================================================
# Document Management Routes
# =============================================================================

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all documents",
    description="Get a paginated list of all documents in the knowledge base.",
)
async def list_documents(
    page: int = 1,
    page_size: int = 20,
    collection: Optional[str] = None,
    file_type: Optional[str] = None,
    user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> DocumentListResponse:
    """List documents in the knowledge base.
    
    Supports filtering by collection and file type.
    """
    # TODO: Implement actual document listing from vector store
    
    return DocumentListResponse(
        documents=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentInfo,
    summary="Get document info",
    description="Get detailed information about a specific document.",
)
async def get_document(
    document_id: str,
    user: User = Depends(get_current_user),
) -> DocumentInfo:
    """Get document details.
    
    Returns metadata, chunk count, and other information about a document.
    """
    # TODO: Implement actual document retrieval
    
    raise HTTPException(
        status_code=404,
        detail=f"Document {document_id} not found"
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document",
    description="Remove a document and all its chunks from the knowledge base.",
)
async def delete_document(
    document_id: str,
    user: User = Depends(get_current_user),
    _: None = Depends(rate_limit(requests=10, window=60)),
) -> DeleteDocumentResponse:
    """Delete a document.
    
    Removes the document and all associated chunks from the vector store.
    This action cannot be undone.
    """
    # TODO: Implement actual document deletion
    
    raise HTTPException(
        status_code=404,
        detail=f"Document {document_id} not found"
    )


# =============================================================================
# Collection Management Routes
# =============================================================================

@router.get(
    "/collections",
    summary="List collections",
    description="Get all collections in the knowledge base.",
)
async def list_collections(
    user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> dict:
    """List all collections.
    
    Collections are logical groupings of documents.
    """
    # TODO: Implement actual collection listing
    
    return {
        "collections": [
            {
                "name": "default",
                "document_count": 0,
                "chunk_count": 0,
            }
        ]
    }


@router.get(
    "/collections/{collection_name}/stats",
    summary="Get collection stats",
    description="Get statistics for a specific collection.",
)
async def get_collection_stats(
    collection_name: str,
    user: User = Depends(get_current_user),
) -> dict:
    """Get collection statistics.
    
    Returns document count, chunk count, and other metrics.
    """
    # TODO: Implement actual stats retrieval
    
    return {
        "name": collection_name,
        "document_count": 0,
        "chunk_count": 0,
        "total_tokens": 0,
        "embedding_dimensions": 1536,
    }
"""
API Module

FastAPI-based REST API for the Enterprise Knowledge Agent.

Quick Start:
    ```bash
    # Start the server
    uvicorn src.api.main:app --reload
    
    # Or run directly
    python -m src.api.main
    ```

Components:
    - main.py: FastAPI application factory
    - dependencies.py: Dependency injection
    - routes/: API endpoint handlers
    - middleware/: Request processing middleware
    - schemas/: Pydantic request/response models

API Endpoints:
    Health:
        GET  /health           - Basic health check
        GET  /health/ready     - Readiness check
        GET  /health/live      - Liveness check
        GET  /metrics          - Prometheus metrics
    
    Chat:
        POST /api/v1/chat          - Chat with knowledge base
        POST /api/v1/chat/stream   - Streaming chat (SSE)
        GET  /api/v1/chat/conversations/{id} - Get conversation
    
    Search:
        POST /api/v1/search        - Search documents
        POST /api/v1/search/similar - Find similar docs
    
    Ingest:
        POST /api/v1/ingest        - Upload document
        POST /api/v1/ingest/url    - Ingest from URL
        POST /api/v1/ingest/text   - Ingest raw text
        GET  /api/v1/ingest/{job_id}/status - Job status
        GET  /api/v1/documents     - List documents
        GET  /api/v1/documents/{id} - Get document
        DELETE /api/v1/documents/{id} - Delete document
"""

from src.api.main import app, create_app

__all__ = ["app", "create_app"]
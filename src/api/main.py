"""
FastAPI Application Entry Point

Main application setup with:
    - Route registration
    - Middleware configuration
    - Error handlers
    - Lifecycle management

Running the server:
    ```bash
    # Development
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
    
    # Production
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
    
    # With Gunicorn
    gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
    ```
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.api.routes import (
    health_router,
    chat_router,
    search_router,
    ingest_router,
)
from src.api.middleware import (
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    RateLimitConfig,
    get_rate_limiter,
    setup_request_logging,
)
from src.api.dependencies import get_settings, cleanup_providers
from src.api.schemas import ErrorResponse

# Setup logging
logger = logging.getLogger(__name__)
setup_request_logging()


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize resources (DB connections, caches, etc.)
    - Shutdown: Clean up resources
    """
    # Startup
    logger.info("Starting Enterprise Knowledge Agent API...")
    
    settings = get_settings()
    logger.info(f"API Version: {settings.api_version}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info(f"Default LLM: {settings.default_llm_provider}/{settings.default_llm_model}")
    
    # Initialize rate limiter
    rate_limiter = get_rate_limiter()
    await rate_limiter.start()
    
    # Initialize SSE connection manager
    from src.api.streaming import get_connection_manager
    connection_manager = get_connection_manager()
    await connection_manager.start()
    logger.info("SSE Connection Manager started")
    
    # TODO: Initialize other resources
    # - Database connections
    # - Vector store connections
    # - Cache connections
    
    logger.info("API startup complete")
    
    logger.info("Pre-initializing RAG pipeline...")
    from src.api.dependencies import get_rag_pipeline
    rag = await get_rag_pipeline()
    rag._ensure_initialized()
    logger.info("RAG pipeline ready")

    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down API...")
    
    # Stop SSE connection manager
    await connection_manager.stop()
    logger.info("SSE Connection Manager stopped")
    
    # Clean up providers
    await cleanup_providers()
    
    # Stop rate limiter
    await rate_limiter.stop()
    
    # TODO: Clean up other resources
    
    logger.info("API shutdown complete")


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        description="""
Enterprise Knowledge Agent API

A production-ready RAG (Retrieval-Augmented Generation) system with:
- Multi-LLM support (OpenAI, Anthropic, Ollama)
- Hybrid search (Vector + BM25)
- Document ingestion and management
- Streaming responses
- Authentication and rate limiting

## Quick Start

1. Get an API key from your administrator
2. Include the key in requests: `X-API-Key: your-key`
3. Start chatting: `POST /api/v1/chat`

## Authentication

This API supports two authentication methods:
- **API Key**: Include `X-API-Key` header
- **JWT Bearer**: Include `Authorization: Bearer <token>` header

## Rate Limits

- Chat: 30 requests/minute
- Search: 60 requests/minute
- Ingest: 10 requests/minute
        """,
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # -------------------------------------------------------------------------
    # CORS Middleware
    # -------------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production!
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # -------------------------------------------------------------------------
    # Custom Middleware (order matters - first added = outermost)
    # -------------------------------------------------------------------------
    
    # Request logging (outermost - logs all requests)
    app.add_middleware(RequestLoggingMiddleware)
    
    # Rate limiting
    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            config=RateLimitConfig(
                requests_per_minute=settings.rate_limit_requests_per_minute,
            ),
        )
    
    # -------------------------------------------------------------------------
    # Route Registration
    # -------------------------------------------------------------------------
    
    # Health check routes (no prefix, no auth required)
    app.include_router(health_router)
    
    # API v1 routes
    api_v1_prefix = "/api/v1"
    
    app.include_router(
        chat_router,
        prefix=api_v1_prefix,
    )
    
    app.include_router(
        search_router,
        prefix=api_v1_prefix,
    )
    
    app.include_router(
        ingest_router,
        prefix=api_v1_prefix,
    )
    
    # -------------------------------------------------------------------------
    # Exception Handlers
    # -------------------------------------------------------------------------
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with consistent format."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"http_{exc.status_code}",
                message=exc.detail,
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors with detailed feedback."""
        errors = exc.errors()
        
        # Format error details
        details = {}
        for error in errors:
            field = ".".join(str(loc) for loc in error["loc"])
            details[field] = error["msg"]
        
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                message="Request validation failed",
                details=details,
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.exception(f"Unexpected error: {exc}")
        
        # Don't expose internal errors in production
        message = str(exc) if settings.debug else "An unexpected error occurred"
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                message=message,
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )
    
    # -------------------------------------------------------------------------
    # Root Endpoint
    # -------------------------------------------------------------------------
    
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint - redirect to docs."""
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "docs": "/docs",
            "health": "/health",
        }
    
    return app


# Create the application instance
app = create_app()


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
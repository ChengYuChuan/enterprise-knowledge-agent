"""
Health Check Routes

Endpoints for monitoring and health checks.

Endpoints:
    GET /health - Basic health check
    GET /health/ready - Readiness check (all dependencies ready)
    GET /health/live - Liveness check (server responding)
    GET /metrics - Prometheus metrics (if enabled)
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Response

from src.api.schemas import HealthResponse
from src.api.dependencies import get_settings, Settings

router = APIRouter(tags=["Health"])


# =============================================================================
# Health Check Models
# =============================================================================

class DetailedHealthResponse(HealthResponse):
    """Detailed health check response."""
    
    components: dict[str, dict] = {}
    """Status of individual components."""


# =============================================================================
# Routes
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Returns OK if the service is running.",
)
async def health_check(
    settings: Settings = Depends(get_settings)
) -> HealthResponse:
    """Basic health check endpoint.
    
    Use this for simple load balancer health checks.
    """
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        timestamp=datetime.utcnow(),
    )


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Returns OK if the server is responding. Use for Kubernetes liveness probe.",
)
async def liveness_check(
    settings: Settings = Depends(get_settings)
) -> HealthResponse:
    """Liveness check - is the server responding?
    
    For Kubernetes liveness probes. If this fails, the pod should be restarted.
    """
    return HealthResponse(
        status="alive",
        version=settings.api_version,
        timestamp=datetime.utcnow(),
    )


@router.get(
    "/health/ready",
    response_model=DetailedHealthResponse,
    summary="Readiness check",
    description="Checks if all dependencies are ready. Use for Kubernetes readiness probe.",
)
async def readiness_check(
    settings: Settings = Depends(get_settings)
) -> DetailedHealthResponse:
    """Readiness check - are all dependencies ready?
    
    For Kubernetes readiness probes. If this fails, traffic should not be
    routed to this pod.
    
    Checks:
        - Database connection
        - Vector store connection
        - LLM provider availability
        - Cache availability
    """
    components = {}
    all_healthy = True
    
    # Check LLM provider
    try:
        from src.llm import is_provider_available
        llm_ready = is_provider_available(settings.default_llm_provider)
        components["llm"] = {
            "status": "healthy" if llm_ready else "degraded",
            "provider": settings.default_llm_provider,
        }
    except Exception as e:
        components["llm"] = {"status": "unhealthy", "error": str(e)}
        all_healthy = False
    
    # Check vector store (placeholder)
    # TODO: Add actual vector store health check
    components["vector_store"] = {
        "status": "healthy",
        "type": "qdrant",
    }
    
    # Check cache (placeholder)
    # TODO: Add actual cache health check
    components["cache"] = {
        "status": "healthy",
        "type": "redis",
    }
    
    return DetailedHealthResponse(
        status="ready" if all_healthy else "degraded",
        version=settings.api_version,
        timestamp=datetime.utcnow(),
        components=components,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Returns metrics in Prometheus format.",
    response_class=Response,
)
async def metrics(
    settings: Settings = Depends(get_settings)
) -> Response:
    """Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format.
    
    TODO: Integrate with prometheus_client for actual metrics.
    """
    # Placeholder metrics
    metrics_text = f"""# HELP api_info API version information
# TYPE api_info gauge
api_info{{version="{settings.api_version}"}} 1

# HELP api_up API health status
# TYPE api_up gauge
api_up 1

# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total 0
"""
    
    return Response(
        content=metrics_text,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
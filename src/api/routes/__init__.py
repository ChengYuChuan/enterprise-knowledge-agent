"""
API Routes

Route handlers for all API endpoints.

Organization:
    - health.py: Health checks and metrics
    - chat.py: Chat and conversation endpoints
    - search.py: Document search endpoints
    - ingest.py: Document ingestion and management
"""

from src.api.routes.health import router as health_router
from src.api.routes.chat import router as chat_router
from src.api.routes.search import router as search_router
from src.api.routes.ingest import router as ingest_router

__all__ = [
    "health_router",
    "chat_router",
    "search_router",
    "ingest_router",
]
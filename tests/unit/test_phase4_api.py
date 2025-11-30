"""
Unit tests for Phase 4: FastAPI REST API

Tests API routes, schemas, and dependencies.
Run with: poetry run pytest tests/unit/test_phase4_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api import create_app
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client):
        """Test main health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    def test_live_check(self, client):
        """Test liveness probe endpoint."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
    
    def test_ready_check(self, client):
        """Test readiness probe endpoint."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_swagger_ui(self, client):
        """Test Swagger UI is available."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "swagger" in response.text.lower()
    
    def test_redoc(self, client):
        """Test ReDoc is available."""
        response = client.get("/redoc")
        
        assert response.status_code == 200


class TestChatSchemas:
    """Test chat request/response schemas."""
    
    def test_chat_request_minimal(self):
        """Test creating minimal chat request."""
        request = ChatRequest(message="Hello")
        
        assert request.message == "Hello"
        assert request.conversation_id is None
        assert request.provider is None
    
    def test_chat_request_full(self):
        """Test creating full chat request."""
        request = ChatRequest(
            message="Hello",
            conversation_id="conv-123",
            provider="openai",
            model="gpt-4",
            temperature=0.5,
        )
        
        assert request.message == "Hello"
        assert request.conversation_id == "conv-123"
        assert request.provider == "openai"


class TestSearchSchemas:
    """Test search request/response schemas."""
    
    def test_search_request_minimal(self):
        """Test creating minimal search request."""
        request = SearchRequest(query="test query")
        
        assert request.query == "test query"
        assert request.top_k == 10  # actual default
    
    def test_search_request_with_options(self):
        """Test creating search request with options."""
        request = SearchRequest(
            query="test",
            top_k=20,
        )
        
        assert request.query == "test"
        assert request.top_k == 20


class TestChatEndpoint:
    """Test chat endpoint (with mocked LLM)."""
    
    def test_chat_requires_auth(self, client):
        """Test that chat endpoint requires authentication."""
        response = client.post("/api/v1/chat", json={"message": "test"})
        
        # Should return 401 Unauthorized without API key
        assert response.status_code == 401


class TestSearchEndpoint:
    """Test search endpoint (with mocked retrieval)."""
    
    def test_search_requires_auth(self, client):
        """Test that search endpoint requires authentication."""
        response = client.post("/api/v1/search", json={"query": "test"})
        
        # Should return 401 Unauthorized without API key
        assert response.status_code == 401


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_redirect_or_info(self, client):
        """Test root endpoint returns info or redirects."""
        response = client.get("/", follow_redirects=False)
        
        # Should either return 200 with info or 307 redirect to /docs
        assert response.status_code in (200, 307)

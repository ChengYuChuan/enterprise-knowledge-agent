"""
Chat Schemas

Pydantic models for chat-related endpoints.
Supports both single-response and streaming chat interactions.

Endpoints:
    POST /api/v1/chat - Single response chat
    POST /api/v1/chat/stream - Streaming chat (SSE)
"""

from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator

from src.api.schemas.common import (
    BaseSchema,
    SourceDocument,
    TokenUsage,
    ProviderEnum,
)


# =============================================================================
# Request Models
# =============================================================================

class ChatMessage(BaseSchema):
    """A single message in the conversation."""
    role: str = Field(
        description="Message role: 'user', 'assistant', or 'system'"
    )
    content: str = Field(
        description="Message content"
    )
    
    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"user", "assistant", "system"}
        if v.lower() not in allowed:
            raise ValueError(f"role must be one of {allowed}")
        return v.lower()
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "user",
                "content": "What is RAG?"
            }
        }
    )


class ChatRequest(BaseSchema):
    """Request body for chat endpoint."""
    message: str = Field(
        min_length=1,
        max_length=32000,
        description="User's message/query"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="ID to continue an existing conversation"
    )
    history: Optional[list[ChatMessage]] = Field(
        default=None,
        max_length=50,
        description="Previous messages for context (if no conversation_id)"
    )
    
    # RAG options
    use_rag: bool = Field(
        default=True,
        description="Whether to use RAG for context retrieval"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve for RAG"
    )
    
    # LLM options
    provider: Optional[ProviderEnum] = Field(
        default=None,
        description="LLM provider override (uses default if not specified)"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model override (uses provider default if not specified)"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature for generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=16000,
        description="Maximum tokens in response"
    )
    
    # System prompt
    system_prompt: Optional[str] = Field(
        default=None,
        max_length=4000,
        description="Custom system prompt (overrides default)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "What were the key findings in the Q3 report?",
                "conversation_id": None,
                "use_rag": True,
                "top_k": 5,
                "temperature": 0.7
            }
        }
    )


class StreamChatRequest(ChatRequest):
    """Request body for streaming chat endpoint.
    
    Inherits all fields from ChatRequest with streaming-specific defaults.
    """
    
    # Streaming doesn't need max_tokens limit as aggressively
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=32000,
        description="Maximum tokens in response"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Explain the revenue trends over the past year.",
                "use_rag": True,
                "top_k": 5
            }
        }
    )


# =============================================================================
# Response Models
# =============================================================================

class ChatResponse(BaseSchema):
    """Response from chat endpoint."""
    response: str = Field(
        description="Generated response text"
    )
    conversation_id: str = Field(
        description="Conversation ID for continuation"
    )
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Source documents used (if RAG enabled)"
    )
    usage: Optional[TokenUsage] = Field(
        default=None,
        description="Token usage statistics"
    )
    model: str = Field(
        description="Model used for generation"
    )
    provider: str = Field(
        description="Provider used for generation"
    )
    latency_ms: float = Field(
        ge=0,
        description="Response latency in milliseconds"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "Based on the Q3 report, the key findings include...",
                "conversation_id": "conv_abc123",
                "sources": [
                    {
                        "document_id": "doc_123",
                        "filename": "q3_report.pdf",
                        "content": "Revenue increased by 25%...",
                        "score": 0.92,
                        "metadata": {}
                    }
                ],
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 200,
                    "total_tokens": 350
                },
                "model": "gpt-4",
                "provider": "openai",
                "latency_ms": 1250.5
            }
        }
    )


class StreamChunk(BaseSchema):
    """A single chunk in a streaming response.
    
    Sent as Server-Sent Events (SSE) with event type 'message'.
    """
    content: str = Field(
        description="Text content delta"
    )
    done: bool = Field(
        default=False,
        description="Whether this is the final chunk"
    )
    
    # Only present on final chunk
    sources: Optional[list[SourceDocument]] = Field(
        default=None,
        description="Source documents (only on final chunk)"
    )
    usage: Optional[TokenUsage] = Field(
        default=None,
        description="Token usage (only on final chunk)"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID (only on final chunk)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Based on",
                "done": False
            }
        }
    )


# =============================================================================
# Conversation Models
# =============================================================================

class ConversationInfo(BaseSchema):
    """Information about a conversation."""
    conversation_id: str = Field(
        description="Unique conversation identifier"
    )
    message_count: int = Field(
        ge=0,
        description="Number of messages in conversation"
    )
    created_at: str = Field(
        description="When the conversation started"
    )
    last_message_at: str = Field(
        description="When the last message was sent"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "conversation_id": "conv_abc123",
                "message_count": 5,
                "created_at": "2024-01-15T10:00:00Z",
                "last_message_at": "2024-01-15T10:30:00Z"
            }
        }
    )
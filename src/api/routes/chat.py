"""
Chat Routes

Endpoints for conversational interactions with the knowledge base.

Endpoints:
    POST /chat - Single response chat
    POST /chat/stream - Streaming chat (SSE)
    GET /chat/conversations/{id} - Get conversation history
"""

import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ChatRequest,
    StreamChatRequest,
    ChatResponse,
    StreamChunk,
    SourceDocument,
    TokenUsage,
)
from src.api.middleware import User, get_current_user, rate_limit
from src.api.dependencies import (
    get_settings,
    Settings,
    get_llm_provider,
    get_rag_pipeline,
    get_conversation_store,
    RAGPipeline,
    ConversationStore,
)
from src.llm import BaseLLMProvider, Message, LLMResponse

router = APIRouter(prefix="/chat", tags=["Chat"])


# =============================================================================
# Helper Functions
# =============================================================================

def build_messages(
    user_message: str,
    system_prompt: str = None,
    history: list = None,
    rag_context: str = None,
) -> list[Message]:
    """Build message list for LLM.
    
    Args:
        user_message: Current user message.
        system_prompt: Optional system prompt.
        history: Previous conversation messages.
        rag_context: Retrieved context from RAG.
    
    Returns:
        List of Message objects.
    """
    messages = []
    
    # System prompt
    if system_prompt:
        messages.append(Message.system(system_prompt))
    else:
        # Default system prompt
        default_prompt = (
            "You are a helpful AI assistant with access to a knowledge base. "
            "Answer questions based on the provided context when available. "
            "If you don't know the answer, say so clearly."
        )
        messages.append(Message.system(default_prompt))
    
    # Add RAG context if available
    if rag_context:
        context_message = f"Here is relevant context from the knowledge base:\n\n{rag_context}"
        messages.append(Message.system(context_message))
    
    # Add conversation history
    if history:
        for msg in history:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(Message.user(content))
                elif role == "assistant":
                    messages.append(Message.assistant(content))
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                if msg.role == "user":
                    messages.append(Message.user(msg.content))
                elif msg.role == "assistant":
                    messages.append(Message.assistant(msg.content))
    
    # Add current user message
    messages.append(Message.user(user_message))
    
    return messages


def format_rag_context(sources: list[dict]) -> str:
    """Format RAG sources into context string.
    
    Args:
        sources: List of source documents from RAG.
    
    Returns:
        Formatted context string.
    """
    if not sources:
        return ""
    
    context_parts = []
    for i, source in enumerate(sources, 1):
        content = source.get("content", "")
        filename = source.get("filename", "Unknown")
        context_parts.append(f"[Source {i}: {filename}]\n{content}")
    
    return "\n\n".join(context_parts)


# =============================================================================
# Routes
# =============================================================================

@router.post(
    "",
    response_model=ChatResponse,
    summary="Chat with the knowledge base",
    description="Send a message and receive a response, optionally using RAG for context.",
)
async def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    llm: BaseLLMProvider = Depends(get_llm_provider),
    rag: RAGPipeline = Depends(get_rag_pipeline),
    conversations: ConversationStore = Depends(get_conversation_store),
    settings: Settings = Depends(get_settings),
    _: None = Depends(rate_limit(requests=30, window=60)),  # 30 req/min for chat
) -> ChatResponse:
    """Chat endpoint with optional RAG.
    
    Flow:
    1. Retrieve relevant context from knowledge base (if use_rag=True)
    2. Build conversation context
    3. Generate response from LLM
    4. Return response with sources
    """
    start_time = time.perf_counter()
    
    # Get or create conversation
    conversation_id = request.conversation_id
    history = []
    
    if conversation_id:
        stored_history = await conversations.get(conversation_id)
        if stored_history:
            history = stored_history
    else:
        conversation_id = await conversations.create()
    
    # Retrieve RAG context if enabled
    sources = []
    rag_context = ""
    
    if request.use_rag:
        try:
            rag_result = await rag.query(
                request.message,
                top_k=request.top_k,
            )
            sources = rag_result.get("sources", [])
            rag_context = format_rag_context(sources)
        except Exception as e:
            # Log error but continue without RAG
            import logging
            logging.getLogger(__name__).warning(f"RAG query failed: {e}")
    
    # Build messages
    messages = build_messages(
        user_message=request.message,
        system_prompt=request.system_prompt,
        history=request.history or history,
        rag_context=rag_context,
    )
    
    # Generate response
    try:
        llm_response: LLMResponse = await llm.generate(
            messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM generation failed: {str(e)}"
        )
    
    # Save to conversation history
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": llm_response.content})
    await conversations.save(conversation_id, history)
    
    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Build source documents
    source_docs = [
        SourceDocument(
            document_id=s.get("document_id", ""),
            filename=s.get("filename", ""),
            chunk_id=s.get("chunk_id"),
            content=s.get("content", ""),
            score=s.get("score", 0.0),
            metadata=s.get("metadata", {}),
        )
        for s in sources
    ]
    
    return ChatResponse(
        response=llm_response.content,
        conversation_id=conversation_id,
        sources=source_docs,
        usage=TokenUsage(
            prompt_tokens=llm_response.usage.prompt_tokens,
            completion_tokens=llm_response.usage.completion_tokens,
            total_tokens=llm_response.usage.total_tokens,
        ) if llm_response.usage.is_available else None,
        model=llm_response.model,
        provider=str(llm.provider_type.value),
        latency_ms=latency_ms,
    )


@router.post(
    "/stream",
    summary="Streaming chat",
    description="Stream a response using Server-Sent Events (SSE).",
)
async def chat_stream(
    request: StreamChatRequest,
    http_request: Request,
    user: User = Depends(get_current_user),
    llm: BaseLLMProvider = Depends(get_llm_provider),
    rag: RAGPipeline = Depends(get_rag_pipeline),
    conversations: ConversationStore = Depends(get_conversation_store),
    settings: Settings = Depends(get_settings),
    _: None = Depends(rate_limit(requests=20, window=60)),  # 20 req/min for streaming
):
    """Streaming chat endpoint.
    
    Uses Server-Sent Events (SSE) to stream the response with proper
    connection management, heartbeats, and error handling.
    
    Event Types:
        - start: Stream started (includes model/provider info)
        - sources: RAG sources (sent before content if RAG enabled)
        - message: Content chunk
        - done: Stream completed (includes usage stats)
        - error: Error occurred
        - heartbeat: Keep-alive ping
    
    Example Event Sequence:
        ```
        event: start
        data: {"conversation_id": "conv_123", "model": "gpt-4", "provider": "openai"}
        
        event: sources
        data: {"sources": [...], "query": "...", "retrieval_time_ms": 150}
        
        event: message
        data: {"content": "Based on", "index": 0}
        
        event: message
        data: {"content": " the documents", "index": 1}
        
        event: done
        data: {"conversation_id": "conv_123", "usage": {...}, "latency_ms": 1500}
        ```
    """
    from src.api.streaming import (
        SSEStreamHandler,
        SSEEvent,
        stream_llm_response,
        get_stream_handler,
    )
    
    # Get or create conversation
    conversation_id = request.conversation_id
    history = []
    
    if conversation_id:
        stored_history = await conversations.get(conversation_id)
        if stored_history:
            history = stored_history
    else:
        conversation_id = await conversations.create()
    
    # Retrieve RAG context if enabled
    sources = []
    rag_context = ""
    retrieval_time_ms = 0
    
    if request.use_rag:
        retrieval_start = time.perf_counter()
        try:
            rag_result = await rag.query(
                request.message,
                top_k=request.top_k,
            )
            sources = rag_result.get("sources", [])
            rag_context = format_rag_context(sources)
            retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"RAG query failed: {e}")
    
    # Build messages
    messages = build_messages(
        user_message=request.message,
        system_prompt=request.system_prompt,
        history=request.history or history,
        rag_context=rag_context,
    )
    
    # Callback to save conversation on completion
    async def on_complete(full_response: str):
        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": full_response})
        await conversations.save(conversation_id, history)
    
    # Format sources for SSE
    formatted_sources = [
        {
            "document_id": s.get("document_id", ""),
            "filename": s.get("filename", ""),
            "content": s.get("content", "")[:500],  # Truncate for streaming
            "score": s.get("score", 0.0),
        }
        for s in sources
    ] if sources else None
    
    # Create LLM stream
    llm_stream = llm.stream(
        messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    
    # Create SSE event generator
    event_generator = stream_llm_response(
        llm_stream=llm_stream,
        conversation_id=conversation_id,
        model=llm.config.model,
        provider=str(llm.provider_type.value),
        sources=formatted_sources,
        on_complete=on_complete,
    )
    
    # Return SSE response with connection management
    handler = get_stream_handler()
    return await handler.create_response(
        event_generator=event_generator,
        user_id=user.id,
        conversation_id=conversation_id,
        request=http_request,
    )


@router.get(
    "/conversations/{conversation_id}",
    summary="Get conversation history",
    description="Retrieve the message history for a conversation.",
)
async def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    conversations: ConversationStore = Depends(get_conversation_store),
) -> dict:
    """Get conversation history.
    
    Returns the full message history for a conversation.
    """
    history = await conversations.get(conversation_id)
    
    if history is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found"
        )
    
    return {
        "conversation_id": conversation_id,
        "messages": history,
        "message_count": len(history),
    }
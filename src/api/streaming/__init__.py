"""
SSE Streaming Module

Server-Sent Events implementation for real-time streaming responses.

Components:
    - events.py: Event type definitions and formatting
    - manager.py: Connection management and tracking
    - sse.py: Core streaming utilities

Quick Start:
    ```python
    from src.api.streaming import (
        SSEStreamHandler,
        SSEEvent,
        stream_llm_response,
    )
    
    # Create a stream handler
    handler = SSEStreamHandler()
    
    # Stream LLM response
    async def chat_stream(request):
        async def generate():
            async for chunk in llm.stream(messages):
                yield SSEEvent.message(chunk.content)
            yield SSEEvent.done(conversation_id)
        
        return handler.create_response(
            generate(),
            user_id=user.id,
            conversation_id=conv_id,
        )
    ```

Event Types:
    - start: Stream started
    - message: Content chunk
    - sources: RAG sources
    - done: Stream completed
    - error: Error occurred
    - heartbeat: Keep-alive ping
"""

# Events
from src.api.streaming.events import (
    SSEEventType,
    SSEEvent,
    format_sse_event,
    # Event payloads
    StartEvent,
    MessageEvent,
    ThinkingEvent,
    SourcesEvent,
    MetadataEvent,
    DoneEvent,
    ErrorEvent,
    HeartbeatEvent,
)

# Connection Manager
from src.api.streaming.manager import (
    SSEConnectionManager,
    ConnectionInfo,
    ConnectionLimitError,
    get_connection_manager,
    set_connection_manager,
)

# SSE Handler
from src.api.streaming.sse import (
    SSEConfig,
    SSEStreamHandler,
    stream_llm_response,
    get_stream_handler,
)

__all__ = [
    # Event types
    "SSEEventType",
    "SSEEvent",
    "format_sse_event",
    # Event payloads
    "StartEvent",
    "MessageEvent",
    "ThinkingEvent",
    "SourcesEvent",
    "MetadataEvent",
    "DoneEvent",
    "ErrorEvent",
    "HeartbeatEvent",
    # Connection manager
    "SSEConnectionManager",
    "ConnectionInfo",
    "ConnectionLimitError",
    "get_connection_manager",
    "set_connection_manager",
    # SSE handler
    "SSEConfig",
    "SSEStreamHandler",
    "stream_llm_response",
    "get_stream_handler",
]
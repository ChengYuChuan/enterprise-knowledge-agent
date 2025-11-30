"""
SSE Core Implementation

Core utilities for Server-Sent Events streaming.

Features:
    - Async generator for SSE streams
    - Automatic heartbeat injection
    - Error handling and recovery
    - Connection state management
    - Proper cleanup on disconnection

Usage:
    ```python
    from src.api.streaming import SSEStreamHandler
    
    handler = SSEStreamHandler(connection_manager)
    
    async def chat_stream(request):
        async def generate():
            async for chunk in llm.stream(messages):
                yield SSEEvent.message(chunk.content)
            yield SSEEvent.done(conversation_id)
        
        return handler.stream_response(
            generate(),
            user_id=user.id,
            conversation_id=conv_id,
        )
    ```
"""

import asyncio
import time
import logging
from typing import AsyncGenerator, Optional, Callable, Any, Tuple
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.api.streaming.events import (
    SSEEventType,
    SSEEvent,
    format_sse_event,
)
from src.api.streaming.manager import (
    SSEConnectionManager,
    ConnectionInfo,
    ConnectionLimitError,
    get_connection_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SSEConfig:
    """SSE streaming configuration."""
    
    # Heartbeat settings
    heartbeat_enabled: bool = True
    heartbeat_interval: float = 15.0  # seconds
    
    # Timeout settings
    stream_timeout: float = 300.0     # 5 minutes max stream duration
    chunk_timeout: float = 30.0       # Max time to wait for a chunk
    
    # Retry settings
    retry_interval: int = 3000        # Client retry interval (ms)
    
    # Buffer settings
    max_buffer_size: int = 100        # Max buffered events


# =============================================================================
# SSE Stream Handler
# =============================================================================

class SSEStreamHandler:
    """Handler for SSE streaming responses.
    
    Provides:
        - Connection management integration
        - Automatic heartbeats
        - Error handling
        - Timeout management
        - Proper cleanup
    
    Usage:
        ```python
        handler = SSEStreamHandler()
        
        async def generate_stream():
            for chunk in data:
                yield SSEEvent.message(chunk)
            yield SSEEvent.done(conv_id)
        
        return handler.create_response(
            generate_stream(),
            user_id="user_123",
            conversation_id="conv_456",
        )
        ```
    """
    
    def __init__(
        self,
        connection_manager: SSEConnectionManager = None,
        config: SSEConfig = None,
    ):
        """Initialize the stream handler.
        
        Args:
            connection_manager: Connection manager instance.
            config: SSE configuration.
        """
        self.connection_manager = connection_manager or get_connection_manager()
        self.config = config or SSEConfig()
    
    async def create_response(
        self,
        event_generator: AsyncGenerator[Tuple[SSEEventType, str], None],
        user_id: str,
        conversation_id: str,
        request: Optional[Request] = None,
    ) -> EventSourceResponse:
        """Create an SSE response from an event generator.
        
        Args:
            event_generator: Async generator yielding (event_type, data) tuples.
            user_id: User identifier for connection tracking.
            conversation_id: Conversation identifier.
            request: Optional FastAPI request for client disconnect detection.
        
        Returns:
            EventSourceResponse for FastAPI.
        """
        return EventSourceResponse(
            self._wrap_generator(
                event_generator,
                user_id,
                conversation_id,
                request,
            ),
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    
    async def _wrap_generator(
        self,
        event_generator: AsyncGenerator[Tuple[SSEEventType, str], None],
        user_id: str,
        conversation_id: str,
        request: Optional[Request],
    ) -> AsyncGenerator[dict, None]:
        """Wrap the event generator with connection management and heartbeats.
        
        Yields:
            Dicts in sse-starlette format: {"event": "...", "data": "..."}
        """
        start_time = time.time()
        chunk_index = 0
        
        try:
            async with self.connection_manager.connection(
                user_id, conversation_id
            ) as conn:
                
                # Create heartbeat task if enabled
                heartbeat_task = None
                if self.config.heartbeat_enabled:
                    heartbeat_queue: asyncio.Queue = asyncio.Queue()
                    heartbeat_task = asyncio.create_task(
                        self._heartbeat_producer(heartbeat_queue, conn)
                    )
                
                try:
                    # Yield retry interval hint
                    yield {
                        "event": "retry",
                        "data": str(self.config.retry_interval),
                    }
                    
                    # Process events
                    async for event_type, data in self._with_timeout(
                        event_generator,
                        conn,
                        heartbeat_queue if self.config.heartbeat_enabled else None,
                    ):
                        # Check for cancellation
                        if conn.cancelled:
                            logger.debug(f"Connection {conn.connection_id} cancelled")
                            break
                        
                        # Check for client disconnect
                        if request and await request.is_disconnected():
                            logger.debug(f"Client disconnected for {conn.connection_id}")
                            break
                        
                        # Check stream timeout
                        if time.time() - start_time > self.config.stream_timeout:
                            logger.warning(f"Stream timeout for {conn.connection_id}")
                            yield {
                                "event": SSEEventType.ERROR.value,
                                "data": SSEEvent.error(
                                    "timeout",
                                    "Stream timeout exceeded",
                                    retryable=True,
                                )[1],
                            }
                            break
                        
                        # Update connection stats
                        conn.update_activity(len(data))
                        
                        # Yield the event
                        yield {
                            "event": event_type.value,
                            "id": f"{conn.connection_id}_{chunk_index}",
                            "data": data,
                        }
                        
                        chunk_index += 1
                        
                finally:
                    # Cancel heartbeat task
                    if heartbeat_task:
                        heartbeat_task.cancel()
                        try:
                            await heartbeat_task
                        except asyncio.CancelledError:
                            pass
                
        except ConnectionLimitError as e:
            logger.warning(f"Connection limit exceeded for user {user_id}: {e}")
            yield {
                "event": SSEEventType.ERROR.value,
                "data": SSEEvent.error(
                    "connection_limit",
                    str(e),
                    code="429",
                    retryable=True,
                )[1],
            }
            
        except Exception as e:
            logger.exception(f"Error in SSE stream: {e}")
            yield {
                "event": SSEEventType.ERROR.value,
                "data": SSEEvent.error(
                    "internal_error",
                    str(e),
                    retryable=False,
                )[1],
            }
    
    async def _with_timeout(
        self,
        generator: AsyncGenerator[Tuple[SSEEventType, str], None],
        conn: ConnectionInfo,
        heartbeat_queue: Optional[asyncio.Queue],
    ) -> AsyncGenerator[Tuple[SSEEventType, str], None]:
        """Wrap generator with timeout and heartbeat interleaving.
        
        This merges the content stream with heartbeat events.
        """
        gen_iter = generator.__aiter__()
        gen_next = asyncio.create_task(gen_iter.__anext__())
        
        try:
            while True:
                # Wait for either: next chunk, heartbeat, or timeout
                wait_tasks = [gen_next]
                
                if heartbeat_queue:
                    heartbeat_wait = asyncio.create_task(heartbeat_queue.get())
                    wait_tasks.append(heartbeat_wait)
                else:
                    heartbeat_wait = None
                
                try:
                    done, pending = await asyncio.wait(
                        wait_tasks,
                        timeout=self.config.chunk_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    
                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, StopAsyncIteration):
                            pass
                    
                    if not done:
                        # Timeout - yield heartbeat and continue
                        yield SSEEvent.heartbeat()
                        gen_next = asyncio.create_task(gen_iter.__anext__())
                        continue
                    
                    # Process completed task
                    completed_task = done.pop()
                    
                    if heartbeat_wait and completed_task == heartbeat_wait:
                        # Heartbeat ready
                        yield SSEEvent.heartbeat()
                        gen_next = asyncio.create_task(gen_iter.__anext__())
                        
                    else:
                        # Generator yielded
                        try:
                            result = completed_task.result()
                            yield result
                            gen_next = asyncio.create_task(gen_iter.__anext__())
                        except StopAsyncIteration:
                            break
                            
                except asyncio.TimeoutError:
                    # Yield heartbeat on timeout
                    yield SSEEvent.heartbeat()
                    
        finally:
            # Clean up
            gen_next.cancel()
            try:
                await gen_next
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
    
    async def _heartbeat_producer(
        self,
        queue: asyncio.Queue,
        conn: ConnectionInfo,
    ):
        """Produce heartbeat events at regular intervals."""
        try:
            while not conn.cancelled:
                await asyncio.sleep(self.config.heartbeat_interval)
                if not conn.cancelled:
                    await queue.put(True)
        except asyncio.CancelledError:
            pass


# =============================================================================
# Helper Functions
# =============================================================================

async def stream_llm_response(
    llm_stream: AsyncGenerator,
    conversation_id: str,
    model: str,
    provider: str,
    sources: list[dict] = None,
    on_complete: Callable[[str], Any] = None,
) -> AsyncGenerator[Tuple[SSEEventType, str], None]:
    """Stream LLM response as SSE events.
    
    This is a helper to convert LLM streaming output to SSE events.
    
    Args:
        llm_stream: Async generator from LLM provider.
        conversation_id: Conversation ID for tracking.
        model: Model name.
        provider: Provider name.
        sources: Optional RAG sources to include.
        on_complete: Optional callback when stream completes.
    
    Yields:
        Tuples of (SSEEventType, json_data).
    """
    start_time = time.time()
    full_response = ""
    chunk_index = 0
    usage = None
    
    # Start event
    yield SSEEvent.start(conversation_id, model, provider)
    
    # Sources event (if available)
    if sources:
        yield SSEEvent.sources(
            sources=sources,
            query="",  # Could be passed in
            retrieval_time_ms=0,
        )
    
    # Stream content
    try:
        async for chunk in llm_stream:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            full_response += content
            
            finish_reason = None
            if hasattr(chunk, 'is_final') and chunk.is_final:
                finish_reason = getattr(chunk, 'finish_reason', 'stop')
            if hasattr(chunk, 'usage'):
                usage = chunk.usage
            
            yield SSEEvent.message(
                content=content,
                index=chunk_index,
                finish_reason=finish_reason,
            )
            
            chunk_index += 1
            
    except Exception as e:
        logger.error(f"Error during LLM streaming: {e}")
        yield SSEEvent.error(
            error="llm_error",
            message=str(e),
            retryable=False,
        )
        return
    
    # Done event
    latency_ms = (time.time() - start_time) * 1000
    
    usage_dict = None
    if usage:
        usage_dict = {
            "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
            "completion_tokens": getattr(usage, 'completion_tokens', 0),
            "total_tokens": getattr(usage, 'total_tokens', 0),
        }
    
    yield SSEEvent.done(
        conversation_id=conversation_id,
        finish_reason="stop",
        usage=usage_dict,
        latency_ms=latency_ms,
    )
    
    # Callback
    if on_complete:
        try:
            result = on_complete(full_response)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in on_complete callback: {e}")


# =============================================================================
# Global Handler
# =============================================================================

_stream_handler: Optional[SSEStreamHandler] = None


def get_stream_handler() -> SSEStreamHandler:
    """Get the global stream handler instance."""
    global _stream_handler
    if _stream_handler is None:
        _stream_handler = SSEStreamHandler()
    return _stream_handler
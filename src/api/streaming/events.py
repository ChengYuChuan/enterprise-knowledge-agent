"""
SSE Event Definitions

Defines all event types used in Server-Sent Events streaming.

Event Types:
    - message: Content chunk from LLM
    - sources: Retrieved sources (sent before content)
    - done: Stream completion
    - error: Error occurred
    - heartbeat: Keep-alive ping

Protocol:
    Each event follows the SSE format:
    ```
    event: <type>
    id: <optional-id>
    data: <json-payload>
    
    ```
    (Note: blank line terminates each event)
"""

from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json


# =============================================================================
# Event Types
# =============================================================================

class SSEEventType(str, Enum):
    """Server-Sent Event types."""
    
    # Content events
    MESSAGE = "message"          # Text content chunk
    THINKING = "thinking"        # Model thinking/reasoning (if exposed)
    
    # Metadata events
    SOURCES = "sources"          # RAG sources
    METADATA = "metadata"        # Additional metadata
    
    # Control events
    START = "start"              # Stream started
    DONE = "done"                # Stream completed
    ERROR = "error"              # Error occurred
    
    # Connection events
    HEARTBEAT = "heartbeat"      # Keep-alive ping
    RETRY = "retry"              # Reconnection hint


# =============================================================================
# Event Payloads
# =============================================================================

@dataclass
class BaseEvent:
    """Base class for all SSE events."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class StartEvent(BaseEvent):
    """Stream start event - sent at the beginning of a stream."""
    conversation_id: str
    model: str
    provider: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class MessageEvent(BaseEvent):
    """Content chunk event - the main streaming content."""
    content: str
    index: int = 0                    # Chunk index
    finish_reason: Optional[str] = None  # null, "stop", "length", etc.


@dataclass
class ThinkingEvent(BaseEvent):
    """Thinking/reasoning event (for models that expose this)."""
    content: str
    step: Optional[int] = None


@dataclass
class SourcesEvent(BaseEvent):
    """RAG sources event - sent when sources are retrieved."""
    sources: list[dict]
    query: str
    retrieval_time_ms: float


@dataclass
class MetadataEvent(BaseEvent):
    """Additional metadata event."""
    key: str
    value: Any


@dataclass
class DoneEvent(BaseEvent):
    """Stream completion event."""
    conversation_id: str
    finish_reason: str = "stop"
    usage: Optional[dict] = None      # Token usage stats
    latency_ms: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ErrorEvent(BaseEvent):
    """Error event."""
    error: str
    message: str
    code: Optional[str] = None
    retryable: bool = False


@dataclass
class HeartbeatEvent(BaseEvent):
    """Keep-alive heartbeat event."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# =============================================================================
# Event Factory
# =============================================================================

class SSEEvent:
    """Factory for creating SSE events."""
    
    @staticmethod
    def start(
        conversation_id: str,
        model: str,
        provider: str,
    ) -> tuple[SSEEventType, str]:
        """Create a stream start event."""
        event = StartEvent(
            conversation_id=conversation_id,
            model=model,
            provider=provider,
        )
        return SSEEventType.START, event.to_json()
    
    @staticmethod
    def message(
        content: str,
        index: int = 0,
        finish_reason: Optional[str] = None,
    ) -> tuple[SSEEventType, str]:
        """Create a message content event."""
        event = MessageEvent(
            content=content,
            index=index,
            finish_reason=finish_reason,
        )
        return SSEEventType.MESSAGE, event.to_json()
    
    @staticmethod
    def thinking(
        content: str,
        step: Optional[int] = None,
    ) -> tuple[SSEEventType, str]:
        """Create a thinking event."""
        event = ThinkingEvent(content=content, step=step)
        return SSEEventType.THINKING, event.to_json()
    
    @staticmethod
    def sources(
        sources: list[dict],
        query: str,
        retrieval_time_ms: float,
    ) -> tuple[SSEEventType, str]:
        """Create a sources event."""
        event = SourcesEvent(
            sources=sources,
            query=query,
            retrieval_time_ms=retrieval_time_ms,
        )
        return SSEEventType.SOURCES, event.to_json()
    
    @staticmethod
    def metadata(
        key: str,
        value: Any,
    ) -> tuple[SSEEventType, str]:
        """Create a metadata event."""
        event = MetadataEvent(key=key, value=value)
        return SSEEventType.METADATA, event.to_json()
    
    @staticmethod
    def done(
        conversation_id: str,
        finish_reason: str = "stop",
        usage: Optional[dict] = None,
        latency_ms: Optional[float] = None,
    ) -> tuple[SSEEventType, str]:
        """Create a stream completion event."""
        event = DoneEvent(
            conversation_id=conversation_id,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
        )
        return SSEEventType.DONE, event.to_json()
    
    @staticmethod
    def error(
        error: str,
        message: str,
        code: Optional[str] = None,
        retryable: bool = False,
    ) -> tuple[SSEEventType, str]:
        """Create an error event."""
        event = ErrorEvent(
            error=error,
            message=message,
            code=code,
            retryable=retryable,
        )
        return SSEEventType.ERROR, event.to_json()
    
    @staticmethod
    def heartbeat() -> tuple[SSEEventType, str]:
        """Create a heartbeat event."""
        event = HeartbeatEvent()
        return SSEEventType.HEARTBEAT, event.to_json()


# =============================================================================
# SSE Formatter
# =============================================================================

def format_sse_event(
    event_type: SSEEventType,
    data: str,
    event_id: Optional[str] = None,
    retry: Optional[int] = None,
) -> str:
    """Format an SSE event string.
    
    Args:
        event_type: Type of the event.
        data: JSON data payload.
        event_id: Optional event ID for client tracking.
        retry: Optional retry interval in milliseconds.
    
    Returns:
        Formatted SSE event string.
    
    Example output:
        event: message
        id: 123
        data: {"content": "Hello"}
        
    """
    lines = []
    
    # Event type
    lines.append(f"event: {event_type.value}")
    
    # Optional ID
    if event_id:
        lines.append(f"id: {event_id}")
    
    # Optional retry
    if retry is not None:
        lines.append(f"retry: {retry}")
    
    # Data (handle multi-line)
    for line in data.split("\n"):
        lines.append(f"data: {line}")
    
    # Terminate with blank line
    lines.append("")
    lines.append("")
    
    return "\n".join(lines)
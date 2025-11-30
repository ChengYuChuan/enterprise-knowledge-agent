"""
SSE Connection Manager

Manages active SSE connections for:
    - Connection tracking
    - Graceful shutdown
    - Heartbeat management
    - Connection limits

Features:
    - Per-user connection limits
    - Automatic cleanup of stale connections
    - Metrics collection
    - Graceful cancellation on shutdown
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Set, Callable, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from weakref import WeakSet

logger = logging.getLogger(__name__)


# =============================================================================
# Connection Info
# =============================================================================

@dataclass
class ConnectionInfo:
    """Information about an active SSE connection."""
    
    connection_id: str
    user_id: str
    conversation_id: str
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    chunks_sent: int = 0
    bytes_sent: int = 0
    
    # Cancellation
    cancelled: bool = False
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    
    def update_activity(self, bytes_count: int = 0):
        """Update last activity timestamp."""
        self.last_activity = time.time()
        self.chunks_sent += 1
        self.bytes_sent += bytes_count
    
    def cancel(self):
        """Cancel this connection."""
        self.cancelled = True
        self.cancel_event.set()
    
    @property
    def duration_seconds(self) -> float:
        """Get connection duration in seconds."""
        return time.time() - self.started_at
    
    @property
    def is_stale(self) -> bool:
        """Check if connection is stale (no activity for 60 seconds)."""
        return time.time() - self.last_activity > 60


# =============================================================================
# Connection Manager
# =============================================================================

class SSEConnectionManager:
    """Manages active SSE connections.
    
    Features:
        - Track active connections
        - Enforce per-user limits
        - Clean up stale connections
        - Provide metrics
    
    Usage:
        ```python
        manager = SSEConnectionManager()
        
        async with manager.connection(user_id, conv_id) as conn:
            for chunk in stream:
                if conn.cancelled:
                    break
                yield chunk
                conn.update_activity(len(chunk))
        ```
    """
    
    def __init__(
        self,
        max_connections_per_user: int = 5,
        max_total_connections: int = 1000,
        heartbeat_interval: float = 30.0,
        stale_timeout: float = 60.0,
    ):
        """Initialize the connection manager.
        
        Args:
            max_connections_per_user: Max concurrent connections per user.
            max_total_connections: Max total concurrent connections.
            heartbeat_interval: Seconds between heartbeats.
            stale_timeout: Seconds before a connection is considered stale.
        """
        self.max_connections_per_user = max_connections_per_user
        self.max_total_connections = max_total_connections
        self.heartbeat_interval = heartbeat_interval
        self.stale_timeout = stale_timeout
        
        # Connection tracking
        self._connections: Dict[str, ConnectionInfo] = {}
        self._user_connections: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics
        self._total_connections_created = 0
        self._total_connections_closed = 0
    
    async def start(self):
        """Start the connection manager."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SSE Connection Manager started")
    
    async def stop(self):
        """Stop the connection manager and cancel all connections."""
        self._running = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active connections
        async with self._lock:
            for conn in self._connections.values():
                conn.cancel()
            self._connections.clear()
            self._user_connections.clear()
        
        logger.info("SSE Connection Manager stopped")
    
    async def _cleanup_loop(self):
        """Background task to clean up stale connections."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._cleanup_stale()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_stale(self):
        """Remove stale connections."""
        async with self._lock:
            stale = [
                conn_id for conn_id, conn in self._connections.items()
                if conn.is_stale
            ]
            
            for conn_id in stale:
                await self._remove_connection_unsafe(conn_id)
                logger.debug(f"Cleaned up stale connection: {conn_id}")
    
    def _generate_connection_id(self) -> str:
        """Generate a unique connection ID."""
        import uuid
        return f"sse_{uuid.uuid4().hex[:12]}"
    
    async def _add_connection(
        self,
        user_id: str,
        conversation_id: str,
    ) -> ConnectionInfo:
        """Add a new connection."""
        async with self._lock:
            # Check total limit
            if len(self._connections) >= self.max_total_connections:
                raise ConnectionLimitError(
                    f"Maximum total connections ({self.max_total_connections}) reached"
                )
            
            # Check user limit
            user_conns = self._user_connections.get(user_id, set())
            if len(user_conns) >= self.max_connections_per_user:
                raise ConnectionLimitError(
                    f"Maximum connections per user ({self.max_connections_per_user}) reached"
                )
            
            # Create connection
            conn_id = self._generate_connection_id()
            conn = ConnectionInfo(
                connection_id=conn_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
            
            # Track connection
            self._connections[conn_id] = conn
            if user_id not in self._user_connections:
                self._user_connections[user_id] = set()
            self._user_connections[user_id].add(conn_id)
            
            self._total_connections_created += 1
            
            logger.debug(f"Added connection {conn_id} for user {user_id}")
            return conn
    
    async def _remove_connection(self, connection_id: str):
        """Remove a connection."""
        async with self._lock:
            await self._remove_connection_unsafe(connection_id)
    
    async def _remove_connection_unsafe(self, connection_id: str):
        """Remove a connection (must hold lock)."""
        conn = self._connections.pop(connection_id, None)
        if conn:
            # Remove from user tracking
            user_conns = self._user_connections.get(conn.user_id)
            if user_conns:
                user_conns.discard(connection_id)
                if not user_conns:
                    del self._user_connections[conn.user_id]
            
            self._total_connections_closed += 1
            
            logger.debug(
                f"Removed connection {connection_id} "
                f"(duration: {conn.duration_seconds:.1f}s, "
                f"chunks: {conn.chunks_sent})"
            )
    
    @asynccontextmanager
    async def connection(
        self,
        user_id: str,
        conversation_id: str,
    ):
        """Context manager for SSE connections.
        
        Usage:
            ```python
            async with manager.connection(user_id, conv_id) as conn:
                for chunk in stream:
                    if conn.cancelled:
                        break
                    yield chunk
            ```
        
        Args:
            user_id: User identifier.
            conversation_id: Conversation identifier.
        
        Yields:
            ConnectionInfo object for tracking.
        
        Raises:
            ConnectionLimitError: If connection limits are exceeded.
        """
        conn = await self._add_connection(user_id, conversation_id)
        try:
            yield conn
        finally:
            await self._remove_connection(conn.connection_id)
    
    def get_connection(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get a connection by ID."""
        return self._connections.get(connection_id)
    
    def get_user_connections(self, user_id: str) -> list[ConnectionInfo]:
        """Get all connections for a user."""
        conn_ids = self._user_connections.get(user_id, set())
        return [self._connections[cid] for cid in conn_ids if cid in self._connections]
    
    async def cancel_user_connections(self, user_id: str):
        """Cancel all connections for a user."""
        async with self._lock:
            conn_ids = list(self._user_connections.get(user_id, set()))
            for conn_id in conn_ids:
                conn = self._connections.get(conn_id)
                if conn:
                    conn.cancel()
    
    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    
    @property
    def active_connections(self) -> int:
        """Number of active connections."""
        return len(self._connections)
    
    @property
    def metrics(self) -> dict:
        """Get connection metrics."""
        return {
            "active_connections": len(self._connections),
            "unique_users": len(self._user_connections),
            "total_created": self._total_connections_created,
            "total_closed": self._total_connections_closed,
            "max_per_user": self.max_connections_per_user,
            "max_total": self.max_total_connections,
        }


# =============================================================================
# Exceptions
# =============================================================================

class ConnectionLimitError(Exception):
    """Raised when connection limits are exceeded."""
    pass


# =============================================================================
# Global Instance
# =============================================================================

_connection_manager: Optional[SSEConnectionManager] = None


def get_connection_manager() -> SSEConnectionManager:
    """Get the global connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = SSEConnectionManager()
    return _connection_manager


def set_connection_manager(manager: SSEConnectionManager):
    """Set the global connection manager instance."""
    global _connection_manager
    _connection_manager = manager
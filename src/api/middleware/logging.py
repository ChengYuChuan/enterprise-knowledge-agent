"""
Request Logging Middleware

Structured logging for all API requests with:
    - Request/Response details
    - Timing information
    - Error tracking
    - Request ID propagation

Log Format (JSON):
    {
        "timestamp": "2024-01-15T10:30:00.000Z",
        "request_id": "req_abc123",
        "method": "POST",
        "path": "/api/v1/chat",
        "status_code": 200,
        "latency_ms": 150.5,
        "client_ip": "192.168.1.1",
        "user_id": "user_123"
    }

Usage:
    ```python
    from src.api.middleware.logging import RequestLoggingMiddleware, get_request_id
    
    app.add_middleware(RequestLoggingMiddleware)
    
    @router.get("/example")
    async def example(request: Request):
        request_id = get_request_id(request)
        logger.info(f"Processing {request_id}")
    ```
"""

import time
import uuid
import json
import logging
from typing import Optional, Callable
from contextvars import ContextVar
from dataclasses import dataclass, asdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Context variable for request ID (accessible anywhere in the request lifecycle)
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LoggingConfig:
    """Logging middleware configuration."""
    
    # Request ID settings
    request_id_header: str = "X-Request-ID"
    generate_request_id: bool = True
    
    # What to log
    log_request_body: bool = False       # Can be expensive/sensitive
    log_response_body: bool = False      # Can be expensive/sensitive
    log_headers: bool = False            # Contains auth tokens
    
    # Body size limits (if logging bodies)
    max_body_log_size: int = 10000       # Truncate large bodies
    
    # Paths to skip logging
    skip_paths: set[str] = None
    
    # Sensitive headers to redact
    sensitive_headers: set[str] = None
    
    # Log level for different status codes
    log_level_by_status: dict[int, str] = None
    
    def __post_init__(self):
        if self.skip_paths is None:
            self.skip_paths = {"/health", "/metrics"}
        
        if self.sensitive_headers is None:
            self.sensitive_headers = {
                "authorization",
                "x-api-key",
                "cookie",
                "set-cookie",
            }
        
        if self.log_level_by_status is None:
            self.log_level_by_status = {
                200: "INFO",
                201: "INFO",
                204: "INFO",
                400: "WARNING",
                401: "WARNING",
                403: "WARNING",
                404: "INFO",
                429: "WARNING",
                500: "ERROR",
                502: "ERROR",
                503: "ERROR",
            }


# =============================================================================
# Log Record
# =============================================================================

@dataclass
class RequestLogRecord:
    """Structured log record for a request."""
    
    # Identifiers
    request_id: str
    
    # Request info
    method: str
    path: str
    query_string: Optional[str] = None
    
    # Response info
    status_code: int = 0
    
    # Timing
    latency_ms: float = 0.0
    
    # Client info
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    user_id: Optional[str] = None
    
    # Optional details
    request_headers: Optional[dict] = None
    response_headers: Optional[dict] = None
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    
    # Error info
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


# =============================================================================
# Request ID Utilities
# =============================================================================

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:12]}"


def get_request_id(request: Request = None) -> Optional[str]:
    """Get the current request ID.
    
    Can be called from anywhere in the request lifecycle.
    
    Args:
        request: Optional request object (uses context var if not provided).
    
    Returns:
        Request ID string or None.
    """
    # Try context variable first
    ctx_id = request_id_ctx.get()
    if ctx_id:
        return ctx_id
    
    # Try request state
    if request and hasattr(request.state, "request_id"):
        return request.state.request_id
    
    return None


def set_request_id(request_id: str) -> None:
    """Set the request ID in context."""
    request_id_ctx.set(request_id)


# =============================================================================
# Structured Logger
# =============================================================================

class StructuredLogger:
    """Logger that outputs structured JSON logs."""
    
    def __init__(self, name: str = "api"):
        self.logger = logging.getLogger(name)
    
    def log(self, level: str, record: RequestLogRecord):
        """Log a request record at the specified level."""
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(record.to_json())
    
    def log_request(self, record: RequestLogRecord, config: LoggingConfig = None):
        """Log a request with appropriate level based on status code."""
        config = config or LoggingConfig()
        
        # Determine log level
        level = config.log_level_by_status.get(
            record.status_code,
            "ERROR" if record.status_code >= 500 else "INFO"
        )
        
        self.log(level, record)


# Global logger instance
_logger = StructuredLogger()


# =============================================================================
# Middleware
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request logging."""
    
    def __init__(
        self,
        app,
        config: LoggingConfig = None,
        logger: StructuredLogger = None
    ):
        super().__init__(app)
        self.config = config or LoggingConfig()
        self.logger = logger or _logger
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip logging for certain paths
        if request.url.path in self.config.skip_paths:
            return await call_next(request)
        
        # Generate or extract request ID
        request_id = request.headers.get(self.config.request_id_header)
        if not request_id and self.config.generate_request_id:
            request_id = generate_request_id()
        
        # Store in context and request state
        set_request_id(request_id)
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.perf_counter()
        
        # Build initial log record
        record = RequestLogRecord(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_string=str(request.url.query) if request.url.query else None,
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )
        
        # Log request headers if configured
        if self.config.log_headers:
            record.request_headers = self._redact_headers(dict(request.headers))
        
        # Log request body if configured
        if self.config.log_request_body:
            record.request_body = await self._get_body(request)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Update record with response info
            record.status_code = response.status_code
            record.latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Try to get user ID from request state (set by auth middleware)
            if hasattr(request.state, "user") and request.state.user:
                record.user_id = request.state.user.id
            
            # Log response headers if configured
            if self.config.log_headers:
                record.response_headers = self._redact_headers(dict(response.headers))
            
            # Add request ID to response headers
            response.headers[self.config.request_id_header] = request_id
            
            # Log the request
            self.logger.log_request(record, self.config)
            
            return response
            
        except Exception as e:
            # Log error
            record.status_code = 500
            record.latency_ms = (time.perf_counter() - start_time) * 1000
            record.error = str(e)
            record.error_type = type(e).__name__
            
            self.logger.log("ERROR", record)
            
            raise
        
        finally:
            # Clear context
            request_id_ctx.set(None)
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Extract client IP from request."""
        # Check X-Forwarded-For header (for proxied requests)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Direct connection
        if request.client:
            return request.client.host
        
        return None
    
    def _redact_headers(self, headers: dict) -> dict:
        """Redact sensitive headers."""
        redacted = {}
        for key, value in headers.items():
            if key.lower() in self.config.sensitive_headers:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        return redacted
    
    async def _get_body(self, request: Request) -> Optional[str]:
        """Get request body for logging."""
        try:
            body = await request.body()
            if len(body) > self.config.max_body_log_size:
                return f"[TRUNCATED: {len(body)} bytes]"
            return body.decode("utf-8", errors="replace")
        except Exception:
            return None


# =============================================================================
# Access Log Formatter
# =============================================================================

class AccessLogFormatter(logging.Formatter):
    """Custom formatter for access logs in JSON format."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Try to parse as JSON (from our structured logger)
        try:
            log_dict = json.loads(record.getMessage())
            log_dict["timestamp"] = self.formatTime(record)
            log_dict["level"] = record.levelname
            return json.dumps(log_dict)
        except (json.JSONDecodeError, TypeError):
            # Fallback to standard formatting
            return super().format(record)


# =============================================================================
# Setup Functions
# =============================================================================

def setup_request_logging(
    logger_name: str = "api",
    level: int = logging.INFO,
    handler: logging.Handler = None
) -> logging.Logger:
    """Set up the request logger with JSON formatting.
    
    Args:
        logger_name: Name for the logger.
        level: Logging level.
        handler: Optional custom handler.
    
    Returns:
        Configured logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    if handler is None:
        handler = logging.StreamHandler()
    
    handler.setFormatter(AccessLogFormatter())
    logger.addHandler(handler)
    
    return logger
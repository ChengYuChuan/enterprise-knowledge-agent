"""
Rate Limiting Middleware

Implements token bucket rate limiting to protect the API from abuse.

Features:
    - Per-client rate limiting (by IP or API key)
    - Configurable limits per endpoint
    - Sliding window algorithm for smooth limiting
    - Redis backend support for distributed deployments

Rate Limit Headers (following RFC draft):
    - X-RateLimit-Limit: Maximum requests allowed
    - X-RateLimit-Remaining: Requests remaining in window
    - X-RateLimit-Reset: Unix timestamp when limit resets
    - Retry-After: Seconds to wait (when limited)

Usage:
    ```python
    from src.api.middleware.rate_limit import RateLimiter, rate_limit
    
    # As middleware (global)
    app.add_middleware(RateLimitMiddleware, limiter=RateLimiter())
    
    # As dependency (per-route)
    @router.post("/chat")
    async def chat(
        request: ChatRequest,
        _: None = Depends(rate_limit(requests=10, window=60))
    ):
        pass
    ```
"""

import time
import hashlib
import asyncio
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict

from fastapi import Request, HTTPException, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    
    # Default limits
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    
    # Burst allowance (tokens above limit for handling bursts)
    burst_multiplier: float = 1.5
    
    # Client identification
    use_api_key: bool = True      # Use API key if available
    use_ip: bool = True           # Fallback to IP address
    
    # Redis settings (for distributed deployments)
    redis_url: Optional[str] = None
    redis_prefix: str = "ratelimit:"
    
    # Whitelist/Blacklist
    whitelist: set[str] = None    # Never rate limit these clients
    blacklist: set[str] = None    # Always reject these clients
    
    # Path-specific overrides
    path_limits: dict[str, int] = None  # {"/api/v1/chat": 30}
    
    def __post_init__(self):
        if self.whitelist is None:
            self.whitelist = set()
        if self.blacklist is None:
            self.blacklist = set()
        if self.path_limits is None:
            self.path_limits = {}


# =============================================================================
# In-Memory Rate Limiter
# =============================================================================

@dataclass
class TokenBucket:
    """Token bucket for rate limiting.
    
    Implements the token bucket algorithm:
    - Tokens are added at a fixed rate
    - Requests consume tokens
    - Requests are rejected when bucket is empty
    """
    capacity: float           # Maximum tokens
    tokens: float             # Current tokens
    refill_rate: float        # Tokens per second
    last_update: float = field(default_factory=time.time)
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume.
        
        Returns:
            True if tokens were consumed, False if not enough tokens.
        """
        now = time.time()
        
        # Refill tokens based on time elapsed
        elapsed = now - self.last_update
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_update = now
        
        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    @property
    def remaining(self) -> int:
        """Get remaining tokens (as integer)."""
        return int(self.tokens)
    
    @property
    def reset_time(self) -> float:
        """Get time until bucket is full again."""
        if self.tokens >= self.capacity:
            return 0
        tokens_needed = self.capacity - self.tokens
        return tokens_needed / self.refill_rate


class InMemoryRateLimiter:
    """In-memory rate limiter using token buckets.
    
    Suitable for single-instance deployments.
    For distributed deployments, use RedisRateLimiter.
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop the cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def _cleanup_loop(self):
        """Periodically clean up expired buckets."""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            await self._cleanup_expired()
    
    async def _cleanup_expired(self):
        """Remove buckets that haven't been used in a while."""
        async with self._lock:
            now = time.time()
            expired = [
                key for key, bucket in self._buckets.items()
                if now - bucket.last_update > 3600  # 1 hour
            ]
            for key in expired:
                del self._buckets[key]
            
            if expired:
                logger.debug(f"Cleaned up {len(expired)} expired rate limit buckets")
    
    def _get_bucket(
        self,
        client_id: str,
        requests_per_minute: int = None
    ) -> TokenBucket:
        """Get or create a token bucket for a client."""
        if requests_per_minute is None:
            requests_per_minute = self.config.requests_per_minute
        
        if client_id not in self._buckets:
            capacity = requests_per_minute * self.config.burst_multiplier
            refill_rate = requests_per_minute / 60.0  # tokens per second
            
            self._buckets[client_id] = TokenBucket(
                capacity=capacity,
                tokens=capacity,  # Start full
                refill_rate=refill_rate,
            )
        
        return self._buckets[client_id]
    
    async def is_allowed(
        self,
        client_id: str,
        path: str = None,
        tokens: int = 1
    ) -> tuple[bool, dict]:
        """Check if a request is allowed.
        
        Args:
            client_id: Client identifier.
            path: Request path (for path-specific limits).
            tokens: Tokens to consume.
        
        Returns:
            Tuple of (allowed, headers_dict)
        """
        # Check whitelist/blacklist
        if client_id in self.config.whitelist:
            return True, {}
        
        if client_id in self.config.blacklist:
            return False, {"Retry-After": "3600"}
        
        # Get path-specific limit
        limit = self.config.requests_per_minute
        if path and path in self.config.path_limits:
            limit = self.config.path_limits[path]
        
        # Get bucket and try to consume
        bucket = self._get_bucket(client_id, limit)
        allowed = bucket.consume(tokens)
        
        # Build rate limit headers
        headers = {
            "X-RateLimit-Limit": str(int(bucket.capacity)),
            "X-RateLimit-Remaining": str(bucket.remaining),
            "X-RateLimit-Reset": str(int(time.time() + bucket.reset_time)),
        }
        
        if not allowed:
            headers["Retry-After"] = str(int(bucket.reset_time) + 1)
        
        return allowed, headers


# =============================================================================
# Redis Rate Limiter (for distributed deployments)
# =============================================================================

class RedisRateLimiter:
    """Redis-backed rate limiter for distributed deployments.
    
    Uses Redis sorted sets to implement sliding window rate limiting.
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._redis = None
    
    async def start(self):
        """Initialize Redis connection."""
        if not self.config.redis_url:
            raise ValueError("Redis URL not configured")
        
        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(self.config.redis_url)
            await self._redis.ping()
            logger.info("Connected to Redis for rate limiting")
        except ImportError:
            raise RuntimeError("redis package not installed. Run: pip install redis")
    
    async def stop(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
    
    async def is_allowed(
        self,
        client_id: str,
        path: str = None,
        tokens: int = 1
    ) -> tuple[bool, dict]:
        """Check if request is allowed using sliding window."""
        if not self._redis:
            raise RuntimeError("Redis not initialized")
        
        # Check whitelist/blacklist
        if client_id in self.config.whitelist:
            return True, {}
        if client_id in self.config.blacklist:
            return False, {"Retry-After": "3600"}
        
        # Get limit
        limit = self.config.requests_per_minute
        if path and path in self.config.path_limits:
            limit = self.config.path_limits[path]
        
        now = time.time()
        window = 60  # 1 minute window
        key = f"{self.config.redis_prefix}{client_id}"
        
        # Use pipeline for atomic operations
        pipe = self._redis.pipeline()
        
        # Remove old entries outside the window
        pipe.zremrangebyscore(key, 0, now - window)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {f"{now}:{tokens}": now})
        
        # Set expiry
        pipe.expire(key, window * 2)
        
        results = await pipe.execute()
        current_count = results[1]
        
        allowed = current_count < limit
        remaining = max(0, limit - current_count - 1) if allowed else 0
        
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(now + window)),
        }
        
        if not allowed:
            headers["Retry-After"] = str(window)
        
        return allowed, headers


# =============================================================================
# Rate Limiter Factory
# =============================================================================

def create_rate_limiter(config: RateLimitConfig = None) -> InMemoryRateLimiter:
    """Create a rate limiter based on configuration.
    
    Returns Redis limiter if redis_url is configured, otherwise in-memory.
    """
    config = config or RateLimitConfig()
    
    if config.redis_url:
        return RedisRateLimiter(config)
    
    return InMemoryRateLimiter(config)


# Global rate limiter instance
_rate_limiter: Optional[InMemoryRateLimiter] = None


def get_rate_limiter() -> InMemoryRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = create_rate_limiter()
    return _rate_limiter


def set_rate_limiter(limiter: InMemoryRateLimiter) -> None:
    """Set the global rate limiter instance."""
    global _rate_limiter
    _rate_limiter = limiter


# =============================================================================
# Client Identification
# =============================================================================

def get_client_id(request: Request, config: RateLimitConfig = None) -> str:
    """Extract client identifier from request.
    
    Priority:
    1. API key (if configured and present)
    2. User ID from auth (if authenticated)
    3. X-Forwarded-For header (if behind proxy)
    4. Client IP address
    """
    config = config or RateLimitConfig()
    
    # Try API key
    if config.use_api_key:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # Hash the key for privacy
            return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
    
    # Try user ID from request state (set by auth middleware)
    if hasattr(request.state, "user") and request.state.user:
        return f"user:{request.state.user.id}"
    
    # Try IP address
    if config.use_ip:
        # Check X-Forwarded-For for proxied requests
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP (original client)
            ip = forwarded.split(",")[0].strip()
            return f"ip:{ip}"
        
        # Direct connection
        if request.client:
            return f"ip:{request.client.host}"
    
    # Fallback
    return "unknown"


# =============================================================================
# Middleware
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware.
    
    Applies rate limiting to all requests and adds rate limit headers.
    """
    
    def __init__(
        self,
        app,
        limiter: InMemoryRateLimiter = None,
        config: RateLimitConfig = None
    ):
        super().__init__(app)
        self.limiter = limiter or get_rate_limiter()
        self.config = config or RateLimitConfig()
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for certain paths
        if request.url.path in {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}:
            return await call_next(request)
        
        # Get client identifier
        client_id = get_client_id(request, self.config)
        
        # Check rate limit
        allowed, headers = await self.limiter.is_allowed(
            client_id,
            path=request.url.path
        )
        
        if not allowed:
            return Response(
                content='{"error": "rate_limit_exceeded", "message": "Too many requests"}',
                status_code=429,
                media_type="application/json",
                headers=headers,
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value
        
        return response


# =============================================================================
# Dependency for Per-Route Rate Limiting
# =============================================================================

def rate_limit(
    requests: int = 60,
    window: int = 60,
    key_func: Callable[[Request], str] = None
):
    """Dependency factory for per-route rate limiting.
    
    Args:
        requests: Maximum requests allowed in the window.
        window: Time window in seconds.
        key_func: Custom function to extract client key.
    
    Usage:
        ```python
        @router.post("/expensive-operation")
        async def expensive_op(
            _: None = Depends(rate_limit(requests=5, window=60))
        ):
            # Only 5 requests per minute allowed
            pass
        ```
    """
    # Create a dedicated limiter for this route
    config = RateLimitConfig(requests_per_minute=requests)
    limiter = InMemoryRateLimiter(config)
    
    async def check_rate_limit(request: Request):
        if key_func:
            client_id = key_func(request)
        else:
            client_id = get_client_id(request)
        
        allowed, headers = await limiter.is_allowed(client_id)
        
        if not allowed:
            retry_after = headers.get("Retry-After", "60")
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers=headers,
            )
        
        return None
    
    return check_rate_limit
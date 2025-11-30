"""
API Middleware

Middleware components for the FastAPI application.

Components:
    - auth: Authentication (API Key, JWT)
    - rate_limit: Request rate limiting
    - logging: Structured request logging

Usage:
    ```python
    from fastapi import FastAPI
    from src.api.middleware import (
        AuthMiddleware,
        RateLimitMiddleware,
        RequestLoggingMiddleware,
    )
    
    app = FastAPI()
    
    # Order matters! Logging should be first to capture all requests
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)
    ```
"""

# Authentication
from src.api.middleware.auth import (
    # Config
    AuthConfig,
    configure_auth,
    get_auth_config,
    # User model
    User,
    # Dependencies
    get_current_user,
    get_optional_user,
    require_api_key,
    require_jwt,
    require_roles,
    # Utilities
    generate_api_key,
    create_jwt_token,
    # Middleware
    AuthMiddleware,
)

# Rate limiting
from src.api.middleware.rate_limit import (
    # Config
    RateLimitConfig,
    # Limiters
    InMemoryRateLimiter,
    RedisRateLimiter,
    create_rate_limiter,
    get_rate_limiter,
    set_rate_limiter,
    # Dependencies
    rate_limit,
    get_client_id,
    # Middleware
    RateLimitMiddleware,
)

# Logging
from src.api.middleware.logging import (
    # Config
    LoggingConfig,
    # Utilities
    get_request_id,
    set_request_id,
    generate_request_id,
    # Logging
    RequestLogRecord,
    StructuredLogger,
    setup_request_logging,
    # Middleware
    RequestLoggingMiddleware,
)

__all__ = [
    # Auth
    "AuthConfig",
    "configure_auth",
    "get_auth_config",
    "User",
    "get_current_user",
    "get_optional_user",
    "require_api_key",
    "require_jwt",
    "require_roles",
    "generate_api_key",
    "create_jwt_token",
    "AuthMiddleware",
    # Rate limit
    "RateLimitConfig",
    "InMemoryRateLimiter",
    "RedisRateLimiter",
    "create_rate_limiter",
    "get_rate_limiter",
    "set_rate_limiter",
    "rate_limit",
    "get_client_id",
    "RateLimitMiddleware",
    # Logging
    "LoggingConfig",
    "get_request_id",
    "set_request_id",
    "generate_request_id",
    "RequestLogRecord",
    "StructuredLogger",
    "setup_request_logging",
    "RequestLoggingMiddleware",
]
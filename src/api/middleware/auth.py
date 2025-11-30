"""
Authentication Middleware

Supports multiple authentication methods:
    1. API Key authentication (simple, for service-to-service)
    2. JWT Bearer token (for user authentication)

Security Design:
    - API keys are validated against a set of allowed keys
    - JWTs are validated using RS256 or HS256 algorithms
    - All auth info is stored in request.state for downstream use

Usage in routes:
    ```python
    from src.api.middleware.auth import get_current_user, require_api_key
    
    @router.get("/protected")
    async def protected_route(user: User = Depends(get_current_user)):
        return {"user": user.id}
    
    @router.get("/service-only")
    async def service_route(api_key: str = Depends(require_api_key)):
        return {"key": api_key[:8] + "..."}
    ```
"""

import os
import time
import hashlib
import secrets
from typing import Optional, Annotated
from dataclasses import dataclass

from fastapi import Request, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AuthConfig:
    """Authentication configuration."""
    
    # API Key settings
    api_key_header: str = "X-API-Key"
    api_keys: set[str] = None  # Set of valid API keys
    
    # JWT settings
    jwt_secret: str = None       # For HS256
    jwt_public_key: str = None   # For RS256
    jwt_algorithm: str = "HS256"
    jwt_issuer: Optional[str] = None
    jwt_audience: Optional[str] = None
    
    # General settings
    enabled: bool = True
    exclude_paths: set[str] = None  # Paths that don't require auth
    
    def __post_init__(self):
        if self.api_keys is None:
            # Load from environment
            keys_str = os.getenv("API_KEYS", "")
            self.api_keys = set(k.strip() for k in keys_str.split(",") if k.strip())
        
        if self.jwt_secret is None:
            self.jwt_secret = os.getenv("JWT_SECRET", "")
        
        if self.exclude_paths is None:
            self.exclude_paths = {
                "/",
                "/health",
                "/api/v1/health",
                "/docs",
                "/redoc",
                "/openapi.json",
            }


# Global config instance
_auth_config = AuthConfig()


def get_auth_config() -> AuthConfig:
    """Get the current auth configuration."""
    return _auth_config


def configure_auth(config: AuthConfig) -> None:
    """Update the authentication configuration."""
    global _auth_config
    _auth_config = config


# =============================================================================
# User Model
# =============================================================================

@dataclass
class User:
    """Authenticated user information."""
    id: str
    email: Optional[str] = None
    name: Optional[str] = None
    roles: list[str] = None
    metadata: dict = None
    auth_method: str = "api_key"  # "api_key" or "jwt"
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = []
        if self.metadata is None:
            self.metadata = {}
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles
    
    def has_any_role(self, roles: list[str]) -> bool:
        """Check if user has any of the specified roles."""
        return bool(set(self.roles) & set(roles))


# =============================================================================
# Security Schemes
# =============================================================================

# API Key header
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key for authentication"
)

# JWT Bearer token
jwt_bearer = HTTPBearer(
    auto_error=False,
    description="JWT Bearer token"
)


# =============================================================================
# API Key Authentication
# =============================================================================

def hash_api_key(key: str) -> str:
    """Hash an API key for secure comparison."""
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(api_key: str, config: AuthConfig = None) -> bool:
    """Verify an API key against the allowed set."""
    if config is None:
        config = get_auth_config()
    
    if not config.api_keys:
        logger.warning("No API keys configured, all keys will be rejected")
        return False
    
    return api_key in config.api_keys


async def get_api_key(
    api_key: Annotated[Optional[str], Security(api_key_header)]
) -> Optional[str]:
    """Extract API key from request header."""
    return api_key


async def require_api_key(
    api_key: Annotated[Optional[str], Depends(get_api_key)]
) -> str:
    """Require a valid API key.
    
    Use as a dependency in routes that require API key authentication.
    
    Raises:
        HTTPException: If API key is missing or invalid.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if not verify_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key


# =============================================================================
# JWT Authentication
# =============================================================================

def decode_jwt(token: str, config: AuthConfig = None) -> dict:
    """Decode and validate a JWT token.
    
    Args:
        token: The JWT token string.
        config: Authentication configuration.
    
    Returns:
        Decoded token payload.
    
    Raises:
        HTTPException: If token is invalid or expired.
    """
    if config is None:
        config = get_auth_config()
    
    try:
        import jwt
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="JWT support not installed. Run: pip install PyJWT"
        )
    
    try:
        # Choose key based on algorithm
        if config.jwt_algorithm.startswith("RS"):
            key = config.jwt_public_key
        else:
            key = config.jwt_secret
        
        if not key:
            raise HTTPException(
                status_code=500,
                detail="JWT secret/key not configured"
            )
        
        # Decode with validation
        decode_options = {
            "verify_signature": True,
            "verify_exp": True,
            "verify_iat": True,
            "require": ["exp", "iat", "sub"],
        }
        
        payload = jwt.decode(
            token,
            key,
            algorithms=[config.jwt_algorithm],
            issuer=config.jwt_issuer,
            audience=config.jwt_audience,
            options=decode_options,
        )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_jwt_token(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Security(jwt_bearer)]
) -> Optional[str]:
    """Extract JWT token from Authorization header."""
    if credentials is None:
        return None
    return credentials.credentials


async def require_jwt(
    token: Annotated[Optional[str], Depends(get_jwt_token)]
) -> dict:
    """Require a valid JWT token.
    
    Use as a dependency in routes that require JWT authentication.
    
    Returns:
        Decoded JWT payload.
    
    Raises:
        HTTPException: If token is missing or invalid.
    """
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Bearer token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return decode_jwt(token)


# =============================================================================
# Combined Authentication
# =============================================================================

async def get_current_user(
    request: Request,
    api_key: Annotated[Optional[str], Depends(get_api_key)] = None,
    jwt_token: Annotated[Optional[str], Depends(get_jwt_token)] = None,
) -> User:
    """Get the current authenticated user.
    
    Supports both API key and JWT authentication.
    Checks API key first, then JWT.
    
    Use as a dependency in routes that require authentication:
    ```python
    @router.get("/me")
    async def get_me(user: User = Depends(get_current_user)):
        return {"id": user.id}
    ```
    
    Returns:
        User object with authentication info.
    
    Raises:
        HTTPException: If no valid authentication is provided.
    """
    config = get_auth_config()
    
    # Check if auth is disabled
    if not config.enabled:
        return User(id="anonymous", auth_method="none")
    
    # Check if path is excluded
    if request.url.path in config.exclude_paths:
        return User(id="anonymous", auth_method="none")
    
    # Try API key first
    if api_key and verify_api_key(api_key, config):
        # For API keys, we create a simple user with the key hash as ID
        key_id = hash_api_key(api_key)[:16]
        return User(
            id=f"apikey_{key_id}",
            roles=["api_user"],
            auth_method="api_key",
        )
    
    # Try JWT
    if jwt_token:
        payload = decode_jwt(jwt_token, config)
        return User(
            id=payload.get("sub"),
            email=payload.get("email"),
            name=payload.get("name"),
            roles=payload.get("roles", []),
            metadata=payload,
            auth_method="jwt",
        )
    
    # No valid authentication
    raise HTTPException(
        status_code=401,
        detail="Authentication required",
        headers={"WWW-Authenticate": 'Bearer, ApiKey'},
    )


async def get_optional_user(
    request: Request,
    api_key: Annotated[Optional[str], Depends(get_api_key)] = None,
    jwt_token: Annotated[Optional[str], Depends(get_jwt_token)] = None,
) -> Optional[User]:
    """Get current user if authenticated, None otherwise.
    
    Use for endpoints that work with or without authentication.
    """
    try:
        return await get_current_user(request, api_key, jwt_token)
    except HTTPException:
        return None


# =============================================================================
# Role-Based Access Control
# =============================================================================

def require_roles(*roles: str):
    """Dependency factory for role-based access control.
    
    Usage:
        ```python
        @router.get("/admin")
        async def admin_route(user: User = Depends(require_roles("admin"))):
            return {"admin": True}
        
        @router.get("/editor")
        async def editor_route(user: User = Depends(require_roles("admin", "editor"))):
            return {"can_edit": True}
        ```
    """
    async def check_roles(user: User = Depends(get_current_user)) -> User:
        if not user.has_any_role(list(roles)):
            raise HTTPException(
                status_code=403,
                detail=f"Required roles: {', '.join(roles)}"
            )
        return user
    
    return check_roles


# =============================================================================
# Middleware
# =============================================================================

class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware for logging authentication info.
    
    This middleware doesn't enforce authentication - that's done by
    dependencies. It logs auth attempts for monitoring.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract auth info for logging
        api_key = request.headers.get("X-API-Key")
        auth_header = request.headers.get("Authorization")
        
        # Log auth attempt (redact sensitive info)
        if api_key:
            logger.debug(f"API key auth attempt: {api_key[:8]}...")
        elif auth_header:
            logger.debug(f"Bearer auth attempt: {auth_header[:20]}...")
        
        response = await call_next(request)
        return response


# =============================================================================
# Utility Functions
# =============================================================================

def generate_api_key(prefix: str = "sk") -> str:
    """Generate a new API key.
    
    Format: {prefix}_{random_32_chars}
    Example: sk_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
    
    Args:
        prefix: Key prefix for identification.
    
    Returns:
        New API key string.
    """
    random_part = secrets.token_hex(16)  # 32 hex chars
    return f"{prefix}_{random_part}"


def create_jwt_token(
    user_id: str,
    email: Optional[str] = None,
    roles: Optional[list[str]] = None,
    expires_in: int = 3600,
    config: AuthConfig = None,
) -> str:
    """Create a new JWT token.
    
    Args:
        user_id: User identifier (will be 'sub' claim).
        email: Optional email.
        roles: Optional list of roles.
        expires_in: Token lifetime in seconds.
        config: Auth configuration.
    
    Returns:
        Encoded JWT token string.
    """
    if config is None:
        config = get_auth_config()
    
    try:
        import jwt
    except ImportError:
        raise RuntimeError("PyJWT not installed")
    
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + expires_in,
    }
    
    if email:
        payload["email"] = email
    if roles:
        payload["roles"] = roles
    if config.jwt_issuer:
        payload["iss"] = config.jwt_issuer
    if config.jwt_audience:
        payload["aud"] = config.jwt_audience
    
    return jwt.encode(
        payload,
        config.jwt_secret,
        algorithm=config.jwt_algorithm,
    )
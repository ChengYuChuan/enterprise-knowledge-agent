"""
LLM Provider Exceptions

Custom exception hierarchy for LLM-related errors.
Following the principle of specific exceptions over generic ones.

Exception Hierarchy:
    LLMError (base)
    ├── LLMConnectionError      - Network/connection issues
    ├── LLMAuthenticationError  - API key/auth failures  
    ├── LLMRateLimitError       - Rate limiting hit
    ├── LLMContextLengthError   - Token limit exceeded
    ├── LLMResponseError        - Invalid/unexpected response
    └── LLMProviderNotFoundError - Unknown provider requested
"""

from typing import Optional


class LLMError(Exception):
    """Base exception for all LLM-related errors.
    
    Attributes:
        message: Human-readable error description.
        provider: Name of the LLM provider that raised the error.
        original_error: The underlying exception, if any.
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.provider = provider
        self.original_error = original_error
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message with provider context."""
        parts = []
        if self.provider:
            parts.append(f"[{self.provider}]")
        parts.append(self.message)
        if self.original_error:
            parts.append(f"(Caused by: {type(self.original_error).__name__})")
        return " ".join(parts)


class LLMConnectionError(LLMError):
    """Raised when connection to LLM provider fails.
    
    Common causes:
    - Network connectivity issues
    - Provider service downtime
    - DNS resolution failures
    - Timeout exceeded
    """
    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication with LLM provider fails.
    
    Common causes:
    - Invalid API key
    - Expired credentials
    - Insufficient permissions
    - Account suspended
    """
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded.
    
    Attributes:
        retry_after: Suggested wait time in seconds before retry.
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
        retry_after: Optional[float] = None
    ):
        super().__init__(message, provider, original_error)
        self.retry_after = retry_after


class LLMContextLengthError(LLMError):
    """Raised when input exceeds the model's context length.
    
    Attributes:
        max_tokens: Maximum allowed tokens for the model.
        actual_tokens: Actual token count of the input.
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
        max_tokens: Optional[int] = None,
        actual_tokens: Optional[int] = None
    ):
        super().__init__(message, provider, original_error)
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens


class LLMResponseError(LLMError):
    """Raised when LLM returns an invalid or unexpected response.
    
    Common causes:
    - Malformed JSON response
    - Missing required fields
    - Content filtering triggered
    - Model returned empty response
    """
    pass


class LLMProviderNotFoundError(LLMError):
    """Raised when requested provider is not available.
    
    Attributes:
        available_providers: List of valid provider names.
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        available_providers: Optional[list[str]] = None
    ):
        # Set available_providers BEFORE calling super().__init__()
        # because _format_message() needs it
        self.available_providers = available_providers or []
        super().__init__(message, provider)
    
    def _format_message(self) -> str:
        base_msg = super()._format_message()
        if self.available_providers:
            return f"{base_msg} Available providers: {', '.join(self.available_providers)}"
        return base_msg


class LLMStreamError(LLMError):
    """Raised when streaming response encounters an error.
    
    This can happen mid-stream, so partial data may have been received.
    
    Attributes:
        partial_response: Any data received before the error occurred.
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
        partial_response: Optional[str] = None
    ):
        super().__init__(message, provider, original_error)
        self.partial_response = partial_response
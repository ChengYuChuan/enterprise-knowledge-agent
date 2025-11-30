"""
LLM Provider Base Classes

This module defines the abstract interface for all LLM providers.
Using the Strategy Pattern, different providers (OpenAI, Anthropic, Ollama)
can be swapped without changing the consuming code.

Design Decisions:
    1. Abstract Base Class (ABC) enforces interface contract
    2. Pydantic models for type-safe configuration and responses
    3. AsyncGenerator for streaming to enable real-time token delivery
    4. Unified message format compatible with chat-based models

Example Usage:
    ```python
    # The consuming code works with any provider
    async def chat(provider: BaseLLMProvider, user_input: str):
        messages = [Message(role="user", content=user_input)]
        response = await provider.generate(messages)
        return response.content
    
    # Swap providers via configuration, not code changes
    provider = OpenAIProvider(config)  # or AnthropicProvider(config)
    result = await chat(provider, "Hello!")
    ```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Optional, Any
import time


# =============================================================================
# Enums and Constants
# =============================================================================

class MessageRole(str, Enum):
    """Standard roles for chat messages.
    
    These align with the common convention used by most LLM APIs:
    - system: Instructions/context for the model
    - user: Human input
    - assistant: Model's previous responses
    - tool: Results from tool/function calls (used in agent scenarios)
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ProviderType(str, Enum):
    """Supported LLM provider types.
    
    Add new providers here when implementing additional integrations.
    """
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


# =============================================================================
# Data Classes (using dataclasses for simplicity, can migrate to Pydantic)
# =============================================================================

@dataclass
class Message:
    """A single message in a conversation.
    
    Attributes:
        role: The role of the message sender.
        content: The text content of the message.
        name: Optional name for the sender (useful for multi-agent scenarios).
        tool_call_id: ID linking to a tool call (for tool role messages).
        metadata: Additional provider-specific data.
    
    Example:
        >>> msg = Message(role=MessageRole.USER, content="What is RAG?")
        >>> msg.to_dict()
        {'role': 'user', 'content': 'What is RAG?'}
    """
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls.
        
        Only includes non-None fields to avoid API errors.
        """
        result = {
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content
        }
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result
    
    @classmethod
    def system(cls, content: str) -> "Message":
        """Factory method for system messages."""
        return cls(role=MessageRole.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> "Message":
        """Factory method for user messages."""
        return cls(role=MessageRole.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Factory method for assistant messages."""
        return cls(role=MessageRole.ASSISTANT, content=content)


@dataclass
class LLMConfig:
    """Configuration for an LLM provider.
    
    This is a base configuration that applies to all providers.
    Provider-specific configs should extend this.
    
    Attributes:
        model: The model identifier (e.g., "gpt-4", "claude-3-opus").
        temperature: Randomness in generation (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling parameter.
        timeout: Request timeout in seconds.
        api_key: API key for authentication (loaded from env if not provided).
        base_url: Override the default API endpoint.
        extra_params: Provider-specific parameters.
    
    Example:
        >>> config = LLMConfig(model="gpt-4", temperature=0.7, max_tokens=1000)
    """
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    timeout: float = 30.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.top_p}")


@dataclass
class TokenUsage:
    """Token usage statistics from an LLM call.
    
    Useful for cost tracking and debugging context length issues.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def is_available(self) -> bool:
        """Check if usage data was provided."""
        return self.total_tokens > 0


@dataclass
class LLMResponse:
    """Response from an LLM generation call.
    
    Attributes:
        content: The generated text content.
        model: The model that generated the response.
        usage: Token usage statistics.
        finish_reason: Why generation stopped (e.g., "stop", "length", "content_filter").
        latency_ms: Time taken for the request in milliseconds.
        raw_response: The original response from the provider (for debugging).
    
    Example:
        >>> response = await provider.generate(messages)
        >>> print(f"Generated {response.usage.completion_tokens} tokens in {response.latency_ms}ms")
    """
    content: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: Optional[str] = None
    latency_ms: float = 0.0
    raw_response: Optional[dict[str, Any]] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if generation completed normally (not truncated)."""
        return self.finish_reason in ("stop", "end_turn", None)


@dataclass
class StreamChunk:
    """A single chunk from a streaming response.
    
    Attributes:
        content: The text delta for this chunk.
        is_final: Whether this is the last chunk.
        finish_reason: Why streaming ended (only set on final chunk).
        usage: Token usage (only available on final chunk for some providers).
    """
    content: str
    is_final: bool = False
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None


# =============================================================================
# Abstract Base Class
# =============================================================================

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    All LLM providers must inherit from this class and implement
    the abstract methods. This ensures a consistent interface
    across different providers (Strategy Pattern).
    
    The class provides:
    - Abstract methods that must be implemented
    - Concrete helper methods for common operations
    - Context manager support for resource cleanup
    
    Subclass Implementation Guide:
        1. Implement __init__ to accept LLMConfig
        2. Implement generate() for single-response generation
        3. Implement stream() for streaming generation
        4. Implement embed() if the provider supports embeddings
        5. Override _validate_messages() if provider has specific requirements
    
    Example Implementation:
        ```python
        class MyProvider(BaseLLMProvider):
            def __init__(self, config: LLMConfig):
                super().__init__(config)
                self._client = MyAPIClient(api_key=config.api_key)
            
            async def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
                # Implementation here
                pass
            
            async def stream(self, messages: list[Message], **kwargs) -> AsyncGenerator[StreamChunk, None]:
                # Implementation here
                pass
        ```
    """
    
    # Class-level provider type identifier
    provider_type: ProviderType
    
    def __init__(self, config: LLMConfig):
        """Initialize the provider with configuration.
        
        Args:
            config: Provider configuration including model, temperature, etc.
        """
        self.config = config
        self._initialized = False
    
    # -------------------------------------------------------------------------
    # Abstract Methods - Must be implemented by subclasses
    # -------------------------------------------------------------------------
    
    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate a response for the given messages.
        
        This is the primary method for non-streaming generation.
        
        Args:
            messages: List of conversation messages.
            **kwargs: Additional provider-specific parameters.
                Common overrides: temperature, max_tokens, top_p
        
        Returns:
            LLMResponse containing the generated content and metadata.
        
        Raises:
            LLMConnectionError: If connection to provider fails.
            LLMAuthenticationError: If API key is invalid.
            LLMRateLimitError: If rate limit is exceeded.
            LLMContextLengthError: If input exceeds model's context.
            LLMResponseError: If response is invalid or unexpected.
        """
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response for the given messages.
        
        Yields chunks as they are received from the provider,
        enabling real-time display of generation.
        
        Args:
            messages: List of conversation messages.
            **kwargs: Additional provider-specific parameters.
        
        Yields:
            StreamChunk objects containing content deltas.
        
        Raises:
            LLMStreamError: If streaming fails mid-response.
            (Plus all exceptions from generate())
        
        Example:
            ```python
            async for chunk in provider.stream(messages):
                print(chunk.content, end="", flush=True)
                if chunk.is_final:
                    print(f"\\nFinished: {chunk.finish_reason}")
            ```
        """
        pass
    
    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        **kwargs
    ) -> list[list[float]]:
        """Generate embeddings for the given texts.
        
        Note: Not all providers support embeddings. Implementations
        should raise NotImplementedError if not supported.
        
        Args:
            texts: List of texts to embed.
            **kwargs: Additional parameters (e.g., embedding model).
        
        Returns:
            List of embedding vectors, one per input text.
        
        Raises:
            NotImplementedError: If provider doesn't support embeddings.
            LLMError: For other API-related errors.
        """
        pass
    
    # -------------------------------------------------------------------------
    # Concrete Methods - Shared functionality
    # -------------------------------------------------------------------------
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Convenience method to generate from a simple text prompt.
        
        Wraps the prompt in a user message and returns just the content.
        
        Args:
            prompt: The text prompt to send.
            **kwargs: Passed to generate().
        
        Returns:
            The generated text content.
        """
        messages = [Message.user(prompt)]
        response = await self.generate(messages, **kwargs)
        return response.content
    
    async def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[Message]] = None,
        **kwargs
    ) -> LLMResponse:
        """High-level chat interface with optional system prompt and history.
        
        Args:
            user_message: The current user message.
            system_prompt: Optional system instructions.
            history: Optional list of previous messages.
            **kwargs: Passed to generate().
        
        Returns:
            LLMResponse from the model.
        """
        messages = []
        
        if system_prompt:
            messages.append(Message.system(system_prompt))
        
        if history:
            messages.extend(history)
        
        messages.append(Message.user(user_message))
        
        return await self.generate(messages, **kwargs)
    
    def _validate_messages(self, messages: list[Message]) -> None:
        """Validate messages before sending to provider.
        
        Override in subclasses for provider-specific validation.
        
        Args:
            messages: Messages to validate.
        
        Raises:
            ValueError: If messages are invalid.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        for msg in messages:
            if not msg.content and msg.role != MessageRole.ASSISTANT:
                raise ValueError(f"Message content cannot be empty for role {msg.role}")
    
    def _merge_config(self, **kwargs) -> dict[str, Any]:
        """Merge runtime kwargs with base config.
        
        Runtime parameters override config values.
        
        Args:
            **kwargs: Runtime parameter overrides.
        
        Returns:
            Merged configuration dictionary.
        """
        base = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        # Extra params from config
        base.update(self.config.extra_params)
        # Runtime overrides take precedence
        base.update({k: v for k, v in kwargs.items() if v is not None})
        return base
    
    def _measure_latency(self, start_time: float) -> float:
        """Calculate latency in milliseconds.
        
        Args:
            start_time: Start time from time.perf_counter().
        
        Returns:
            Latency in milliseconds.
        """
        return (time.perf_counter() - start_time) * 1000
    
    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------
    
    async def __aenter__(self) -> "BaseLLMProvider":
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self) -> None:
        """Initialize provider resources.
        
        Override in subclasses if initialization is needed
        (e.g., connection pooling, health checks).
        """
        self._initialized = True
    
    async def close(self) -> None:
        """Clean up provider resources.
        
        Override in subclasses if cleanup is needed
        (e.g., closing HTTP clients).
        """
        self._initialized = False
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    @property
    def model_name(self) -> str:
        """Get the configured model name."""
        return self.config.model
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"
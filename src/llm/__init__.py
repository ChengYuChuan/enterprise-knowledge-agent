"""
LLM Provider Module

This module provides a unified interface for multiple LLM providers.
It implements the Strategy Pattern to allow seamless switching between
different providers (OpenAI, Anthropic, Ollama) without code changes.

Quick Start:
    ```python
    from src.llm import LLMConfig, Message, OpenAIProvider
    
    # Create a provider directly
    config = LLMConfig(model="gpt-4")
    provider = OpenAIProvider(config)
    
    # Use the unified interface
    messages = [Message.user("What is RAG?")]
    response = await provider.generate(messages)
    print(response.content)
    ```

Factory Pattern (recommended for production):
    ```python
    from src.llm import get_provider, LLMConfig
    
    # Create by name
    config = LLMConfig(model="gpt-4")
    provider = get_provider("openai", config)
    
    # From environment variables
    provider = get_provider_from_env()
    
    # From YAML configuration
    provider = get_provider_from_yaml("configs/llm_configs/openai.yaml")
    ```

Architecture:
    BaseLLMProvider (Abstract)
    ├── OpenAIProvider
    ├── AnthropicProvider
    └── OllamaProvider

Exports:
    - Base classes: BaseLLMProvider, LLMConfig, Message, LLMResponse
    - Providers: OpenAIProvider, AnthropicProvider, OllamaProvider
    - Data types: MessageRole, ProviderType, TokenUsage, StreamChunk
    - Exceptions: LLMError and subclasses
    - Factory: get_provider, get_provider_from_env, get_provider_from_yaml
    - Registry: register_provider, list_providers, is_provider_available
"""

# Base classes and data types
from src.llm.base import (
    # Core types
    BaseLLMProvider,
    LLMConfig,
    Message,
    MessageRole,
    LLMResponse,
    StreamChunk,
    TokenUsage,
    ProviderType,
)

# Provider implementations
from src.llm.providers import (
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
)

# Exceptions
from src.llm.exceptions import (
    LLMError,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMContextLengthError,
    LLMResponseError,
    LLMProviderNotFoundError,
    LLMStreamError,
)

# Factory functions
from src.llm.factory import (
    get_provider,
    get_provider_from_env,
    get_provider_from_yaml,
    list_providers,
    is_provider_available,
    register_provider,
    unregister_provider,
)

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "LLMConfig",
    "Message",
    "LLMResponse",
    "StreamChunk",
    "TokenUsage",
    # Enums
    "MessageRole",
    "ProviderType",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    # Exceptions
    "LLMError",
    "LLMConnectionError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMContextLengthError",
    "LLMResponseError",
    "LLMProviderNotFoundError",
    "LLMStreamError",
    # Factory functions
    "get_provider",
    "get_provider_from_env",
    "get_provider_from_yaml",
    "list_providers",
    "is_provider_available",
    "register_provider",
    "unregister_provider",
]
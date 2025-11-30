"""
OpenAI LLM Provider

Adapter for OpenAI's Chat Completions API.
Supports GPT-3.5, GPT-4, GPT-4-Turbo, and future models.

API Documentation:
    https://platform.openai.com/docs/api-reference/chat

Supported Features:
    - Chat completions (generate)
    - Streaming responses (stream)
    - Embeddings (embed)
    - Function/Tool calling (via extra_params)

Example Usage:
    ```python
    from src.llm import LLMConfig, Message
    from src.llm.providers.openai_provider import OpenAIProvider
    
    config = LLMConfig(
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        api_key="sk-..."  # Or set OPENAI_API_KEY env var
    )
    
    async with OpenAIProvider(config) as provider:
        messages = [Message.user("Explain RAG in simple terms")]
        response = await provider.generate(messages)
        print(response.content)
    ```
"""

import os
import time
from typing import AsyncGenerator, Optional, Any

from src.llm.base import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
    Message,
    MessageRole,
    ProviderType,
    StreamChunk,
    TokenUsage,
)
from src.llm.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMContextLengthError,
    LLMRateLimitError,
    LLMResponseError,
    LLMStreamError,
)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider implementation.
    
    This class adapts OpenAI's Chat Completions API to our unified
    BaseLLMProvider interface.
    
    Key Adaptations:
        - Converts our Message objects to OpenAI's message format
        - Maps OpenAI-specific errors to our exception hierarchy
        - Handles both streaming and non-streaming responses
        - Supports embeddings via text-embedding-ada-002 or newer models
    
    Attributes:
        provider_type: Always ProviderType.OPENAI
        config: The LLMConfig for this provider
        _client: The OpenAI async client instance
    """
    
    provider_type = ProviderType.OPENAI
    
    # Default embedding model (can be overridden in embed() call)
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    
    def __init__(self, config: LLMConfig):
        """Initialize OpenAI provider.
        
        Args:
            config: Provider configuration.
                - api_key: OpenAI API key (or set OPENAI_API_KEY env var)
                - base_url: Optional custom endpoint (for Azure, proxies)
                - model: Model name (gpt-4, gpt-3.5-turbo, etc.)
        
        Raises:
            LLMAuthenticationError: If no API key is available.
        """
        super().__init__(config)
        
        # Resolve API key from config or environment
        self._api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise LLMAuthenticationError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment "
                "variable or pass api_key in config.",
                provider="openai"
            )
        
        self._base_url = config.base_url
        self._client: Optional[Any] = None  # Lazy initialization
    
    async def initialize(self) -> None:
        """Initialize the OpenAI async client.
        
        We use lazy initialization to avoid importing openai until needed.
        This also allows for proper async client setup.
        """
        if self._client is not None:
            return
        
        try:
            # Import here to make openai an optional dependency
            from openai import AsyncOpenAI
            
            client_kwargs = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            
            self._client = AsyncOpenAI(**client_kwargs)
            self._initialized = True
            
        except ImportError:
            raise LLMConnectionError(
                "openai package not installed. Run: pip install openai",
                provider="openai"
            )
    
    async def close(self) -> None:
        """Close the OpenAI client and release resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._initialized = False
    
    # -------------------------------------------------------------------------
    # Core Methods
    # -------------------------------------------------------------------------
    
    async def generate(
        self,
        messages: list[Message],
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion.
        
        Args:
            messages: Conversation messages.
            **kwargs: Override config values (temperature, max_tokens, etc.)
                Also supports OpenAI-specific params:
                - response_format: {"type": "json_object"} for JSON mode
                - tools: List of tool definitions
                - tool_choice: "auto", "none", or specific tool
                - logprobs: Include log probabilities
        
        Returns:
            LLMResponse with generated content and metadata.
        
        Raises:
            LLMAuthenticationError: Invalid API key.
            LLMRateLimitError: Rate limit exceeded.
            LLMContextLengthError: Input too long.
            LLMResponseError: Invalid response from API.
        """
        await self._ensure_initialized()
        self._validate_messages(messages)
        
        # Prepare request parameters
        params = self._merge_config(**kwargs)
        openai_messages = self._convert_messages(messages)
        
        start_time = time.perf_counter()
        
        try:
            response = await self._client.chat.completions.create(
                model=params["model"],
                messages=openai_messages,
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                top_p=params["top_p"],
                stream=False,
                # Pass through any extra OpenAI-specific params
                **{k: v for k, v in params.items() 
                   if k not in ["model", "temperature", "max_tokens", "top_p"]}
            )
            
            # Extract response data
            choice = response.choices[0]
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ) if response.usage else TokenUsage()
            
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason,
                latency_ms=self._measure_latency(start_time),
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )
            
        except Exception as e:
            raise self._convert_exception(e)
    
    async def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat completion.
        
        Yields chunks as they are received from OpenAI.
        The final chunk will have is_final=True and include usage stats.
        
        Args:
            messages: Conversation messages.
            **kwargs: Same as generate().
        
        Yields:
            StreamChunk objects with content deltas.
        
        Example:
            ```python
            full_response = ""
            async for chunk in provider.stream(messages):
                full_response += chunk.content
                print(chunk.content, end="", flush=True)
            ```
        """
        await self._ensure_initialized()
        self._validate_messages(messages)
        
        params = self._merge_config(**kwargs)
        openai_messages = self._convert_messages(messages)
        
        try:
            stream = await self._client.chat.completions.create(
                model=params["model"],
                messages=openai_messages,
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                top_p=params["top_p"],
                stream=True,
                stream_options={"include_usage": True},  # Get usage on final chunk
                **{k: v for k, v in params.items() 
                   if k not in ["model", "temperature", "max_tokens", "top_p"]}
            )
            
            async for chunk in stream:
                # Handle the chunk
                if not chunk.choices:
                    # Final chunk with usage info only
                    if chunk.usage:
                        yield StreamChunk(
                            content="",
                            is_final=True,
                            usage=TokenUsage(
                                prompt_tokens=chunk.usage.prompt_tokens,
                                completion_tokens=chunk.usage.completion_tokens,
                                total_tokens=chunk.usage.total_tokens,
                            )
                        )
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta
                
                # Extract content (may be None for function calls)
                content = delta.content or ""
                
                # Check if this is the final chunk
                is_final = choice.finish_reason is not None
                
                yield StreamChunk(
                    content=content,
                    is_final=is_final,
                    finish_reason=choice.finish_reason,
                )
                
        except Exception as e:
            raise LLMStreamError(
                f"Stream error: {str(e)}",
                provider="openai",
                original_error=e,
            )
    
    async def embed(
        self,
        texts: list[str],
        **kwargs
    ) -> list[list[float]]:
        """Generate embeddings using OpenAI's embedding models.
        
        Args:
            texts: List of texts to embed.
            **kwargs:
                - model: Embedding model (default: text-embedding-3-small)
                - dimensions: Output dimensions (for ada-3 models)
        
        Returns:
            List of embedding vectors.
        
        Example:
            ```python
            embeddings = await provider.embed(["Hello", "World"])
            # embeddings[0] is the vector for "Hello"
            ```
        """
        await self._ensure_initialized()
        
        if not texts:
            return []
        
        model = kwargs.get("model", self.DEFAULT_EMBEDDING_MODEL)
        
        try:
            response = await self._client.embeddings.create(
                model=model,
                input=texts,
                **{k: v for k, v in kwargs.items() if k != "model"}
            )
            
            # Sort by index to ensure order matches input
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
            
        except Exception as e:
            raise self._convert_exception(e)
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    async def _ensure_initialized(self) -> None:
        """Ensure client is initialized before making requests."""
        if self._client is None:
            await self.initialize()
    
    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert our Message objects to OpenAI's format.
        
        OpenAI format:
            {"role": "user", "content": "Hello"}
            {"role": "assistant", "content": "Hi there!"}
            {"role": "system", "content": "You are helpful."}
        
        Args:
            messages: Our Message objects.
        
        Returns:
            List of dicts in OpenAI format.
        """
        return [msg.to_dict() for msg in messages]
    
    def _convert_exception(self, error: Exception) -> Exception:
        """Convert OpenAI exceptions to our exception hierarchy.
        
        This is the core of the Adapter pattern for error handling.
        
        Args:
            error: The original OpenAI exception.
        
        Returns:
            Our corresponding LLMError subclass.
        """
        error_message = str(error)
        error_type = type(error).__name__
        
        # Import OpenAI exceptions for type checking
        try:
            from openai import (
                AuthenticationError,
                RateLimitError,
                APIConnectionError,
                BadRequestError,
            )
            
            if isinstance(error, AuthenticationError):
                return LLMAuthenticationError(
                    "Invalid OpenAI API key",
                    provider="openai",
                    original_error=error,
                )
            
            if isinstance(error, RateLimitError):
                # Try to extract retry-after from headers
                retry_after = None
                if hasattr(error, 'response') and error.response:
                    retry_after = error.response.headers.get('retry-after')
                    if retry_after:
                        retry_after = float(retry_after)
                
                return LLMRateLimitError(
                    "OpenAI rate limit exceeded",
                    provider="openai",
                    original_error=error,
                    retry_after=retry_after,
                )
            
            if isinstance(error, APIConnectionError):
                return LLMConnectionError(
                    f"Failed to connect to OpenAI: {error_message}",
                    provider="openai",
                    original_error=error,
                )
            
            if isinstance(error, BadRequestError):
                # Check for context length errors
                if "maximum context length" in error_message.lower():
                    return LLMContextLengthError(
                        error_message,
                        provider="openai",
                        original_error=error,
                    )
                return LLMResponseError(
                    f"Bad request: {error_message}",
                    provider="openai",
                    original_error=error,
                )
                
        except ImportError:
            pass  # openai not installed, fall through to generic handling
        
        # Generic fallback
        return LLMResponseError(
            f"OpenAI error ({error_type}): {error_message}",
            provider="openai",
            original_error=error,
        )
"""
Anthropic LLM Provider

Adapter for Anthropic's Messages API (Claude models).
Supports Claude 3 family: Opus, Sonnet, Haiku.

API Documentation:
    https://docs.anthropic.com/en/api/messages

Key Differences from OpenAI:
    1. System prompt is a separate parameter, not in messages
    2. Different streaming event structure
    3. No native embedding API (raises NotImplementedError)
    4. Uses "max_tokens" as required parameter

Example Usage:
    ```python
    from src.llm import LLMConfig, Message
    from src.llm.providers.anthropic_provider import AnthropicProvider
    
    config = LLMConfig(
        model="claude-3-sonnet-20240229",
        temperature=0.7,
        max_tokens=1000,
        api_key="sk-ant-..."  # Or set ANTHROPIC_API_KEY env var
    )
    
    async with AnthropicProvider(config) as provider:
        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("Explain RAG in simple terms"),
        ]
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


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider implementation.
    
    This class adapts Anthropic's Messages API to our unified
    BaseLLMProvider interface.
    
    Key Adaptations:
        - Extracts system messages and passes as separate parameter
        - Converts streaming events to our StreamChunk format
        - Maps Anthropic-specific errors to our exception hierarchy
    
    Model Names:
        - claude-3-opus-20240229 (most capable)
        - claude-3-sonnet-20240229 (balanced)
        - claude-3-haiku-20240307 (fastest)
        - claude-3-5-sonnet-20240620 (latest)
    
    Attributes:
        provider_type: Always ProviderType.ANTHROPIC
        config: The LLMConfig for this provider
        _client: The Anthropic async client instance
    """
    
    provider_type = ProviderType.ANTHROPIC
    
    # Anthropic's default max tokens if not specified
    DEFAULT_MAX_TOKENS = 4096
    
    def __init__(self, config: LLMConfig):
        """Initialize Anthropic provider.
        
        Args:
            config: Provider configuration.
                - api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
                - base_url: Optional custom endpoint
                - model: Model name (claude-3-sonnet-20240229, etc.)
        
        Raises:
            LLMAuthenticationError: If no API key is available.
        """
        super().__init__(config)
        
        # Resolve API key from config or environment
        self._api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise LLMAuthenticationError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY "
                "environment variable or pass api_key in config.",
                provider="anthropic"
            )
        
        self._base_url = config.base_url
        self._client: Optional[Any] = None
    
    async def initialize(self) -> None:
        """Initialize the Anthropic async client."""
        if self._client is not None:
            return
        
        try:
            from anthropic import AsyncAnthropic
            
            client_kwargs = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            
            self._client = AsyncAnthropic(**client_kwargs)
            self._initialized = True
            
        except ImportError:
            raise LLMConnectionError(
                "anthropic package not installed. Run: pip install anthropic",
                provider="anthropic"
            )
    
    async def close(self) -> None:
        """Close the Anthropic client."""
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
        """Generate a message completion.
        
        Args:
            messages: Conversation messages. System messages are extracted
                and passed separately to the API.
            **kwargs: Override config values or Anthropic-specific params:
                - metadata: Request metadata
                - stop_sequences: Custom stop sequences
        
        Returns:
            LLMResponse with generated content and metadata.
        """
        await self._ensure_initialized()
        self._validate_messages(messages)
        
        params = self._merge_config(**kwargs)
        system_prompt, anthropic_messages = self._convert_messages(messages)
        
        start_time = time.perf_counter()
        
        try:
            # Build request kwargs
            request_kwargs = {
                "model": params["model"],
                "messages": anthropic_messages,
                "max_tokens": params.get("max_tokens", self.DEFAULT_MAX_TOKENS),
                "temperature": params["temperature"],
                "top_p": params["top_p"],
            }
            
            # Add system prompt if present
            if system_prompt:
                request_kwargs["system"] = system_prompt
            
            # Add any extra params (stop_sequences, metadata, etc.)
            for key in ["stop_sequences", "metadata", "top_k"]:
                if key in params:
                    request_kwargs[key] = params[key]
            
            response = await self._client.messages.create(**request_kwargs)
            
            # Extract content (Anthropic returns a list of content blocks)
            content = self._extract_content(response.content)
            
            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=response.stop_reason,
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
        """Stream a message completion.
        
        Anthropic uses Server-Sent Events with different event types:
        - message_start: Initial message metadata
        - content_block_start: Start of a content block
        - content_block_delta: Text delta
        - content_block_stop: End of content block
        - message_delta: Final stats (stop_reason, usage)
        - message_stop: End of message
        
        We adapt these to our simpler StreamChunk format.
        
        Args:
            messages: Conversation messages.
            **kwargs: Same as generate().
        
        Yields:
            StreamChunk objects with content deltas.
        """
        await self._ensure_initialized()
        self._validate_messages(messages)
        
        params = self._merge_config(**kwargs)
        system_prompt, anthropic_messages = self._convert_messages(messages)
        
        try:
            request_kwargs = {
                "model": params["model"],
                "messages": anthropic_messages,
                "max_tokens": params.get("max_tokens", self.DEFAULT_MAX_TOKENS),
                "temperature": params["temperature"],
                "top_p": params["top_p"],
            }
            
            if system_prompt:
                request_kwargs["system"] = system_prompt
            
            # Use streaming context manager
            async with self._client.messages.stream(**request_kwargs) as stream:
                async for event in stream:
                    # Handle different event types
                    if event.type == "content_block_delta":
                        # This is where we get the actual text
                        if hasattr(event.delta, 'text'):
                            yield StreamChunk(
                                content=event.delta.text,
                                is_final=False,
                            )
                    
                    elif event.type == "message_delta":
                        # Final event with stop reason and usage
                        usage = None
                        if hasattr(event, 'usage') and event.usage:
                            usage = TokenUsage(
                                prompt_tokens=0,  # Not available in delta
                                completion_tokens=event.usage.output_tokens,
                                total_tokens=event.usage.output_tokens,
                            )
                        
                        yield StreamChunk(
                            content="",
                            is_final=True,
                            finish_reason=event.delta.stop_reason if hasattr(event.delta, 'stop_reason') else None,
                            usage=usage,
                        )
                        
        except Exception as e:
            raise LLMStreamError(
                f"Stream error: {str(e)}",
                provider="anthropic",
                original_error=e,
            )
    
    async def embed(
        self,
        texts: list[str],
        **kwargs
    ) -> list[list[float]]:
        """Generate embeddings.
        
        Note: Anthropic does not provide an embedding API.
        Use OpenAI or a dedicated embedding provider instead.
        
        Raises:
            NotImplementedError: Always, as Anthropic has no embedding API.
        """
        raise NotImplementedError(
            "Anthropic does not provide an embedding API. "
            "Use OpenAI's text-embedding-3-small or a dedicated embedding provider."
        )
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    async def _ensure_initialized(self) -> None:
        """Ensure client is initialized before making requests."""
        if self._client is None:
            await self.initialize()
    
    def _convert_messages(
        self, 
        messages: list[Message]
    ) -> tuple[Optional[str], list[dict[str, Any]]]:
        """Convert our Message objects to Anthropic's format.
        
        Key difference: Anthropic expects system message as a separate
        parameter, not in the messages list.
        
        Args:
            messages: Our Message objects.
        
        Returns:
            Tuple of (system_prompt, messages_list)
        """
        system_prompt = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Anthropic only supports one system prompt
                # If multiple, concatenate them
                if system_prompt is None:
                    system_prompt = msg.content
                else:
                    system_prompt += "\n\n" + msg.content
            else:
                # Map our roles to Anthropic's expected roles
                role = msg.role.value if isinstance(msg.role, MessageRole) else msg.role
                # Anthropic only accepts "user" and "assistant"
                if role == "tool":
                    role = "user"  # Tool results go as user messages
                
                anthropic_messages.append({
                    "role": role,
                    "content": msg.content,
                })
        
        return system_prompt, anthropic_messages
    
    def _extract_content(self, content_blocks: list) -> str:
        """Extract text from Anthropic's content blocks.
        
        Anthropic returns content as a list of blocks, e.g.:
        [{"type": "text", "text": "Hello!"}]
        
        Args:
            content_blocks: List of content blocks from response.
        
        Returns:
            Concatenated text content.
        """
        texts = []
        for block in content_blocks:
            if hasattr(block, 'text'):
                texts.append(block.text)
            elif isinstance(block, dict) and 'text' in block:
                texts.append(block['text'])
        return "".join(texts)
    
    def _convert_exception(self, error: Exception) -> Exception:
        """Convert Anthropic exceptions to our exception hierarchy."""
        error_message = str(error)
        error_type = type(error).__name__
        
        try:
            from anthropic import (
                AuthenticationError,
                RateLimitError,
                APIConnectionError,
                BadRequestError,
            )
            
            if isinstance(error, AuthenticationError):
                return LLMAuthenticationError(
                    "Invalid Anthropic API key",
                    provider="anthropic",
                    original_error=error,
                )
            
            if isinstance(error, RateLimitError):
                return LLMRateLimitError(
                    "Anthropic rate limit exceeded",
                    provider="anthropic",
                    original_error=error,
                )
            
            if isinstance(error, APIConnectionError):
                return LLMConnectionError(
                    f"Failed to connect to Anthropic: {error_message}",
                    provider="anthropic",
                    original_error=error,
                )
            
            if isinstance(error, BadRequestError):
                if "token" in error_message.lower() and "limit" in error_message.lower():
                    return LLMContextLengthError(
                        error_message,
                        provider="anthropic",
                        original_error=error,
                    )
                return LLMResponseError(
                    f"Bad request: {error_message}",
                    provider="anthropic",
                    original_error=error,
                )
                
        except ImportError:
            pass
        
        return LLMResponseError(
            f"Anthropic error ({error_type}): {error_message}",
            provider="anthropic",
            original_error=error,
        )
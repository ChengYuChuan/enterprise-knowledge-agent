"""
Ollama LLM Provider

Adapter for Ollama's local LLM server.
Supports any model available through Ollama (Llama, Mistral, etc.)

API Documentation:
    https://github.com/ollama/ollama/blob/main/docs/api.md

Key Features:
    - No API key required (local server)
    - Supports both chat and embeddings
    - Custom model parameters via extra_params
    - Uses httpx for async HTTP requests

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull a model: ollama pull llama3
    3. Start server (auto-starts on install): ollama serve

Example Usage:
    ```python
    from src.llm import LLMConfig, Message
    from src.llm.providers.ollama_provider import OllamaProvider
    
    config = LLMConfig(
        model="llama3",  # or mistral, codellama, etc.
        temperature=0.7,
        base_url="http://localhost:11434",  # Default Ollama URL
    )
    
    async with OllamaProvider(config) as provider:
        messages = [Message.user("Explain RAG")]
        response = await provider.generate(messages)
        print(response.content)
    ```
"""

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
    LLMConnectionError,
    LLMResponseError,
    LLMStreamError,
)


class OllamaProvider(BaseLLMProvider):
    """Ollama API provider implementation.
    
    This class adapts Ollama's REST API to our unified
    BaseLLMProvider interface.
    
    Key Features:
        - No authentication required (local server)
        - Supports any Ollama-compatible model
        - Native embedding support via /api/embeddings
        - Customizable model options (context length, etc.)
    
    Popular Models:
        - llama3: Meta's Llama 3 (8B, 70B)
        - mistral: Mistral 7B
        - codellama: Code-specialized Llama
        - phi3: Microsoft's Phi-3
        - gemma: Google's Gemma
    
    Attributes:
        provider_type: Always ProviderType.OLLAMA
        config: The LLMConfig for this provider
        _client: httpx AsyncClient instance
    """
    
    provider_type = ProviderType.OLLAMA
    
    # Default Ollama server URL
    DEFAULT_BASE_URL = "http://localhost:11434"
    
    # Default embedding model (if different from chat model)
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
    
    def __init__(self, config: LLMConfig):
        """Initialize Ollama provider.
        
        Args:
            config: Provider configuration.
                - model: Ollama model name (llama3, mistral, etc.)
                - base_url: Ollama server URL (default: http://localhost:11434)
                - extra_params: Model-specific options (num_ctx, num_gpu, etc.)
        """
        super().__init__(config)
        
        self._base_url = (config.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._client: Optional[Any] = None
    
    async def initialize(self) -> None:
        """Initialize the HTTP client."""
        if self._client is not None:
            return
        
        try:
            import httpx
            
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self.config.timeout, connect=10.0),
            )
            self._initialized = True
            
            # Verify connection by checking if model exists
            await self._check_model_available()
            
        except ImportError:
            raise LLMConnectionError(
                "httpx package not installed. Run: pip install httpx",
                provider="ollama"
            )
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
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
            **kwargs: Override config values or Ollama-specific options:
                - num_ctx: Context window size
                - num_predict: Max tokens to generate
                - top_k: Top-k sampling
                - repeat_penalty: Repetition penalty
                - seed: Random seed for reproducibility
        
        Returns:
            LLMResponse with generated content and metadata.
        """
        await self._ensure_initialized()
        self._validate_messages(messages)
        
        params = self._merge_config(**kwargs)
        ollama_messages = self._convert_messages(messages)
        
        # Build options dict for Ollama
        options = self._build_options(params)
        
        start_time = time.perf_counter()
        
        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": params["model"],
                    "messages": ollama_messages,
                    "stream": False,
                    "options": options,
                },
            )
            
            if response.status_code != 200:
                raise LLMResponseError(
                    f"Ollama returned status {response.status_code}: {response.text}",
                    provider="ollama"
                )
            
            data = response.json()
            
            # Extract usage info if available
            usage = TokenUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            )
            
            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", params["model"]),
                usage=usage,
                finish_reason="stop" if data.get("done") else None,
                latency_ms=self._measure_latency(start_time),
                raw_response=data,
            )
            
        except Exception as e:
            if isinstance(e, (LLMConnectionError, LLMResponseError)):
                raise
            raise self._convert_exception(e)
    
    async def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat completion.
        
        Ollama streams JSON objects, one per line (NDJSON format).
        Each object contains a "message" with "content" delta.
        
        Args:
            messages: Conversation messages.
            **kwargs: Same as generate().
        
        Yields:
            StreamChunk objects with content deltas.
        """
        await self._ensure_initialized()
        self._validate_messages(messages)
        
        params = self._merge_config(**kwargs)
        ollama_messages = self._convert_messages(messages)
        options = self._build_options(params)
        
        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": params["model"],
                    "messages": ollama_messages,
                    "stream": True,
                    "options": options,
                },
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise LLMResponseError(
                        f"Ollama returned status {response.status_code}: {error_text}",
                        provider="ollama"
                    )
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        import json
                        data = json.loads(line)
                    except Exception:
                        continue
                    
                    # Extract content from message
                    content = data.get("message", {}).get("content", "")
                    is_done = data.get("done", False)
                    
                    # Build usage for final chunk
                    usage = None
                    if is_done:
                        usage = TokenUsage(
                            prompt_tokens=data.get("prompt_eval_count", 0),
                            completion_tokens=data.get("eval_count", 0),
                            total_tokens=(
                                data.get("prompt_eval_count", 0) + 
                                data.get("eval_count", 0)
                            ),
                        )
                    
                    yield StreamChunk(
                        content=content,
                        is_final=is_done,
                        finish_reason="stop" if is_done else None,
                        usage=usage,
                    )
                    
        except Exception as e:
            if isinstance(e, (LLMConnectionError, LLMResponseError, LLMStreamError)):
                raise
            raise LLMStreamError(
                f"Stream error: {str(e)}",
                provider="ollama",
                original_error=e,
            )
    
    async def embed(
        self,
        texts: list[str],
        **kwargs
    ) -> list[list[float]]:
        """Generate embeddings using Ollama's embedding API.
        
        Args:
            texts: List of texts to embed.
            **kwargs:
                - model: Embedding model (default: nomic-embed-text)
        
        Returns:
            List of embedding vectors.
        
        Note:
            Requires an embedding-capable model. Popular choices:
            - nomic-embed-text
            - mxbai-embed-large
            - all-minilm
        """
        await self._ensure_initialized()
        
        if not texts:
            return []
        
        model = kwargs.get("model", self.DEFAULT_EMBEDDING_MODEL)
        embeddings = []
        
        try:
            # Ollama's embedding API processes one text at a time
            for text in texts:
                response = await self._client.post(
                    "/api/embeddings",
                    json={
                        "model": model,
                        "prompt": text,
                    },
                )
                
                if response.status_code != 200:
                    raise LLMResponseError(
                        f"Embedding failed: {response.text}",
                        provider="ollama"
                    )
                
                data = response.json()
                embeddings.append(data.get("embedding", []))
            
            return embeddings
            
        except Exception as e:
            if isinstance(e, LLMResponseError):
                raise
            raise self._convert_exception(e)
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    async def _ensure_initialized(self) -> None:
        """Ensure client is initialized."""
        if self._client is None:
            await self.initialize()
    
    async def _check_model_available(self) -> None:
        """Check if the configured model is available on the Ollama server."""
        try:
            response = await self._client.get("/api/tags")
            
            if response.status_code != 200:
                raise LLMConnectionError(
                    f"Failed to connect to Ollama at {self._base_url}. "
                    "Is Ollama running? Start with: ollama serve",
                    provider="ollama"
                )
            
            # Check if our model is in the list
            data = response.json()
            available_models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            
            model_name = self.config.model.split(":")[0]  # Handle tags like "llama3:8b"
            
            if model_name not in available_models and available_models:
                # Not a fatal error, just a warning (model might still work)
                import logging
                logging.getLogger(__name__).warning(
                    f"Model '{self.config.model}' not found locally. "
                    f"Available: {available_models}. "
                    f"Pull with: ollama pull {self.config.model}"
                )
                
        except Exception as e:
            if isinstance(e, LLMConnectionError):
                raise
            raise LLMConnectionError(
                f"Failed to connect to Ollama at {self._base_url}: {e}",
                provider="ollama",
                original_error=e,
            )
    
    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert our Message objects to Ollama's format.
        
        Ollama format is similar to OpenAI:
            {"role": "user", "content": "Hello"}
            {"role": "assistant", "content": "Hi!"}
            {"role": "system", "content": "You are helpful."}
        
        Args:
            messages: Our Message objects.
        
        Returns:
            List of dicts in Ollama format.
        """
        return [msg.to_dict() for msg in messages]
    
    def _build_options(self, params: dict[str, Any]) -> dict[str, Any]:
        """Build Ollama-specific options dict.
        
        Maps our generic params to Ollama's options format.
        
        Args:
            params: Merged configuration parameters.
        
        Returns:
            Options dict for Ollama API.
        """
        options = {
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 1.0),
        }
        
        # Map max_tokens to num_predict
        if "max_tokens" in params:
            options["num_predict"] = params["max_tokens"]
        
        # Add any Ollama-specific options from extra_params
        ollama_options = [
            "num_ctx", "num_gpu", "num_thread",
            "top_k", "repeat_penalty", "seed",
            "stop", "mirostat", "mirostat_eta", "mirostat_tau",
        ]
        
        for opt in ollama_options:
            if opt in params:
                options[opt] = params[opt]
        
        return options
    
    def _convert_exception(self, error: Exception) -> Exception:
        """Convert httpx exceptions to our exception hierarchy."""
        error_message = str(error)
        
        try:
            import httpx
            
            if isinstance(error, httpx.ConnectError):
                return LLMConnectionError(
                    f"Cannot connect to Ollama at {self._base_url}. "
                    "Is Ollama running? Start with: ollama serve",
                    provider="ollama",
                    original_error=error,
                )
            
            if isinstance(error, httpx.TimeoutException):
                return LLMConnectionError(
                    f"Timeout connecting to Ollama: {error_message}",
                    provider="ollama",
                    original_error=error,
                )
                
        except ImportError:
            pass
        
        return LLMResponseError(
            f"Ollama error: {error_message}",
            provider="ollama",
            original_error=error,
        )
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    async def list_models(self) -> list[dict[str, Any]]:
        """List all models available on the Ollama server.
        
        Returns:
            List of model info dicts with name, size, modified date, etc.
        
        Example:
            ```python
            models = await provider.list_models()
            for m in models:
                print(f"{m['name']}: {m['size'] / 1e9:.1f}GB")
            ```
        """
        await self._ensure_initialized()
        
        response = await self._client.get("/api/tags")
        if response.status_code != 200:
            raise LLMResponseError(
                f"Failed to list models: {response.text}",
                provider="ollama"
            )
        
        return response.json().get("models", [])
    
    async def pull_model(self, model_name: str) -> AsyncGenerator[dict[str, Any], None]:
        """Pull a model from Ollama's registry.
        
        Args:
            model_name: Name of the model to pull (e.g., "llama3:8b")
        
        Yields:
            Progress updates as dicts with status, completed, total bytes.
        
        Example:
            ```python
            async for progress in provider.pull_model("llama3"):
                print(f"Downloading: {progress.get('completed', 0) / 1e6:.0f}MB")
            ```
        """
        await self._ensure_initialized()
        
        async with self._client.stream(
            "POST",
            "/api/pull",
            json={"name": model_name, "stream": True},
        ) as response:
            async for line in response.aiter_lines():
                if line.strip():
                    import json
                    yield json.loads(line)
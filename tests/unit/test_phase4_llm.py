"""
Unit tests for Phase 4: Multi-LLM + API Layer

Tests LLM provider abstraction, factory pattern, and basic API functionality.
Run with: poetry run pytest tests/unit/test_phase4_llm.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm import (
    # Base classes
    BaseLLMProvider,
    LLMConfig,
    Message,
    MessageRole,
    LLMResponse,
    TokenUsage,
    ProviderType,
    # Providers
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    # Factory
    get_provider,
    list_providers,
    is_provider_available,
    # Exceptions
    LLMError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
)


class TestLLMConfig:
    """Test cases for LLMConfig dataclass."""
    
    def test_create_default_config(self):
        """Test creating config with defaults."""
        config = LLMConfig(model="gpt-4")
        
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.top_p == 1.0
    
    def test_create_custom_config(self):
        """Test creating config with custom values."""
        config = LLMConfig(
            model="claude-3-sonnet",
            temperature=0.5,
            max_tokens=1000,
            api_key="test-key",
        )
        
        assert config.model == "claude-3-sonnet"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.api_key == "test-key"
    
    def test_temperature_validation_low(self):
        """Test that temperature < 0 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between"):
            LLMConfig(model="gpt-4", temperature=-0.1)
    
    def test_temperature_validation_high(self):
        """Test that temperature > 2 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between"):
            LLMConfig(model="gpt-4", temperature=2.5)
    
    def test_max_tokens_validation(self):
        """Test that max_tokens < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            LLMConfig(model="gpt-4", max_tokens=0)
    
    def test_top_p_validation(self):
        """Test that top_p outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be between"):
            LLMConfig(model="gpt-4", top_p=1.5)


class TestMessage:
    """Test cases for Message class."""
    
    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message.user("Hello, world!")
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
    
    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message.system("You are a helpful assistant.")
        
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are a helpful assistant."
    
    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message.assistant("I can help with that.")
        
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "I can help with that."
    
    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message.user("Test content")
        d = msg.to_dict()
        
        assert d["role"] == "user"
        assert d["content"] == "Test content"


class TestProviderFactory:
    """Test cases for provider factory functions."""
    
    def test_list_providers(self):
        """Test listing available providers."""
        providers = list_providers()
        
        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" in providers
    
    def test_is_provider_available(self):
        """Test checking provider availability."""
        assert is_provider_available("openai") is True
        assert is_provider_available("anthropic") is True
        assert is_provider_available("ollama") is True
        assert is_provider_available("nonexistent") is False
    
    def test_get_provider_openai(self):
        """Test getting OpenAI provider."""
        config = LLMConfig(model="gpt-4", api_key="test-key")
        provider = get_provider("openai", config)
        
        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_type == ProviderType.OPENAI
    
    def test_get_provider_anthropic(self):
        """Test getting Anthropic provider."""
        config = LLMConfig(model="claude-3-sonnet", api_key="test-key")
        provider = get_provider("anthropic", config)
        
        assert isinstance(provider, AnthropicProvider)
        assert provider.provider_type == ProviderType.ANTHROPIC
    
    def test_get_provider_ollama(self):
        """Test getting Ollama provider."""
        config = LLMConfig(model="llama3")
        provider = get_provider("ollama", config)
        
        assert isinstance(provider, OllamaProvider)
        assert provider.provider_type == ProviderType.OLLAMA
    
    def test_get_provider_not_found(self):
        """Test that nonexistent provider raises error."""
        from src.llm import LLMProviderNotFoundError
        
        config = LLMConfig(model="test")
        with pytest.raises(LLMProviderNotFoundError):
            get_provider("nonexistent", config)


class TestOpenAIProvider:
    """Test cases for OpenAI provider."""
    
    def test_provider_type(self):
        """Test that provider type is correct."""
        config = LLMConfig(model="gpt-4", api_key="test")
        provider = OpenAIProvider(config)
        
        assert provider.provider_type == ProviderType.OPENAI
    
    def test_no_api_key_raises_error(self):
        """Test that missing API key raises auth error."""
        config = LLMConfig(model="gpt-4")
        
        # Clear environment variable if set
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMAuthenticationError):
                OpenAIProvider(config)


class TestAnthropicProvider:
    """Test cases for Anthropic provider."""
    
    def test_provider_type(self):
        """Test that provider type is correct."""
        config = LLMConfig(model="claude-3-sonnet", api_key="test")
        provider = AnthropicProvider(config)
        
        assert provider.provider_type == ProviderType.ANTHROPIC
    
    def test_no_api_key_raises_error(self):
        """Test that missing API key raises auth error."""
        config = LLMConfig(model="claude-3-sonnet")
        
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMAuthenticationError):
                AnthropicProvider(config)


class TestOllamaProvider:
    """Test cases for Ollama provider."""
    
    def test_provider_type(self):
        """Test that provider type is correct."""
        config = LLMConfig(model="llama3")
        provider = OllamaProvider(config)
        
        assert provider.provider_type == ProviderType.OLLAMA
    
    def test_default_base_url(self):
        """Test that default base URL is set."""
        config = LLMConfig(model="llama3")
        provider = OllamaProvider(config)
        
        # Ollama doesn't require API key
        assert provider is not None


class TestLLMResponse:
    """Test cases for LLMResponse dataclass."""
    
    def test_create_response(self):
        """Test creating a response."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            usage=usage,
            finish_reason="stop",
        )
        
        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.usage.total_tokens == 30
        assert response.is_complete is True
    
    def test_response_truncated(self):
        """Test detecting truncated response."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            finish_reason="length",
        )
        
        assert response.is_complete is False


class TestTokenUsage:
    """Test cases for TokenUsage dataclass."""
    
    def test_usage_available(self):
        """Test usage is available when tokens > 0."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.is_available is True
    
    def test_usage_not_available(self):
        """Test usage is not available when tokens = 0."""
        usage = TokenUsage()
        assert usage.is_available is False


# =============================================================================
# Integration-style tests (mock external APIs)
# =============================================================================

class TestProviderGenerate:
    """Test provider generate() method with mocked APIs."""
    
    @pytest.mark.asyncio
    async def test_openai_generate_mock(self):
        """Test OpenAI generate with mocked client."""
        config = LLMConfig(model="gpt-4", api_key="test-key")
        provider = OpenAIProvider(config)
        
        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Mocked response"), finish_reason="stop")]
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        
        with patch.object(provider, "_client") as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            
            messages = [Message.user("Hello")]
            response = await provider.generate(messages)
            
            assert response.content == "Mocked response"
            assert response.model == "gpt-4"

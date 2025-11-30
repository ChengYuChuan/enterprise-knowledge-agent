"""
LLM Provider Factory

This module implements the Factory Pattern for creating LLM providers.
It provides a registry-based approach that allows:
    1. Creating providers by name (string-driven)
    2. Dynamic registration of new providers
    3. Configuration-based provider instantiation

Design Pattern: Factory + Registry

Usage:
    ```python
    from src.llm import get_provider, LLMConfig
    
    # Simple usage - create by name
    config = LLMConfig(model="gpt-4")
    provider = get_provider("openai", config)
    
    # From environment/config file
    provider = get_provider_from_env()
    
    # List available providers
    providers = list_providers()
    print(providers)  # ['openai', 'anthropic', 'ollama']
    ```

Extending with Custom Providers:
    ```python
    from src.llm import register_provider, BaseLLMProvider
    
    class MyCustomProvider(BaseLLMProvider):
        # ... implementation
    
    register_provider("my_custom", MyCustomProvider)
    
    # Now it's available
    provider = get_provider("my_custom", config)
    ```
"""

import os
from typing import Type, Optional, Any
import logging

from src.llm.base import BaseLLMProvider, LLMConfig, ProviderType
from src.llm.exceptions import LLMProviderNotFoundError, LLMAuthenticationError

# Lazy imports to avoid circular dependencies and make dependencies optional
# The actual provider classes are imported when first needed

logger = logging.getLogger(__name__)


# =============================================================================
# Provider Registry
# =============================================================================

class ProviderRegistry:
    """Registry for LLM provider classes.
    
    This class maintains a mapping from provider names to their
    implementation classes. It supports:
        - Built-in providers (openai, anthropic, ollama)
        - Custom provider registration
        - Alias support (e.g., "gpt" -> "openai")
    
    Thread Safety:
        The registry is not thread-safe for writes. Register all
        custom providers at application startup before concurrent access.
    
    Attributes:
        _providers: Dict mapping names to provider classes.
        _aliases: Dict mapping alias names to canonical names.
    """
    
    def __init__(self):
        self._providers: dict[str, Type[BaseLLMProvider]] = {}
        self._aliases: dict[str, str] = {}
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Lazy initialization of built-in providers.
        
        We defer imports until first use to:
        1. Avoid circular imports
        2. Make provider dependencies optional
        3. Speed up initial module load
        """
        if self._initialized:
            return
        
        # Register built-in providers
        try:
            from src.llm.providers.openai_provider import OpenAIProvider
            self._providers["openai"] = OpenAIProvider
            self._aliases["gpt"] = "openai"
            self._aliases["chatgpt"] = "openai"
        except ImportError as e:
            logger.debug(f"OpenAI provider not available: {e}")
        
        try:
            from src.llm.providers.anthropic_provider import AnthropicProvider
            self._providers["anthropic"] = AnthropicProvider
            self._aliases["claude"] = "anthropic"
        except ImportError as e:
            logger.debug(f"Anthropic provider not available: {e}")
        
        try:
            from src.llm.providers.ollama_provider import OllamaProvider
            self._providers["ollama"] = OllamaProvider
            self._aliases["local"] = "ollama"
        except ImportError as e:
            logger.debug(f"Ollama provider not available: {e}")
        
        self._initialized = True
    
    def register(
        self,
        name: str,
        provider_class: Type[BaseLLMProvider],
        aliases: Optional[list[str]] = None
    ) -> None:
        """Register a provider class.
        
        Args:
            name: Canonical name for the provider (lowercase).
            provider_class: The provider class (must inherit BaseLLMProvider).
            aliases: Optional list of alternative names.
        
        Raises:
            TypeError: If provider_class doesn't inherit from BaseLLMProvider.
            ValueError: If name is already registered.
        
        Example:
            ```python
            registry.register("my_provider", MyProvider, aliases=["mp", "custom"])
            ```
        """
        # Validate
        if not isinstance(provider_class, type):
            raise TypeError(f"Expected a class, got {type(provider_class)}")
        
        if not issubclass(provider_class, BaseLLMProvider):
            raise TypeError(
                f"{provider_class.__name__} must inherit from BaseLLMProvider"
            )
        
        name = name.lower()
        
        if name in self._providers:
            logger.warning(f"Overwriting existing provider: {name}")
        
        self._providers[name] = provider_class
        logger.info(f"Registered LLM provider: {name}")
        
        # Register aliases
        if aliases:
            for alias in aliases:
                alias = alias.lower()
                self._aliases[alias] = name
                logger.debug(f"Registered alias: {alias} -> {name}")
    
    def unregister(self, name: str) -> bool:
        """Unregister a provider.
        
        Args:
            name: Provider name to remove.
        
        Returns:
            True if provider was removed, False if it didn't exist.
        """
        name = name.lower()
        
        # Remove aliases pointing to this provider
        aliases_to_remove = [
            alias for alias, target in self._aliases.items()
            if target == name
        ]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        if name in self._providers:
            del self._providers[name]
            return True
        return False
    
    def get(self, name: str) -> Type[BaseLLMProvider]:
        """Get a provider class by name.
        
        Args:
            name: Provider name or alias.
        
        Returns:
            The provider class.
        
        Raises:
            LLMProviderNotFoundError: If provider is not registered.
        """
        self._ensure_initialized()
        
        name = name.lower()
        
        # Check aliases first
        if name in self._aliases:
            name = self._aliases[name]
        
        if name not in self._providers:
            raise LLMProviderNotFoundError(
                f"Unknown provider: '{name}'",
                provider=name,
                available_providers=list(self._providers.keys())
            )
        
        return self._providers[name]
    
    def list_providers(self) -> list[str]:
        """List all registered provider names.
        
        Returns:
            List of canonical provider names (not aliases).
        """
        self._ensure_initialized()
        return list(self._providers.keys())
    
    def list_aliases(self) -> dict[str, str]:
        """List all registered aliases.
        
        Returns:
            Dict mapping aliases to canonical names.
        """
        self._ensure_initialized()
        return dict(self._aliases)
    
    def is_available(self, name: str) -> bool:
        """Check if a provider is available.
        
        Args:
            name: Provider name or alias.
        
        Returns:
            True if provider is registered.
        """
        self._ensure_initialized()
        name = name.lower()
        
        if name in self._aliases:
            name = self._aliases[name]
        
        return name in self._providers


# Global registry instance
_registry = ProviderRegistry()


# =============================================================================
# Public Factory Functions
# =============================================================================

def get_provider(
    provider_name: str,
    config: LLMConfig,
) -> BaseLLMProvider:
    """Create an LLM provider by name.
    
    This is the primary factory function for creating providers.
    
    Args:
        provider_name: Name of the provider ("openai", "anthropic", "ollama")
            or an alias ("gpt", "claude", "local").
        config: Configuration for the provider.
    
    Returns:
        An instance of the requested provider.
    
    Raises:
        LLMProviderNotFoundError: If provider is not registered.
        LLMAuthenticationError: If required credentials are missing.
    
    Example:
        ```python
        config = LLMConfig(model="gpt-4", temperature=0.7)
        provider = get_provider("openai", config)
        
        # Using alias
        provider = get_provider("claude", LLMConfig(model="claude-3-sonnet-20240229"))
        ```
    """
    provider_class = _registry.get(provider_name)
    return provider_class(config)


def get_provider_from_env(
    provider_name: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """Create a provider using environment variables for configuration.
    
    This is convenient for deployment where configuration comes from
    environment variables rather than code.
    
    Environment Variables:
        LLM_PROVIDER: Provider name (default: "openai")
        LLM_MODEL: Model name (default: provider-specific)
        LLM_TEMPERATURE: Temperature (default: 0.7)
        LLM_MAX_TOKENS: Max tokens (default: 2048)
        OPENAI_API_KEY: OpenAI API key
        ANTHROPIC_API_KEY: Anthropic API key
    
    Args:
        provider_name: Override LLM_PROVIDER env var.
        model: Override LLM_MODEL env var.
        **kwargs: Additional config overrides.
    
    Returns:
        Configured provider instance.
    
    Example:
        ```bash
        export LLM_PROVIDER=openai
        export LLM_MODEL=gpt-4
        export OPENAI_API_KEY=sk-...
        ```
        
        ```python
        provider = get_provider_from_env()
        # Equivalent to: get_provider("openai", LLMConfig(model="gpt-4"))
        ```
    """
    # Resolve provider
    provider_name = provider_name or os.getenv("LLM_PROVIDER", "openai")
    
    # Determine default model based on provider
    default_models = {
        "openai": "gpt-4",
        "anthropic": "claude-3-sonnet-20240229",
        "ollama": "llama3",
    }
    
    # Resolve model
    resolved_name = provider_name.lower()
    if resolved_name in _registry._aliases:
        resolved_name = _registry._aliases[resolved_name]
    
    default_model = default_models.get(resolved_name, "gpt-4")
    model = model or os.getenv("LLM_MODEL", default_model)
    
    # Build config from environment
    config = LLMConfig(
        model=model,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        top_p=float(os.getenv("LLM_TOP_P", "1.0")),
        timeout=float(os.getenv("LLM_TIMEOUT", "30.0")),
        # API keys are handled by individual providers
        **kwargs
    )
    
    return get_provider(provider_name, config)


def list_providers() -> list[str]:
    """List all available provider names.
    
    Returns:
        List of registered provider names.
    
    Example:
        ```python
        providers = list_providers()
        print(providers)  # ['openai', 'anthropic', 'ollama']
        ```
    """
    return _registry.list_providers()


def is_provider_available(name: str) -> bool:
    """Check if a provider is available.
    
    Args:
        name: Provider name or alias.
    
    Returns:
        True if provider is registered and available.
    """
    return _registry.is_available(name)


def register_provider(
    name: str,
    provider_class: Type[BaseLLMProvider],
    aliases: Optional[list[str]] = None
) -> None:
    """Register a custom provider.
    
    Use this to add your own LLM provider implementations.
    
    Args:
        name: Canonical name for the provider.
        provider_class: Class inheriting from BaseLLMProvider.
        aliases: Optional alternative names.
    
    Example:
        ```python
        from src.llm import register_provider, BaseLLMProvider
        
        class MyCloudProvider(BaseLLMProvider):
            async def generate(self, messages, **kwargs):
                # Custom implementation
                pass
            
            async def stream(self, messages, **kwargs):
                # Custom implementation
                pass
            
            async def embed(self, texts, **kwargs):
                # Custom implementation
                pass
        
        register_provider("mycloud", MyCloudProvider, aliases=["mc"])
        
        # Now available via factory
        provider = get_provider("mycloud", config)
        provider = get_provider("mc", config)  # Alias works too
        ```
    """
    _registry.register(name, provider_class, aliases)


def unregister_provider(name: str) -> bool:
    """Unregister a provider.
    
    Args:
        name: Provider name to remove.
    
    Returns:
        True if removed, False if didn't exist.
    """
    return _registry.unregister(name)


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config_from_yaml(config_path: str) -> dict[str, Any]:
    """Load LLM configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        Configuration dictionary.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If YAML is invalid.
    
    Example YAML:
        ```yaml
        provider: openai
        model: gpt-4
        temperature: 0.7
        max_tokens: 2048
        ```
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML config. Run: pip install pyyaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: expected dict, got {type(config)}")
    
    return config


def get_provider_from_yaml(config_path: str) -> BaseLLMProvider:
    """Create a provider from a YAML configuration file.
    
    Args:
        config_path: Path to YAML config file.
    
    Returns:
        Configured provider instance.
    
    Example:
        ```python
        provider = get_provider_from_yaml("configs/llm_configs/openai.yaml")
        ```
    
    YAML Format:
        ```yaml
        provider: openai
        model: gpt-4
        temperature: 0.7
        max_tokens: 2048
        api_key: ${OPENAI_API_KEY}  # Supports env var substitution
        extra_params:
          response_format:
            type: json_object
        ```
    """
    config_dict = load_config_from_yaml(config_path)
    
    # Extract provider name
    provider_name = config_dict.pop("provider", "openai")
    
    # Handle environment variable substitution in strings
    config_dict = _substitute_env_vars(config_dict)
    
    # Extract extra_params if present
    extra_params = config_dict.pop("extra_params", {})
    
    # Build LLMConfig
    config = LLMConfig(
        model=config_dict.get("model", "gpt-4"),
        temperature=config_dict.get("temperature", 0.7),
        max_tokens=config_dict.get("max_tokens", 2048),
        top_p=config_dict.get("top_p", 1.0),
        timeout=config_dict.get("timeout", 30.0),
        api_key=config_dict.get("api_key"),
        base_url=config_dict.get("base_url"),
        extra_params=extra_params,
    )
    
    return get_provider(provider_name, config)


def _substitute_env_vars(config: Any) -> Any:
    """Recursively substitute ${VAR} patterns with environment variables.
    
    Args:
        config: Configuration value (dict, list, or scalar).
    
    Returns:
        Config with environment variables substituted.
    
    Example:
        ${OPENAI_API_KEY} -> actual value from environment
        ${VAR:-default} -> VAR value or "default" if not set
    """
    import re
    
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    
    if isinstance(config, list):
        return [_substitute_env_vars(v) for v in config]
    
    if isinstance(config, str):
        # Pattern: ${VAR} or ${VAR:-default}
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            value = os.getenv(var_name)
            
            if value is None:
                if default is not None:
                    return default
                # Return original pattern if var not found and no default
                return match.group(0)
            return value
        
        return re.sub(pattern, replacer, config)
    
    return config
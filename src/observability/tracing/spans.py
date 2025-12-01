"""
Span Builder Utilities

This module provides builder classes for creating spans with pre-configured
attributes for common RAG pipeline operations. These builders enforce
consistent attribute naming and make it easier to add observability.

Design:
    Each builder follows the Builder pattern, allowing fluent configuration
    of span attributes before creating the span.

Usage:
    ```python
    from src.observability.tracing.spans import LLMSpanBuilder

    # Build and execute with context manager
    builder = LLMSpanBuilder(tracer)
        .with_provider("openai")
        .with_model("gpt-4")
        .with_temperature(0.7)
    
    with builder.build("chat_completion") as span:
        response = await llm.generate(prompt)
        span.set_completion(response.text)
        span.set_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
    ```

Semantic Conventions:
    This module follows OpenInference semantic conventions for LLM tracing,
    which are compatible with Arize Phoenix and other observability platforms.
    
    Reference: https://github.com/Arize-ai/openinference
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

from src.observability.base import SpanKind, SpanStatus

if TYPE_CHECKING:
    from src.observability.tracing.phoenix import PhoenixSpan, PhoenixTracer


# =============================================================================
# OpenInference Semantic Attribute Keys
# =============================================================================

class SemanticAttributes:
    """
    OpenInference semantic attribute keys.
    
    These follow the OpenInference specification for LLM observability.
    """
    
    # LLM attributes
    LLM_PROVIDER = "llm.provider"
    LLM_MODEL_NAME = "llm.model_name"
    LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
    LLM_TEMPERATURE = "llm.invocation_parameters.temperature"
    LLM_MAX_TOKENS = "llm.invocation_parameters.max_tokens"
    LLM_TOP_P = "llm.invocation_parameters.top_p"
    LLM_STOP_SEQUENCES = "llm.invocation_parameters.stop_sequences"
    
    # Token counts
    LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
    LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
    LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
    
    # Input/Output
    INPUT_VALUE = "input.value"
    INPUT_MIME_TYPE = "input.mime_type"
    OUTPUT_VALUE = "output.value"
    OUTPUT_MIME_TYPE = "output.mime_type"
    
    # Messages (for chat completions)
    LLM_INPUT_MESSAGES = "llm.input_messages"
    LLM_OUTPUT_MESSAGES = "llm.output_messages"
    MESSAGE_ROLE = "message.role"
    MESSAGE_CONTENT = "message.content"
    MESSAGE_FUNCTION_CALL_NAME = "message.function_call_name"
    MESSAGE_FUNCTION_CALL_ARGUMENTS = "message.function_call_arguments"
    MESSAGE_TOOL_CALLS = "message.tool_calls"
    
    # Retrieval attributes
    RETRIEVAL_DOCUMENTS = "retrieval.documents"
    DOCUMENT_ID = "document.id"
    DOCUMENT_CONTENT = "document.content"
    DOCUMENT_SCORE = "document.score"
    DOCUMENT_METADATA = "document.metadata"
    
    # Embedding attributes
    EMBEDDING_MODEL_NAME = "embedding.model_name"
    EMBEDDING_EMBEDDINGS = "embedding.embeddings"
    EMBEDDING_TEXT = "embedding.text"
    EMBEDDING_VECTOR = "embedding.vector"
    
    # Reranking attributes
    RERANKER_MODEL_NAME = "reranker.model_name"
    RERANKER_TOP_K = "reranker.top_k"
    RERANKER_INPUT_DOCUMENTS = "reranker.input_documents"
    RERANKER_OUTPUT_DOCUMENTS = "reranker.output_documents"
    
    # Tool attributes
    TOOL_NAME = "tool.name"
    TOOL_DESCRIPTION = "tool.description"
    TOOL_PARAMETERS = "tool.parameters"
    
    # Error attributes
    EXCEPTION_TYPE = "exception.type"
    EXCEPTION_MESSAGE = "exception.message"
    EXCEPTION_STACKTRACE = "exception.stacktrace"


# =============================================================================
# Data Classes for Structured Attributes
# =============================================================================

@dataclass
class Message:
    """Represents a chat message."""
    
    role: str
    content: str
    function_call_name: Optional[str] = None
    function_call_arguments: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "role": self.role,
            "content": self.content,
        }
        if self.function_call_name:
            result["function_call_name"] = self.function_call_name
        if self.function_call_arguments:
            result["function_call_arguments"] = self.function_call_arguments
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result


@dataclass
class Document:
    """Represents a retrieved document."""
    
    id: str
    content: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "content": self.content,
        }
        if self.score is not None:
            result["score"] = self.score
        if self.metadata:
            result["metadata"] = self.metadata
        return result


# =============================================================================
# Span Builders
# =============================================================================

class BaseSpanBuilder:
    """
    Base class for span builders.
    
    Provides common functionality for all span builders.
    """
    
    def __init__(self, tracer: "PhoenixTracer"):
        """
        Initialize the builder.
        
        Args:
            tracer: The tracer to use for creating spans.
        """
        self._tracer = tracer
        self._attributes: Dict[str, Any] = {}
        self._kind = SpanKind.INTERNAL
    
    def with_attribute(self, key: str, value: Any) -> "BaseSpanBuilder":
        """
        Add a custom attribute.
        
        Args:
            key: Attribute key.
            value: Attribute value.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[key] = value
        return self
    
    @contextmanager
    def build(self, name: str) -> Generator["PhoenixSpan", None, None]:
        """
        Build and yield the span.
        
        Args:
            name: Span name.
            
        Yields:
            The created span.
        """
        with self._tracer.span(name, kind=self._kind, attributes=self._attributes) as span:
            yield span


class LLMSpanBuilder(BaseSpanBuilder):
    """
    Builder for LLM call spans.
    
    Creates spans with attributes following OpenInference LLM conventions.
    
    Example:
        ```python
        builder = LLMSpanBuilder(tracer)
            .with_provider("openai")
            .with_model("gpt-4")
            .with_temperature(0.7)
            .with_max_tokens(1000)
            .with_input_messages([
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hello!"),
            ])
        
        with builder.build("chat_completion") as span:
            response = await client.chat.completions.create(...)
            
            # Record output
            span.set_attribute(SemanticAttributes.OUTPUT_VALUE, response.choices[0].message.content)
            span.set_attribute(SemanticAttributes.LLM_TOKEN_COUNT_PROMPT, response.usage.prompt_tokens)
            span.set_attribute(SemanticAttributes.LLM_TOKEN_COUNT_COMPLETION, response.usage.completion_tokens)
        ```
    """
    
    def __init__(self, tracer: "PhoenixTracer"):
        """Initialize the LLM span builder."""
        super().__init__(tracer)
        self._kind = SpanKind.LLM
    
    def with_provider(self, provider: str) -> "LLMSpanBuilder":
        """
        Set the LLM provider.
        
        Args:
            provider: Provider name (e.g., "openai", "anthropic").
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.LLM_PROVIDER] = provider
        return self
    
    def with_model(self, model: str) -> "LLMSpanBuilder":
        """
        Set the model name.
        
        Args:
            model: Model name (e.g., "gpt-4", "claude-3-opus").
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.LLM_MODEL_NAME] = model
        return self
    
    def with_temperature(self, temperature: float) -> "LLMSpanBuilder":
        """
        Set the temperature parameter.
        
        Args:
            temperature: Temperature value (0.0 - 2.0).
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.LLM_TEMPERATURE] = temperature
        return self
    
    def with_max_tokens(self, max_tokens: int) -> "LLMSpanBuilder":
        """
        Set the max tokens parameter.
        
        Args:
            max_tokens: Maximum tokens to generate.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.LLM_MAX_TOKENS] = max_tokens
        return self
    
    def with_top_p(self, top_p: float) -> "LLMSpanBuilder":
        """
        Set the top_p parameter.
        
        Args:
            top_p: Top-p sampling value.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.LLM_TOP_P] = top_p
        return self
    
    def with_input(self, value: str, mime_type: str = "text/plain") -> "LLMSpanBuilder":
        """
        Set the input value.
        
        Args:
            value: Input text/prompt.
            mime_type: MIME type of the input.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.INPUT_VALUE] = value
        self._attributes[SemanticAttributes.INPUT_MIME_TYPE] = mime_type
        return self
    
    def with_input_messages(self, messages: List[Message]) -> "LLMSpanBuilder":
        """
        Set the input messages for chat completions.
        
        Args:
            messages: List of Message objects.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.LLM_INPUT_MESSAGES] = [
            m.to_dict() for m in messages
        ]
        return self


class RetrievalSpanBuilder(BaseSpanBuilder):
    """
    Builder for retrieval operation spans.
    
    Creates spans with attributes for tracking document retrieval.
    
    Example:
        ```python
        builder = RetrievalSpanBuilder(tracer)
            .with_query("What is machine learning?")
            .with_top_k(10)
            .with_strategy("hybrid")
        
        with builder.build("hybrid_search") as span:
            results = await retriever.search(query)
            
            # Record retrieved documents
            documents = [
                Document(id=r.id, content=r.content, score=r.score)
                for r in results
            ]
            span.set_attribute(SemanticAttributes.RETRIEVAL_DOCUMENTS, 
                             [d.to_dict() for d in documents])
        ```
    """
    
    def __init__(self, tracer: "PhoenixTracer"):
        """Initialize the retrieval span builder."""
        super().__init__(tracer)
        self._kind = SpanKind.RETRIEVER
    
    def with_query(self, query: str) -> "RetrievalSpanBuilder":
        """
        Set the search query.
        
        Args:
            query: Search query text.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.INPUT_VALUE] = query
        return self
    
    def with_top_k(self, top_k: int) -> "RetrievalSpanBuilder":
        """
        Set the number of results to retrieve.
        
        Args:
            top_k: Number of top results.
            
        Returns:
            Self for method chaining.
        """
        self._attributes["retrieval.top_k"] = top_k
        return self
    
    def with_strategy(self, strategy: str) -> "RetrievalSpanBuilder":
        """
        Set the retrieval strategy.
        
        Args:
            strategy: Strategy name (e.g., "vector", "bm25", "hybrid").
            
        Returns:
            Self for method chaining.
        """
        self._attributes["retrieval.strategy"] = strategy
        return self
    
    def with_filters(self, filters: Dict[str, Any]) -> "RetrievalSpanBuilder":
        """
        Set metadata filters.
        
        Args:
            filters: Filter criteria.
            
        Returns:
            Self for method chaining.
        """
        self._attributes["retrieval.filters"] = filters
        return self
    
    def with_documents(self, documents: List[Document]) -> "RetrievalSpanBuilder":
        """
        Set the retrieved documents.
        
        Args:
            documents: List of Document objects.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.RETRIEVAL_DOCUMENTS] = [
            d.to_dict() for d in documents
        ]
        return self


class EmbeddingSpanBuilder(BaseSpanBuilder):
    """
    Builder for embedding generation spans.
    
    Creates spans for tracking embedding operations.
    
    Example:
        ```python
        builder = EmbeddingSpanBuilder(tracer)
            .with_model("text-embedding-3-small")
            .with_texts(["Hello", "World"])
        
        with builder.build("generate_embeddings") as span:
            embeddings = await embedding_model.embed(texts)
            span.set_attribute("embedding.dimensions", len(embeddings[0]))
        ```
    """
    
    def __init__(self, tracer: "PhoenixTracer"):
        """Initialize the embedding span builder."""
        super().__init__(tracer)
        self._kind = SpanKind.EMBEDDING
    
    def with_model(self, model: str) -> "EmbeddingSpanBuilder":
        """
        Set the embedding model name.
        
        Args:
            model: Model name.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.EMBEDDING_MODEL_NAME] = model
        return self
    
    def with_texts(self, texts: List[str]) -> "EmbeddingSpanBuilder":
        """
        Set the texts to embed.
        
        Args:
            texts: List of texts.
            
        Returns:
            Self for method chaining.
        """
        self._attributes["embedding.num_texts"] = len(texts)
        # Store first few texts for debugging (truncated)
        self._attributes["embedding.sample_texts"] = [
            t[:100] + "..." if len(t) > 100 else t
            for t in texts[:3]
        ]
        return self
    
    def with_dimensions(self, dimensions: int) -> "EmbeddingSpanBuilder":
        """
        Set the embedding dimensions.
        
        Args:
            dimensions: Vector dimensions.
            
        Returns:
            Self for method chaining.
        """
        self._attributes["embedding.dimensions"] = dimensions
        return self


class RerankerSpanBuilder(BaseSpanBuilder):
    """
    Builder for reranking operation spans.
    
    Creates spans for tracking reranking operations.
    
    Example:
        ```python
        builder = RerankerSpanBuilder(tracer)
            .with_model("cross-encoder/ms-marco-MiniLM-L-6-v2")
            .with_query("machine learning")
            .with_num_documents(10)
            .with_top_k(5)
        
        with builder.build("rerank") as span:
            reranked = await reranker.rerank(query, documents)
            span.set_attribute("reranker.output_count", len(reranked))
        ```
    """
    
    def __init__(self, tracer: "PhoenixTracer"):
        """Initialize the reranker span builder."""
        super().__init__(tracer)
        self._kind = SpanKind.RERANKER
    
    def with_model(self, model: str) -> "RerankerSpanBuilder":
        """
        Set the reranker model name.
        
        Args:
            model: Model name.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.RERANKER_MODEL_NAME] = model
        return self
    
    def with_query(self, query: str) -> "RerankerSpanBuilder":
        """
        Set the query for reranking.
        
        Args:
            query: Query text.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.INPUT_VALUE] = query
        return self
    
    def with_num_documents(self, count: int) -> "RerankerSpanBuilder":
        """
        Set the number of input documents.
        
        Args:
            count: Number of documents.
            
        Returns:
            Self for method chaining.
        """
        self._attributes["reranker.input_count"] = count
        return self
    
    def with_top_k(self, top_k: int) -> "RerankerSpanBuilder":
        """
        Set the number of results to return.
        
        Args:
            top_k: Number of top results.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.RERANKER_TOP_K] = top_k
        return self


class ToolSpanBuilder(BaseSpanBuilder):
    """
    Builder for tool execution spans.
    
    Creates spans for tracking tool/function calls.
    
    Example:
        ```python
        builder = ToolSpanBuilder(tracer)
            .with_name("search_documents")
            .with_parameters({"query": "test", "top_k": 5})
        
        with builder.build("tool_execution") as span:
            result = await tool.execute(**params)
            span.set_attribute(SemanticAttributes.OUTPUT_VALUE, str(result))
        ```
    """
    
    def __init__(self, tracer: "PhoenixTracer"):
        """Initialize the tool span builder."""
        super().__init__(tracer)
        self._kind = SpanKind.TOOL
    
    def with_name(self, name: str) -> "ToolSpanBuilder":
        """
        Set the tool name.
        
        Args:
            name: Tool name.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.TOOL_NAME] = name
        return self
    
    def with_description(self, description: str) -> "ToolSpanBuilder":
        """
        Set the tool description.
        
        Args:
            description: Tool description.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.TOOL_DESCRIPTION] = description
        return self
    
    def with_parameters(self, parameters: Dict[str, Any]) -> "ToolSpanBuilder":
        """
        Set the tool parameters.
        
        Args:
            parameters: Tool input parameters.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.TOOL_PARAMETERS] = parameters
        self._attributes[SemanticAttributes.INPUT_VALUE] = str(parameters)
        return self


class AgentSpanBuilder(BaseSpanBuilder):
    """
    Builder for agent operation spans.
    
    Creates spans for tracking agent reasoning and actions.
    
    Example:
        ```python
        builder = AgentSpanBuilder(tracer)
            .with_input("What is machine learning?")
            .with_max_iterations(5)
        
        with builder.build("agent_run") as span:
            response = await agent.run(query)
            span.set_attribute("agent.iterations", response.iterations)
            span.set_attribute("agent.tools_used", response.tools_used)
        ```
    """
    
    def __init__(self, tracer: "PhoenixTracer"):
        """Initialize the agent span builder."""
        super().__init__(tracer)
        self._kind = SpanKind.AGENT
    
    def with_input(self, input_text: str) -> "AgentSpanBuilder":
        """
        Set the agent input.
        
        Args:
            input_text: Input query or instruction.
            
        Returns:
            Self for method chaining.
        """
        self._attributes[SemanticAttributes.INPUT_VALUE] = input_text
        return self
    
    def with_max_iterations(self, max_iterations: int) -> "AgentSpanBuilder":
        """
        Set the maximum iterations.
        
        Args:
            max_iterations: Maximum number of reasoning iterations.
            
        Returns:
            Self for method chaining.
        """
        self._attributes["agent.max_iterations"] = max_iterations
        return self
    
    def with_conversation_id(self, conversation_id: str) -> "AgentSpanBuilder":
        """
        Set the conversation ID.
        
        Args:
            conversation_id: Conversation/session identifier.
            
        Returns:
            Self for method chaining.
        """
        self._attributes["conversation.id"] = conversation_id
        return self
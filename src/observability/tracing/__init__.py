"""
Tracing Submodule

This module provides distributed tracing capabilities with Arize Phoenix
integration for LLM observability.

Components:
    - PhoenixTracer: Arize Phoenix implementation of the Tracer interface
    - Span definitions: Pre-defined spans for common RAG operations

Usage:
    ```python
    from src.observability.tracing import PhoenixTracer

    tracer = PhoenixTracer(endpoint="http://localhost:6006")

    with tracer.span("llm_call", kind=SpanKind.LLM) as span:
        span.set_attribute("llm.model", "gpt-4")
        # ... make LLM call ...
    ```
"""

from src.observability.tracing.phoenix import PhoenixTracer, PhoenixSpan
from src.observability.tracing.spans import (
    LLMSpanBuilder,
    RetrievalSpanBuilder,
    EmbeddingSpanBuilder,
    RerankerSpanBuilder,
)

__all__ = [
    "PhoenixTracer",
    "PhoenixSpan",
    "LLMSpanBuilder",
    "RetrievalSpanBuilder",
    "EmbeddingSpanBuilder",
    "RerankerSpanBuilder",
]
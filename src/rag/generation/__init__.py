"""
RAG generation module.

This module handles response synthesis and citation generation.
"""

from .synthesizer import ResponseSynthesizer, SynthesizedResponse

__all__ = [
    "ResponseSynthesizer",
    "SynthesizedResponse",
]
"""
Response synthesizer for generating answers with citations.

This module provides functionality to synthesize final responses
from retrieved contexts, including automatic citation generation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SynthesizedResponse:
    """
    Response with answer, sources, and metadata.
    
    Attributes:
        answer: Generated answer text.
        sources: List of source chunks used.
        confidence: Confidence score (0-1).
        num_sources: Number of sources cited.
        has_answer: Whether a valid answer was generated.
    """
    
    answer: str
    sources: list[dict]
    confidence: float = 1.0
    num_sources: int = 0
    has_answer: bool = True
    
    def __post_init__(self) -> None:
        """Calculate derived fields."""
        self.num_sources = len(self.sources)


class ResponseSynthesizer:
    """
    Synthesizer for generating responses from retrieved contexts.
    
    In this phase (without LLM), we provide:
    1. Context aggregation
    2. Automatic citation extraction
    3. Relevance-based answer confidence
    
    In Phase 4, this will be extended with actual LLM generation.
    
    Attributes:
        min_confidence: Minimum confidence threshold for returning answer.
        max_sources: Maximum number of sources to cite.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        max_sources: int = 5
    ) -> None:
        """
        Initialize the response synthesizer.
        
        Args:
            min_confidence: Minimum confidence threshold (0-1).
            max_sources: Maximum number of sources to include.
        """
        self.min_confidence = min_confidence
        self.max_sources = max_sources
    
    def synthesize(
        self,
        query: str,
        search_results: list[dict]
    ) -> SynthesizedResponse:
        """
        Synthesize a response from search results.
        
        Currently returns formatted context with citations.
        Will be replaced with LLM generation in Phase 4.
        
        Args:
            query: Original user query.
            search_results: List of retrieved chunks with scores.
            
        Returns:
            SynthesizedResponse: Synthesized answer with metadata.
        """
        if not search_results:
            return SynthesizedResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                confidence=0.0,
                has_answer=False
            )
        
        # Calculate average relevance as confidence
        avg_score = sum(r.get("score", 0) for r in search_results) / len(search_results)
        confidence = min(avg_score, 1.0)  # Normalize to 0-1
        
        # Check if confidence is too low
        if confidence < self.min_confidence:
            return SynthesizedResponse(
                answer=(
                    "I found some potentially relevant information, but I'm not "
                    "confident it fully answers your question. Please rephrase or "
                    "provide more details."
                ),
                sources=search_results[:self.max_sources],
                confidence=confidence,
                has_answer=False
            )
        
        # Generate response with citations
        answer = self._format_context_with_citations(
            query=query,
            results=search_results[:self.max_sources]
        )
        
        return SynthesizedResponse(
            answer=answer,
            sources=search_results[:self.max_sources],
            confidence=confidence,
            has_answer=True
        )
    
    def _format_context_with_citations(
        self,
        query: str,
        results: list[dict]
    ) -> str:
        """
        Format retrieved contexts with citation markers.
        
        Args:
            query: Original query (for context).
            results: Search results to format.
            
        Returns:
            str: Formatted response with citations.
        """
        response_parts = [
            f"Based on the available information regarding '{query}':\n"
        ]
        
        # Add each result as a cited paragraph
        for i, result in enumerate(results, start=1):
            text = result["text"].strip()
            metadata = result.get("metadata", {})
            
            # Get source information
            source_name = metadata.get("filename", "Unknown Source")
            
            # Format citation
            citation = f"[{i}]"
            
            response_parts.append(
                f"\n{citation} {text}\n"
                f"   Source: {source_name}"
            )
        
        # Add references section
        response_parts.append("\n\nReferences:")
        for i, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source_name = metadata.get("filename", "Unknown")
            file_type = metadata.get("file_type", "")
            
            response_parts.append(f"[{i}] {source_name} {file_type}")
        
        return "\n".join(response_parts)
    
    def extract_citations(self, response: SynthesizedResponse) -> list[dict]:
        """
        Extract structured citation information from response.
        
        Args:
            response: Synthesized response with sources.
            
        Returns:
            list[dict]: List of citation dictionaries.
        """
        citations = []
        
        for i, source in enumerate(response.sources, start=1):
            metadata = source.get("metadata", {})
            
            citation = {
                "id": i,
                "text": source["text"][:200] + "..." if len(source["text"]) > 200 else source["text"],
                "source": metadata.get("filename", "Unknown"),
                "file_type": metadata.get("file_type", ""),
                "score": source.get("score", 0.0),
            }
            
            citations.append(citation)
        
        return citations
    
    def format_for_display(self, response: SynthesizedResponse) -> str:
        """
        Format response for clean display (without citations).
        
        Args:
            response: Synthesized response.
            
        Returns:
            str: Formatted response text.
        """
        if not response.has_answer:
            return response.answer
        
        # Extract main content (remove citation markers and source info)
        lines = response.answer.split("\n")
        content_lines = [
            line for line in lines 
            if not line.strip().startswith("Source:") 
            and not line.strip().startswith("References:")
            and not line.strip().startswith("[")
        ]
        
        return "\n".join(content_lines).strip()
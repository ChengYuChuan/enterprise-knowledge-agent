"""
Cross-encoder reranker implementation.

This module provides reranking functionality using cross-encoder models
to improve retrieval quality by reordering initial search results.
"""

from typing import Optional

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder based reranker for improving search results.
    
    Cross-encoders jointly encode the query and document, producing
    a relevance score. This is more accurate than bi-encoders but
    also more computationally expensive.
    
    Attributes:
        model: CrossEncoder model instance.
        model_name: Name of the model being used.
    """
    
    DEFAULT_MODEL = "BAAI/bge-reranker-base"
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name (default: BAAI/bge-reranker-base).
            device: Device to run model on ('cpu', 'cuda', or None for auto).
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        
        # Initialize cross-encoder
        # Note: First time will download the model
        self.model = CrossEncoder(
            self.model_name,
            max_length=512,
            device=device
        )
    
    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: Optional[int] = None
    ) -> list[dict]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Original search query.
            results: List of search results to rerank. Each result should have
                    a 'text' field.
            top_k: Number of top results to return (default: all results).
            
        Returns:
            list[dict]: Reranked results with updated scores and ranks.
            
        Raises:
            ValueError: If results is empty or missing required fields.
        """
        if not results:
            return []
        
        # Validate result format
        if not all("text" in r for r in results):
            raise ValueError("All results must contain 'text' field")
        
        # Prepare query-document pairs
        pairs = [[query, result["text"]] for result in results]
        
        # Get reranker scores
        scores = self.model.predict(pairs)
        
        # Update results with new scores
        reranked_results = []
        for result, score in zip(results, scores):
            # Create a copy to avoid modifying original
            reranked = result.copy()
            
            # Store original score if present
            if "score" in reranked:
                reranked["original_score"] = reranked["score"]
            
            # Update with reranker score
            reranked["score"] = float(score)
            reranked_results.append(reranked)
        
        # Sort by new scores
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked_results):
            result["rank"] = i + 1
        
        # Return top-k if specified
        if top_k is not None:
            reranked_results = reranked_results[:top_k]
        
        return reranked_results
    
    def get_model_info(self) -> dict:
        """
        Get information about the reranker model.
        
        Returns:
            dict: Model information including name and device.
        """
        return {
            "model_name": self.model_name,
            "device": str(self.model.device),
            "max_length": self.model.max_length,
        }
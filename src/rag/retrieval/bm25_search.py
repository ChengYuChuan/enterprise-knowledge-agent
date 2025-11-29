"""
BM25-based keyword search implementation.

This module provides BM25 (Best Match 25) ranking function for keyword-based
document retrieval. BM25 is a probabilistic ranking function that scores
documents based on query term frequency.
"""

import re
from typing import Optional

from rank_bm25 import BM25Okapi


class BM25Search:
    """
    BM25-based keyword search for document chunks.
    
    BM25 (Best Matching 25) is a bag-of-words retrieval function that ranks
    documents based on the query terms appearing in each document, regardless
    of their proximity within the document.
    
    Attributes:
        corpus_texts: List of document texts in the corpus.
        tokenized_corpus: Tokenized version of the corpus.
        bm25: BM25Okapi instance for scoring.
        chunk_ids: IDs corresponding to each document in the corpus.
    """
    
    def __init__(self) -> None:
        """Initialize an empty BM25 search instance."""
        self.corpus_texts: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: list[str] = []
        self.metadata: list[dict] = []
    
    def index_chunks(self, chunks: list[dict]) -> None:
        """
        Index a list of chunks for BM25 search.
        
        Args:
            chunks: List of chunk dictionaries containing 'chunk_id', 'text', 
                   and 'metadata'.
        """
        self.corpus_texts = [chunk["text"] for chunk in chunks]
        self.chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.metadata = [chunk.get("metadata", {}) for chunk in chunks]
        
        # Tokenize corpus
        self.tokenized_corpus = [self._tokenize(text) for text in self.corpus_texts]
        
        # Create BM25 index
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        score_threshold: Optional[float] = None
    ) -> list[dict]:
        """
        Search for documents using BM25 scoring.
        
        Args:
            query: Search query string.
            top_k: Number of top results to return.
            score_threshold: Minimum BM25 score threshold (optional).
            
        Returns:
            list[dict]: List of search results with chunk_id, text, score, and metadata.
            
        Raises:
            ValueError: If index is empty (no chunks indexed).
        """
        if self.bm25 is None or not self.corpus_texts:
            raise ValueError("No chunks indexed. Call index_chunks() first.")
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            
            # Apply score threshold if specified
            if score_threshold is not None and score < score_threshold:
                continue
            
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.corpus_texts[idx],
                "score": score,
                "metadata": self.metadata[idx],
                "rank": len(results) + 1,
            })
        
        return results
    
    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 processing.
        
        Simple tokenization: lowercase, split on non-alphanumeric characters.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            list[str]: List of tokens.
        """
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def get_corpus_stats(self) -> dict:
        """
        Get statistics about the indexed corpus.
        
        Returns:
            dict: Statistics including document count, average length, etc.
        """
        if not self.corpus_texts:
            return {
                "document_count": 0,
                "total_tokens": 0,
                "avg_tokens_per_doc": 0,
            }
        
        total_tokens = sum(len(tokens) for tokens in self.tokenized_corpus)
        
        return {
            "document_count": len(self.corpus_texts),
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": total_tokens / len(self.corpus_texts),
        }
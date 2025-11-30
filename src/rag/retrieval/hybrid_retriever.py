"""
Hybrid retrieval combining vector search and BM25.

This module implements hybrid search that combines:
1. Vector similarity search (semantic)
2. BM25 keyword search (lexical)
3. Reciprocal Rank Fusion (RRF) for score combination
4. Optional cross-encoder reranking

FIXED: When alpha=1.0 or alpha=0.0, skip RRF and use raw scores directly.
"""

from typing import Optional

from .bm25_search import BM25Search
from .embedder import MockEmbedder
from .reranker import Reranker
from .vector_store import QdrantVectorStore


class HybridRetriever:
    """
    Hybrid retriever combining vector and BM25 search.
    
    Implements the following pipeline:
    1. Vector search (semantic similarity)
    2. BM25 search (keyword matching)
    3. Reciprocal Rank Fusion (RRF) to combine results
    4. Optional reranking with cross-encoder
    
    Special cases:
    - alpha=1.0: Pure vector search (skips RRF, uses raw cosine scores)
    - alpha=0.0: Pure BM25 search (skips RRF, uses raw BM25 scores)
    - 0 < alpha < 1: Hybrid with RRF fusion
    
    Attributes:
        vector_store: Vector store for semantic search.
        embedder: Embedder for query encoding.
        bm25_search: BM25 searcher for keyword search.
        alpha: Weight for vector search (0=pure BM25, 1=pure vector).
        use_reranking: Whether to apply reranking.
        reranker: Cross-encoder reranker (if use_reranking=True).
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedder: MockEmbedder,
        alpha: float = 0.5,
        use_reranking: bool = False,
        reranker_model: Optional[str] = None,
    ) -> None:
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store: Vector store instance.
            embedder: Embedder instance.
            alpha: Weight for vector search (0-1). 
                   0 = pure BM25, 1 = pure vector, 0.5 = balanced.
            use_reranking: Whether to use cross-encoder reranking.
            reranker_model: Model name for reranker (if use_reranking=True).
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_search = BM25Search()
        self.alpha = alpha
        self.use_reranking = use_reranking
        
        # Initialize reranker if needed
        self.reranker: Optional[Reranker] = None
        if use_reranking:
            self.reranker = Reranker(model_name=reranker_model)
    
    def index_for_bm25(self, chunks: list[dict]) -> None:
        """
        Index chunks for BM25 search.
        
        This should be called after documents are ingested into vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_id', 'text', 'metadata'.
        """
        self.bm25_search.index_chunks(chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: int = 20,
    ) -> list[dict]:
        """
        Perform hybrid search.
        
        IMPORTANT: This is a SYNCHRONOUS method (not async).
        
        Args:
            query: Search query string.
            top_k: Number of final results to return.
            retrieve_k: Number of results to retrieve from each method before fusion.
            
        Returns:
            list[dict]: Ranked search results.
        """
        # Special case: Pure vector search (alpha=1.0)
        if self.alpha == 1.0:
            return self._pure_vector_search(query, top_k)
        
        # Special case: Pure BM25 search (alpha=0.0)
        if self.alpha == 0.0:
            return self._pure_bm25_search(query, top_k)
        
        # Standard hybrid search with RRF
        return self._hybrid_search_with_rrf(query, top_k, retrieve_k)
    
    def _pure_vector_search(self, query: str, top_k: int) -> list[dict]:
        """
        Pure vector search (bypasses RRF).
        
        Returns results with original cosine similarity scores.
        """
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=None,
        )
        
        # Add rank information
        for rank, result in enumerate(results, start=1):
            result['rank'] = rank
            result['vector_rank'] = rank
            result['bm25_rank'] = None
            result['vector_score'] = result['score']
            result['bm25_score'] = 0
        
        return results
    
    def _pure_bm25_search(self, query: str, top_k: int) -> list[dict]:
        """
        Pure BM25 search (bypasses RRF).
        
        Returns results with original BM25 scores.
        """
        results = self.bm25_search.search(
            query=query,
            top_k=top_k,
        )
        
        # Add rank information
        for rank, result in enumerate(results, start=1):
            result['rank'] = rank
            result['vector_rank'] = None
            result['bm25_rank'] = rank
            result['vector_score'] = 0
            result['bm25_score'] = result['score']
        
        return results
    
    def _hybrid_search_with_rrf(
        self,
        query: str,
        top_k: int,
        retrieve_k: int,
    ) -> list[dict]:
        """
        Hybrid search with RRF fusion.
        """
        # Step 1: Vector search
        query_embedding = self.embedder.embed_text(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=retrieve_k,
            score_threshold=None,
        )
        
        # Step 2: BM25 search
        bm25_results = self.bm25_search.search(
            query=query,
            top_k=retrieve_k,
        )
        
        # Step 3: Reciprocal Rank Fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
        )
        
        # Step 4: Optional reranking
        if self.use_reranking and self.reranker is not None:
            # Rerank top results
            rerank_candidates = fused_results[:retrieve_k]
            fused_results = self.reranker.rerank(
                query=query,
                results=rerank_candidates,
                top_k=top_k,
            )
        else:
            # Just take top-k
            fused_results = fused_results[:top_k]
        
        return fused_results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF formula: RRF_score(d) = sum(1 / (k + rank_i(d)))
        
        Args:
            vector_results: Results from vector search.
            bm25_results: Results from BM25 search.
            k: Constant for RRF formula (default: 60).
            
        Returns:
            list[dict]: Fused and ranked results.
        """
        # Build chunk_id to result mapping
        all_chunks: dict[str, dict] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result["chunk_id"]
            
            # Initialize if new
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "vector_score": result.get("score", 0),
                    "bm25_score": 0,
                    "vector_rank": rank,
                    "bm25_rank": None,
                    "rrf_score": 0,
                }
            
            # Add RRF score from vector search
            all_chunks[chunk_id]["rrf_score"] += self.alpha / (k + rank)
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result["chunk_id"]
            
            # Initialize if new
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "vector_score": 0,
                    "bm25_score": result.get("score", 0),
                    "vector_rank": None,
                    "bm25_rank": rank,
                    "rrf_score": 0,
                }
            else:
                # Update BM25 info
                all_chunks[chunk_id]["bm25_score"] = result.get("score", 0)
                all_chunks[chunk_id]["bm25_rank"] = rank
            
            # Add RRF score from BM25 search
            all_chunks[chunk_id]["rrf_score"] += (1 - self.alpha) / (k + rank)
        
        # Convert to list and sort by RRF score
        fused_results = list(all_chunks.values())
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # Update final ranks and use RRF score as main score
        for rank, result in enumerate(fused_results, start=1):
            result["rank"] = rank
            result["score"] = result["rrf_score"]
        
        return fused_results
    
    def get_stats(self) -> dict:
        """
        Get statistics about the hybrid retriever.
        
        Returns:
            dict: Statistics including alpha, reranking status, etc.
        """
        stats = {
            "alpha": self.alpha,
            "use_reranking": self.use_reranking,
            "vector_store_stats": self.vector_store.get_collection_info(),
            "bm25_stats": self.bm25_search.get_corpus_stats(),
        }
        
        if self.reranker is not None:
            stats["reranker_info"] = self.reranker.get_model_info()
        
        return stats
"""
Search Routes

Endpoints for searching the knowledge base.

Endpoints:
    POST /search - Search documents
    POST /search/similar - Find similar documents
"""

import time
from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    SimilarDocumentsRequest,
)
from src.api.middleware import User, get_current_user, rate_limit
from src.api.dependencies import (
    get_settings,
    Settings,
    get_rag_pipeline,
    RAGPipeline,
)

router = APIRouter(prefix="/search", tags=["Search"])


# =============================================================================
# Routes
# =============================================================================

@router.post(
    "",
    response_model=SearchResponse,
    summary="Search the knowledge base",
    description="Search for relevant documents using semantic, keyword, or hybrid search.",
)
async def search(
    request: SearchRequest,
    user: User = Depends(get_current_user),
    rag: RAGPipeline = Depends(get_rag_pipeline),
    settings: Settings = Depends(get_settings),
    _: None = Depends(rate_limit(requests=60, window=60)),  # 60 req/min for search
) -> SearchResponse:
    """Search endpoint.
    
    Supports three search modes:
    - semantic: Vector similarity search
    - keyword: BM25 keyword search
    - hybrid: Combined vector + keyword with RRF fusion
    
    Optional reranking can be applied to improve result quality.
    """
    start_time = time.perf_counter()
    
    try:
        # Perform search
        # TODO: Implement actual search with the RAG pipeline
        # For now, return placeholder results
        
        results = await rag.search(
            request.query,
            top_k=request.top_k,
        )
        
        # Transform results to response format
        search_results = []
        for i, result in enumerate(results):
            search_results.append(SearchResult(
                document_id=result.get("document_id", f"doc_{i}"),
                chunk_id=result.get("chunk_id", f"chunk_{i}"),
                filename=result.get("filename", "unknown"),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                semantic_score=result.get("semantic_score"),
                keyword_score=result.get("keyword_score"),
                rerank_score=result.get("rerank_score"),
                metadata=result.get("metadata", {}),
                highlights=result.get("highlights"),
            ))
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return SearchResponse(
            results=search_results,
            total=len(search_results),
            query=request.query,
            mode=request.mode,
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post(
    "/similar",
    response_model=SearchResponse,
    summary="Find similar documents",
    description="Find documents similar to a given document.",
)
async def find_similar(
    request: SimilarDocumentsRequest,
    user: User = Depends(get_current_user),
    rag: RAGPipeline = Depends(get_rag_pipeline),
    settings: Settings = Depends(get_settings),
    _: None = Depends(rate_limit(requests=30, window=60)),
) -> SearchResponse:
    """Find documents similar to a given document.
    
    Uses the document's embedding to find semantically similar content.
    """
    start_time = time.perf_counter()
    
    try:
        # TODO: Implement similar document search
        # For now, return empty results
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return SearchResponse(
            results=[],
            total=0,
            query=f"similar to {request.document_id}",
            mode="semantic",
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Similar search failed: {str(e)}"
        )
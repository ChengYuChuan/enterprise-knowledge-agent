"""
Knowledge base query tool.

This tool allows the agent to search the enterprise knowledge base
and retrieve relevant information.
"""

from typing import Any, Optional

from src.config import get_settings
from src.rag.generation import ResponseSynthesizer
from src.rag.retrieval import HybridRetriever, get_embedder, QdrantVectorStore

from .base import BaseTool, ToolCategory, ToolParameter, ToolResult


class QueryKnowledgeBaseTool(BaseTool):
    """
    Tool for querying the enterprise knowledge base.
    
    Uses the hybrid retrieval pipeline (vector + BM25 + reranking)
    to find and return relevant information.
    
    Use this tool when:
    - User asks about company policies, procedures, or documentation
    - User needs information from internal knowledge base
    - Questions about HR policies, technical docs, FAQs, etc.
    
    Do NOT use this tool when:
    - User asks for real-time data (stock prices, weather, etc.)
    - User wants to perform calculations
    - User asks general knowledge questions not related to the company
    """
    
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        synthesizer: Optional[ResponseSynthesizer] = None,
    ) -> None:
        """
        Initialize the knowledge base tool.
        
        Args:
            retriever: Hybrid retriever instance. If None, creates default.
            synthesizer: Response synthesizer instance. If None, creates default.
        """
        super().__init__()
        
        # Initialize retriever if not provided
        if retriever is None:
            settings = get_settings()
            vector_store = QdrantVectorStore(
                url=settings.qdrant.url,
                collection_name=settings.qdrant.collection_name,
                vector_size=settings.qdrant.vector_size,
            )
            embedder = get_embedder(dimension=settings.qdrant.vector_size)
            self.retriever = HybridRetriever(
                vector_store=vector_store,
                embedder=embedder,
                alpha=1.0,  # Pure vector search (avoids RRF score issues)
                use_reranking=False,  # Can enable for better quality
            )
        else:
            self.retriever = retriever
        
        # Initialize synthesizer if not provided
        if synthesizer is None:
            # Lower threshold to work with vector similarity scores (0.3-0.9)
            self.synthesizer = ResponseSynthesizer(min_confidence=0.3)
        else:
            self.synthesizer = synthesizer
        
        # Flag to track if BM25 is indexed
        self._bm25_indexed = False
    
    @property
    def name(self) -> str:
        """Tool name."""
        return "query_knowledge_base"
    
    @property
    def description(self) -> str:
        """Tool description for LLM."""
        return """Query the enterprise knowledge base for information.

Use this tool to search through company documentation including:
- HR policies (vacation, remote work, benefits)
- Technical documentation
- FAQs and procedures
- Internal guidelines

The tool returns relevant excerpts with citations.

Parameters:
- query (required): Natural language question or search query
- top_k (optional): Number of results to return (default: 5)

Example usage:
- "What is the vacation policy for new employees?"
- "How do I request remote work?"
- "What are the company's technical documentation standards?"
"""
    
    @property
    def parameters(self) -> list[ToolParameter]:
        """Tool parameters."""
        return [
            ToolParameter(
                name="query",
                type=str,
                description="Natural language question or search query",
                required=True,
            ),
            ToolParameter(
                name="top_k",
                type=int,
                description="Number of results to return",
                required=False,
                default=5,
            ),
        ]
    
    def get_category(self) -> ToolCategory:
        """Get tool category."""
        return ToolCategory.KNOWLEDGE
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the knowledge base query.
        
        Args:
            query: Search query string.
            top_k: Number of results to return.
            
        Returns:
            ToolResult: Query results with synthesized answer.
        """
        # Validate parameters
        self.validate_parameters(**kwargs)
        
        query = kwargs["query"]
        top_k = kwargs.get("top_k", 5)
        
        try:
            # Check if vector store has any data
            collection_info = self.retriever.vector_store.get_collection_info()
            if collection_info["points_count"] == 0:
                return ToolResult(
                    success=False,
                    error=(
                        "Knowledge base is empty. No documents have been indexed. "
                        "Please ingest documents using 'python src/cli.py reset' or "
                        "'python scripts/ingest_all_samples.py'"
                    ),
                    metadata={
                        "query": query,
                        "total_chunks": 0,
                    }
                )
            
            # Ensure BM25 index is built (lazy initialization)
            if not self._bm25_indexed:
                await self._ensure_bm25_index()
            
            # Perform hybrid search
            search_results = self.retriever.search(
                query=query,
                top_k=top_k,
                retrieve_k=top_k * 2,  # Retrieve more for better fusion
            )
            
            # Synthesize response
            response = self.synthesizer.synthesize(
                query=query,
                search_results=search_results
            )
            
            # Format result
            if response.has_answer:
                return ToolResult(
                    success=True,
                    data={
                        "answer": response.answer,
                        "sources": [
                            {
                                "text": src["text"][:200] + "..." 
                                if len(src["text"]) > 200 else src["text"],
                                "filename": src.get("metadata", {}).get("filename", "Unknown"),
                                "score": src.get("score", 0.0),
                            }
                            for src in response.sources
                        ],
                        "confidence": response.confidence,
                        "num_sources": response.num_sources,
                    },
                    metadata={
                        "query": query,
                        "top_k": top_k,
                        "total_results": len(search_results),
                        "total_chunks_in_db": collection_info["points_count"],
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    data={"answer": response.answer},
                    error=(
                        f"No relevant information found or confidence too low "
                        f"(confidence: {response.confidence:.2f}, threshold: "
                        f"{self.synthesizer.min_confidence}). "
                        f"Try rephrasing your query or check if the knowledge base "
                        f"contains relevant documents."
                    ),
                    metadata={
                        "query": query,
                        "confidence": response.confidence,
                        "min_confidence": self.synthesizer.min_confidence,
                        "total_results": len(search_results),
                        "total_chunks_in_db": collection_info["points_count"],
                    }
                )
        
        except Exception as e:
            import traceback
            return ToolResult(
                success=False,
                error=f"Failed to query knowledge base: {str(e)}",
                metadata={
                    "query": query,
                    "traceback": traceback.format_exc(),
                }
            )
    
    async def _ensure_bm25_index(self) -> None:
        """
        Ensure BM25 index is built.
        
        This retrieves all chunks from the vector store and indexes them
        for BM25 search. In production, this should be done during ingestion
        or cached.
        """
        try:
            # Get all chunks from vector store
            # Note: This is a simplification. In production, you'd want
            # a more efficient method or persistent BM25 index.
            dummy_embedding = self.retriever.embedder.embed_text("dummy")
            all_chunks = self.retriever.vector_store.search(
                query_embedding=dummy_embedding,
                top_k=1000,  # Get many chunks
                score_threshold=None
            )
            
            if all_chunks:
                self.retriever.index_for_bm25(all_chunks)
                self._bm25_indexed = True
        
        except Exception as e:
            # If indexing fails, continue with vector-only search
            # The hybrid retriever will gracefully degrade
            pass
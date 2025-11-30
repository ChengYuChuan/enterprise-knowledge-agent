"""
Administrative tools for knowledge base management.

These tools allow agents to manage the knowledge base, including
viewing statistics and getting metadata about documents.
"""

from typing import Any

from src.config import get_settings
from src.rag.retrieval import QdrantVectorStore

from .base import BaseTool, ToolCategory, ToolParameter, ToolResult


class GetKnowledgeBaseStatsTool(BaseTool):
    """
    Tool for retrieving knowledge base statistics.
    
    Use this tool when:
    - User asks "how many documents are in the knowledge base?"
    - User wants to know the current state of the index
    - User asks about storage or capacity
    
    This is a read-only administrative tool.
    """
    
    def __init__(self, vector_store: QdrantVectorStore = None) -> None:
        """
        Initialize the stats tool.
        
        Args:
            vector_store: Vector store instance. If None, creates default.
        """
        super().__init__()
        
        if vector_store is None:
            settings = get_settings()
            self.vector_store = QdrantVectorStore(
                url=settings.qdrant.url,
                collection_name=settings.qdrant.collection_name,
                vector_size=settings.qdrant.vector_size,
            )
        else:
            self.vector_store = vector_store
    
    @property
    def name(self) -> str:
        """Tool name."""
        return "get_knowledge_base_stats"
    
    @property
    def description(self) -> str:
        """Tool description."""
        return """Get statistics about the knowledge base.

Returns information about:
- Total number of indexed document chunks
- Vector dimension
- Collection name
- Storage metrics

Use this when user asks about the size or state of the knowledge base.

Example queries:
- "How many documents are indexed?"
- "What's the current state of the knowledge base?"
- "Show me knowledge base statistics"
"""
    
    @property
    def parameters(self) -> list[ToolParameter]:
        """Tool parameters (none required)."""
        return []
    
    def get_category(self) -> ToolCategory:
        """Get tool category."""
        return ToolCategory.ADMIN
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Get knowledge base statistics.
        
        Returns:
            ToolResult: Statistics about the knowledge base.
        """
        try:
            stats = self.vector_store.get_collection_info()
            
            return ToolResult(
                success=True,
                data={
                    "collection_name": stats["name"],
                    "total_chunks": stats["points_count"],
                    "vector_dimension": stats["vector_size"],
                    "distance_metric": stats["distance"],
                    "status": "operational" if stats["points_count"] > 0 else "empty",
                },
                metadata={
                    "raw_stats": stats
                }
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to retrieve statistics: {str(e)}"
            )


class SearchDocumentsTool(BaseTool):
    """
    Tool for searching documents by metadata.
    
    Unlike query_knowledge_base which does semantic search,
    this tool searches for documents based on metadata filters
    (e.g., filename, file type, department).
    
    Use this when:
    - User asks "what documents are in the knowledge base?"
    - User wants to find documents by name or type
    - User asks "do we have documentation about X?"
    """
    
    def __init__(self, vector_store: QdrantVectorStore = None) -> None:
        """
        Initialize the document search tool.
        
        Args:
            vector_store: Vector store instance. If None, creates default.
        """
        super().__init__()
        
        if vector_store is None:
            settings = get_settings()
            self.vector_store = QdrantVectorStore(
                url=settings.qdrant.url,
                collection_name=settings.qdrant.collection_name,
                vector_size=settings.qdrant.vector_size,
            )
        else:
            self.vector_store = vector_store
    
    @property
    def name(self) -> str:
        """Tool name."""
        return "search_documents"
    
    @property
    def description(self) -> str:
        """Tool description."""
        return """Search for documents by metadata (filename, type, etc).

This tool finds documents based on metadata filters rather than
content search. Useful for:
- Finding specific documents by name
- Listing documents of a certain type
- Discovering what documentation exists

Parameters:
- filename_pattern (optional): Search for files matching this pattern
- file_type (optional): Filter by file extension (.md, .pdf, .txt)

Example queries:
- "What vacation-related documents do we have?"
- "Show me all PDF documents"
- "Find the remote work guidelines"

Note: For content-based search, use query_knowledge_base instead.
"""
    
    @property
    def parameters(self) -> list[ToolParameter]:
        """Tool parameters."""
        return [
            ToolParameter(
                name="filename_pattern",
                type=str,
                description="Pattern to match in filenames (case-insensitive)",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="file_type",
                type=str,
                description="File extension to filter by (e.g., '.md', '.pdf')",
                required=False,
                default=None,
            ),
        ]
    
    def get_category(self) -> ToolCategory:
        """Get tool category."""
        return ToolCategory.SEARCH
    
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Search for documents by metadata.
        
        Args:
            filename_pattern: Optional filename pattern.
            file_type: Optional file type filter.
            
        Returns:
            ToolResult: List of matching documents.
        """
        try:
            filename_pattern = kwargs.get("filename_pattern")
            file_type = kwargs.get("file_type")
            
            # For now, we'll do a simple retrieval and filter
            # In production, this should use Qdrant's filter capabilities
            
            # Get a sample of documents using a dummy query
            from src.rag.retrieval import MockEmbedder
            embedder = MockEmbedder(dimension=self.vector_store.vector_size)
            dummy_embedding = embedder.embed_text("dummy")
            
            # Retrieve many documents
            all_results = self.vector_store.search(
                query_embedding=dummy_embedding,
                top_k=100,
                score_threshold=None,
            )
            
            # Extract unique documents
            seen_files = set()
            documents = []
            
            for result in all_results:
                metadata = result.get("metadata", {})
                filename = metadata.get("filename", "Unknown")
                
                # Skip duplicates
                if filename in seen_files:
                    continue
                
                # Apply filters
                if filename_pattern:
                    if filename_pattern.lower() not in filename.lower():
                        continue
                
                if file_type:
                    if not filename.endswith(file_type):
                        continue
                
                seen_files.add(filename)
                documents.append({
                    "filename": filename,
                    "file_type": metadata.get("file_type", "unknown"),
                    "file_size": metadata.get("file_size", 0),
                    "created_at": metadata.get("created_at", "unknown"),
                })
            
            return ToolResult(
                success=True,
                data={
                    "documents": documents,
                    "total_count": len(documents),
                },
                metadata={
                    "filename_pattern": filename_pattern,
                    "file_type": file_type,
                }
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to search documents: {str(e)}",
                metadata={
                    "filename_pattern": filename_pattern,
                    "file_type": file_type,
                }
            )
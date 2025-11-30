"""
MCP Server for Enterprise Knowledge Agent.

This module implements the Model Context Protocol (MCP) server,
exposing agent capabilities to MCP-compatible clients like Claude Desktop.

The MCP server provides:
1. Tools: Executable functions for knowledge queries and admin operations
2. Resources: Queryable data sources (documents, statistics)

Example Usage:
    Run server:
    ```bash
    python -m src.mcp_server.server
    ```

Reference:
    - MCP Specification: https://spec.modelcontextprotocol.io/
    - FastMCP: https://github.com/jlowin/fastmcp
"""

import asyncio
import logging
from typing import Any, Optional

from fastmcp import FastMCP

from src.agent.react import ReActConfig, ReActEngine
from src.agent.tools import (
    GetKnowledgeBaseStatsTool,
    QueryKnowledgeBaseTool,
    SearchDocumentsTool,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Enterprise Knowledge Agent")


# ============================================================================
# TOOLS - Executable functions exposed via MCP
# ============================================================================


@mcp.tool()
async def query_knowledge_base(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Query the enterprise knowledge base for information.

    Use this tool to search through company documentation including:
    - HR policies (vacation, remote work, benefits)
    - Technical documentation
    - FAQs and procedures
    - Internal guidelines

    The tool returns relevant excerpts with citations.

    Args:
        query: Natural language question or search query.
        top_k: Number of results to return (default: 5).

    Returns:
        dict: Query results with answer, sources, and metadata.

    Example:
        >>> result = await query_knowledge_base("What is the vacation policy?")
        >>> print(result["answer"])
    """
    logger.info(f"[MCP Tool] query_knowledge_base called: query='{query}', top_k={top_k}")

    try:
        # Initialize tool
        tool = QueryKnowledgeBaseTool()

        # Execute query
        result = await tool.execute(query=query, top_k=top_k)

        # Format response
        if result.success:
            return {
                "success": True,
                "answer": result.data["answer"],
                "sources": result.data["sources"],
                "confidence": result.data["confidence"],
                "num_sources": result.data["num_sources"],
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "metadata": result.metadata,
            }

    except Exception as e:
        logger.error(f"[MCP Tool] Error in query_knowledge_base: {e}")
        return {
            "success": False,
            "error": f"Tool execution failed: {str(e)}",
        }


@mcp.tool()
async def get_knowledge_base_stats() -> dict[str, Any]:
    """
    Get statistics about the knowledge base.

    Returns information about:
    - Total number of indexed document chunks
    - Vector dimension
    - Collection name
    - Storage metrics

    Use this when user asks about the size or state of the knowledge base.

    Returns:
        dict: Statistics about the knowledge base.

    Example:
        >>> stats = await get_knowledge_base_stats()
        >>> print(f"Total chunks: {stats['total_chunks']}")
    """
    logger.info("[MCP Tool] get_knowledge_base_stats called")

    try:
        tool = GetKnowledgeBaseStatsTool()
        result = await tool.execute()

        if result.success:
            return {
                "success": True,
                "collection_name": result.data["collection_name"],
                "total_chunks": result.data["total_chunks"],
                "vector_dimension": result.data["vector_dimension"],
                "distance_metric": result.data["distance_metric"],
                "status": result.data["status"],
            }
        else:
            return {
                "success": False,
                "error": result.error,
            }

    except Exception as e:
        logger.error(f"[MCP Tool] Error in get_knowledge_base_stats: {e}")
        return {
            "success": False,
            "error": f"Tool execution failed: {str(e)}",
        }


@mcp.tool()
async def search_documents(
    filename_pattern: Optional[str] = None,
    file_type: Optional[str] = None,
) -> dict[str, Any]:
    """
    Search for documents by metadata (filename, type, etc).

    This tool finds documents based on metadata filters rather than
    content search. Useful for:
    - Finding specific documents by name
    - Listing documents of a certain type
    - Discovering what documentation exists

    Args:
        filename_pattern: Search for files matching this pattern (case-insensitive).
        file_type: Filter by file extension (e.g., '.md', '.pdf').

    Returns:
        dict: List of matching documents with metadata.

    Example:
        >>> docs = await search_documents(filename_pattern="vacation")
        >>> print(f"Found {len(docs['documents'])} documents")
    """
    logger.info(
        f"[MCP Tool] search_documents called: "
        f"pattern='{filename_pattern}', type='{file_type}'"
    )

    try:
        tool = SearchDocumentsTool()
        result = await tool.execute(
            filename_pattern=filename_pattern,
            file_type=file_type,
        )

        if result.success:
            return {
                "success": True,
                "documents": result.data["documents"],
                "total_count": result.data["total_count"],
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "metadata": result.metadata,
            }

    except Exception as e:
        logger.error(f"[MCP Tool] Error in search_documents: {e}")
        return {
            "success": False,
            "error": f"Tool execution failed: {str(e)}",
        }


@mcp.tool()
async def agent_query(
    query: str,
    max_iterations: int = 5,
    enable_reflection: bool = True,
) -> dict[str, Any]:
    """
    Execute a query using the full ReAct agent with reasoning capabilities.

    This is the most powerful tool, using multi-step reasoning to:
    - Break down complex questions
    - Use multiple tools as needed
    - Reflect on answer quality
    - Handle comparisons and multi-step tasks

    Use this for:
    - Complex questions requiring reasoning
    - Queries needing multiple information sources
    - Comparison tasks
    - Questions where simple search isn't enough

    Args:
        query: User query to process.
        max_iterations: Maximum reasoning steps (default: 5).
        enable_reflection: Whether to self-reflect on answers (default: True).

    Returns:
        dict: Complete execution trace with final answer and metadata.

    Example:
        >>> result = await agent_query(
        ...     "Compare vacation policies for new vs senior employees"
        ... )
        >>> print(result["final_answer"])
    """
    logger.info(
        f"[MCP Tool] agent_query called: query='{query}', "
        f"max_iterations={max_iterations}"
    )

    try:
        # Configure ReAct engine
        config = ReActConfig(
            max_iterations=max_iterations,
            enable_reflection=enable_reflection,
            verbose=False,  # Don't spam logs in production
        )

        # Initialize engine
        engine = ReActEngine(config=config)

        # Execute query
        trace = await engine.run(query)

        # Return trace data
        return {
            "success": trace.success,
            "final_answer": trace.final_answer,
            "total_steps": trace.get_step_count(),
            "tools_used": list(trace.get_unique_tools()),
            "duration_ms": trace.total_duration_ms,
            "error": trace.error,
            "metadata": trace.metadata,
        }

    except Exception as e:
        logger.error(f"[MCP Tool] Error in agent_query: {e}")
        return {
            "success": False,
            "error": f"Agent execution failed: {str(e)}",
        }


# ============================================================================
# RESOURCES - Queryable data sources
# ============================================================================


@mcp.resource("documents://list")
async def list_documents() -> str:
    """
    List all documents in the knowledge base.

    Returns:
        str: Formatted list of documents with metadata.
    """
    logger.info("[MCP Resource] list_documents accessed")

    try:
        tool = SearchDocumentsTool()
        result = await tool.execute()

        if not result.success:
            return f"Error: {result.error}"

        documents = result.data["documents"]
        total = result.data["total_count"]

        lines = [f"# Knowledge Base Documents ({total} total)\n"]

        for doc in documents:
            lines.append(
                f"- **{doc['filename']}** ({doc['file_type']}) - "
                f"{doc['file_size']} bytes"
            )

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[MCP Resource] Error listing documents: {e}")
        return f"Error: {str(e)}"


@mcp.resource("stats://knowledge-base")
async def knowledge_base_stats() -> str:
    """
    Get current knowledge base statistics.

    Returns:
        str: Formatted statistics.
    """
    logger.info("[MCP Resource] knowledge_base_stats accessed")

    try:
        tool = GetKnowledgeBaseStatsTool()
        result = await tool.execute()

        if not result.success:
            return f"Error: {result.error}"

        data = result.data

        return f"""# Knowledge Base Statistics

- **Collection**: {data['collection_name']}
- **Total Chunks**: {data['total_chunks']:,}
- **Vector Dimension**: {data['vector_dimension']}
- **Distance Metric**: {data['distance_metric']}
- **Status**: {data['status']}
"""

    except Exception as e:
        logger.error(f"[MCP Resource] Error getting stats: {e}")
        return f"Error: {str(e)}"


# ============================================================================
# SERVER LIFECYCLE
# ============================================================================


async def startup():
    """Initialize server on startup."""
    logger.info("=== MCP Server Starting ===")
    logger.info("Enterprise Knowledge Agent MCP Server")
    
    # Register agent tools to global registry
    from src.agent.tools import register_default_tools
    try:
        register_default_tools()
        logger.info("Agent tools registered to global registry")
    except ValueError as e:
        # Tools already registered
        logger.info(f"Agent tools already registered: {e}")
    
    # Get tool/resource counts (need to await)
    tools = await mcp.list_tools()
    resources = await mcp.list_resources()
    
    logger.info(f"MCP tools registered: {len(tools)}")
    logger.info(f"MCP resources registered: {len(resources)}")
    logger.info("=== Server Ready ===")


async def shutdown():
    """Cleanup on server shutdown."""
    logger.info("=== MCP Server Shutting Down ===")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Run the MCP server."""
    logger.info("Starting MCP Server...")

    # Run startup
    asyncio.run(startup())

    # Run server with stdio transport
    # This allows Claude Desktop to communicate with the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
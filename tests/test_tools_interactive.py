#!/usr/bin/env python
"""
Interactive tool testing script.

Run with: python scripts/test_tools_interactive.py
"""

import asyncio
import sys

from src.agent.tools import (
    GetKnowledgeBaseStatsTool,
    QueryKnowledgeBaseTool,
    SearchDocumentsTool,
)


async def test_query_tool():
    """Test the query knowledge base tool."""
    print("\n" + "=" * 70)
    print("Testing QueryKnowledgeBaseTool")
    print("=" * 70)
    
    tool = QueryKnowledgeBaseTool()
    
    # Test queries
    test_queries = [
        ("vacation policy", "Should find vacation policy documents"),
        ("remote work", "Should find remote work guidelines"),
        ("nonexistent query xyz123", "Should gracefully handle no results"),
    ]
    
    for query, description in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected: {description}")
        print("-" * 70)
        
        result = await tool.execute(query=query, top_k=3)
        
        if result.success:
            print("✓ SUCCESS")
            print(f"  Confidence: {result.data['confidence']:.3f}")
            print(f"  Sources: {result.data['num_sources']}")
            
            # Show answer preview
            answer = result.data['answer']
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"\n  Answer preview:\n  {preview}")
            
            # Show sources
            if result.data['sources']:
                print(f"\n  Top source:")
                src = result.data['sources'][0]
                print(f"  - File: {src['filename']}")
                print(f"  - Score: {src['score']:.3f}")
                print(f"  - Text: {src['text'][:100]}...")
        else:
            print("✗ FAILED")
            print(f"  Error: {result.error}")
            if result.metadata:
                print(f"  Metadata: {result.metadata}")


async def test_stats_tool():
    """Test the stats tool."""
    print("\n" + "=" * 70)
    print("Testing GetKnowledgeBaseStatsTool")
    print("=" * 70)
    
    tool = GetKnowledgeBaseStatsTool()
    result = await tool.execute()
    
    if result.success:
        print("✓ SUCCESS")
        for key, value in result.data.items():
            print(f"  {key}: {value}")
    else:
        print("✗ FAILED")
        print(f"  Error: {result.error}")


async def test_search_tool():
    """Test the document search tool."""
    print("\n" + "=" * 70)
    print("Testing SearchDocumentsTool")
    print("=" * 70)
    
    tool = SearchDocumentsTool()
    
    # Test 1: Search all documents
    print("\nTest 1: Search all documents")
    print("-" * 70)
    result = await tool.execute()
    
    if result.success:
        print(f"✓ Found {result.data['total_count']} documents")
        for doc in result.data['documents'][:5]:
            print(f"  - {doc['filename']}")
    else:
        print(f"✗ Failed: {result.error}")
    
    # Test 2: Search by filename pattern
    print("\nTest 2: Search for 'vacation' documents")
    print("-" * 70)
    result = await tool.execute(filename_pattern="vacation")
    
    if result.success:
        print(f"✓ Found {result.data['total_count']} matching documents")
        for doc in result.data['documents']:
            print(f"  - {doc['filename']}")
    else:
        print(f"✗ Failed: {result.error}")
    
    # Test 3: Search by file type
    print("\nTest 3: Search for .md files")
    print("-" * 70)
    result = await tool.execute(file_type=".md")
    
    if result.success:
        print(f"✓ Found {result.data['total_count']} .md files")
        for doc in result.data['documents'][:5]:
            print(f"  - {doc['filename']}")
    else:
        print(f"✗ Failed: {result.error}")


async def main():
    """Run all tool tests."""
    print("\n" + "🧪" * 35)
    print("AGENT TOOLS INTERACTIVE TEST")
    print("🧪" * 35)
    
    # First, check if knowledge base has data
    stats_tool = GetKnowledgeBaseStatsTool()
    stats = await stats_tool.execute()
    
    if stats.success and stats.data['total_chunks'] == 0:
        print("\n" + "⚠️ " * 35)
        print("WARNING: Knowledge base is EMPTY!")
        print("⚠️ " * 35)
        print("\nPlease run one of the following commands first:")
        print("1. poetry run python src/cli.py reset")
        print("2. poetry run python scripts/ingest_all_samples.py")
        print("\nExiting...")
        sys.exit(1)
    
    # Run tests
    await test_stats_tool()
    await test_search_tool()
    await test_query_tool()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
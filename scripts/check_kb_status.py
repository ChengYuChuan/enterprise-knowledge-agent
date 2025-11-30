#!/usr/bin/env python
"""
Diagnostic script to check knowledge base status.

Run with: python scripts/check_kb_status.py
"""

import asyncio

from src.agent.tools import (
    GetKnowledgeBaseStatsTool,
    QueryKnowledgeBaseTool,
    SearchDocumentsTool,
)


async def main():
    """Run diagnostics on the knowledge base."""
    print("=" * 70)
    print("Knowledge Base Diagnostic Report")
    print("=" * 70)
    
    # Test 1: Check statistics
    print("\n1. Checking knowledge base statistics...")
    stats_tool = GetKnowledgeBaseStatsTool()
    stats_result = await stats_tool.execute()
    
    if stats_result.success:
        print("   ✓ Vector store is accessible")
        print(f"   - Collection: {stats_result.data['collection_name']}")
        print(f"   - Total chunks: {stats_result.data['total_chunks']}")
        print(f"   - Vector dimension: {stats_result.data['vector_dimension']}")
        print(f"   - Status: {stats_result.data['status']}")
        
        if stats_result.data['total_chunks'] == 0:
            print("\n   ⚠️  WARNING: No documents indexed!")
            print("   To fix this, run:")
            print("   1. poetry run python src/cli.py reset")
            print("   2. Or: poetry run python scripts/ingest_all_samples.py")
            return
    else:
        print(f"   ✗ Failed to get stats: {stats_result.error}")
        return
    
    # Test 2: Search for documents
    print("\n2. Searching for documents...")
    search_tool = SearchDocumentsTool()
    search_result = await search_tool.execute()
    
    if search_result.success:
        docs = search_result.data['documents']
        print(f"   ✓ Found {len(docs)} unique documents")
        
        if docs:
            print("\n   Documents in knowledge base:")
            for doc in docs[:5]:  # Show first 5
                print(f"   - {doc['filename']} ({doc['file_type']})")
            
            if len(docs) > 5:
                print(f"   ... and {len(docs) - 5} more")
    else:
        print(f"   ✗ Search failed: {search_result.error}")
    
    # Test 3: Try a query
    print("\n3. Testing knowledge base query...")
    query_tool = QueryKnowledgeBaseTool()
    
    test_queries = [
        "vacation policy",
        "remote work",
        "employee benefits"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        result = await query_tool.execute(query=query, top_k=3)
        
        if result.success:
            print(f"   ✓ Success! Confidence: {result.data['confidence']:.2f}")
            print(f"   Found {result.data['num_sources']} sources")
            
            # Show first source excerpt
            if result.data['sources']:
                first_source = result.data['sources'][0]
                print(f"\n   Top result (score: {first_source['score']:.3f}):")
                print(f"   {first_source['text'][:150]}...")
                print(f"   Source: {first_source['filename']}")
        else:
            print(f"   ✗ Failed: {result.error}")
            if result.metadata:
                print(f"   Confidence was: {result.metadata.get('confidence', 'N/A')}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Diagnostic Summary")
    print("=" * 70)
    
    if stats_result.data['total_chunks'] == 0:
        print("❌ Knowledge base is EMPTY - needs data ingestion")
        print("\nNext steps:")
        print("1. Run: poetry run python src/cli.py reset")
        print("2. This will ingest sample documents from examples/")
    elif stats_result.data['total_chunks'] < 10:
        print("⚠️  Knowledge base has very few chunks")
        print("   Consider ingesting more documents")
    else:
        print("✅ Knowledge base appears operational")
        print(f"   {stats_result.data['total_chunks']} chunks indexed")


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python
"""
Test script for MCP server functionality.

This script verifies that all MCP server components work correctly:
1. Server initialization
2. Tool execution
3. Resource access
4. Error handling

Run with: python scripts/test_mcp_server.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_server_initialization():
    """Test that MCP server initializes correctly."""
    print("\n" + "="*70)
    print("Test 1: Server Initialization")
    print("="*70)
    
    try:
        from src.mcp_server.server import mcp
        
        print(f"✓ Server instance created")
        
        # Get tools list (need to await)
        tools = await mcp.list_tools()
        print(f"  Tools available: {len(tools)}")
        
        # Get resources list (need to await)
        resources = await mcp.list_resources()
        print(f"  Resources available: {len(resources)}")
        
        # List tools
        if tools:
            print("\n  Registered Tools:")
            for tool in tools:
                print(f"    - {tool.name}")
        
        # List resources
        if resources:
            print("\n  Registered Resources:")
            for resource in resources:
                print(f"    - {resource.uri}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_knowledge_base_stats():
    """Test get_knowledge_base_stats tool."""
    print("\n" + "="*70)
    print("Test 2: Knowledge Base Stats Tool")
    print("="*70)
    
    try:
        from src.mcp_server.server import get_knowledge_base_stats
        
        print("Calling get_knowledge_base_stats()...")
        result = await get_knowledge_base_stats()
        
        if result.get("success"):
            print("✓ Tool executed successfully")
            print(f"  Collection: {result.get('collection_name')}")
            print(f"  Total chunks: {result.get('total_chunks')}")
            print(f"  Status: {result.get('status')}")
            return True
        else:
            print(f"✗ Tool failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_query_knowledge_base():
    """Test query_knowledge_base tool."""
    print("\n" + "="*70)
    print("Test 3: Query Knowledge Base Tool")
    print("="*70)
    
    try:
        from src.mcp_server.server import query_knowledge_base
        
        test_query = "What is the vacation policy?"
        print(f"Query: '{test_query}'")
        
        result = await query_knowledge_base(test_query, top_k=3)
        
        if result.get("success"):
            print("✓ Tool executed successfully")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")
            print(f"  Sources: {result.get('num_sources')}")
            
            answer = result.get('answer', '')
            print(f"\n  Answer preview:")
            print(f"  {answer[:200]}...")
            
            return True
        else:
            print(f"✗ Tool failed: {result.get('error')}")
            print(f"  Metadata: {result.get('metadata')}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_search_documents():
    """Test search_documents tool."""
    print("\n" + "="*70)
    print("Test 4: Search Documents Tool")
    print("="*70)
    
    try:
        from src.mcp_server.server import search_documents
        
        print("Searching for documents with 'vacation' in filename...")
        result = await search_documents(filename_pattern="vacation")
        
        if result.get("success"):
            print("✓ Tool executed successfully")
            total = result.get('total_count', 0)
            print(f"  Found {total} matching documents")
            
            if result.get('documents'):
                print("\n  Documents:")
                for doc in result['documents'][:3]:
                    print(f"    - {doc['filename']}")
            
            return True
        else:
            print(f"✗ Tool failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_query():
    """Test agent_query tool."""
    print("\n" + "="*70)
    print("Test 5: Agent Query Tool")
    print("="*70)
    
    try:
        from src.mcp_server.server import agent_query
        
        test_query = "What is the vacation policy for new employees?"
        print(f"Query: '{test_query}'")
        
        result = await agent_query(
            query=test_query,
            max_iterations=3,
            enable_reflection=True
        )
        
        if result.get("success"):
            print("✓ Agent executed successfully")
            print(f"  Total steps: {result.get('total_steps')}")
            print(f"  Duration: {result.get('duration_ms', 0):.0f}ms")
            
            tools_used = result.get('tools_used', [])
            if tools_used:
                print(f"  Tools used: {', '.join(tools_used)}")
            
            answer = result.get('final_answer', '')
            print(f"\n  Answer preview:")
            print(f"  {answer[:200]}...")
            
            return True
        else:
            print(f"✗ Agent failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_resources():
    """Test MCP resources."""
    print("\n" + "="*70)
    print("Test 6: MCP Resources")
    print("="*70)
    
    try:
        from src.mcp_server.server import list_documents, knowledge_base_stats
        
        print("\n6.1: list_documents resource...")
        docs_content = await list_documents()
        print(f"  Content length: {len(docs_content)} characters")
        print(f"  Preview:\n{docs_content[:200]}...")
        
        print("\n6.2: knowledge_base_stats resource...")
        stats_content = await knowledge_base_stats()
        print(f"  Content length: {len(stats_content)} characters")
        print(f"  Preview:\n{stats_content[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "🧪"*35)
    print("MCP SERVER TEST SUITE")
    print("🧪"*35)
    print("\nThis will test all MCP server components.\n")
    
    from src.agent.tools import register_default_tools
    print("Registering agent tools...")
    try:
        register_default_tools()
        print("✓ Agent tools registered\n")
    except ValueError:
        print("✓ Agent tools already registered\n")

    # Check knowledge base status first
    from src.agent.tools import GetKnowledgeBaseStatsTool
    
    print("Checking knowledge base status...")
    stats_tool = GetKnowledgeBaseStatsTool()
    stats_result = await stats_tool.execute()
    
    if stats_result.success and stats_result.data["total_chunks"] == 0:
        print("\n" + "⚠️ "*35)
        print("WARNING: Knowledge base is EMPTY!")
        print("⚠️ "*35)
        print("\nSome tests will fail without data.")
        print("To populate the KB, run:")
        print("  poetry run python src/cli.py reset\n")
        
        response = input("Continue anyway? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Exiting...")
            return
    
    # Run tests
    tests = [
        ("Server Initialization", test_server_initialization),
        ("Knowledge Base Stats", test_knowledge_base_stats),
        ("Query Knowledge Base", test_query_knowledge_base),
        ("Search Documents", test_search_documents),
        ("Agent Query", test_agent_query),
        ("Resources", test_resources),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append((test_name, False))
        
        # Add separator
        print()
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("-"*70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python
"""
Quick test to verify QueryKnowledgeBaseTool configuration.
"""

import asyncio


async def main():
    print("=" * 70)
    print("Testing QueryKnowledgeBaseTool Configuration")
    print("=" * 70)
    
    from src.agent.tools import QueryKnowledgeBaseTool
    
    # Create tool
    print("\nCreating QueryKnowledgeBaseTool...")
    tool = QueryKnowledgeBaseTool()
    
    # Check configuration
    print(f"\nRetriever type: {type(tool.retriever).__name__}")
    print(f"Alpha (vector weight): {tool.retriever.alpha}")
    print(f"Use reranking: {tool.retriever.use_reranking}")
    
    print(f"\nSynthesizer min_confidence: {tool.synthesizer.min_confidence}")
    
    # Expected values for pure vector search
    if tool.retriever.alpha == 1.0:
        print("\n✓ Correct! Using pure vector search (alpha=1.0)")
    elif tool.retriever.alpha == 0.5:
        print("\n✗ Wrong! Still using hybrid (alpha=0.5)")
        print("   This will cause RRF score issues")
    else:
        print(f"\n⚠️  Unexpected alpha value: {tool.retriever.alpha}")
    
    # Test query
    print("\n" + "─" * 70)
    print("Running test query...")
    print("─" * 70)
    
    result = await tool.execute(query="vacation policy")
    
    print(f"\nSuccess: {result.success}")
    
    if result.success:
        print(f"Confidence: {result.data['confidence']:.4f}")
        print(f"Sources: {result.data['num_sources']}")
        print(f"\nAnswer (first 150 chars):")
        print(f"{result.data['answer'][:150]}...")
    else:
        print(f"Error: {result.error}")
        if result.metadata:
            print(f"Confidence: {result.metadata.get('confidence', 'N/A')}")
            print(f"Threshold: {result.metadata.get('min_confidence', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
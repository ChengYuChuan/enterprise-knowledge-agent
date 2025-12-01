#!/usr/bin/env python3
"""
Test RAGPipeline.query() directly - same code path as chat endpoint
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


async def main():
    print(f"\n{'='*60}")
    print("  RAGPipeline Query Test")
    print("  (Same code path as /api/v1/chat)")
    print('='*60)
    
    # Import and initialize RAGPipeline exactly like the API does
    from src.api.dependencies import RAGPipeline
    
    print("\n1. Initializing RAGPipeline...")
    rag = RAGPipeline()
    rag._ensure_initialized()
    
    # Check retriever settings
    print(f"\n2. Retriever Configuration:")
    print(f"   Alpha: {rag._retriever.alpha}")
    print(f"   Use Reranking: {rag._retriever.use_reranking}")
    
    # Get BM25 stats
    bm25_stats = rag._retriever.bm25_search.get_corpus_stats()
    print(f"   BM25 Corpus Size: {bm25_stats.get('corpus_size', 0)}")
    
    # Test queries
    queries = [
        "What are the remote work guidelines?",
        "How many vacation days do new employees get?",
        "What is Qdrant?",
    ]
    
    print(f"\n{'='*60}")
    print("  Query Tests")
    print('='*60)
    
    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")
        
        # Call query() exactly like chat endpoint does
        result = await rag.query(query=query, top_k=5)
        
        sources = result.get("sources", [])
        print(f"   Sources returned: {len(sources)}")
        
        if sources:
            for i, src in enumerate(sources[:3], 1):
                filename = src.get("filename", "unknown")
                score = src.get("score", 0)
                content = src.get("content", "")[:60]
                print(f"   [{i}] {filename} (score: {score:.4f})")
                print(f"       {content}...")
        else:
            print("   ❌ No sources returned!")
            
            # Debug: Test retriever directly
            print("\n   Debug - Testing retriever.search() directly:")
            direct_results = rag._retriever.search(query=query, top_k=5)
            print(f"   Direct search returned: {len(direct_results)} results")
            
            if direct_results:
                for j, r in enumerate(direct_results[:2], 1):
                    print(f"   [{j}] Score: {r.get('score', 0):.4f}")
                    print(f"       Text: {r.get('text', '')[:60]}...")
    
    print(f"\n{'='*60}")
    print("  Test Complete")
    print('='*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
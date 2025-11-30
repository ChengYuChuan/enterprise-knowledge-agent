#!/usr/bin/env python
"""
Test pure vector search vs hybrid search to identify the issue.
"""

import asyncio


async def test_pure_vector_search():
    """Test using only vector search (no BM25)."""
    print("=" * 70)
    print("Test 1: Pure Vector Search (No BM25)")
    print("=" * 70)
    
    from src.config import get_settings
    from src.rag.retrieval import get_embedder, QdrantVectorStore
    
    settings = get_settings()
    
    # Initialize components
    vector_store = QdrantVectorStore(
        url=settings.qdrant.url,
        collection_name=settings.qdrant.collection_name,
        vector_size=settings.qdrant.vector_size,
    )
    
    embedder = get_embedder(dimension=settings.qdrant.vector_size)
    
    # Query
    query = "vacation policy for new employees"
    print(f"\nQuery: '{query}'")
    
    # Generate embedding
    query_embedding = embedder.embed_text(query)
    
    # Pure vector search
    print("\nSearching with pure vector similarity...")
    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=5,
        score_threshold=None,
    )
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Source: {result['metadata'].get('filename', 'unknown')}")
    
    return results


async def test_hybrid_search():
    """Test using hybrid search (Vector + BM25)."""
    print("\n" + "=" * 70)
    print("Test 2: Hybrid Search (Vector + BM25 + RRF)")
    print("=" * 70)
    
    from src.config import get_settings
    from src.rag.retrieval import HybridRetriever, get_embedder, QdrantVectorStore
    
    settings = get_settings()
    
    # Initialize
    vector_store = QdrantVectorStore(
        url=settings.qdrant.url,
        collection_name=settings.qdrant.collection_name,
        vector_size=settings.qdrant.vector_size,
    )
    
    embedder = get_embedder(dimension=settings.qdrant.vector_size)
    
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        alpha=0.5,  # Balanced
        use_reranking=False,
    )
    
    # Index for BM25
    print("\nBuilding BM25 index...")
    dummy_embedding = embedder.embed_text("dummy")
    all_chunks = vector_store.search(
        query_embedding=dummy_embedding,
        top_k=1000,
        score_threshold=None
    )
    
    retriever.index_for_bm25(all_chunks)
    print(f"✓ Indexed {len(all_chunks)} chunks for BM25")
    
    # Query
    query = "vacation policy for new employees"
    print(f"\nQuery: '{query}'")
    
    print("\nSearching with hybrid retrieval...")
    results = retriever.search(
        query=query,
        top_k=5,
        retrieve_k=10,
    )
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. RRF Score: {result['score']:.4f}")
        print(f"   Vector Score: {result.get('vector_score', 'N/A')}")
        print(f"   BM25 Score: {result.get('bm25_score', 'N/A')}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Source: {result['metadata'].get('filename', 'unknown')}")
    
    return results


async def test_with_synthesizer():
    """Test the full pipeline with response synthesizer."""
    print("\n" + "=" * 70)
    print("Test 3: Full Pipeline (Hybrid + Synthesizer)")
    print("=" * 70)
    
    from src.agent.tools import QueryKnowledgeBaseTool
    
    tool = QueryKnowledgeBaseTool()
    
    query = "vacation policy for new employees"
    print(f"\nQuery: '{query}'")
    
    result = await tool.execute(query=query, top_k=5)
    
    if result.success:
        print(f"\n✓ Success!")
        print(f"   Confidence: {result.data['confidence']:.4f}")
        print(f"   Sources: {result.data['num_sources']}")
        print(f"\n   Answer (first 200 chars):")
        print(f"   {result.data['answer'][:200]}...")
    else:
        print(f"\n✗ Failed!")
        print(f"   Error: {result.error}")
        if result.metadata:
            print(f"   Confidence: {result.metadata.get('confidence', 'N/A')}")
            print(f"   Min threshold: {result.metadata.get('min_confidence', 'N/A')}")


async def main():
    """Run all tests."""
    print("\n" + "🔬" * 35)
    print("Vector vs Hybrid Search Comparison")
    print("🔬" * 35)
    print("\nThis will identify where the low scores come from.\n")
    
    # Test 1: Pure vector
    vector_results = await test_pure_vector_search()
    
    # Test 2: Hybrid
    hybrid_results = await test_hybrid_search()
    
    # Test 3: Full pipeline
    await test_with_synthesizer()
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    if vector_results:
        vector_top_score = vector_results[0]['score']
        print(f"\nPure vector top score: {vector_top_score:.4f}")
        
        if vector_top_score > 0.6:
            print("   ✓ Vector search is working well!")
        else:
            print("   ✗ Vector search scores are low")
            print("   This suggests embeddings might still be mismatched")
    
    if hybrid_results:
        hybrid_top_score = hybrid_results[0]['score']
        print(f"Hybrid (RRF) top score: {hybrid_top_score:.4f}")
        
        if hybrid_top_score < vector_top_score:
            print(f"   ⚠️  Hybrid score is LOWER than vector score!")
            print(f"   RRF is pulling the score down by {(vector_top_score - hybrid_top_score):.4f}")
            print("\n   Possible causes:")
            print("   1. BM25 not finding good matches")
            print("   2. Alpha weighting issue")
            print("   3. RRF fusion reducing scores")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    if vector_results and vector_results[0]['score'] > 0.6:
        print("\n✓ Vector search is working!")
        print("\nIf hybrid search is giving low confidence:")
        print("1. Try using ONLY vector search (disable hybrid)")
        print("2. Or adjust alpha parameter (increase towards 1.0)")
        print("3. Or rebuild BM25 index")
    else:
        print("\n✗ Even pure vector search has low scores")
        print("The embeddings might not be properly matched")


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python
"""
Deep trace of the search flow to find where scores are being changed.
"""

import asyncio


async def main():
    print("=" * 70)
    print("Deep Search Flow Trace")
    print("=" * 70)
    
    from src.config import get_settings
    from src.rag.retrieval import get_embedder, QdrantVectorStore, HybridRetriever
    from src.rag.generation import ResponseSynthesizer
    
    settings = get_settings()
    
    # Step 1: Direct vector store search
    print("\n[1] Direct Vector Store Search")
    print("─" * 70)
    
    vector_store = QdrantVectorStore(
        url=settings.qdrant.url,
        collection_name=settings.qdrant.collection_name,
        vector_size=settings.qdrant.vector_size,
    )
    
    embedder = get_embedder(dimension=settings.qdrant.vector_size)
    
    query = "vacation policy"
    query_embedding = embedder.embed_text(query)
    
    direct_results = vector_store.search(
        query_embedding=query_embedding,
        top_k=5,
        score_threshold=None,
    )
    
    print(f"Query: '{query}'")
    print(f"Direct vector search results:")
    for i, r in enumerate(direct_results[:3], 1):
        print(f"  {i}. Score: {r['score']:.4f} | {r['text'][:80]}...")
    
    # Step 2: HybridRetriever with alpha=1.0 (should be same as direct)
    print("\n[2] HybridRetriever Search (alpha=1.0)")
    print("─" * 70)
    
    hybrid = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        alpha=1.0,
        use_reranking=False,
    )
    
    # Check if BM25 needs indexing
    print("Checking BM25 index...")
    if not hasattr(hybrid.bm25_search, 'bm25') or hybrid.bm25_search.bm25 is None:
        print("  BM25 not indexed, indexing now...")
        dummy_embedding = embedder.embed_text("dummy")
        all_chunks = vector_store.search(
            query_embedding=dummy_embedding,
            top_k=1000,
            score_threshold=None
        )
        hybrid.index_for_bm25(all_chunks)
        print(f"  Indexed {len(all_chunks)} chunks")
    else:
        print("  BM25 already indexed")
    
    hybrid_results = hybrid.search(
        query=query,
        top_k=5,
        retrieve_k=10,
    )
    
    print(f"\nHybrid search results (alpha=1.0):")
    for i, r in enumerate(hybrid_results[:3], 1):
        print(f"  {i}. Score: {r['score']:.4f} | Vector: {r.get('vector_score', 'N/A'):.4f} | {r['text'][:60]}...")
    
    # Step 3: Synthesizer
    print("\n[3] Response Synthesizer")
    print("─" * 70)
    
    synthesizer = ResponseSynthesizer(min_confidence=0.3)
    
    response = synthesizer.synthesize(
        query=query,
        search_results=hybrid_results
    )
    
    print(f"Synthesizer output:")
    print(f"  Has answer: {response.has_answer}")
    print(f"  Confidence: {response.confidence:.4f}")
    print(f"  Num sources: {response.num_sources}")
    
    # Step 4: Check confidence calculation
    print("\n[4] Confidence Calculation Analysis")
    print("─" * 70)
    
    if hybrid_results:
        scores = [r.get('score', 0) for r in hybrid_results]
        avg_score = sum(scores) / len(scores)
        
        print(f"Score statistics:")
        print(f"  Scores: {[f'{s:.4f}' for s in scores[:5]]}")
        print(f"  Average: {avg_score:.4f}")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")
        
        print(f"\nSynthesizer confidence calculation:")
        print(f"  Input avg score: {avg_score:.4f}")
        print(f"  Normalized confidence: {min(avg_score, 1.0):.4f}")
        print(f"  Min threshold: {synthesizer.min_confidence}")
        
        if min(avg_score, 1.0) >= synthesizer.min_confidence:
            print(f"  ✓ Should pass threshold")
        else:
            print(f"  ✗ Below threshold")
    
    # Step 5: Compare with QueryKnowledgeBaseTool
    print("\n[5] QueryKnowledgeBaseTool (for comparison)")
    print("─" * 70)
    
    from src.agent.tools import QueryKnowledgeBaseTool
    
    tool = QueryKnowledgeBaseTool()
    result = await tool.execute(query=query)
    
    print(f"Tool result:")
    print(f"  Success: {result.success}")
    if result.metadata:
        print(f"  Confidence: {result.metadata.get('confidence', 'N/A')}")
    if not result.success:
        print(f"  Error: {result.error}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    direct_score = direct_results[0]['score'] if direct_results else 0
    hybrid_score = hybrid_results[0]['score'] if hybrid_results else 0
    synth_confidence = response.confidence
    
    print(f"\nScore comparison:")
    print(f"  1. Direct vector:     {direct_score:.4f}")
    print(f"  2. Hybrid (α=1.0):    {hybrid_score:.4f}")
    print(f"  3. Synthesizer conf:  {synth_confidence:.4f}")
    
    if abs(direct_score - hybrid_score) < 0.01:
        print("\n✓ Step 1→2: Hybrid is correctly using pure vector (scores match)")
    else:
        print(f"\n✗ Step 1→2: Score mismatch! Hybrid changed score by {abs(direct_score - hybrid_score):.4f}")
    
    if abs(hybrid_score - synth_confidence) < 0.01:
        print("✓ Step 2→3: Synthesizer preserves score")
    else:
        print(f"✗ Step 2→3: Synthesizer changed score by {abs(hybrid_score - synth_confidence):.4f}")
        
        # This is expected - synthesizer averages scores
        avg_hybrid = sum(r['score'] for r in hybrid_results) / len(hybrid_results)
        print(f"  (This is expected - synthesizer uses average: {avg_hybrid:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
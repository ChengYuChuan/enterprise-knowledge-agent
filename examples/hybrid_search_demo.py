"""
Example: Hybrid Search with Reranking

This example demonstrates Phase 2 capabilities:
1. BM25 keyword search
2. Vector semantic search
3. Hybrid search with RRF fusion
4. Cross-encoder reranking
5. Response synthesis with citations

Run with: python examples/hybrid_search_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.rag.ingestion import IngestionPipeline, LoaderFactory
from src.rag.retrieval import (
    HybridRetriever,
    MockEmbedder,
    QdrantVectorStore,
)
from src.rag.generation import ResponseSynthesizer


def main():
    """Run the hybrid search demo."""
    print("=" * 70)
    print("Phase 2 Demo: Hybrid Search with Reranking")
    print("=" * 70)
    
    # Initialize components
    print("\n1. Initializing components...")
    settings = get_settings()
    
    vector_store = QdrantVectorStore(
        url=settings.qdrant.url,
        collection_name=settings.qdrant.collection_name,
        vector_size=settings.qdrant.vector_size,
    )
    
    embedder = MockEmbedder(dimension=settings.qdrant.vector_size)
    
    # Initialize hybrid retriever (with reranking disabled for demo speed)
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        alpha=0.5,  # Balanced between vector and BM25
        use_reranking=False,  # Set to True to enable reranking
    )
    
    # Initialize response synthesizer
    synthesizer = ResponseSynthesizer(min_confidence=0.4)
    
    # Check if we have indexed documents
    collection_info = vector_store.get_collection_info()
    
    if collection_info["points_count"] == 0:
        print("\n⚠️  No documents in vector store.")
        print("   Please ingest documents first using:")
        print("   python src/cli.py ingest <file_path>")
        return
    
    print(f"   ✓ Vector store ready ({collection_info['points_count']} chunks)")
    
    # Index chunks for BM25
    print("\n2. Building BM25 index...")
    # Retrieve all chunks from vector store for BM25 indexing
    # Note: In production, you'd want a more efficient way to do this
    dummy_embedding = embedder.embed_text("dummy")
    all_chunks = vector_store.search(
        query_embedding=dummy_embedding,
        top_k=1000,  # Get many chunks
        score_threshold=None
    )
    
    hybrid_retriever.index_for_bm25(all_chunks)
    print(f"   ✓ BM25 index built ({len(all_chunks)} documents)")
    
    # Demo queries
    queries = [
        "What is the vacation policy for new employees?",
        "How many vacation days do senior employees get?",
        "What are the remote work requirements?",
    ]
    
    print("\n" + "=" * 70)
    print("Running Demo Queries")
    print("=" * 70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: {query}")
        print('─' * 70)
        
        # Perform hybrid search
        results = hybrid_retriever.search(
            query=query,
            top_k=3,
            retrieve_k=10
        )
        
        print(f"\nFound {len(results)} results:")
        
        # Display raw results
        for j, result in enumerate(results, 1):
            print(f"\n  Result {j} (score: {result['score']:.4f}):")
            print(f"  {result['text'][:150]}...")
            if 'vector_rank' in result and 'bm25_rank' in result:
                print(f"  Vector rank: {result['vector_rank']}, BM25 rank: {result['bm25_rank']}")
        
        # Synthesize response
        print("\n" + "─" * 70)
        print("Synthesized Response:")
        print("─" * 70)
        
        response = synthesizer.synthesize(query=query, search_results=results)
        
        print(f"\nConfidence: {response.confidence:.2f}")
        print(f"Has Answer: {response.has_answer}")
        print(f"\n{response.answer}")
    
    # Show statistics
    print("\n" + "=" * 70)
    print("Retriever Statistics")
    print("=" * 70)
    
    stats = hybrid_retriever.get_stats()
    print(f"\nAlpha (vector weight): {stats['alpha']}")
    print(f"Reranking enabled: {stats['use_reranking']}")
    print(f"\nBM25 Statistics:")
    print(f"  Documents: {stats['bm25_stats']['document_count']}")
    print(f"  Avg tokens/doc: {stats['bm25_stats']['avg_tokens_per_doc']:.1f}")
    print(f"\nVector Store Statistics:")
    print(f"  Points: {stats['vector_store_stats']['points_count']}")
    print(f"  Vector size: {stats['vector_store_stats']['vector_size']}")


if __name__ == "__main__":
    main()
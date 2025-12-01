#!/usr/bin/env python3
"""
Quick Qdrant diagnostic - test search directly
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

def main():
    from qdrant_client import QdrantClient
    
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base")
    
    print(f"\n{'='*60}")
    print("  Qdrant Direct Diagnostic")
    print('='*60)
    
    client = QdrantClient(url=url)
    
    # 1. Check collection
    info = client.get_collection(collection)
    print(f"\n✅ Collection: {collection}")
    print(f"   Points count: {info.points_count}")
    print(f"   Vector size: {info.config.params.vectors.size}")
    
    # 2. Scroll through ALL points to see what's there
    print(f"\n{'='*60}")
    print("  All Points in Collection")
    print('='*60)
    
    results, _ = client.scroll(
        collection_name=collection,
        limit=20,
        with_payload=True,
        with_vectors=False,
    )
    
    print(f"\nFound {len(results)} points:\n")
    
    for i, point in enumerate(results, 1):
        payload = point.payload or {}
        text = payload.get("text", "")[:150]
        metadata = payload.get("metadata", {})
        print(f"[{i}] ID: {point.id}")
        print(f"    Metadata: {metadata}")
        print(f"    Text preview: {text}...")
        print()
    
    # 3. Test vector search
    print(f"\n{'='*60}")
    print("  Vector Search Test")
    print('='*60)
    
    from src.rag.retrieval.openai_embedder import OpenAIEmbedder
    
    embedder = OpenAIEmbedder(dimension=1536)
    
    queries = [
        "remote work guidelines",
        "vacation days",
        "What is Qdrant?",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")
        
        # Generate embedding
        query_embedding = embedder.embed_text(query)
        
        # Search
        search_results = client.query_points(
            collection_name=collection,
            query=query_embedding,
            limit=3,
        ).points
        
        if search_results:
            print(f"   Found {len(search_results)} results:")
            for j, r in enumerate(search_results, 1):
                text = r.payload.get("text", "")[:80]
                print(f"   [{j}] Score: {r.score:.4f} | {text}...")
        else:
            print("   ❌ No results found!")
    
    print(f"\n{'='*60}")
    print("  Diagnostic Complete")
    print('='*60 + "\n")


if __name__ == "__main__":
    main()
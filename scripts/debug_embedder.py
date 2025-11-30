#!/usr/bin/env python
"""
Debug script to verify which embedder is being used.

This will show exactly which embedder is initialized during ingestion.
"""

import sys
from pathlib import Path


def test_pipeline_embedder():
    """Test which embedder IngestionPipeline uses."""
    print("=" * 70)
    print("Testing IngestionPipeline Embedder")
    print("=" * 70)
    
    from src.rag.ingestion import IngestionPipeline
    
    print("\nCreating IngestionPipeline...")
    pipeline = IngestionPipeline()
    
    print(f"\nEmbedder type: {type(pipeline.embedder).__name__}")
    print(f"Embedder module: {type(pipeline.embedder).__module__}")
    
    # Check if it's OpenAI
    if hasattr(pipeline.embedder, 'is_using_mock'):
        is_mock = pipeline.embedder.is_using_mock()
        print(f"Is using mock: {is_mock}")
    
    # Test embedding
    print("\nTesting embedding generation...")
    test_text = "vacation policy"
    embedding = pipeline.embedder.embed_text(test_text)
    
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Check if values look like OpenAI (should be between -1 and 1, with variety)
    import statistics
    std_dev = statistics.stdev(embedding[:100])
    print(f"Standard deviation (first 100): {std_dev:.4f}")
    
    if std_dev < 0.01:
        print("\n⚠️  WARNING: Low variance suggests MockEmbedder!")
        print("   OpenAI embeddings typically have higher variance.")
    else:
        print("\n✓ Variance looks good for OpenAI embeddings")
    
    return pipeline.embedder


def test_retriever_embedder():
    """Test which embedder QueryKnowledgeBaseTool uses."""
    print("\n" + "=" * 70)
    print("Testing QueryKnowledgeBaseTool Embedder")
    print("=" * 70)
    
    from src.agent.tools import QueryKnowledgeBaseTool
    
    print("\nCreating QueryKnowledgeBaseTool...")
    tool = QueryKnowledgeBaseTool()
    
    print(f"\nRetriever embedder type: {type(tool.retriever.embedder).__name__}")
    print(f"Retriever embedder module: {type(tool.retriever.embedder).__module__}")
    
    # Check if it's OpenAI
    if hasattr(tool.retriever.embedder, 'is_using_mock'):
        is_mock = tool.retriever.embedder.is_using_mock()
        print(f"Is using mock: {is_mock}")
    
    return tool.retriever.embedder


def compare_embeddings():
    """Compare embeddings from ingestion and retrieval."""
    print("\n" + "=" * 70)
    print("Comparing Embeddings")
    print("=" * 70)
    
    from src.rag.ingestion import IngestionPipeline
    from src.agent.tools import QueryKnowledgeBaseTool
    
    pipeline = IngestionPipeline()
    tool = QueryKnowledgeBaseTool()
    
    test_text = "vacation policy for new employees"
    
    print(f"\nTest text: '{test_text}'")
    
    # Get embeddings from both
    print("\nGenerating embedding with IngestionPipeline embedder...")
    emb1 = pipeline.embedder.embed_text(test_text)
    
    print("Generating embedding with QueryKnowledgeBaseTool embedder...")
    emb2 = tool.retriever.embedder.embed_text(test_text)
    
    # Compare
    print(f"\nIngestion embedding (first 5): {emb1[:5]}")
    print(f"Query embedding (first 5):     {emb2[:5]}")
    
    # Calculate similarity
    import math
    dot_product = sum(a * b for a, b in zip(emb1, emb2))
    mag1 = math.sqrt(sum(a * a for a in emb1))
    mag2 = math.sqrt(sum(b * b for b in emb2))
    similarity = dot_product / (mag1 * mag2)
    
    print(f"\nCosine similarity: {similarity:.6f}")
    
    if similarity > 0.99:
        print("✓ Embeddings are nearly identical (same embedder!)")
    elif similarity > 0.9:
        print("⚠️  Embeddings are similar but not identical")
    else:
        print("✗ Embeddings are very different (different embedders!)")
        print("   This explains the low confidence scores!")
    
    return similarity


def check_get_embedder():
    """Check what get_embedder() actually returns."""
    print("\n" + "=" * 70)
    print("Testing get_embedder() Function")
    print("=" * 70)
    
    import os
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"\nOPENAI_API_KEY in environment: {bool(api_key)}")
    if api_key:
        print(f"API key (first 20 chars): {api_key[:20]}...")
    
    # Test get_embedder
    from src.rag.retrieval import get_embedder
    
    print("\nCalling get_embedder()...")
    embedder = get_embedder()
    
    print(f"Returned type: {type(embedder).__name__}")
    
    if hasattr(embedder, 'is_using_mock'):
        print(f"Is using mock: {embedder.is_using_mock()}")
    
    return embedder


def main():
    """Run all diagnostics."""
    print("\n" + "🔬" * 35)
    print("Embedder Deep Diagnostic")
    print("🔬" * 35)
    print("\nThis will identify exactly which embedders are being used.\n")
    
    try:
        # Test 1: Check get_embedder
        check_get_embedder()
        
        # Test 2: Check pipeline
        pipeline_emb = test_pipeline_embedder()
        
        # Test 3: Check retriever
        retriever_emb = test_retriever_embedder()
        
        # Test 4: Compare embeddings
        similarity = compare_embeddings()
        
        # Summary
        print("\n" + "=" * 70)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 70)
        
        pipeline_type = type(pipeline_emb).__name__
        retriever_type = type(retriever_emb).__name__
        
        print(f"\n1. IngestionPipeline uses: {pipeline_type}")
        print(f"2. QueryKnowledgeBaseTool uses: {retriever_type}")
        print(f"3. Embedding similarity: {similarity:.4f}")
        
        if pipeline_type == retriever_type == "OpenAIEmbedder":
            print("\n✓ Both are using OpenAIEmbedder!")
            if similarity < 0.99:
                print("⚠️  But embeddings differ - check API keys or models")
        elif pipeline_type == "MockEmbedder" or retriever_type == "MockEmbedder":
            print("\n✗ At least one is using MockEmbedder!")
            print("   This explains the low confidence scores.")
            print("\n   Possible causes:")
            print("   1. OPENAI_API_KEY not properly loaded")
            print("   2. get_embedder() not being used")
            print("   3. Different code paths for ingestion vs query")
        else:
            print(f"\n⚠️  Different embedders: {pipeline_type} vs {retriever_type}")
        
    except Exception as e:
        print(f"\n✗ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Test OpenAI API connection and embeddings.

This script verifies:
1. API key is properly configured
2. OpenAI SDK is installed
3. Embedding API works
4. Cost estimation

Run with: python scripts/test_openai_connection.py
"""

import os
import sys


def check_api_key():
    """Check if OpenAI API key is configured."""
    print("=" * 70)
    print("Step 1: Checking API Key Configuration")
    print("=" * 70)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("✗ OPENAI_API_KEY not found in environment!")
        print("\nTo fix this:")
        print("1. Create a .env file in the project root")
        print("2. Add this line: OPENAI_API_KEY=sk-your-key-here")
        print("3. Or run: bash scripts/setup_api_keys.sh")
        return False
    
    # Check key format
    if not api_key.startswith("sk-"):
        print(f"⚠️  Warning: API key doesn't start with 'sk-': {api_key[:10]}...")
        print("   This might not be a valid OpenAI API key.")
    else:
        print(f"✓ API key found: {api_key[:7]}...{api_key[-4:]}")
    
    return True


def check_openai_sdk():
    """Check if OpenAI SDK is installed."""
    print("\n" + "=" * 70)
    print("Step 2: Checking OpenAI SDK Installation")
    print("=" * 70)
    
    try:
        import openai
        print(f"✓ OpenAI SDK installed (version: {openai.__version__})")
        return True
    except ImportError:
        print("✗ OpenAI SDK not installed!")
        print("\nTo fix this:")
        print("  poetry add openai")
        return False


def test_embedding_api():
    """Test the embedding API with a simple request."""
    print("\n" + "=" * 70)
    print("Step 3: Testing Embedding API")
    print("=" * 70)
    
    try:
        from openai import OpenAI
        
        client = OpenAI()  # Uses OPENAI_API_KEY from environment
        
        # Test with a simple text
        test_text = "This is a test embedding request"
        print(f"\nTest text: '{test_text}'")
        print("Requesting embedding...")
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=test_text,
        )
        
        embedding = response.data[0].embedding
        
        print(f"✓ Embedding received!")
        print(f"  Dimension: {len(embedding)}")
        print(f"  Sample values: {embedding[:5]}")
        print(f"  Tokens used: {response.usage.total_tokens}")
        
        # Calculate cost
        cost_per_million = 0.02
        cost = (response.usage.total_tokens / 1_000_000) * cost_per_million
        print(f"  Cost: ${cost:.8f} (~{cost * 1000000:.2f} cents per million tokens)")
        
        return True
    
    except Exception as e:
        print(f"✗ API request failed: {e}")
        
        # Provide helpful error messages
        error_str = str(e)
        if "authentication" in error_str.lower() or "api_key" in error_str.lower():
            print("\n💡 This looks like an authentication error.")
            print("   Check that your API key is correct.")
        elif "quota" in error_str.lower():
            print("\n💡 This looks like a quota error.")
            print("   Check your OpenAI account billing settings.")
        
        return False


def estimate_project_cost():
    """Estimate cost for the full project."""
    print("\n" + "=" * 70)
    print("Step 4: Project Cost Estimation")
    print("=" * 70)
    
    # Assumptions
    num_docs = 10
    avg_tokens_per_doc = 500
    total_tokens = num_docs * avg_tokens_per_doc
    
    # Embedding cost
    embedding_cost_per_million = 0.02
    embedding_cost = (total_tokens / 1_000_000) * embedding_cost_per_million
    
    print(f"\nAssuming:")
    print(f"  - {num_docs} documents")
    print(f"  - {avg_tokens_per_doc} tokens per document average")
    print(f"  - Total tokens: {total_tokens:,}")
    
    print(f"\nEmbedding Cost (text-embedding-3-small):")
    print(f"  - Rate: ${embedding_cost_per_million} per 1M tokens")
    print(f"  - Initial ingestion: ${embedding_cost:.6f}")
    
    # Query cost (embeddings only, no LLM yet)
    queries_per_day = 100
    tokens_per_query = 20
    daily_query_tokens = queries_per_day * tokens_per_query
    daily_cost = (daily_query_tokens / 1_000_000) * embedding_cost_per_million
    
    print(f"\nQuery Cost (embeddings only):")
    print(f"  - {queries_per_day} queries per day")
    print(f"  - {tokens_per_query} tokens per query")
    print(f"  - Daily cost: ${daily_cost:.6f}")
    print(f"  - Monthly cost: ${daily_cost * 30:.4f}")
    
    print(f"\n💡 Total estimated cost for development: ~$0.001 (< 1 cent)")
    print(f"   This is extremely affordable for a portfolio project!")


def main():
    """Run all checks."""
    print("\n" + "🔍" * 35)
    print("OpenAI API Connection Test")
    print("🔍" * 35)
    
    # Step 1: Check API key
    if not check_api_key():
        sys.exit(1)
    
    # Step 2: Check SDK
    if not check_openai_sdk():
        sys.exit(1)
    
    # Step 3: Test API
    if not test_embedding_api():
        sys.exit(1)
    
    # Step 4: Cost estimation
    estimate_project_cost()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ All Checks Passed!")
    print("=" * 70)
    print("\nYour OpenAI API is properly configured and working.")
    print("\nNext steps:")
    print("1. Copy openai_embedder.py to src/rag/retrieval/")
    print("2. Update knowledge_tools.py to use OpenAI embedder")
    print("3. Re-ingest documents: poetry run python src/cli.py reset")
    print("4. Test: poetry run python scripts/check_kb_status.py")
    print("=" * 70)


if __name__ == "__main__":
    # Load environment variables from .env
    from pathlib import Path
    env_file = Path(".env")
    
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Loaded environment from .env file\n")
    else:
        print("⚠️  No .env file found. Using system environment variables.\n")
    
    main()

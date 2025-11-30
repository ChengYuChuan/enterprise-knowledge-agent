#!/usr/bin/env python
"""
Direct OpenAI API test (no environment variables).

This bypasses all environment loading to test if the API itself works.
"""

import sys


def test_direct_openai():
    """Test OpenAI API directly."""
    print("=" * 70)
    print("Direct OpenAI API Test")
    print("=" * 70)
    
    # Check if .env file exists and has key
    from pathlib import Path
    env_file = Path(".env")
    
    if not env_file.exists():
        print("\n✗ .env file not found!")
        print("   Create it with: echo 'OPENAI_API_KEY=sk-...' > .env")
        return False
    
    # Read .env content
    env_content = env_file.read_text()
    
    # Find API key
    api_key = None
    for line in env_content.split('\n'):
        if line.startswith('OPENAI_API_KEY'):
            # Extract key value
            if '=' in line:
                api_key = line.split('=', 1)[1].strip()
                # Remove quotes if present
                api_key = api_key.strip('"').strip("'")
                break
    
    if not api_key:
        print("\n✗ OPENAI_API_KEY not found in .env!")
        print(f"   .env content:\n{env_content}")
        return False
    
    print(f"\n✓ API key found in .env")
    print(f"   Key preview: {api_key[:20]}...")
    
    if api_key == 'sk-your-openai-key-here' or api_key.startswith('sk-your'):
        print("\n✗ API key is still placeholder!")
        print("   Replace it with your real API key from OpenAI")
        return False
    
    # Try using it directly with OpenAI
    print("\n2. Testing direct OpenAI API call...")
    
    try:
        from openai import OpenAI
        
        print("   Creating OpenAI client with API key...")
        client = OpenAI(api_key=api_key)
        
        print("   Making embedding request...")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test",
        )
        
        embedding = response.data[0].embedding
        
        print(f"\n✓ API call successful!")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Tokens used: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ API call failed: {e}")
        
        error_str = str(e).lower()
        if 'api_key' in error_str or 'authentication' in error_str:
            print("\n   This is an authentication error.")
            print("   Your API key might be:")
            print("   - Invalid or expired")
            print("   - Not activated (need to add payment method)")
            print("   - Incorrectly copied")
            print("\n   Steps to fix:")
            print("   1. Go to https://platform.openai.com/api-keys")
            print("   2. Create a new key")
            print("   3. Copy it carefully (should start with 'sk-proj-' or 'sk-')")
            print("   4. Update .env: OPENAI_API_KEY=your-new-key")
        
        return False


def main():
    """Run test."""
    print("\n" + "🔑" * 35)
    print("Direct API Key Test")
    print("🔑" * 35)
    print("\nThis bypasses all environment loading.\n")
    
    success = test_direct_openai()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ SUCCESS - OpenAI API is working!")
        print("=" * 70)
        print("\nYour API key is valid and working.")
        print("\nIf you're still getting low confidence:")
        print("1. Make sure the key is in .env")
        print("2. Re-run: poetry run python src/cli.py reset")
        print("3. Check: poetry run python scripts/check_kb_status.py")
    else:
        print("\n" + "=" * 70)
        print("❌ FAILED - Please fix the issues above")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
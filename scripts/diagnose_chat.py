#!/usr/bin/env python3
"""
Chat Endpoint Diagnostic Script

This script checks all prerequisites for the chat endpoint to work correctly.
Run this before testing the chat API to identify configuration issues.

Usage:
    poetry run python scripts/diagnose_chat.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_status(name: str, status: bool, details: str = ""):
    """Print a status line with check/cross mark."""
    mark = "✅" if status else "❌"
    print(f"  {mark} {name}")
    if details:
        print(f"     └─ {details}")


def check_environment_variables():
    """Check required environment variables."""
    print_header("1. Environment Variables")
    
    # Load .env file
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print_status(".env file", True, str(env_path))
    else:
        print_status(".env file", False, "Not found! Copy .env.example to .env")
    
    # Required variables
    required = {
        "API_KEYS": "Authentication keys for API access",
        "OPENAI_API_KEY": "Required for embeddings and LLM (if using OpenAI)",
        "QDRANT_URL": "Qdrant vector database URL",
    }
    
    for var, description in required.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "KEY" in var or "SECRET" in var:
                display = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display = value
            print_status(var, True, display)
        else:
            print_status(var, False, f"Not set! ({description})")
    
    # Optional but useful
    optional = ["LLM_PROVIDER", "LLM_MODEL", "QDRANT_COLLECTION_NAME"]
    print("\n  Optional variables:")
    for var in optional:
        value = os.getenv(var)
        if value:
            print_status(var, True, value)
        else:
            print(f"     ○ {var}: using default")


def check_qdrant_connection():
    """Check Qdrant connectivity."""
    print_header("2. Qdrant Vector Database")
    
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base")
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url)
        
        # Check connection
        print_status("Connection", True, qdrant_url)
        
        # Check collection
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if collection_name in collection_names:
            info = client.get_collection(collection_name)
            print_status(
                f"Collection '{collection_name}'", 
                True, 
                f"{info.points_count} vectors, dim={info.config.params.vectors.size}"
            )
        else:
            print_status(
                f"Collection '{collection_name}'", 
                False, 
                f"Not found! Available: {collection_names}"
            )
            
    except Exception as e:
        print_status("Connection", False, str(e))
        print("\n  ⚠️  Make sure Qdrant is running:")
        print("     docker run -p 6333:6333 qdrant/qdrant")


def check_openai_api():
    """Check OpenAI API connectivity."""
    print_header("3. OpenAI API")
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print_status("API Key", False, "OPENAI_API_KEY not set")
        return
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test with a simple embedding request
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="test"
        )
        
        print_status("API Key", True, f"{api_key[:8]}...")
        print_status("Embeddings", True, f"dimension={len(response.data[0].embedding)}")
        
        # Test models list
        models = client.models.list()
        gpt4_available = any("gpt-4" in m.id for m in models.data)
        print_status("GPT-4 Access", gpt4_available, 
                    "Available" if gpt4_available else "Not available (check API tier)")
        
    except Exception as e:
        print_status("API Connection", False, str(e))


def check_llm_provider():
    """Check LLM provider configuration."""
    print_header("4. LLM Provider Configuration")
    
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4")
    
    print_status("Provider", True, provider)
    print_status("Model", True, model)
    
    try:
        from src.llm import get_provider, LLMConfig, list_providers
        
        available = list_providers()
        print_status("Registered Providers", True, ", ".join(available))
        
        if provider.lower() in available:
            print_status(f"Provider '{provider}'", True, "Available")
        else:
            print_status(f"Provider '{provider}'", False, f"Not in {available}")
            
    except Exception as e:
        print_status("LLM Factory", False, str(e))


def check_api_server():
    """Check if API server can be started."""
    print_header("5. API Server Configuration")
    
    try:
        from src.api.dependencies import get_settings
        settings = get_settings()
        
        print_status("Settings Loaded", True)
        print(f"     ├─ Version: {settings.api_version}")
        print(f"     ├─ Debug: {settings.debug}")
        print(f"     ├─ LLM Provider: {settings.default_llm_provider}")
        print(f"     └─ LLM Model: {settings.default_llm_model}")
        
    except Exception as e:
        print_status("Settings", False, str(e))
    
    # Check API key validation
    api_keys = os.getenv("API_KEYS", "")
    if api_keys:
        keys_list = [k.strip() for k in api_keys.split(",") if k.strip()]
        print_status("API Keys Configured", True, f"{len(keys_list)} key(s)")
        
        # Check if dev key is present
        if "dev-test-key-123" in keys_list:
            print_status("Test Key 'dev-test-key-123'", True, "Available for testing")
    else:
        print_status("API Keys", False, "No API_KEYS configured")


def check_imports():
    """Check that all required modules can be imported."""
    print_header("6. Required Imports")
    
    modules = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("openai", "OpenAI client"),
        ("qdrant_client", "Qdrant vector database client"),
        ("pydantic", "Data validation"),
        ("src.api.main", "API application"),
        ("src.api.routes.chat", "Chat routes"),
        ("src.llm", "LLM providers"),
        ("src.rag.retrieval.hybrid_retriever", "Hybrid retriever"),
    ]
    
    for module_name, description in modules:
        try:
            __import__(module_name)
            print_status(module_name, True, description)
        except ImportError as e:
            print_status(module_name, False, str(e))


def print_test_command():
    """Print the test curl command."""
    print_header("7. Test Command")
    
    print("""
  Once everything is ready, test with:
  
  # Start the server
  poetry run uvicorn src.api.main:app --reload --port 8000
  
  # In another terminal, test the chat endpoint
  curl -X POST http://localhost:8000/api/v1/chat \\
       -H "Content-Type: application/json" \\
       -H "X-API-Key: dev-test-key-123" \\
       -d '{"message": "What is Qdrant?", "use_rag": true}'
""")


def main():
    """Run all diagnostic checks."""
    print("\n" + "="*60)
    print("  Chat Endpoint Diagnostic Tool")
    print("  Enterprise Knowledge Agent")
    print("="*60)
    
    check_environment_variables()
    check_qdrant_connection()
    check_openai_api()
    check_llm_provider()
    check_api_server()
    check_imports()
    print_test_command()
    
    print("\n" + "="*60)
    print("  Diagnostic Complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
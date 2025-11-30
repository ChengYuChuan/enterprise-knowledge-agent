#!/usr/bin/env python3
"""
SSE Client Example

Demonstrates how to consume SSE streams from the Enterprise Knowledge Agent API.

Usage:
    python sse_client.py "What is RAG?"

Requirements:
    pip install httpx sseclient-py
"""

import asyncio
import json
import sys
from typing import Optional

import httpx


# Configuration
API_BASE = "http://localhost:8000/api/v1"
API_KEY = "your-api-key"  # Replace with actual API key


async def stream_chat(
    message: str,
    conversation_id: Optional[str] = None,
    use_rag: bool = True,
    top_k: int = 5,
) -> str:
    """Stream a chat response using SSE.
    
    Args:
        message: User message.
        conversation_id: Optional conversation ID for continuation.
        use_rag: Whether to use RAG.
        top_k: Number of documents to retrieve.
    
    Returns:
        Full response text.
    """
    request_body = {
        "message": message,
        "conversation_id": conversation_id,
        "use_rag": use_rag,
        "top_k": top_k,
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
        "Accept": "text/event-stream",
    }
    
    full_response = ""
    new_conversation_id = None
    sources = []
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{API_BASE}/chat/stream",
            json=request_body,
            headers=headers,
        ) as response:
            response.raise_for_status()
            
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                
                # Process complete events
                while "\n\n" in buffer:
                    event_str, buffer = buffer.split("\n\n", 1)
                    event = parse_sse_event(event_str)
                    if event:
                        result = handle_event(event)
                        if result.get("content"):
                            full_response += result["content"]
                        if result.get("conversation_id"):
                            new_conversation_id = result["conversation_id"]
                        if result.get("sources"):
                            sources = result["sources"]
    
    return {
        "response": full_response,
        "conversation_id": new_conversation_id,
        "sources": sources,
    }


def parse_sse_event(event_str: str) -> Optional[dict]:
    """Parse an SSE event string.
    
    Args:
        event_str: Raw SSE event string.
    
    Returns:
        Parsed event dict or None.
    """
    event = {"type": "message", "data": None}
    
    for line in event_str.strip().split("\n"):
        if line.startswith("event:"):
            event["type"] = line[6:].strip()
        elif line.startswith("data:"):
            try:
                event["data"] = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                event["data"] = line[5:].strip()
        elif line.startswith("id:"):
            event["id"] = line[3:].strip()
    
    return event if event.get("data") is not None else None


def handle_event(event: dict) -> dict:
    """Handle an SSE event.
    
    Args:
        event: Parsed SSE event.
    
    Returns:
        Dict with any extracted data.
    """
    result = {}
    event_type = event.get("type")
    data = event.get("data", {})
    
    if event_type == "start":
        print(f"\n🚀 Stream started (model: {data.get('model')}, provider: {data.get('provider')})")
        result["conversation_id"] = data.get("conversation_id")
        
    elif event_type == "sources":
        sources = data.get("sources", [])
        if sources:
            print(f"\n📚 Retrieved {len(sources)} sources:")
            for i, source in enumerate(sources, 1):
                print(f"   {i}. {source.get('filename')} (score: {source.get('score', 0):.2f})")
        result["sources"] = sources
        
    elif event_type == "message":
        content = data.get("content", "")
        if content:
            print(content, end="", flush=True)
            result["content"] = content
            
    elif event_type == "done":
        print(f"\n\n✅ Stream completed")
        if data.get("latency_ms"):
            print(f"   Latency: {data['latency_ms']:.0f}ms")
        if data.get("usage"):
            usage = data["usage"]
            print(f"   Tokens: {usage.get('total_tokens', 0)} total "
                  f"({usage.get('prompt_tokens', 0)} prompt, "
                  f"{usage.get('completion_tokens', 0)} completion)")
        result["conversation_id"] = data.get("conversation_id")
        
    elif event_type == "error":
        print(f"\n❌ Error: {data.get('message')}")
        
    elif event_type == "heartbeat":
        # Keep-alive, silently ignore
        pass
    
    return result


async def interactive_chat():
    """Run an interactive chat session."""
    print("=" * 60)
    print("🤖 Enterprise Knowledge Agent - Interactive Chat")
    print("=" * 60)
    print("Type 'quit' to exit, 'new' to start a new conversation")
    print()
    
    conversation_id = None
    
    while True:
        try:
            message = input("You: ").strip()
            
            if not message:
                continue
            
            if message.lower() == "quit":
                print("\nGoodbye! 👋")
                break
            
            if message.lower() == "new":
                conversation_id = None
                print("\n🆕 Starting new conversation\n")
                continue
            
            print("\nAssistant: ", end="")
            
            result = await stream_chat(
                message=message,
                conversation_id=conversation_id,
            )
            
            conversation_id = result.get("conversation_id")
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


async def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Single message mode
        message = " ".join(sys.argv[1:])
        print(f"📝 Query: {message}")
        print("-" * 40)
        print("\nAssistant: ", end="")
        
        result = await stream_chat(message)
        
        if result.get("sources"):
            print("\n\n📚 Sources used:")
            for i, source in enumerate(result["sources"], 1):
                print(f"   {i}. {source.get('filename')}")
    else:
        # Interactive mode
        await interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())
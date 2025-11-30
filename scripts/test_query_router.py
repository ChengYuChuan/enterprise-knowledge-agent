#!/usr/bin/env python
"""
Interactive Query Router Test Script.

This script allows you to test the query router with different queries
and see the routing decisions in real-time.

Run with: python scripts/test_query_router.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.router import QueryRouter


def print_decision(decision, query: str) -> None:
    """
    Print a routing decision in a formatted way.
    
    Args:
        decision: RouteDecision to print.
        query: Original query string.
    """
    print("\n" + "─" * 70)
    print(f"Query: '{query}'")
    print("─" * 70)
    
    print(f"\n📋 Routing Decision:")
    print(f"   Intent:     {decision.intent.value}")
    print(f"   Strategy:   {decision.strategy.value}")
    print(f"   Confidence: {decision.confidence:.2f}")
    
    if decision.tools:
        print(f"\n🔧 Tools to use:")
        for i, tool in enumerate(decision.tools, 1):
            print(f"   {i}. {tool}")
    else:
        print(f"\n🔧 Tools: None (direct response)")
    
    print(f"\n💭 Reasoning:")
    print(f"   {decision.reasoning}")
    
    if decision.metadata:
        print(f"\n📊 Metadata:")
        for key, value in decision.metadata.items():
            if isinstance(value, list):
                print(f"   {key}: {', '.join(str(v) for v in value)}")
            else:
                print(f"   {key}: {value}")
    
    # Action recommendations
    print(f"\n✨ Recommended Action:")
    if decision.should_use_tools():
        print(f"   → Execute tools: {', '.join(decision.tools)}")
    elif decision.needs_clarification():
        msg = decision.metadata.get("clarification_message", "Ask for clarification")
        print(f"   → Ask user: {msg}")
    else:
        print(f"   → Provide direct response (no tools needed)")


def test_predefined_queries():
    """Test a set of predefined queries."""
    print("\n" + "=" * 70)
    print("Testing Predefined Queries")
    print("=" * 70)
    
    router = QueryRouter()
    
    # Test queries covering different intents
    test_queries = [
        # Chitchat
        ("Hello, how are you?", "Should be CHITCHAT"),
        ("Thanks for your help!", "Should be CHITCHAT"),
        
        # Admin requests
        ("How many documents are in the knowledge base?", "Should be ADMIN_REQUEST (stats)"),
        ("Show me all PDF files", "Should be ADMIN_REQUEST (search)"),
        
        # Comparisons
        ("Compare vacation policies for new and senior employees", "Should be COMPARISON"),
        ("What's the difference between remote work and office work policies?", "Should be COMPARISON"),
        
        # Complex reasoning
        ("Why do we have different vacation days for different levels?", "Should be COMPLEX_REASONING"),
        ("Explain the rationale behind the remote work policy", "Should be COMPLEX_REASONING"),
        
        # Factual lookup
        ("What is the vacation policy?", "Should be FACTUAL_LOOKUP"),
        ("When can new employees take vacation?", "Should be FACTUAL_LOOKUP"),
        
        # Knowledge query
        ("Tell me about remote work guidelines", "Should be KNOWLEDGE_QUERY"),
        ("vacation accrual for experienced employees", "Should be KNOWLEDGE_QUERY"),
        
        # Unclear
        ("it", "Should be UNCLEAR"),
        ("hm", "Should be UNCLEAR"),
    ]
    
    for query, expected in test_queries:
        print(f"\n{'▼' * 35}")
        print(f"Expected: {expected}")
        
        decision = router.route(query)
        print_decision(decision, query)


def interactive_mode():
    """Interactive mode for testing custom queries."""
    print("\n" + "=" * 70)
    print("Interactive Query Router Test")
    print("=" * 70)
    print("\nEnter queries to see how they're routed.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    router = QueryRouter()
    
    while True:
        try:
            # Get user input
            query = input("\n🔍 Enter query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Route the query
            decision = router.route(query)
            
            # Print decision
            print_decision(decision, query)
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main entry point."""
    print("\n" + "🧭" * 35)
    print("QUERY ROUTER TEST SUITE")
    print("🧭" * 35)
    
    # Check if user wants interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        # Run predefined tests
        test_predefined_queries()
        
        # Offer interactive mode
        print("\n" + "=" * 70)
        print("\n💡 Tip: Run with --interactive flag for interactive testing:")
        print("   python scripts/test_query_router.py --interactive")
        print("=" * 70)


if __name__ == "__main__":
    main()
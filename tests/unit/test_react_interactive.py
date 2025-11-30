#!/usr/bin/env python
"""
Interactive ReAct Engine Test Script.

This script allows you to manually test the ReAct engine with different
queries and see the complete reasoning trace.

Run with: python scripts/test_react_interactive.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.react import ReActConfig, ReActEngine


async def test_single_query(query: str, verbose: bool = True) -> None:
    """
    Test a single query through the ReAct engine.
    
    Args:
        query: Query to test.
        verbose: Whether to show verbose output.
    """
    print("\n" + "="*70)
    print(f"Testing Query: {query}")
    print("="*70)
    
    # Create engine with verbose mode
    config = ReActConfig(
        max_iterations=5,
        enable_reflection=True,
        verbose=verbose,
    )
    
    engine = ReActEngine(config=config)
    
    # Run query
    trace = await engine.run(query)
    
    # Display results
    print("\n" + "-"*70)
    print("EXECUTION RESULTS")
    print("-"*70)
    
    print(f"\nSuccess: {'✅ Yes' if trace.success else '❌ No'}")
    print(f"Total Steps: {trace.get_step_count()}")
    print(f"Duration: {trace.total_duration_ms:.0f}ms")
    
    if trace.has_errors():
        print(f"Failed Steps: {len(trace.get_failed_steps())}")
    
    # Show tools used
    tools_used = trace.get_unique_tools()
    if tools_used:
        print(f"Tools Used: {', '.join(tools_used)}")
    
    # Show steps
    print("\n" + "-"*70)
    print("STEP-BY-STEP TRACE")
    print("-"*70)
    
    for step in trace.steps:
        print(f"\n{'▼'*35}")
        print(f"Step {step.step_number} - {step.status.value.upper()}")
        print(f"{'▼'*35}")
        
        print(f"\n💭 Thought:")
        print(f"   {step.thought.content}")
        print(f"   Type: {step.thought.reasoning_type}")
        print(f"   Confidence: {step.thought.confidence:.2f}")
        
        print(f"\n🔧 Action:")
        if step.action.is_tool_call():
            print(f"   Tool: {step.action.tool_name}")
            print(f"   Parameters: {step.action.parameters}")
        else:
            print(f"   Type: {step.action.action_type.value}")
        
        if step.observation:
            print(f"\n👀 Observation:")
            print(f"   Success: {step.observation.success}")
            if step.observation.success:
                content = step.observation.content[:200]
                if len(step.observation.content) > 200:
                    content += "..."
                print(f"   Content: {content}")
            else:
                print(f"   Error: {step.observation.error}")
        
        if step.duration_ms > 0:
            print(f"\n⏱️  Duration: {step.duration_ms:.0f}ms")
    
    # Show final answer
    print("\n" + "="*70)
    print("FINAL ANSWER")
    print("="*70)
    print(f"\n{trace.final_answer}\n")
    
    if trace.error:
        print(f"\n⚠️  Error: {trace.error}")
    
    # Show reflection if available
    if "reflection" in trace.metadata:
        reflection = trace.metadata["reflection"]
        print("\n" + "-"*70)
        print("SELF-REFLECTION")
        print("-"*70)
        print(f"Confidence: {reflection['confidence']:.2f}")
        print(f"Assessment: {reflection['content']}")


async def run_predefined_tests():
    """Run a set of predefined test queries."""
    print("\n" + "🧪"*35)
    print("REACT ENGINE TEST SUITE")
    print("🧪"*35)
    
    test_queries = [
        # Chitchat
        ("Hello, how are you?", "Chitchat query"),
        
        # Simple factual lookup
        ("What is the vacation policy?", "Simple knowledge query"),
        
        # Admin request
        ("How many documents are in the knowledge base?", "Admin request"),
        
        # Comparison
        ("Compare vacation days for new vs senior employees", "Comparison query"),
        
        # Complex reasoning
        ("Why do senior employees get more vacation days?", "Complex reasoning"),
    ]
    
    for i, (query, description) in enumerate(test_queries, 1):
        print(f"\n\n{'#'*70}")
        print(f"Test {i}/{len(test_queries)}: {description}")
        print(f"{'#'*70}")
        
        await test_single_query(query, verbose=False)
        
        # Add separator between tests
        if i < len(test_queries):
            input("\nPress Enter to continue to next test...")


async def interactive_mode():
    """Interactive mode for custom queries."""
    print("\n" + "="*70)
    print("REACT ENGINE - INTERACTIVE MODE")
    print("="*70)
    print("\nEnter queries to test the ReAct engine.")
    print("Commands:")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'verbose' to toggle verbose mode")
    print("  - Type 'stats' to show KB statistics")
    print()
    
    verbose = False
    
    while True:
        try:
            # Get user input
            query = input("\n🔍 Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'verbose':
                verbose = not verbose
                print(f"✓ Verbose mode: {'ON' if verbose else 'OFF'}")
                continue
            
            if query.lower() == 'stats':
                from src.agent.tools import GetKnowledgeBaseStatsTool
                
                print("\nFetching KB statistics...")
                tool = GetKnowledgeBaseStatsTool()
                result = await tool.execute()
                
                if result.success:
                    print("\n📊 Knowledge Base Statistics:")
                    for key, value in result.data.items():
                        print(f"   {key}: {value}")
                else:
                    print(f"❌ Error: {result.error}")
                continue
            
            # Run query
            await test_single_query(query, verbose=verbose)
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main entry point."""
    print("\n" + "🤖"*35)
    print("REACT ENGINE TESTING TOOL")
    print("🤖"*35)
    
    # Check if knowledge base has data
    from src.agent.tools import GetKnowledgeBaseStatsTool
    
    print("\nChecking knowledge base status...")
    stats_tool = GetKnowledgeBaseStatsTool()
    stats_result = await stats_tool.execute()
    
    if stats_result.success:
        chunk_count = stats_result.data["total_chunks"]
        print(f"✓ Knowledge base loaded: {chunk_count} chunks indexed")
        
        if chunk_count == 0:
            print("\n⚠️  WARNING: Knowledge base is empty!")
            print("   Queries will not return meaningful results.")
            print("\n   To populate the KB, run:")
            print("   poetry run python src/cli.py reset")
    else:
        print(f"❌ Error checking KB: {stats_result.error}")
    
    # Choose mode
    print("\n" + "-"*70)
    print("Select testing mode:")
    print("  1. Run predefined test suite")
    print("  2. Interactive mode (custom queries)")
    print("  3. Quick single test")
    print("-"*70)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = input("\nChoice (1/2/3): ").strip()
    
    if mode == "1":
        await run_predefined_tests()
    elif mode == "2":
        await interactive_mode()
    elif mode == "3":
        test_query = input("\nEnter test query: ").strip()
        if test_query:
            await test_single_query(test_query, verbose=True)
    else:
        print("Invalid choice. Running interactive mode...")
        await interactive_mode()
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
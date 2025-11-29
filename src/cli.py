"""
Command-line interface for the Knowledge Agent.

Enhanced CLI with reset and clear commands.
"""

import argparse
from pathlib import Path

from src.rag.ingestion.pipeline import IngestionPipeline
from src.rag.retrieval import Retriever


def ingest_command(args: argparse.Namespace) -> None:
    """
    Handle the ingest command.

    Args:
        args: Parsed command-line arguments.
    """
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    print(f"Ingesting file: {file_path}")

    pipeline = IngestionPipeline(chunking_strategy=args.strategy)
    result = pipeline.ingest_file(file_path)

    print("\n✅ Ingestion complete!")
    print(f"  - Chunks created: {result['chunks_created']}")
    print(f"  - Total characters: {result['total_characters']}")
    print(f"  - Collection: {result['collection_name']}")

    # Show stats
    stats = pipeline.get_stats()
    print(f"\nCollection stats:")
    print(f"  - Total points: {stats['points_count']}")
    print(f"  - Vector size: {stats['vector_size']}")


def search_command(args: argparse.Namespace) -> None:
    """
    Handle the search command.

    Args:
        args: Parsed command-line arguments.
    """
    query = args.query

    print(f"Searching for: {query}")

    retriever = Retriever()
    results = retriever.search(query=query, top_k=args.top_k)

    print(f"\n📚 Found {len(results)} results:\n")
    print(retriever.format_results(results))


def stats_command(args: argparse.Namespace) -> None:
    """
    Handle the stats command.

    Args:
        args: Parsed command-line arguments.
    """
    pipeline = IngestionPipeline()
    stats = pipeline.get_stats()

    print("📊 Vector Store Statistics:")
    print(f"  - Collection: {stats['name']}")
    print(f"  - Total chunks: {stats['points_count']}")
    print(f"  - Vector dimension: {stats['vector_size']}")
    print(f"  - Distance metric: {stats['distance']}")


def clear_command(args: argparse.Namespace) -> None:
    """
    Handle the clear command - delete all data.

    Args:
        args: Parsed command-line arguments.
    """
    pipeline = IngestionPipeline()
    
    # Get current stats
    stats = pipeline.get_stats()
    points_count = stats['points_count']
    
    if points_count == 0:
        print("ℹ️  Vector store is already empty.")
        return
    
    # Confirm deletion
    if not args.force:
        print(f"⚠️  This will delete {points_count} chunks from the vector store.")
        confirmation = input("Are you sure? (yes/no): ")
        
        if confirmation.lower() not in ['yes', 'y']:
            print("❌ Cancelled.")
            return
    
    # Delete collection
    print(f"🗑️  Deleting collection '{stats['name']}'...")
    pipeline.vector_store.delete_collection()
    
    # Recreate empty collection
    print("♻️  Recreating empty collection...")
    from src.config import get_settings
    from src.rag.retrieval.vector_store import QdrantVectorStore
    
    settings = get_settings()
    new_store = QdrantVectorStore(
        url=settings.qdrant.url,
        collection_name=settings.qdrant.collection_name,
        vector_size=settings.qdrant.vector_size,
    )
    
    print(f"✅ Vector store cleared! ({points_count} chunks deleted)")
    print("\n💡 You can now ingest documents with:")
    print("   python src/cli.py ingest <file_path>")


def reset_command(args: argparse.Namespace) -> None:
    """
    Handle the reset command - clear and reingest sample data.

    Args:
        args: Parsed command-line arguments.
    """
    print("🔄 Resetting knowledge base...")
    print("=" * 70)
    
    # Step 1: Clear existing data
    print("\n1. Clearing existing data...")
    pipeline = IngestionPipeline()
    stats = pipeline.get_stats()
    
    if stats['points_count'] > 0:
        pipeline.vector_store.delete_collection()
        print(f"   ✓ Deleted {stats['points_count']} chunks")
    else:
        print("   ✓ Vector store was empty")
    
    # Recreate collection
    from src.config import get_settings
    from src.rag.retrieval.vector_store import QdrantVectorStore
    
    settings = get_settings()
    pipeline.vector_store = QdrantVectorStore(
        url=settings.qdrant.url,
        collection_name=settings.qdrant.collection_name,
        vector_size=settings.qdrant.vector_size,
    )
    
    # Step 2: Ingest sample documents
    print("\n2. Ingesting sample documents...")
    sample_dir = Path("examples/sample_documents")
    
    if not sample_dir.exists():
        print("   ⚠️  Sample documents not found!")
        return
    
    # Find all supported files
    supported_extensions = [".md", ".txt", ".pdf"]
    all_files = []
    
    for ext in supported_extensions:
        all_files.extend(sample_dir.rglob(f"*{ext}"))
    
    if not all_files:
        print("   ⚠️  No sample documents found!")
        return
    
    print(f"   Found {len(all_files)} files to ingest\n")
    
    # Ingest each file
    total_chunks = 0
    for i, file_path in enumerate(all_files, 1):
        try:
            print(f"   [{i}/{len(all_files)}] {file_path.name}...", end=" ")
            result = pipeline.ingest_file(file_path)
            chunks = result["chunks_created"]
            total_chunks += chunks
            print(f"✓ ({chunks} chunks)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Show final stats
    print("\n" + "=" * 70)
    print("✅ Reset complete!")
    print(f"   Total files: {len(all_files)}")
    print(f"   Total chunks: {total_chunks}")
    print("\n💡 Try searching:")
    print("   python src/cli.py search 'vacation policy'")
    print("=" * 70)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enterprise Knowledge Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document")
    ingest_parser.add_argument("file", type=str, help="Path to the document file")
    ingest_parser.add_argument(
        "--strategy",
        type=str,
        default="semantic",
        choices=["fixed", "sentence", "semantic"],
        help="Chunking strategy to use",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the knowledge base")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return"
    )

    # Stats command
    subparsers.add_parser("stats", help="Show vector store statistics")
    
    # Clear command
    clear_parser = subparsers.add_parser(
        "clear", 
        help="Clear all data from vector store"
    )
    clear_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    # Reset command
    subparsers.add_parser(
        "reset",
        help="Clear and reingest sample documents"
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "search":
        search_command(args)
    elif args.command == "stats":
        stats_command(args)
    elif args.command == "clear":
        clear_command(args)
    elif args.command == "reset":
        reset_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
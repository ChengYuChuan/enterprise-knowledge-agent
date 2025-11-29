"""
Command-line interface for the Knowledge Agent.

Simple CLI for testing document ingestion and retrieval.
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

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "search":
        search_command(args)
    elif args.command == "stats":
        stats_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
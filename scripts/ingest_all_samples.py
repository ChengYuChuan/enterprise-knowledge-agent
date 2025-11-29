#!/usr/bin/env python
"""
Batch ingestion script for sample documents.

This script ingests all sample documents in the examples directory.
Run with: python scripts/ingest_all_samples.py
"""

from pathlib import Path

from src.rag.ingestion import IngestionPipeline


def main():
    """Ingest all sample documents."""
    print("🚀 Batch Ingestion: Sample Documents")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = IngestionPipeline(chunking_strategy="semantic")
    
    # Find all sample documents
    sample_dir = Path("examples/sample_documents")
    
    if not sample_dir.exists():
        print("❌ Sample documents directory not found!")
        return
    
    # Get all supported file types
    supported_extensions = [".md", ".txt", ".pdf"]
    all_files = []
    
    for ext in supported_extensions:
        all_files.extend(sample_dir.rglob(f"*{ext}"))
    
    if not all_files:
        print("❌ No documents found!")
        return
    
    print(f"\n📚 Found {len(all_files)} documents to ingest:\n")
    
    for file in all_files:
        print(f"  • {file.relative_to('examples')}")
    
    print("\n" + "─" * 70)
    print("Starting ingestion...\n")
    
    # Ingest each file
    total_chunks = 0
    successful = 0
    failed = 0
    
    for i, file_path in enumerate(all_files, 1):
        try:
            print(f"[{i}/{len(all_files)}] Ingesting: {file_path.name}...", end=" ")
            
            result = pipeline.ingest_file(file_path)
            chunks_created = result["chunks_created"]
            total_chunks += chunks_created
            successful += 1
            
            print(f"✓ ({chunks_created} chunks)")
            
        except Exception as e:
            failed += 1
            print(f"✗ Error: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 Ingestion Summary")
    print("=" * 70)
    print(f"Total files processed: {len(all_files)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  📦 Total chunks created: {total_chunks}")
    
    # Show collection stats
    print("\n" + "─" * 70)
    print("Vector Store Statistics:")
    print("─" * 70)
    
    stats = pipeline.get_stats()
    print(f"  Collection: {stats['name']}")
    print(f"  Total points: {stats['points_count']}")
    print(f"  Vector dimension: {stats['vector_size']}")
    
    print("\n✅ Batch ingestion complete!")
    print("\n💡 Next steps:")
    print("  1. Run demo: poetry run python examples/hybrid_search_demo.py")
    print("  2. Try search: poetry run python src/cli.py search 'your query'")
    print("=" * 70)


if __name__ == "__main__":
    main()
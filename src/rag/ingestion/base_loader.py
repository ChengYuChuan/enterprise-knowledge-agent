"""
Base document loader interface and common data structures.

This module defines the abstract base class for document loaders
and common data structures used across all loaders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Document:
    """
    Represents a loaded document with content and metadata.

    Attributes:
        content: The raw text content of the document.
        metadata: Dictionary containing document metadata.
        doc_id: Unique identifier for the document.
    """

    content: str
    metadata: dict
    doc_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Generate doc_id if not provided."""
        if self.doc_id is None:
            # Use filename + timestamp as default ID
            filename = self.metadata.get("filename", "unknown")
            timestamp = datetime.now().isoformat()
            self.doc_id = f"{filename}_{timestamp}"


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.

    All document loaders must inherit from this class and implement
    the load() method for their specific file format.
    """

    @abstractmethod
    def load(self, file_path: Path) -> Document:
        """
        Load a document from the given file path.

        Args:
            file_path: Path to the document file.

        Returns:
            Document: Loaded document with content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        pass

    def extract_metadata(self, file_path: Path) -> dict:
        """
        Extract basic metadata from the file.

        Args:
            file_path: Path to the document file.

        Returns:
            dict: Metadata dictionary with common fields.
        """
        stat = file_path.stat()

        return {
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_path": str(file_path.absolute()),
        }

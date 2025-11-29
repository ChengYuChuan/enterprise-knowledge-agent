"""
Text and Markdown document loader implementation.

This module provides functionality to load plain text and Markdown files.
"""

from pathlib import Path

from .base_loader import BaseLoader, Document


class TextLoader(BaseLoader):
    """
    Loader for plain text and Markdown documents.

    Supports .txt and .md file formats with UTF-8 encoding.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}

    def __init__(self, encoding: str = "utf-8") -> None:
        """
        Initialize the text loader.

        Args:
            encoding: Character encoding to use (default: utf-8).
        """
        self.encoding = encoding

    def load(self, file_path: Path) -> Document:
        """
        Load a text or Markdown document.

        Args:
            file_path: Path to the text file.

        Returns:
            Document: Loaded document with content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {file_path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        # Read file content
        content = self._read_text(file_path)

        # Get metadata
        metadata = self.extract_metadata(file_path)
        metadata["encoding"] = self.encoding
        metadata["line_count"] = content.count("\n") + 1
        metadata["is_markdown"] = file_path.suffix.lower() in {".md", ".markdown"}

        return Document(content=content, metadata=metadata)

    def _read_text(self, file_path: Path) -> str:
        """
        Read text content from file.

        Args:
            file_path: Path to the text file.

        Returns:
            str: File content as string.

        Raises:
            UnicodeDecodeError: If file encoding doesn't match specified encoding.
        """
        try:
            with open(file_path, "r", encoding=self.encoding) as f:
                return f.read()
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Failed to decode file with encoding '{self.encoding}': {e}"
            ) from e

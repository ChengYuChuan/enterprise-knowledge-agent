"""
Document loader factory for automatic loader selection.

This module provides a factory function to automatically select
the appropriate loader based on file extension.
"""

from pathlib import Path
from typing import Type

from .base_loader import BaseLoader, Document
from .pdf_loader import PDFLoader
from .text_loader import TextLoader


class LoaderFactory:
    """
    Factory for creating appropriate document loaders.

    Automatically selects the correct loader based on file extension.
    """

    # Map file extensions to loader classes
    LOADER_MAP: dict[str, Type[BaseLoader]] = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".markdown": TextLoader,
    }

    @classmethod
    def get_loader(cls, file_path: Path) -> BaseLoader:
        """
        Get the appropriate loader for the given file.

        Args:
            file_path: Path to the document file.

        Returns:
            BaseLoader: Instantiated loader for the file type.

        Raises:
            ValueError: If the file extension is not supported.
        """
        extension = file_path.suffix.lower()

        if extension not in cls.LOADER_MAP:
            supported = ", ".join(cls.LOADER_MAP.keys())
            raise ValueError(
                f"Unsupported file extension: {extension}. Supported formats: {supported}"
            )

        loader_class = cls.LOADER_MAP[extension]
        return loader_class()

    @classmethod
    def load_document(cls, file_path: Path) -> Document:
        """
        Load a document using the appropriate loader.

        Convenience method that combines loader selection and loading.

        Args:
            file_path: Path to the document file.

        Returns:
            Document: Loaded document with content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        loader = cls.get_loader(file_path)
        return loader.load(file_path)

    @classmethod
    def register_loader(cls, extension: str, loader_class: Type[BaseLoader]) -> None:
        """
        Register a custom loader for a file extension.

        Args:
            extension: File extension (e.g., '.custom').
            loader_class: Loader class to use for this extension.
        """
        if not extension.startswith("."):
            extension = f".{extension}"

        cls.LOADER_MAP[extension.lower()] = loader_class

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """
        Get list of supported file extensions.

        Returns:
            list[str]: List of supported extensions.
        """
        return list(cls.LOADER_MAP.keys())

"""
PDF document loader implementation.

This module provides functionality to load and extract text from PDF files.
"""

from pathlib import Path

from pypdf import PdfReader

from .base_loader import BaseLoader, Document


class PDFLoader(BaseLoader):
    """
    Loader for PDF documents.

    Uses pypdf library to extract text content from PDF files.
    Handles multi-page documents and preserves page structure.
    """

    def load(self, file_path: Path) -> Document:
        """
        Load a PDF document.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Document: Loaded document with extracted text and metadata.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")

        # Extract text from PDF
        content = self._extract_text(file_path)

        # Get metadata
        metadata = self.extract_metadata(file_path)
        metadata.update(self._extract_pdf_metadata(file_path))

        return Document(content=content, metadata=metadata)

    def _extract_text(self, file_path: Path) -> str:
        """
        Extract text content from all pages of the PDF.

        Args:
            file_path: Path to the PDF file.

        Returns:
            str: Concatenated text from all pages.
        """
        reader = PdfReader(str(file_path))
        pages_text = []

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text.strip():  # Only add non-empty pages
                pages_text.append(f"[Page {page_num}]\n{page_text}")

        return "\n\n".join(pages_text)

    def _extract_pdf_metadata(self, file_path: Path) -> dict:
        """
        Extract PDF-specific metadata.

        Args:
            file_path: Path to the PDF file.

        Returns:
            dict: PDF metadata including page count, author, title, etc.
        """
        reader = PdfReader(str(file_path))
        pdf_metadata = reader.metadata or {}

        return {
            "page_count": len(reader.pages),
            "author": pdf_metadata.get("/Author", "Unknown"),
            "title": pdf_metadata.get("/Title", ""),
            "subject": pdf_metadata.get("/Subject", ""),
            "creator": pdf_metadata.get("/Creator", ""),
        }

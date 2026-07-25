from pathlib import Path
from typing import Protocol


class DocumentLoader(Protocol):
    def load(self, source_path: str) -> str:
        """Return the document's plain text, structure-preserving (headings on
        their own lines) so clause_parser can detect boundaries."""
        ...


class PlainTextLoader:
    """Reads a .txt/.md file as-is. Used for fixtures/tests and for pre-extracted
    text; production ingestion of PDFs/DOCX goes through DoclingLoader."""

    def load(self, source_path: str) -> str:
        return Path(source_path).read_text(encoding="utf-8")


class DoclingLoader:
    """Docling-backed loader for real contract files (PDF/DOCX/etc). Docling is
    a heavy optional dependency — imported lazily so importing this module (or
    the rest of the ingestion package) never requires it to be installed."""

    def load(self, source_path: str) -> str:
        from docling.document_converter import DocumentConverter  # lazy import

        converter = DocumentConverter()
        result = converter.convert(source_path)
        return result.document.export_to_markdown()

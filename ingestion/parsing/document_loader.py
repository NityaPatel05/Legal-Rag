"""Document loaders: turn a source file into the plain text clause_parser
segments.

Docling's export is Markdown, not plain text: headings come out as
"## 1. Services" or "**1. Services**", and no clause_parser pattern matches
a line that starts with "#" or "*" -- so every Docling-loaded contract parsed
as ONE clause (preamble only), silently. DoclingLoader therefore strips the
heading markup (strip_markdown_heading_markup) before returning. Only markup
that sits at the start of a line is touched: bold inside body text is left
as-is, since stored text is what citations are matched against and the
model sees the same text. Markdown tables are out of scope."""

import re
from pathlib import Path
from typing import Protocol

# ATX heading: up to 3 spaces of indent, 1-6 "#", whitespace, the text, and
# an optional closing run of "#".
_ATX_HEADING = re.compile(r"^[ ]{0,3}#{1,6}[ \t]+(?P<text>.*?)(?:[ \t]+#+)?[ \t]*$", re.MULTILINE)
# Bold / underline-bold wrapping the start of a line: "**1. Services**" or
# "**4. Law.** Delaware law governs." -> the markers go, the text stays.
_LEADING_EMPHASIS = re.compile(r"^(?P<indent>[ \t]*)(?P<mark>\*\*|__)(?P<text>[^\n]+?)(?P=mark)", re.MULTILINE)


def strip_markdown_heading_markup(markdown: str) -> str:
    """Removes ATX heading markers ("## ") and a bold/underline wrapper at
    the start of a line, leaving the heading text on its own line where
    clause_parser can see it."""
    text = _ATX_HEADING.sub(lambda m: m.group("text"), markdown)
    return _LEADING_EMPHASIS.sub(lambda m: m.group("indent") + m.group("text"), text)


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
        return strip_markdown_heading_markup(result.document.export_to_markdown())

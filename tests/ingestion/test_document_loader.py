"""B20: DoclingLoader returns Markdown; heading markers must not hide
headings from clause_parser. Docling itself is not needed -- the helper is
pure."""

from ingestion.parsing.clause_parser import parse_clauses
from ingestion.parsing.document_loader import strip_markdown_heading_markup

DOCLING_MARKDOWN = (
    "# MASTER SERVICES AGREEMENT\n\n"
    "This Agreement is made by Acme and Globex.\n\n"
    "## 1. Services\n\n"
    "Vendor provides the services.\n\n"
    "**2. Fees**\n\n"
    "Customer pays the fees in **U.S. dollars**.\n\n"
    "### __3. Term__ ###\n\n"
    "One year.\n\n"
    "**4. Law.** Delaware law governs.\n"
)


def test_heading_markup_is_stripped():
    text = strip_markdown_heading_markup(DOCLING_MARKDOWN)
    assert text.splitlines()[0] == "MASTER SERVICES AGREEMENT"
    assert "\n1. Services\n" in text
    assert "\n2. Fees\n" in text
    assert "\n3. Term\n" in text
    assert "\n4. Law. Delaware law governs.\n" in text


def test_inline_emphasis_in_body_text_is_left_alone():
    text = strip_markdown_heading_markup(DOCLING_MARKDOWN)
    assert "Customer pays the fees in **U.S. dollars**." in text


def test_markdown_headings_segment_into_clauses():
    clauses = parse_clauses(strip_markdown_heading_markup(DOCLING_MARKDOWN), document_id="d")
    assert [c.section_id for c in clauses] == [None, "1", "2", "3", "4"]
    assert [c.heading for c in clauses[1:4]] == ["Services", "Fees", "Term"]


def test_raw_markdown_was_one_clause():
    """Documents the bug: without stripping, no heading pattern matches."""
    assert len(parse_clauses(DOCLING_MARKDOWN, document_id="d")) == 1

from pathlib import Path

from ingestion.parsing.clause_parser import parse_clauses
from ingestion.parsing.references import extract_references

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_contract.txt"


def test_extracts_cross_references_from_fixture():
    raw_text = FIXTURE_PATH.read_text(encoding="utf-8")
    clauses = parse_clauses(raw_text, document_id="doc-1")

    references = extract_references(clauses)

    by_source_heading = {
        next(c.heading for c in clauses if c.clause_id == r.source_clause_id): r.referenced_section_id
        for r in references
    }
    assert by_source_heading == {
        "Initial Term": "3",
        "Termination": "2.1",
    }


def test_self_reference_in_captioned_heading_is_excluded():
    text = "Section 5: Miscellaneous\n\nThis clause is governed by Section 5 of this Agreement."
    clauses = parse_clauses(text, document_id="doc-2")

    references = extract_references(clauses)
    assert references == []


def test_no_references_returns_empty_list():
    clauses = parse_clauses("1. Term\n\nThis Agreement lasts one year.", document_id="doc-3")
    assert extract_references(clauses) == []

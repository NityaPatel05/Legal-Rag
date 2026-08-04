from pathlib import Path

from ingestion.parsing.clause_parser import parse_clauses
from ingestion.parsing.defined_terms import extract_defined_terms

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_contract.txt"


def test_extracts_defined_term_from_fixture():
    raw_text = FIXTURE_PATH.read_text(encoding="utf-8")
    clauses = parse_clauses(raw_text, document_id="doc-1")

    terms = extract_defined_terms(clauses)

    assert len(terms) == 1
    assert terms[0].term == "Confidential Information"
    assert "non-public information" in terms[0].definition
    assert terms[0].definition.strip().endswith("technical specifications")
    definitions_clause = next(c for c in clauses if c.heading == "Definitions")
    assert terms[0].defined_in_clause_id == definitions_clause.clause_id


def test_no_definitions_returns_empty_list():
    clauses = parse_clauses("1. Term\n\nThis Agreement lasts one year.", document_id="doc-2")
    assert extract_defined_terms(clauses) == []


def test_multiple_defined_terms_in_one_clause():
    text = (
        '1. Definitions\n\n"Buyer" means Acme Corp. "Seller" means Globex LLC.'
    )
    clauses = parse_clauses(text, document_id="doc-3")

    terms = extract_defined_terms(clauses)
    term_names = {t.term for t in terms}
    assert term_names == {"Buyer", "Seller"}

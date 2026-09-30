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


# --- L6: abbreviation periods don't end a definition ---


def _definitions(text: str) -> dict[str, str]:
    return {t.term: t.definition for t in extract_defined_terms(parse_clauses(text, document_id="d"))}


def test_definition_is_not_cut_at_a_company_suffix():
    defs = _definitions('1. Definitions\n\n"Licensor" means Acme Holdings, Inc. and its Affiliates. Other text.')
    assert defs["Licensor"] == "Acme Holdings, Inc. and its Affiliates"


def test_definition_is_not_cut_at_eg_or_ie():
    defs = _definitions(
        '1. Definitions\n\n"Products" means goods, e.g. Widgets and Gadgets, i.e. Hardware sold hereunder. Next.'
    )
    assert defs["Products"] == "goods, e.g. Widgets and Gadgets, i.e. Hardware sold hereunder"


def test_definition_keeps_us_and_no_abbreviations():
    defs = _definitions('1. Definitions\n\n"Dollars" means U.S. Dollars under Order No. 5 of the Plan. Next.')
    assert defs["Dollars"] == "U.S. Dollars under Order No. 5 of the Plan"


def test_company_suffix_before_the_next_definition_still_ends_it():
    defs = _definitions('1. Definitions\n\n"Buyer" means Acme Corp. "Seller" means Globex Ltd. The rest.')
    assert defs == {"Buyer": "Acme Corp", "Seller": "Globex Ltd"}


def test_parenthetical_definition_after_a_company_suffix_keeps_the_name():
    defs = _definitions('1. Parties\n\nThis Agreement is between Acme Corp. (the "Company") and Globex, Inc. ("Globex").')
    assert defs["Company"] == "This Agreement is between Acme Corp."
    assert defs["Globex"].endswith("and Globex, Inc.")

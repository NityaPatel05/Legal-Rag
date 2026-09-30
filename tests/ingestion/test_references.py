from pathlib import Path

from ingestion.parsing.clause_parser import parse_clauses, split_oversize_clauses
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


# --- B8: references on a clause's first line, ranges ---


def _refs(text: str) -> list[str]:
    return [r.referenced_section_id for r in extract_references(parse_clauses(text, document_id="d"))]


def test_reference_on_an_inline_heading_line_is_extracted():
    """Most CUAD clauses put heading and body on one line."""
    text = "1. Scope. See Section 4.\n\n2. Termination. Either party may terminate under Section 7.2 and Section 9."
    clauses = parse_clauses(text, document_id="d")
    by_clause = {}
    for r in extract_references(clauses):
        by_clause.setdefault(r.source_clause_id, []).append(r.referenced_section_id)
    assert by_clause == {clauses[0].clause_id: ["4"], clauses[1].clause_id: ["7.2", "9"]}


def test_reference_on_first_line_of_a_continuation_passage_is_extracted():
    body = " ".join(f"Sentence {i} states an obligation." for i in range(40))
    text = "1. Big\n\n" + body + " Fees are due under Section 4.\n\n" + body
    passages = split_oversize_clauses(parse_clauses(text, document_id="d"), max_chars=1000)
    refs = extract_references(passages)
    assert [r.referenced_section_id for r in refs] == ["4"]
    assert refs[0].source_clause_id.endswith(".p01")


def test_inline_captioned_self_reference_is_still_excluded():
    assert _refs("Section 5: Miscellaneous. This Section 5 applies, subject to Section 6.") == ["6"]


def test_section_range_is_expanded():
    assert _refs("1. Survival\n\nSections 5 through 9 survive termination.") == ["5", "6", "7", "8", "9"]
    assert _refs("1. Survival\n\nSections 5 to 7 survive.") == ["5", "6", "7"]
    assert _refs("1. Survival\n\nSections 5-7 survive.") == ["5", "6", "7"]
    assert _refs("1. Survival\n\nSections 5.1–5.4 survive.") == ["5.1", "5.2", "5.3", "5.4"]


def test_range_expansion_is_capped_and_needs_matching_prefixes():
    assert _refs("1. Survival\n\nSections 2 through 90 survive.") == ["2", "90"]
    assert _refs("1. Survival\n\nSections 3.1 through 4.2 survive.") == ["3.1", "4.2"]
    assert _refs("1. Survival\n\nArticles II through IV survive.") == ["II", "IV"]


def test_list_with_oxford_comma_is_fully_extracted():
    assert _refs("1. Survival\n\nSections 5, 6, and 8 survive.") == ["5", "6", "8"]


def test_to_followed_by_a_word_is_not_a_range():
    assert _refs("1. Fees\n\nSubject to Section 5 to the extent applicable.") == ["5"]

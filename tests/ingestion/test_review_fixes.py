"""Ingestion-side fixes from the design review (see CLAUDE.md "Review
findings"): oversize-clause splitting, definition/reference patterns that
real contracts use, place-name normalization, document-level metadata, and
unambiguous section lookup."""

from datetime import date

from ingestion.embedding.document_reader import build_section_lookup
from ingestion.metadata.document_profile import (
    DocumentProfile,
    extract_document_profile,
    merge_document_profile,
    parse_iso_date,
)
from ingestion.metadata.normalize import normalize_place
from ingestion.parsing.clause_parser import parse_clauses, split_oversize_clauses
from ingestion.parsing.defined_terms import extract_defined_terms
from ingestion.parsing.references import extract_references
from ingestion.schemas import Clause, ClauseMetadata

# --- split_oversize_clauses ---


def _long_clause_text(paragraphs=4, sentences=40):
    para = " ".join(f"Sentence number {i} states an obligation." for i in range(sentences))
    return "\n\n".join(para for _ in range(paragraphs))


def test_oversize_clause_is_split_into_contiguous_passages_within_its_bounds():
    raw = "1. Big Clause\n\n" + _long_clause_text() + "\n\n2. Next\n\nShort."
    clauses = parse_clauses(raw, document_id="d")
    passages = split_oversize_clauses(clauses, max_chars=1000)

    big = clauses[0]
    parts = [p for p in passages if p.parent_clause_id == big.clause_id]
    assert len(parts) > 1
    assert all(len(p.text) <= 1000 for p in parts)
    assert "".join(p.text for p in parts) == big.text
    assert parts[0].start_offset == big.start_offset and parts[-1].end_offset == big.end_offset
    for p in parts:
        assert p.text == raw[p.start_offset : p.end_offset]
        assert p.section_id == "1" and p.heading == big.heading


def test_passages_prefer_paragraph_then_sentence_boundaries():
    raw = "1. Big\n\n" + _long_clause_text(paragraphs=2, sentences=30)
    passages = split_oversize_clauses(parse_clauses(raw, document_id="d"), max_chars=1200)
    for p in passages[:-1]:
        assert p.text.rstrip().endswith("."), p.text[-40:]


def test_short_clauses_pass_through_unchanged():
    clauses = parse_clauses("1. Term\n\nOne year.", document_id="d")
    assert split_oversize_clauses(clauses, max_chars=1000) == clauses


# --- defined terms / references ---


def test_parenthetical_and_curly_quote_definitions_are_extracted():
    text = (
        "1. Parties\n\nThis agreement is between Acme Corp, a Delaware corporation "
        "(the “Company”), and Globex LLC (\"Distributor\")."
    )
    terms = {t.term: t for t in extract_defined_terms(parse_clauses(text, document_id="d"))}
    assert set(terms) == {"Company", "Distributor"}
    assert "Acme Corp" in terms["Company"].definition


def test_duplicate_definitions_keep_the_first():
    text = '1. Definitions\n\n"Term" means one year. "Term" means two years.'
    terms = extract_defined_terms(parse_clauses(text, document_id="d"))
    assert [(t.term, t.definition) for t in terms] == [("Term", "one year")]


def test_plural_and_list_references_are_all_extracted():
    text = "1. Scope\n\nSubject to Sections 5 and 7.2, and Article IV, the parties agree."
    refs = extract_references(parse_clauses(text, document_id="d"))
    assert [r.referenced_section_id for r in refs] == ["5", "7.2", "IV"]


# --- normalization + document profile ---


def test_normalize_place_collapses_boilerplate_variants():
    variants = ["Delaware", "State of Delaware", "the laws of the State of Delaware.", "  DELAWARE "]
    assert {normalize_place(v) for v in variants} == {"delaware"}
    assert normalize_place("Commonwealth of Massachusetts") == "massachusetts"
    assert normalize_place(None) is None
    assert normalize_place("  ") is None


def test_invalid_llm_dates_become_none_instead_of_crashing():
    assert parse_iso_date("2024-01-15") == date(2024, 1, 15)
    assert parse_iso_date("2024-13-45") is None
    assert parse_iso_date("sometime in 2024") is None
    assert parse_iso_date(None) is None


class _ProfileLLM:
    def __init__(self, response):
        self._response = response

    def generate_json(self, system_prompt, user_prompt):
        return self._response


def test_document_profile_is_normalized_and_merged_without_overwriting():
    profile = extract_document_profile(
        "text",
        llm_client=_ProfileLLM(
            {
                "summary": "Distribution agreement between Acme and Globex.",
                "contract_type": "Distribution Agreement",
                "parties": ["Acme", "Globex"],
                "effective_date": "not a date",
                "governing_law": "State of Delaware",
                "jurisdiction": "New York",
            }
        ),
    )
    assert profile.governing_law == "delaware"
    assert profile.effective_date is None

    # An ordinary clause inherits the contract's governing law...
    plain = merge_document_profile(ClauseMetadata(tenant_id="t", matter_id="m"), profile)
    assert plain.governing_law == "delaware" and plain.jurisdiction == "new york"
    assert plain.contract_type == "Distributor"  # controlled vocabulary (R7)

    # ...but a clause-level value (e.g. an exhibit's own law) is never overwritten.
    exhibit = merge_document_profile(
        ClauseMetadata(tenant_id="t", matter_id="m", governing_law="california"), profile
    )
    assert exhibit.governing_law == "california"


def test_empty_profile_merge_is_a_no_op():
    metadata = ClauseMetadata(tenant_id="t", matter_id="m", clause_type="Termination")
    assert merge_document_profile(metadata, DocumentProfile()) == metadata


# --- section lookup ---


def _c(cid, section, start, parent=None):
    return Clause(
        clause_id=cid, section_id=section, text="x", start_offset=start, end_offset=start + 1, parent_clause_id=parent
    )


def test_section_lookup_drops_ambiguous_ids_but_keeps_passages_of_one_clause():
    clauses = [
        _c("d::clause-001", "1", 0),
        _c("d::clause-005", "1", 500),  # exhibit restarts numbering -> ambiguous
        _c("d::clause-002.p00", "2", 100, parent="d::clause-002"),
        _c("d::clause-002.p01", "2", 150, parent="d::clause-002"),
    ]
    meta = {c.clause_id: ClauseMetadata(tenant_id="t", matter_id="m") for c in clauses}
    lookup = build_section_lookup(clauses, meta)

    assert "1" not in lookup
    assert lookup["2"][0].clause_id == "d::clause-002.p00"

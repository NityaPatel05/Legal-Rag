from query.schemas import RetrievedClause
from query.validation.citation_validator import (
    Citation,
    check_span_supported,
    validate_citation,
    validate_citations,
)
from tests.query.factories import make_clause, make_metadata

GOVERNING_LAW_CLAUSE = make_clause(
    "doc::c4",
    section_id="4",
    heading="Governing Law",
    text="This Agreement shall be governed by the laws of the State of Delaware.",
)
TERMINATION_CLAUSE = make_clause(
    "doc::c3",
    section_id="3",
    heading="Termination",
    text="Either party may terminate this Agreement for cause upon thirty (30) days written notice.",
)
SOURCE_CLAUSES = [GOVERNING_LAW_CLAUSE, TERMINATION_CLAUSE]


def _shown(*clauses):
    return [RetrievedClause(clause=c, metadata=make_metadata(), score=1.0, source="hybrid") for c in clauses]


# --- known-good ---


def test_known_good_citation_is_valid():
    citation = Citation(section_id="4", claim_text="This Agreement is governed by Delaware law.")
    result = validate_citation(
        citation, source_clauses=SOURCE_CLAUSES, shown_clauses=_shown(GOVERNING_LAW_CLAUSE)
    )

    assert result.exists is True
    assert result.in_context is True
    assert result.span_supported is True
    assert result.is_valid is True


def test_known_good_citation_with_matching_number_is_valid():
    citation = Citation(
        section_id="3",
        claim_text="Either party may terminate this Agreement for cause upon thirty (30) days written notice.",
    )
    result = validate_citation(
        citation, source_clauses=SOURCE_CLAUSES, shown_clauses=_shown(TERMINATION_CLAUSE)
    )
    assert result.is_valid is True


# --- known-bad: citation to a section that doesn't exist in the document ---


def test_nonexistent_section_id_fails_exists_check():
    citation = Citation(section_id="99", claim_text="This clause covers indemnification.")
    result = validate_citation(
        citation, source_clauses=SOURCE_CLAUSES, shown_clauses=_shown(GOVERNING_LAW_CLAUSE)
    )

    assert result.exists is False
    assert result.is_valid is False


# --- known-bad: section exists in the document, but wasn't shown to the model ---


def test_section_exists_but_not_shown_fails_in_context_check():
    citation = Citation(
        section_id="3",
        claim_text="Either party may terminate this Agreement for cause upon thirty (30) days written notice.",
    )
    # only the governing-law clause was retrieved/shown -- termination exists
    # in the document but was never in the model's context.
    result = validate_citation(
        citation, source_clauses=SOURCE_CLAUSES, shown_clauses=_shown(GOVERNING_LAW_CLAUSE)
    )

    assert result.exists is True
    assert result.in_context is False
    assert result.is_valid is False


# --- known-bad: right clause, wrong claim (the case most stacks skip) ---


def test_right_clause_off_topic_claim_fails_span_support():
    citation = Citation(section_id="4", claim_text="This Agreement has a five-year initial term.")
    result = validate_citation(
        citation, source_clauses=SOURCE_CLAUSES, shown_clauses=_shown(GOVERNING_LAW_CLAUSE)
    )

    assert result.exists is True
    assert result.in_context is True
    assert result.span_supported is False
    assert result.is_valid is False


def test_right_clause_swapped_number_fails_span_support():
    """The critical case: high word overlap with the real clause (~90%),
    differing only in the cited number -- a claim citing 90 days against a
    clause that actually says 30. Pure word-overlap would pass this; the
    numeric-token check must not."""
    citation = Citation(
        section_id="3",
        claim_text="Either party may terminate this Agreement for cause upon ninety (90) days written notice.",
    )
    result = validate_citation(
        citation, source_clauses=SOURCE_CLAUSES, shown_clauses=_shown(TERMINATION_CLAUSE)
    )

    assert result.exists is True
    assert result.in_context is True
    assert result.span_supported is False
    assert result.is_valid is False


def test_right_clause_flipped_negation_fails_span_support():
    citation = Citation(
        section_id="3",
        claim_text="Neither party may terminate this Agreement without ninety (90) days written notice.",
    )
    result = validate_citation(
        citation, source_clauses=SOURCE_CLAUSES, shown_clauses=_shown(TERMINATION_CLAUSE)
    )
    assert result.span_supported is False


# --- unit-level checks on check_span_supported directly ---


def test_check_span_supported_true_for_verbatim_restatement():
    assert check_span_supported("Delaware law governs this Agreement.", GOVERNING_LAW_CLAUSE.text) is True


def test_check_span_supported_false_for_empty_claim():
    assert check_span_supported("", GOVERNING_LAW_CLAUSE.text) is False


# --- batch validation ---


def test_validate_citations_handles_a_mix_of_valid_and_invalid():
    citations = [
        Citation(section_id="4", claim_text="This Agreement is governed by Delaware law."),
        Citation(section_id="99", claim_text="This clause covers indemnification."),
    ]
    results = validate_citations(
        citations, source_clauses=SOURCE_CLAUSES, shown_clauses=_shown(GOVERNING_LAW_CLAUSE)
    )

    assert [r.is_valid for r in results] == [True, False]

"""Query-side fixes from the design review (see CLAUDE.md "Review
findings")."""

import pytest
from fastapi import HTTPException

from app.dependencies import authorize_tenant, check_auth_configured
from app.main import build_query_response
from ingestion.config import Settings
from ingestion.schemas import DefinedTerm, SectionReference
from query.retrieval.bm25_search import InMemoryBM25Index, tokenize
from query.retrieval.hybrid import hybrid_retrieve
from query.retrieval.reference_expansion import expand_with_references
from query.schemas import PermissionContext, QueryFilters, RetrievedClause
from query.validation.citation_validator import (
    Citation,
    check_quote_in_clause,
    check_span_supported,
    validate_citation,
)
from query.validation.strip_unsupported import Claim
from tests.query.factories import make_clause, make_metadata

# --- citation validator: keys, quotes, numbers, negation ---

TERMINATION = make_clause(
    "doc::clause-003",
    section_id="3",
    text=(
        "Either party may terminate this Agreement for cause upon thirty (30) days written notice. "
        "Fees paid are not refundable."
    ),
)
PREAMBLE = make_clause("doc::clause-000", section_id=None, text="This Agreement is between Acme Corp and Globex LLC.")


def _shown(*clauses):
    return [RetrievedClause(clause=c, metadata=make_metadata(), score=1.0, source="hybrid") for c in clauses]


def test_clause_without_section_id_is_citable_by_clause_id():
    citation = Citation(
        section_id="doc::clause-000",
        claim_text="The Agreement is between Acme Corp and Globex LLC.",
        supporting_quote="between Acme Corp and Globex LLC",
    )
    result = validate_citation(citation, source_clauses=[PREAMBLE], shown_clauses=_shown(PREAMBLE))
    assert result.is_valid
    assert result.resolved_clause_id == "doc::clause-000"


def test_ambiguous_section_id_does_not_resolve():
    exhibit = make_clause("doc::clause-009", section_id="3", text="Exhibit clause three.")
    citation = Citation(section_id="3", claim_text="Either party may terminate for cause.")
    result = validate_citation(
        citation, source_clauses=[TERMINATION, exhibit], shown_clauses=_shown(TERMINATION)
    )
    assert result.exists is False and not result.is_valid


def test_shown_clause_from_another_document_is_not_in_context():
    other_doc = make_clause("other::clause-003", section_id="3", text=TERMINATION.text)
    citation = Citation(section_id="3", claim_text="Either party may terminate for cause upon 30 days notice.")
    result = validate_citation(citation, source_clauses=[TERMINATION], shown_clauses=_shown(other_doc))
    assert result.exists is True
    assert result.in_context is False


def test_fabricated_quote_fails():
    assert not check_span_supported(
        "Either party may terminate at will.",
        TERMINATION.text,
        supporting_quote="Either party may terminate this Agreement at will",
    )


def test_quote_matching_tolerates_whitespace_and_curly_quotes_only():
    clause = "The “Services” shall be\nperformed   diligently."
    assert check_quote_in_clause('The "Services" shall be performed diligently', clause)
    assert not check_quote_in_clause("The Services shall be performed", clause)  # quotes are content
    assert not check_quote_in_clause("performed", clause)  # too short to ground anything


def test_number_words_are_compared_to_digits():
    # Digit-only checking let "ninety days" pass against a 30-day clause.
    assert not check_span_supported("Either party may terminate for cause upon ninety days written notice.", TERMINATION.text)
    assert check_span_supported("Either party may terminate for cause upon 30 days written notice.", TERMINATION.text.split(" Fees")[0])


def test_quote_scopes_negation_check_to_the_relevant_sentence():
    claim = "Either party may terminate for cause upon thirty (30) days written notice."
    # Whole-clause comparison wrongly rejects: "not" appears in a different sentence.
    assert not check_span_supported(claim, TERMINATION.text)
    assert check_span_supported(
        claim, TERMINATION.text, supporting_quote="terminate this Agreement for cause upon thirty (30) days"
    )


def test_quote_path_still_catches_a_flipped_negation():
    assert not check_span_supported(
        "Fees paid are refundable.", TERMINATION.text, supporting_quote="Fees paid are not refundable"
    )


# --- hybrid retrieval: BM25 obeys every filter ---


class _Vec:
    def search(self, query_text, *, filters, top_k):
        return []


class _BM25:
    def __init__(self, results):
        self._results = results

    def search(self, query_text, *, top_k):
        return self._results


def test_bm25_hits_respect_governing_law_and_document_filters():
    def rc(cid, law):
        return RetrievedClause(
            clause=make_clause(cid), metadata=make_metadata(governing_law=law), score=1.0, source="bm25"
        )

    results = hybrid_retrieve(
        "q",
        vector_client=_Vec(),
        bm25_index=_BM25([rc("doc-a::clause-000", "delaware"), rc("doc-a::clause-001", "california"), rc("doc-b::clause-000", "delaware")]),
        permission_context=PermissionContext(tenant_id="tenant-1", matter_id="matter-1"),
        query_filters=QueryFilters(governing_law="State of Delaware", document_id="doc-a"),
        top_k=10,
    )
    assert [r.clause.clause_id for r in results] == ["doc-a::clause-000"]


def test_bm25_tokenizer_strips_punctuation():
    assert tokenize("governed by Delaware. See Section 2.1, party's rights") == [
        "governed", "by", "delaware", "see", "section", "2.1", "party's", "rights",
    ]
    corpus = [(make_clause("d::c0", text="Governed by the laws of Delaware."), make_metadata())]
    assert [r.clause.clause_id for r in InMemoryBM25Index(corpus).search("delaware", top_k=5)] == ["d::c0"]


# --- reference expansion ---


def _entry(cid, section, text="x"):
    return make_clause(cid, section_id=section, text=text), make_metadata()


def test_reference_to_unparsed_subsection_falls_back_to_parent_section():
    top = [RetrievedClause(clause=make_clause("d::c1", section_id="1"), metadata=make_metadata(), score=1, source="h")]
    expanded = expand_with_references(
        top,
        all_clauses_by_section_id={"5": _entry("d::c5", "5")},
        references=[SectionReference(source_clause_id="d::c1", referenced_section_id="5.2", raw_text="Section 5.2")],
    )
    assert [r.clause.clause_id for r in expanded] == ["d::c1", "d::c5"]


def test_defined_term_used_in_retrieved_clause_pulls_in_its_definition():
    top = [
        RetrievedClause(
            clause=make_clause("d::c3", text="The Company shall deliver the Services."),
            metadata=make_metadata(),
            score=1,
            source="h",
        )
    ]
    expanded = expand_with_references(
        top,
        all_clauses_by_section_id={},
        references=[],
        defined_terms=[
            DefinedTerm(term="Services", definition="the work", defined_in_clause_id="d::c1"),
            DefinedTerm(term="Territory", definition="the US", defined_in_clause_id="d::c2"),
        ],
        all_clauses_by_id={"d::c1": _entry("d::c1", "1"), "d::c2": _entry("d::c2", "2")},
    )
    assert [r.clause.clause_id for r in expanded] == ["d::c3", "d::c1"]


def test_expansion_is_capped():
    top = [RetrievedClause(clause=make_clause("d::c0", section_id="0"), metadata=make_metadata(), score=1, source="h")]
    refs = [SectionReference(source_clause_id="d::c0", referenced_section_id=str(i), raw_text="") for i in range(1, 10)]
    lookup = {str(i): _entry(f"d::c{i}", str(i)) for i in range(1, 10)}
    expanded = expand_with_references(top, all_clauses_by_section_id=lookup, references=refs, max_expansions=3)
    assert len(expanded) == 4


# --- API: auth + response only carries validated content ---


def test_authorization_binds_api_key_to_tenant():
    settings = Settings(api_keys={"key-1": "tenant-1"})
    authorize_tenant("tenant-1", "key-1", settings)
    with pytest.raises(HTTPException) as missing:
        authorize_tenant("tenant-1", None, settings)
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as wrong_tenant:
        authorize_tenant("tenant-2", "key-1", settings)
    assert wrong_tenant.value.status_code == 403


def test_no_keys_fails_closed_unless_dev_mode_on_loopback():
    # A5: an empty API_KEYS is no longer "open to anyone".
    with pytest.raises(HTTPException) as not_configured:
        authorize_tenant("any-tenant", None, Settings(api_keys={}, dev_mode=False), "127.0.0.1")
    assert not_configured.value.status_code == 503

    dev = Settings(api_keys={}, dev_mode=True)
    authorize_tenant("any-tenant", None, dev, "127.0.0.1")
    authorize_tenant("any-tenant", None, dev, "::1")
    for host in ("10.0.0.5", "172.17.0.1", None):
        with pytest.raises(HTTPException) as remote:
            authorize_tenant("any-tenant", None, dev, host)
        assert remote.value.status_code == 403


def test_startup_refuses_empty_api_keys_without_dev_mode():
    with pytest.raises(RuntimeError, match="API_KEYS"):
        check_auth_configured(Settings(api_keys={}, dev_mode=False))
    check_auth_configured(Settings(api_keys={}, dev_mode=True))
    check_auth_configured(Settings(api_keys={"k": "t"}, dev_mode=False))


def test_reranker_is_off_by_default():
    # A6: on only when the R3 ablation shows a gain.
    assert Settings(_env_file=None).rerank_enabled is False


def test_flagged_response_carries_no_answer_text():
    response = build_query_response(
        "q",
        "clause_lookup",
        {"final_status": "flagged_for_review", "flagged_reason": "Ambiguous", "final_answer_text": None},
    )
    assert response.answer_text is None
    assert response.citations == []
    assert "not legal advice" in response.disclaimer


def test_answered_response_uses_composed_text_not_model_free_text():
    response = build_query_response(
        "q",
        "clause_lookup",
        {
            "final_status": "answered_with_stripped_claims",
            "final_answer_text": "Delaware law governs.",
            "kept_claims": [Claim(claim_text="Delaware law governs.", citation_section_id="d::c4")],
            "citation_results": [],
        },
    )
    assert response.answer_text == "Delaware law governs."
    assert [c.section_id for c in response.citations] == ["d::c4"]

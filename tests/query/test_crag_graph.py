import pytest

from ingestion.metadata.llm_client import CountingLLMClient
from query.filters.audit_log import InMemoryAuditLogger
from query.filters.permission import PermissionViolation
from query.generation.answer_generator import AnswerGenerationResult, Sufficiency
from query.generation.crag_graph import (
    BLOCKED_AT_INPUT_REASON,
    MAX_RETRIES,
    NOT_FOUND_TEXT,
    _route_after_generate,
    _route_after_validation,
    run_crag,
)
from query.schemas import (
    JailbreakClassification,
    PermissionContext,
    QueryFilters,
    QueryResult,
    RetrievedClause,
    RouteType,
)
from query.validation.citation_validator import Citation, CitationValidationResult
from tests.query.factories import make_clause, make_metadata

# --- routing functions, tested directly as pure functions of state ---


def _answer(sufficiency, claims=None, search_query=None):
    return AnswerGenerationResult(
        answer_text="", claims=claims or [], sufficiency=Sufficiency(sufficiency), search_query=search_query
    )


def test_route_sufficient_goes_to_validation():
    state = {"answer": _answer("sufficient"), "retry_count": 0}
    assert _route_after_generate(state) == "validate_citations"


def test_route_ambiguous_goes_to_flag_for_review():
    state = {"answer": _answer("ambiguous"), "retry_count": 0, "route": "clause_lookup"}
    assert _route_after_generate(state) == "flag_for_review"


def test_route_ambiguous_on_a_contradiction_check_still_validates():
    state = {"answer": _answer("ambiguous"), "retry_count": 0, "route": "contradiction_check"}
    assert _route_after_generate(state) == "validate_citations"


def test_route_insufficient_with_budget_and_a_new_query_retrieves_again():
    state = {"answer": _answer("insufficient", search_query="termination notice"), "retry_count": 0}
    assert _route_after_generate(state) == "rewrite_and_retrieve"


def test_route_insufficient_without_a_new_query_does_not_repeat_the_same_search():
    state = {
        "answer": _answer("insufficient", search_query="Termination  Notice"),
        "retry_count": 0,
        "tried_queries": ["termination notice"],
    }
    assert _route_after_generate(state) == "not_found"


def test_route_insufficient_with_budget_exhausted_is_not_found():
    state = {"answer": _answer("insufficient", search_query="x"), "retry_count": MAX_RETRIES}
    assert _route_after_generate(state) == "not_found"


def _valid_result():
    return CitationValidationResult(
        citation=Citation(section_id="1", claim_text="x"), exists=True, in_context=True, span_supported=True
    )


def _invalid_result():
    return CitationValidationResult(
        citation=Citation(section_id="1", claim_text="x"), exists=True, in_context=True, span_supported=False
    )


def test_route_all_valid_citations_finalizes():
    state = {"citation_results": [_valid_result(), _valid_result()], "retry_count": 0}
    assert _route_after_validation(state) == "finalize_answered"


def test_route_no_citations_flags_for_review():
    """An answer with zero claims verified nothing -- it must not be returned
    as "answered" (the original behavior let uncited free text through)."""
    state = {"citation_results": [], "retry_count": 0}
    assert _route_after_validation(state) == "flag_for_review"


def test_route_invalid_citation_with_budget_remaining_retries():
    state = {"citation_results": [_valid_result(), _invalid_result()], "retry_count": 0}
    assert _route_after_validation(state) == "regenerate_with_feedback"


def test_route_invalid_citation_with_budget_exhausted_strips():
    state = {"citation_results": [_invalid_result()], "retry_count": MAX_RETRIES}
    assert _route_after_validation(state) == "strip_and_finalize"


# --- full graph integration, via run_crag ---

GOV_LAW_TEXT = "This Agreement is governed by the laws of Delaware."
GOV_LAW_CLAIM = {"citation_section_id": "4", "supporting_quote": GOV_LAW_TEXT, "claim_text": GOV_LAW_TEXT}
WRONG_CLAIM = {"citation_section_id": "4", "claim_text": "The initial term is five years."}


class ScriptedLLMClient:
    """The generator is the only LLM step inside run_crag. Responses are
    consumed one per call; the last one repeats."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.prompts: list[str] = []

    @property
    def generate_calls(self):
        return len(self.prompts)

    def generate_json(self, system_prompt, user_prompt):
        assert "legal contract analysis assistant" in system_prompt, system_prompt[:80]
        self.prompts.append(user_prompt)
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


class FixedVectorSearchClient:
    def __init__(self, results):
        self._results = results
        self.queries: list[str] = []

    def search(self, query_text, *, filters, top_k):
        self.queries.append(query_text)
        return self._results[:top_k]


class EmptyBM25Index:
    def search(self, query_text, *, top_k):
        return []


class PassthroughReranker:
    def rerank(self, query_text, candidates, *, top_k):
        return [c.model_copy(update={"source": "hybrid+rerank"}) for c in candidates[:top_k]]


def _permission_context():
    return PermissionContext(tenant_id="tenant-1", matter_id="matter-1")


def _query_result(clauses, jailbreak_flagged=False, route=RouteType.CLAUSE_LOOKUP):
    return QueryResult(
        query_text="What law governs this Agreement?",
        route=route,
        jailbreak_check=JailbreakClassification(is_flagged=jailbreak_flagged, reason="model text" if jailbreak_flagged else None),
        retrieved_clauses=clauses,
    )


def _gov_law_clause():
    return make_clause("doc::c4", section_id="4", heading="Governing Law", text=GOV_LAW_TEXT)


def _shown(clause=None, score=1.0):
    return RetrievedClause(clause=clause or _gov_law_clause(), metadata=make_metadata(), score=score, source="hybrid")


def _common_kwargs(llm_client, vector_client=None, source_clauses=None):
    return dict(
        source_clauses=source_clauses or [_gov_law_clause()],
        all_clauses_by_section_id={},
        references=[],
        llm_client=llm_client,
        vector_client=vector_client or FixedVectorSearchClient([]),
        bm25_index=EmptyBM25Index(),
        reranker=PassthroughReranker(),
        permission_context=_permission_context(),
        query_filters=QueryFilters(),
    )


def test_sufficient_and_valid_citation_is_answered_in_one_call():
    llm_client = ScriptedLLMClient({"sufficiency": "sufficient", "claims": [GOV_LAW_CLAIM]})

    final_state = run_crag(_query_result([_shown()]), **_common_kwargs(llm_client))

    assert final_state["final_status"] == "answered"
    assert final_state["stripped_claims"] == []
    assert llm_client.generate_calls == 1  # grading + answering is one call


def test_missing_grade_is_treated_as_ambiguous_not_sufficient():
    llm_client = ScriptedLLMClient({"claims": [GOV_LAW_CLAIM]})
    final_state = run_crag(_query_result([_shown()]), **_common_kwargs(llm_client))
    assert final_state["final_status"] == "flagged_for_review"
    assert "Ambiguous" in final_state["flagged_reason"]


def test_ambiguous_flags_for_review():
    llm_client = ScriptedLLMClient({"sufficiency": "ambiguous", "claims": [GOV_LAW_CLAIM]})
    final_state = run_crag(_query_result([_shown()]), **_common_kwargs(llm_client))
    assert final_state["final_status"] == "flagged_for_review"
    assert "Ambiguous" in final_state["flagged_reason"]
    assert llm_client.generate_calls == 1


def test_jailbreak_flagged_query_short_circuits_with_a_fixed_message():
    class ExplodingLLMClient:
        def generate_json(self, *args, **kwargs):
            raise AssertionError("graph must not run for a flagged query")

    final_state = run_crag(_query_result([], jailbreak_flagged=True), **_common_kwargs(ExplodingLLMClient()))

    assert final_state["final_status"] == "flagged_for_review"
    # B9: never echo the classifier's (model-written) reason to the client.
    assert final_state["flagged_reason"] == BLOCKED_AT_INPUT_REASON


def test_persistently_insufficient_context_is_not_found_after_retries():
    off_topic = _shown(make_clause("doc::c9", section_id="9", text="Some retrieved but unhelpful clause."))
    llm_client = ScriptedLLMClient(
        {"sufficiency": "insufficient", "search_query": "governing law choice of law"},
        {"sufficiency": "insufficient", "search_query": "laws of which state"},
        {"sufficiency": "insufficient", "search_query": "yet another"},
    )
    vector_client = FixedVectorSearchClient([off_topic])

    final_state = run_crag(_query_result([off_topic]), **_common_kwargs(llm_client, vector_client))

    assert final_state["final_status"] == "not_found"  # B11
    assert final_state["final_answer_text"] == NOT_FOUND_TEXT
    assert final_state["retry_count"] == MAX_RETRIES
    assert llm_client.generate_calls == MAX_RETRIES + 1  # no separate grade / rewrite calls
    assert vector_client.queries == ["governing law choice of law", "laws of which state"]


def test_after_a_rewrite_the_original_question_is_answered_and_evidence_is_added():
    """B2 + B3: the rewritten search query only drives retrieval; the
    generator still answers the user's question, over old + new evidence."""
    first = _shown(make_clause("doc::c1", section_id="1", text="Definitions clause."), score=0.9)
    gov = _shown(score=0.5)
    llm_client = ScriptedLLMClient(
        {"sufficiency": "insufficient", "search_query": "choice of law"},
        {"sufficiency": "sufficient", "claims": [GOV_LAW_CLAIM]},
    )
    final_state = run_crag(
        _query_result([first]),
        **_common_kwargs(llm_client, FixedVectorSearchClient([gov]), source_clauses=[first.clause, gov.clause]),
    )

    assert final_state["final_status"] == "answered"
    assert "Question: What law governs this Agreement?" in llm_client.prompts[1]
    assert "choice of law" not in llm_client.prompts[1]
    shown_ids = [rc.clause.clause_id for rc in final_state["retrieved_clauses"]]
    assert shown_ids == ["doc::c1", "doc::c4"]
    assert any(step.stage.startswith("rewrite1:") for step in final_state["trace"])


def test_rewrite_path_enforces_permission():
    """B1: the rewrite branch used to skip the permission assertion, so a
    clause leaked by the store reached the next prompt."""
    leaked = RetrievedClause(
        clause=make_clause("doc::c7", section_id="7", text="Other tenant's clause."),
        metadata=make_metadata(tenant_id="tenant-2"),
        score=1.0,
        source="vector",
    )
    llm_client = ScriptedLLMClient({"sufficiency": "insufficient", "search_query": "something else"})
    audit_logger = InMemoryAuditLogger()

    with pytest.raises(PermissionViolation):
        run_crag(
            _query_result([_shown()]),
            **_common_kwargs(llm_client, FixedVectorSearchClient([leaked])),
            audit_logger=audit_logger,
        )
    assert llm_client.generate_calls == 1
    assert all("Other tenant" not in p for p in llm_client.prompts)
    assert "permission_violation" in [e.event_type for e in audit_logger.events]


def test_persistently_invalid_citations_exhaust_retries_then_flag_when_nothing_survives():
    llm_client = ScriptedLLMClient({"sufficiency": "sufficient", "claims": [WRONG_CLAIM]})
    vector_client = FixedVectorSearchClient([_shown()])

    final_state = run_crag(_query_result([_shown()]), **_common_kwargs(llm_client, vector_client))

    # Every claim stripped -> nothing verified is left to answer with.
    assert final_state["final_status"] == "flagged_for_review"
    assert final_state["final_answer_text"] is None
    assert final_state["retry_count"] == MAX_RETRIES
    assert len(final_state["stripped_claims"]) == 1
    assert final_state["kept_claims"] == []
    # Citation failures regenerate against the same evidence; they don't
    # retrieve again.
    assert vector_client.queries == []
    assert llm_client.generate_calls == MAX_RETRIES + 1


def test_best_attempt_is_kept_when_a_regeneration_does_worse():
    llm_client = ScriptedLLMClient(
        {"sufficiency": "sufficient", "claims": [GOV_LAW_CLAIM, WRONG_CLAIM]},
        {"sufficiency": "sufficient", "claims": [WRONG_CLAIM]},
    )
    final_state = run_crag(_query_result([_shown()]), **_common_kwargs(llm_client))
    assert final_state["final_status"] == "answered_with_stripped_claims"
    assert final_state["final_answer_text"] == GOV_LAW_TEXT


def test_ambiguous_path_logs_flagged_for_review_event():
    llm_client = ScriptedLLMClient({"sufficiency": "ambiguous"})
    audit_logger = InMemoryAuditLogger()

    run_crag(_query_result([_shown()]), **_common_kwargs(llm_client), audit_logger=audit_logger)

    assert [e.event_type for e in audit_logger.events] == ["flagged_for_review"]
    assert audit_logger.events[0].tenant_id == "tenant-1"


def test_stripped_citation_path_logs_hashes_not_contract_text():
    llm_client = ScriptedLLMClient({"sufficiency": "sufficient", "claims": [WRONG_CLAIM]})
    audit_logger = InMemoryAuditLogger()

    run_crag(_query_result([_shown()]), **_common_kwargs(llm_client), audit_logger=audit_logger)

    assert [e.event_type for e in audit_logger.events] == ["citation_stripped", "flagged_for_review"]
    details = audit_logger.events[0].details
    assert details["stripped_count"] == 1
    assert "five years" not in str(details)  # S7


def test_no_claims_reason_wins_over_ambiguous_on_a_conflict_route():
    """B19."""
    llm_client = ScriptedLLMClient({"sufficiency": "ambiguous", "claims": []})
    final_state = run_crag(
        _query_result([_shown()], route=RouteType.CONTRADICTION_CHECK), **_common_kwargs(llm_client)
    )
    assert final_state["flagged_reason"].startswith("No verifiable claims")


def test_contradiction_verdict_reaches_the_answer_when_two_clauses_are_grounded():
    """B10: the conflict assessment used to live only in the (never returned)
    free-text answer."""
    term = make_clause("doc::c2", section_id="2", text="Either party may terminate on 30 days notice.")
    lock = make_clause("doc::c3", section_id="3", text="Neither party may terminate before the second anniversary.")
    claims = [
        {"citation_section_id": "2", "supporting_quote": term.text, "claim_text": term.text},
        {"citation_section_id": "3", "supporting_quote": lock.text, "claim_text": lock.text},
    ]
    llm_client = ScriptedLLMClient({"sufficiency": "sufficient", "verdict": "Conflict", "claims": claims})
    final_state = run_crag(
        _query_result([_shown(term), _shown(lock)], route=RouteType.CONTRADICTION_CHECK),
        **_common_kwargs(llm_client, source_clauses=[term, lock]),
    )
    assert final_state["final_status"] == "answered"
    assert final_state["verdict"] == "conflict"
    assert "Assessment: the cited clauses conflict" in final_state["final_answer_text"]


def test_verdict_is_not_rendered_for_a_single_grounded_clause():
    llm_client = ScriptedLLMClient({"sufficiency": "sufficient", "verdict": "conflict", "claims": [GOV_LAW_CLAIM]})
    final_state = run_crag(
        _query_result([_shown()], route=RouteType.CONTRADICTION_CHECK), **_common_kwargs(llm_client)
    )
    assert final_state["final_status"] == "answered"
    assert final_state["verdict"] is None
    assert "Assessment" not in final_state["final_answer_text"]


def test_happy_path_llm_budget_is_one_call_inside_crag():
    counting = CountingLLMClient(ScriptedLLMClient({"sufficiency": "sufficient", "claims": [GOV_LAW_CLAIM]}))
    run_crag(_query_result([_shown()]), **_common_kwargs(counting))
    assert counting.calls == 1


def test_run_crag_without_audit_logger_does_not_raise():
    llm_client = ScriptedLLMClient({"sufficiency": "ambiguous"})
    final_state = run_crag(_query_result([_shown()]), **_common_kwargs(llm_client))
    assert final_state["final_status"] == "flagged_for_review"

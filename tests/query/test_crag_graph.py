from query.filters.audit_log import InMemoryAuditLogger
from query.generation.crag_graph import (
    MAX_RETRIES,
    _route_after_sufficiency,
    _route_after_validation,
    run_crag,
)
from query.generation.sufficiency_evaluator import SufficiencyDecision
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


def test_route_sufficient_goes_to_generate():
    state = {"sufficiency": SufficiencyDecision.SUFFICIENT.value, "retry_count": 0}
    assert _route_after_sufficiency(state) == "generate"


def test_route_ambiguous_goes_to_flag_for_review():
    state = {"sufficiency": SufficiencyDecision.AMBIGUOUS.value, "retry_count": 0}
    assert _route_after_sufficiency(state) == "flag_for_review"


def test_route_insufficient_with_budget_remaining_retries():
    state = {"sufficiency": SufficiencyDecision.INSUFFICIENT.value, "retry_count": 0}
    assert _route_after_sufficiency(state) == "rewrite_and_retrieve"


def test_route_insufficient_with_budget_exhausted_escalates_to_review():
    state = {"sufficiency": SufficiencyDecision.INSUFFICIENT.value, "retry_count": MAX_RETRIES}
    assert _route_after_sufficiency(state) == "flag_for_review"


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


def test_route_no_citations_finalizes():
    state = {"citation_results": [], "retry_count": 0}
    assert _route_after_validation(state) == "finalize_answered"


def test_route_invalid_citation_with_budget_remaining_retries():
    state = {"citation_results": [_valid_result(), _invalid_result()], "retry_count": 0}
    assert _route_after_validation(state) == "rewrite_and_retrieve"


def test_route_invalid_citation_with_budget_exhausted_strips():
    state = {"citation_results": [_invalid_result()], "retry_count": MAX_RETRIES}
    assert _route_after_validation(state) == "strip_and_finalize"


# --- full graph integration, via run_crag ---


class ScriptedLLMClient:
    """Dispatches by system-prompt content. sufficiency_responses is consumed
    one per call (so a test can script "insufficient" on attempt 1 and
    "sufficient" on attempt 2, for example)."""

    def __init__(self, sufficiency_responses, generate_response=None, rewrite_response=None):
        self._sufficiency_responses = list(sufficiency_responses)
        self._generate_response = generate_response
        self._rewrite_response = rewrite_response
        self.sufficiency_calls = 0
        self.generate_calls = 0
        self.rewrite_calls = 0

    def generate_json(self, system_prompt, user_prompt):
        if "Rewrite the question" in system_prompt:
            self.rewrite_calls += 1
            return self._rewrite_response
        if "legal contract analysis assistant" in system_prompt:
            self.generate_calls += 1
            return self._generate_response
        if "enough to answer" in system_prompt:
            self.sufficiency_calls += 1
            return self._sufficiency_responses.pop(0)
        raise AssertionError(f"unexpected system prompt: {system_prompt[:80]}")


class FixedVectorSearchClient:
    def __init__(self, results):
        self._results = results

    def search(self, query_text, *, filters, top_k):
        return self._results[:top_k]


class EmptyBM25Index:
    def search(self, query_text, *, top_k):
        return []


class PassthroughReranker:
    def rerank(self, query_text, candidates, *, top_k):
        return [c.model_copy(update={"source": "hybrid+rerank"}) for c in candidates[:top_k]]


def _permission_context():
    return PermissionContext(tenant_id="tenant-1", matter_id="matter-1")


def _query_result(clauses, jailbreak_flagged=False):
    return QueryResult(
        query_text="What law governs this Agreement?",
        route=RouteType.CLAUSE_LOOKUP,
        jailbreak_check=JailbreakClassification(is_flagged=jailbreak_flagged),
        retrieved_clauses=clauses,
    )


def _common_kwargs(llm_client, vector_results=None):
    gov_law_clause = make_clause(
        "doc::c4", section_id="4", heading="Governing Law", text="This Agreement is governed by the laws of Delaware."
    )
    return dict(
        source_clauses=[gov_law_clause],
        all_clauses_by_section_id={},
        references=[],
        llm_client=llm_client,
        vector_client=FixedVectorSearchClient(vector_results or []),
        bm25_index=EmptyBM25Index(),
        reranker=PassthroughReranker(),
        permission_context=_permission_context(),
        query_filters=QueryFilters(),
    )


def test_sufficient_and_valid_citation_reaches_answered():
    gov_law_clause = make_clause(
        "doc::c4", section_id="4", heading="Governing Law", text="This Agreement is governed by the laws of Delaware."
    )
    shown = [RetrievedClause(clause=gov_law_clause, metadata=make_metadata(), score=1.0, source="hybrid")]

    llm_client = ScriptedLLMClient(
        sufficiency_responses=[{"sufficiency": "sufficient"}],
        generate_response={
            "answer_text": "Delaware law governs.",
            "claims": [{"claim_text": "This Agreement is governed by the laws of Delaware.", "citation_section_id": "4"}],
        },
    )

    final_state = run_crag(_query_result(shown), **_common_kwargs(llm_client))

    assert final_state["final_status"] == "answered"
    assert final_state["stripped_claims"] == []
    assert llm_client.rewrite_calls == 0


def test_ambiguous_flags_for_review_without_generating():
    shown = [
        RetrievedClause(
            clause=make_clause("doc::c4", section_id="4"), metadata=make_metadata(), score=1.0, source="hybrid"
        )
    ]

    class ExplodingGenerateLLM(ScriptedLLMClient):
        pass

    llm_client = ExplodingGenerateLLM(
        sufficiency_responses=[{"sufficiency": "ambiguous"}], generate_response=None
    )

    final_state = run_crag(_query_result(shown), **_common_kwargs(llm_client))

    assert final_state["final_status"] == "flagged_for_review"
    assert "Ambiguous" in final_state["flagged_reason"]
    assert llm_client.generate_calls == 0


def test_jailbreak_flagged_query_short_circuits_without_running_graph():
    class ExplodingLLMClient:
        def generate_json(self, *args, **kwargs):
            raise AssertionError("graph must not run for a flagged query")

    final_state = run_crag(
        _query_result([], jailbreak_flagged=True), **_common_kwargs(ExplodingLLMClient())
    )

    assert final_state["final_status"] == "flagged_for_review"
    assert "Blocked at input" in final_state["flagged_reason"]


def test_persistently_insufficient_context_exhausts_retries_then_flags():
    """Non-empty retrieved_clauses throughout, so evaluate_sufficiency
    actually calls the LLM each time rather than short-circuiting on an
    empty list (that shortcut is covered separately in
    test_generation_units.py) -- this exercises the scripted "insufficient"
    branch specifically."""
    off_topic_clause = make_clause("doc::c9", section_id="9", text="Some retrieved but unhelpful clause.")
    shown = [RetrievedClause(clause=off_topic_clause, metadata=make_metadata(), score=1.0, source="hybrid")]

    llm_client = ScriptedLLMClient(
        sufficiency_responses=[{"sufficiency": "insufficient"}] * (MAX_RETRIES + 1),
        rewrite_response={"rewritten_query": "a different phrasing"},
    )

    final_state = run_crag(_query_result(shown), **_common_kwargs(llm_client, vector_results=shown))

    assert final_state["final_status"] == "flagged_for_review"
    assert final_state["retry_count"] == MAX_RETRIES
    assert llm_client.rewrite_calls == MAX_RETRIES
    assert llm_client.sufficiency_calls == MAX_RETRIES + 1
    assert llm_client.generate_calls == 0


def test_persistently_invalid_citations_exhaust_retries_then_strip():
    off_topic_clause = make_clause(
        "doc::c4", section_id="4", heading="Governing Law", text="This Agreement is governed by the laws of Delaware."
    )
    shown = [RetrievedClause(clause=off_topic_clause, metadata=make_metadata(), score=1.0, source="hybrid")]

    llm_client = ScriptedLLMClient(
        sufficiency_responses=[{"sufficiency": "sufficient"}] * (MAX_RETRIES + 1),
        generate_response={
            "answer_text": "The initial term is five years.",
            # citation exists + in-context, but the clause it points at is
            # about governing law, not term length -- span-support must fail.
            "claims": [{"claim_text": "The initial term is five years.", "citation_section_id": "4"}],
        },
        rewrite_response={"rewritten_query": "a different phrasing"},
    )

    final_state = run_crag(_query_result(shown), **_common_kwargs(llm_client, vector_results=shown))

    assert final_state["final_status"] == "answered_with_stripped_claims"
    assert final_state["retry_count"] == MAX_RETRIES
    assert len(final_state["stripped_claims"]) == 1
    assert final_state["kept_claims"] == []


def test_ambiguous_path_logs_flagged_for_review_event():
    shown = [
        RetrievedClause(
            clause=make_clause("doc::c4", section_id="4"), metadata=make_metadata(), score=1.0, source="hybrid"
        )
    ]
    llm_client = ScriptedLLMClient(sufficiency_responses=[{"sufficiency": "ambiguous"}])
    audit_logger = InMemoryAuditLogger()

    run_crag(_query_result(shown), **_common_kwargs(llm_client), audit_logger=audit_logger)

    assert [e.event_type for e in audit_logger.events] == ["flagged_for_review"]
    assert audit_logger.events[0].tenant_id == "tenant-1"


def test_stripped_citation_path_logs_citation_stripped_event():
    off_topic_clause = make_clause(
        "doc::c4", section_id="4", heading="Governing Law", text="This Agreement is governed by the laws of Delaware."
    )
    shown = [RetrievedClause(clause=off_topic_clause, metadata=make_metadata(), score=1.0, source="hybrid")]
    llm_client = ScriptedLLMClient(
        sufficiency_responses=[{"sufficiency": "sufficient"}] * (MAX_RETRIES + 1),
        generate_response={
            "answer_text": "The initial term is five years.",
            "claims": [{"claim_text": "The initial term is five years.", "citation_section_id": "4"}],
        },
        rewrite_response={"rewritten_query": "a different phrasing"},
    )
    audit_logger = InMemoryAuditLogger()

    run_crag(
        _query_result(shown),
        **_common_kwargs(llm_client, vector_results=shown),
        audit_logger=audit_logger,
    )

    assert [e.event_type for e in audit_logger.events] == ["citation_stripped"]
    assert audit_logger.events[0].details["stripped_claim_texts"] == ["The initial term is five years."]


def test_run_crag_without_audit_logger_does_not_raise():
    shown = [
        RetrievedClause(
            clause=make_clause("doc::c4", section_id="4"), metadata=make_metadata(), score=1.0, source="hybrid"
        )
    ]
    llm_client = ScriptedLLMClient(sufficiency_responses=[{"sufficiency": "ambiguous"}])

    final_state = run_crag(_query_result(shown), **_common_kwargs(llm_client))
    assert final_state["final_status"] == "flagged_for_review"

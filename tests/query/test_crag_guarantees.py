"""End-to-end guarantees of the CRAG graph that the original design missed:
only validated claims reach the user, regeneration uses validator feedback,
and the router's decision actually changes behavior."""

from query.generation.crag_graph import _route_after_sufficiency, run_crag
from query.schemas import (
    JailbreakClassification,
    PermissionContext,
    QueryFilters,
    QueryResult,
    RetrievedClause,
    RouteType,
)
from tests.query.factories import make_clause, make_metadata

TERM = make_clause(
    "doc::clause-003",
    section_id="3",
    heading="Termination",
    text="Either party may terminate this Agreement for cause upon thirty (30) days written notice.",
)
LAW = make_clause(
    "doc::clause-004",
    section_id="4",
    heading="Governing Law",
    text="This Agreement is governed by the laws of the State of Delaware.",
)


def _shown(*clauses):
    return [RetrievedClause(clause=c, metadata=make_metadata(), score=1.0, source="hybrid") for c in clauses]


class SequencedLLM:
    """Scripted responses per call kind, consumed in order; the last one
    repeats. Records the user prompts sent to the generator."""

    def __init__(self, *, sufficiency, generate):
        self._sufficiency = list(sufficiency)
        self._generate = list(generate)
        self.generate_prompts: list[str] = []

    def generate_json(self, system_prompt, user_prompt):
        if "legal contract analysis assistant" in system_prompt:
            self.generate_prompts.append(user_prompt)
            return self._generate.pop(0) if len(self._generate) > 1 else self._generate[0]
        if "enough to answer" in system_prompt:
            return self._sufficiency.pop(0) if len(self._sufficiency) > 1 else self._sufficiency[0]
        raise AssertionError(f"unexpected prompt: {system_prompt[:60]}")


class _NoRetrieval:
    def search(self, *args, **kwargs):
        raise AssertionError("retrieval must not re-run in this scenario")


class _Passthrough:
    def rerank(self, query_text, candidates, *, top_k):
        return candidates[:top_k]


def _run(llm, shown, route=RouteType.CLAUSE_LOOKUP):
    result = QueryResult(
        query_text="question",
        route=route,
        jailbreak_check=JailbreakClassification(is_flagged=False),
        retrieved_clauses=shown,
    )
    return run_crag(
        result,
        source_clauses=[TERM, LAW],
        all_clauses_by_section_id={},
        references=[],
        llm_client=llm,
        vector_client=_NoRetrieval(),
        bm25_index=_NoRetrieval(),
        reranker=_Passthrough(),
        permission_context=PermissionContext(tenant_id="tenant-1", matter_id="matter-1"),
        query_filters=QueryFilters(),
    )


GOOD_TERM_CLAIM = {
    "claim_text": "Either party may terminate for cause on thirty (30) days written notice.",
    "citation_section_id": "doc::clause-003",
    "supporting_quote": "terminate this Agreement for cause upon thirty (30) days written notice",
}
BAD_TERM_CLAIM = {
    "claim_text": "Either party may terminate for cause on ninety (90) days written notice.",
    "citation_section_id": "doc::clause-003",
    "supporting_quote": "terminate this Agreement for cause upon thirty (30) days written notice",
}


def test_stripped_claim_never_appears_in_returned_answer_text():
    llm = SequencedLLM(
        sufficiency=[{"sufficiency": "sufficient"}],
        generate=[
            {
                # The model's free text contains the unsupported 90-day claim.
                "answer_text": "You can terminate on 90 days notice; Delaware law governs.",
                "claims": [
                    BAD_TERM_CLAIM,
                    {
                        "claim_text": "The Agreement is governed by Delaware law.",
                        "citation_section_id": "doc::clause-004",
                        "supporting_quote": "governed by the laws of the State of Delaware",
                    },
                ],
            }
        ],
    )
    state = _run(llm, _shown(TERM, LAW))

    assert state["final_status"] == "answered_with_stripped_claims"
    assert "90" not in state["final_answer_text"]
    assert "ninety" not in state["final_answer_text"]
    assert state["final_answer_text"] == "The Agreement is governed by Delaware law."


def test_regeneration_receives_validator_feedback_and_can_recover():
    llm = SequencedLLM(
        sufficiency=[{"sufficiency": "sufficient"}],
        generate=[
            {"answer_text": "x", "claims": [BAD_TERM_CLAIM]},
            {"answer_text": "x", "claims": [GOOD_TERM_CLAIM]},
        ],
    )
    state = _run(llm, _shown(TERM))

    assert state["final_status"] == "answered"
    assert state["retry_count"] == 1
    assert len(llm.generate_prompts) == 2
    assert "failed citation validation" in llm.generate_prompts[1]
    assert "number" in llm.generate_prompts[1]  # the specific failure reason is named


def test_uncited_answer_is_flagged_not_answered():
    llm = SequencedLLM(
        sufficiency=[{"sufficiency": "sufficient"}],
        generate=[{"answer_text": "Termination requires 30 days notice.", "claims": []}],
    )
    state = _run(llm, _shown(TERM))

    assert state["final_status"] == "flagged_for_review"
    assert "No verifiable claims" in state["flagged_reason"]


def test_contradiction_check_route_proceeds_on_ambiguous():
    assert (
        _route_after_sufficiency(
            {"sufficiency": "ambiguous", "retry_count": 0, "route": RouteType.CONTRADICTION_CHECK.value}
        )
        == "generate"
    )
    assert (
        _route_after_sufficiency({"sufficiency": "ambiguous", "retry_count": 0, "route": "clause_lookup"})
        == "flag_for_review"
    )


def test_out_of_scope_route_is_flagged_without_calling_the_llm():
    class Exploding:
        def generate_json(self, *a, **k):
            raise AssertionError("no LLM call expected")

    state = _run(Exploding(), [], route=RouteType.AMBIGUOUS_OUT_OF_SCOPE)
    assert state["final_status"] == "flagged_for_review"
    assert "Out of scope" in state["flagged_reason"]

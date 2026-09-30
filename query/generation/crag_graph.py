"""CRAG-style three-way evaluator + generation orchestration — CLAUDE.md #6,
architecture.md B8-B15. Uses LangGraph for the stateful branching/retry
control. Builds on top of query/pipeline.py's process_query (B1-B7); this
module picks up from its output and owns B8 onward: sufficiency check ->
generate -> citation validate -> (regenerate | rewrite+retrieve | strip |
flag | return).

Retry budget: one SHARED counter (MAX_RETRIES) across both retry triggers,
so a query can't retry 2+2 times:

- sufficiency says "insufficient" -> rewrite the query and re-retrieve,
  then re-check sufficiency (a new query can surface different clauses).
- citation validation fails -> REGENERATE against the same evidence, with
  the validator's failure reasons as feedback. The evidence was already
  judged sufficient; a failed citation is almost always a generation error
  (wrong ID, paraphrase drift, misread number), and rewriting the retrieval
  query -- the original design, whose rewrite prompt says "the search didn't
  return enough" -- throws away the good evidence and fixes the wrong thing.

What the user gets back is built ONLY from validated claims
(final_answer_text). The model's free-text answer_text is never returned:
it is exactly where unvalidated content lives -- the original code filtered
the citation list after stripping but still returned the unstripped
answer_text, so stripped claims reached the user anyway.

Terminal states:
- answered: every claim validated.
- answered_with_stripped_claims: budget exhausted, some claims failed and
  were removed, at least one validated claim remains.
- flagged_for_review: ambiguous evidence; insufficient after all retries;
  no verifiable claims at all (an uncited answer is not an answer); every
  claim stripped; out-of-scope route; or jailbreak blocked at input.

Route handling: an out-of-scope query is flagged without generating. For
comparison / contradiction-check routes, "ambiguous" (clauses conflict) is
the thing being asked about, so those routes proceed to generation.

The jailbreak classifier (B1a) already ran in process_query and is NOT
re-run here — a system-generated query rewrite is a reformulation of an
already-cleared original question, not new untrusted input.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from ingestion.metadata.llm_client import LLMClient
from ingestion.schemas import Clause, ClauseMetadata, DefinedTerm, SectionReference
from query.filters.audit_log import AuditEvent, AuditLogger
from query.generation.answer_generator import AnswerGenerationResult, generate_answer
from query.generation.query_rewriter import rewrite_query
from query.generation.sufficiency_evaluator import SufficiencyDecision, evaluate_sufficiency
from query.retrieval.bm25_search import BM25Index
from query.retrieval.hybrid import hybrid_retrieve
from query.retrieval.passage_context import expand_with_sibling_passages
from query.retrieval.reference_expansion import expand_with_references
from query.retrieval.reorder import reorder_for_generation
from query.retrieval.reranker import Reranker
from query.retrieval.vector_search import VectorSearchClient
from query.schemas import PermissionContext, QueryFilters, QueryResult, RetrievedClause, RouteType
from query.validation.citation_validator import (
    Citation,
    CitationValidationResult,
    EntailmentChecker,
    validate_citations,
)
from query.validation.strip_unsupported import Claim, strip_unsupported_claims

MAX_RETRIES = 2

_CONFLICT_ROUTES = {RouteType.COMPARISON.value, RouteType.CONTRADICTION_CHECK.value}


class CragState(TypedDict, total=False):
    original_query_text: str
    current_query_text: str
    route: str
    permission_context: PermissionContext
    query_filters: QueryFilters
    retrieved_clauses: list[RetrievedClause]
    source_clauses: list[Clause]
    all_clauses_by_section_id: dict[str, tuple[Clause, ClauseMetadata]]
    all_clauses_by_id: dict[str, tuple[Clause, ClauseMetadata]]
    references: list[SectionReference]
    defined_terms: list[DefinedTerm]
    retry_count: int
    sufficiency: str
    answer: AnswerGenerationResult | None
    citation_results: list[CitationValidationResult]
    final_status: str  # "answered" | "answered_with_stripped_claims" | "flagged_for_review"
    final_answer_text: str | None
    flagged_reason: str | None
    kept_claims: list[Claim]
    stripped_claims: list[Claim]
    # dependencies threaded through state so every node is a pure function of it
    llm_client: LLMClient
    vector_client: VectorSearchClient
    bm25_index: BM25Index
    reranker: Reranker
    retrieval_top_k: int
    rerank_top_k: int
    audit_logger: AuditLogger | None
    entailment_checker: EntailmentChecker | None


def _route_enum(state: CragState) -> RouteType | None:
    try:
        return RouteType(state.get("route")) if state.get("route") else None
    except ValueError:
        return None


def _node_evaluate_sufficiency(state: CragState) -> dict:
    decision = evaluate_sufficiency(
        state["current_query_text"],
        state["retrieved_clauses"],
        llm_client=state["llm_client"],
        route=_route_enum(state),
    )
    return {"sufficiency": decision.value}


def _route_after_sufficiency(state: CragState) -> str:
    decision = state["sufficiency"]
    if decision == SufficiencyDecision.SUFFICIENT.value:
        return "generate"
    if decision == SufficiencyDecision.AMBIGUOUS.value:
        if state.get("route") in _CONFLICT_ROUTES:
            return "generate"
        return "flag_for_review"
    if state["retry_count"] < MAX_RETRIES:  # INSUFFICIENT
        return "rewrite_and_retrieve"
    return "flag_for_review"


def _node_generate(state: CragState) -> dict:
    answer = generate_answer(
        state["current_query_text"],
        state["retrieved_clauses"],
        llm_client=state["llm_client"],
        route=_route_enum(state) or RouteType.CLAUSE_LOOKUP,
    )
    return {"answer": answer}


def _node_validate_citations(state: CragState) -> dict:
    answer = state["answer"]
    citations = [
        Citation(section_id=c.citation_section_id, claim_text=c.claim_text, supporting_quote=c.supporting_quote)
        for c in answer.claims
    ]
    results = validate_citations(
        citations,
        source_clauses=state["source_clauses"],
        shown_clauses=state["retrieved_clauses"],
        entailment_checker=state.get("entailment_checker"),
    )
    return {"citation_results": results}


def _route_after_validation(state: CragState) -> str:
    results = state["citation_results"]
    if not results:
        # No claims means nothing was verified -- an uncited answer (or a
        # "not addressed" refusal) goes to a human, never out as "answered".
        return "flag_for_review"
    if all(r.is_valid for r in results):
        return "finalize_answered"
    if state["retry_count"] < MAX_RETRIES:
        return "regenerate_with_feedback"
    return "strip_and_finalize"


def _node_regenerate_with_feedback(state: CragState) -> dict:
    feedback = [
        f'Claim "{r.citation.claim_text}" citing "{r.citation.section_id}": {r.failure_reason}'
        for r in state["citation_results"]
        if not r.is_valid
    ]
    answer = generate_answer(
        state["current_query_text"],
        state["retrieved_clauses"],
        llm_client=state["llm_client"],
        route=_route_enum(state) or RouteType.CLAUSE_LOOKUP,
        feedback=feedback,
    )
    return {"answer": answer, "retry_count": state["retry_count"] + 1}


def _node_rewrite_and_retrieve(state: CragState) -> dict:
    new_query = rewrite_query(
        state["original_query_text"], state["current_query_text"], llm_client=state["llm_client"]
    )
    candidates = hybrid_retrieve(
        new_query,
        vector_client=state["vector_client"],
        bm25_index=state["bm25_index"],
        permission_context=state["permission_context"],
        query_filters=state["query_filters"],
        top_k=state.get("retrieval_top_k", 20),
    )
    reranked = state["reranker"].rerank(new_query, candidates, top_k=state.get("rerank_top_k", 5))
    expanded = expand_with_references(
        reranked,
        all_clauses_by_section_id=state["all_clauses_by_section_id"],
        references=state["references"],
        defined_terms=state.get("defined_terms"),
        all_clauses_by_id=state.get("all_clauses_by_id"),
    )
    reordered = reorder_for_generation(expanded)
    with_context = expand_with_sibling_passages(reordered, all_clauses_by_id=state.get("all_clauses_by_id"))
    return {
        "current_query_text": new_query,
        "retrieved_clauses": with_context,
        "retry_count": state["retry_count"] + 1,
    }


def _log(state: CragState, event_type: str, details: dict | None = None) -> None:
    audit_logger = state.get("audit_logger")
    if not audit_logger:
        return
    permission_context = state.get("permission_context")
    audit_logger.log(
        AuditEvent(
            event_type=event_type,
            tenant_id=permission_context.tenant_id if permission_context else None,
            matter_id=permission_context.matter_id if permission_context else None,
            details=details or {},
        )
    )


def _node_flag_for_review(state: CragState) -> dict:
    if state.get("sufficiency") == SufficiencyDecision.AMBIGUOUS.value:
        reason = "Ambiguous: retrieved context supports multiple readings or conflicting clauses."
    elif state.get("answer") is not None and not state.get("citation_results"):
        reason = "No verifiable claims: the answer cited no contract clauses."
    else:
        reason = f"Insufficient context after {state['retry_count']} retry attempt(s)."
    _log(state, "flagged_for_review", {"reason": reason})
    return {"final_status": "flagged_for_review", "flagged_reason": reason, "final_answer_text": None}


def _claims_from_answer(answer: AnswerGenerationResult) -> list[Claim]:
    return [
        Claim(
            claim_text=c.claim_text,
            citation_section_id=c.citation_section_id,
            supporting_quote=c.supporting_quote,
        )
        for c in answer.claims
    ]


def compose_answer_text(claims: list[Claim]) -> str:
    return " ".join(c.claim_text.strip() for c in claims)


def _node_strip_and_finalize(state: CragState) -> dict:
    outcome = strip_unsupported_claims(_claims_from_answer(state["answer"]), state["citation_results"])
    if outcome.stripped:
        _log(
            state,
            "citation_stripped",
            {"stripped_claim_texts": [c.claim_text for c in outcome.stripped]},
        )
    if not outcome.kept:
        reason = "Every claim failed citation validation after all retries."
        _log(state, "flagged_for_review", {"reason": reason})
        return {
            "final_status": "flagged_for_review",
            "flagged_reason": reason,
            "final_answer_text": None,
            "kept_claims": [],
            "stripped_claims": outcome.stripped,
        }
    return {
        "final_status": "answered_with_stripped_claims",
        "final_answer_text": compose_answer_text(outcome.kept),
        "kept_claims": outcome.kept,
        "stripped_claims": outcome.stripped,
    }


def _node_finalize_answered(state: CragState) -> dict:
    kept = _claims_from_answer(state["answer"])
    return {
        "final_status": "answered",
        "final_answer_text": compose_answer_text(kept),
        "kept_claims": kept,
        "stripped_claims": [],
    }


def build_crag_graph():
    graph = StateGraph(CragState)
    graph.add_node("evaluate_sufficiency", _node_evaluate_sufficiency)
    graph.add_node("generate", _node_generate)
    graph.add_node("validate_citations", _node_validate_citations)
    graph.add_node("regenerate_with_feedback", _node_regenerate_with_feedback)
    graph.add_node("rewrite_and_retrieve", _node_rewrite_and_retrieve)
    graph.add_node("flag_for_review", _node_flag_for_review)
    graph.add_node("strip_and_finalize", _node_strip_and_finalize)
    graph.add_node("finalize_answered", _node_finalize_answered)

    graph.set_entry_point("evaluate_sufficiency")
    graph.add_conditional_edges(
        "evaluate_sufficiency",
        _route_after_sufficiency,
        {
            "generate": "generate",
            "flag_for_review": "flag_for_review",
            "rewrite_and_retrieve": "rewrite_and_retrieve",
        },
    )
    graph.add_edge("generate", "validate_citations")
    graph.add_conditional_edges(
        "validate_citations",
        _route_after_validation,
        {
            "finalize_answered": "finalize_answered",
            "regenerate_with_feedback": "regenerate_with_feedback",
            "strip_and_finalize": "strip_and_finalize",
            "flag_for_review": "flag_for_review",
        },
    )
    graph.add_edge("regenerate_with_feedback", "validate_citations")
    graph.add_edge("rewrite_and_retrieve", "evaluate_sufficiency")
    graph.add_edge("flag_for_review", END)
    graph.add_edge("strip_and_finalize", END)
    graph.add_edge("finalize_answered", END)

    return graph.compile()


_GRAPH = None


def _compiled_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_crag_graph()
    return _GRAPH


def run_crag(
    query_result: QueryResult,
    *,
    source_clauses: list[Clause],
    all_clauses_by_section_id: dict[str, tuple[Clause, ClauseMetadata]],
    references: list[SectionReference],
    llm_client: LLMClient,
    vector_client: VectorSearchClient,
    bm25_index: BM25Index,
    reranker: Reranker,
    permission_context: PermissionContext,
    query_filters: QueryFilters,
    retrieval_top_k: int = 20,
    rerank_top_k: int = 5,
    audit_logger: AuditLogger | None = None,
    defined_terms: list[DefinedTerm] | None = None,
    all_clauses_by_id: dict[str, tuple[Clause, ClauseMetadata]] | None = None,
    entailment_checker: EntailmentChecker | None = None,
) -> CragState:
    base = {
        "original_query_text": query_result.query_text,
        "current_query_text": query_result.query_text,
        "route": query_result.route.value,
        "retry_count": 0,
        "final_answer_text": None,
    }
    if query_result.jailbreak_check.is_flagged:
        # Already logged as "jailbreak_blocked" by process_query -- not
        # re-logged here, this is just the CRAG-side reflection of that.
        return {
            **base,
            "final_status": "flagged_for_review",
            "flagged_reason": f"Blocked at input: {query_result.jailbreak_check.reason}",
        }
    if query_result.route == RouteType.AMBIGUOUS_OUT_OF_SCOPE:
        reason = "Out of scope or too vague to answer from the contract; please rephrase."
        state: CragState = {**base, "permission_context": permission_context, "audit_logger": audit_logger}
        _log(state, "flagged_for_review", {"reason": reason})
        return {**base, "final_status": "flagged_for_review", "flagged_reason": reason}

    initial_state: CragState = {
        **base,
        "permission_context": permission_context,
        "query_filters": query_filters,
        "retrieved_clauses": query_result.retrieved_clauses,
        "source_clauses": source_clauses,
        "all_clauses_by_section_id": all_clauses_by_section_id,
        "all_clauses_by_id": all_clauses_by_id or {},
        "references": references,
        "defined_terms": defined_terms or [],
        "llm_client": llm_client,
        "vector_client": vector_client,
        "bm25_index": bm25_index,
        "reranker": reranker,
        "retrieval_top_k": retrieval_top_k,
        "rerank_top_k": rerank_top_k,
        "audit_logger": audit_logger,
        "entailment_checker": entailment_checker,
    }
    return _compiled_graph().invoke(initial_state)

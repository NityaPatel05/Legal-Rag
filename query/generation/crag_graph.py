"""CRAG-style three-way evaluator + generation orchestration — CLAUDE.md #6,
architecture.md B8-B15. Uses LangGraph for the stateful branching/retry
control (architecture.md §8's stated rationale for choosing LangGraph: "adds
the stateful branching/retry control the CRAG evaluator loop and the 'flag
for attorney review' gate need"). Builds on top of query/pipeline.py's
process_query (B1-B7); this module picks up from its output and owns B8
onward: sufficiency check -> generate -> citation validate -> (retry | strip
| return).

Retry budget: CLAUDE.md #4 and #6 both call for a SHARED 2-attempt cap across
two different triggers — the evaluator's own "insufficient, rewrite and
retry" branch (B11) and the citation validator's "invalid, retry budget
remains, re-retrieve with refined query" branch (B15). Both increment the
same retry_count and both loop back through a fresh retrieval AND a fresh
sufficiency check, not just a fresh generation — a rewritten query can
retrieve a completely different set of clauses, so re-checking sufficiency
before generating again is the safer order, even though the architecture
diagram's B15 draws a direct line back to B4 (retrieval) without explicitly
routing through B8 again. Once the cap is exhausted: the insufficient path
escalates to attorney-review (B10) and the invalid-citation path strips
unsupported claims (B14) rather than looping forever or answering unchecked.

The jailbreak classifier (B1a) already ran in process_query and is NOT
re-run here — a system-generated query rewrite is a reformulation of an
already-cleared original question, not new untrusted input. If the original
query was flagged, run_crag short-circuits before the graph even starts.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from ingestion.metadata.llm_client import LLMClient
from ingestion.schemas import Clause, ClauseMetadata, SectionReference
from query.filters.audit_log import AuditEvent, AuditLogger
from query.generation.answer_generator import AnswerGenerationResult, generate_answer
from query.generation.query_rewriter import rewrite_query
from query.generation.sufficiency_evaluator import SufficiencyDecision, evaluate_sufficiency
from query.retrieval.bm25_search import BM25Index
from query.retrieval.hybrid import hybrid_retrieve
from query.retrieval.reference_expansion import expand_with_references
from query.retrieval.reorder import reorder_for_generation
from query.retrieval.reranker import Reranker
from query.retrieval.vector_search import VectorSearchClient
from query.schemas import PermissionContext, QueryFilters, QueryResult, RetrievedClause
from query.validation.citation_validator import Citation, CitationValidationResult, validate_citations
from query.validation.strip_unsupported import Claim, strip_unsupported_claims

MAX_RETRIES = 2


class CragState(TypedDict, total=False):
    original_query_text: str
    current_query_text: str
    permission_context: PermissionContext
    query_filters: QueryFilters
    retrieved_clauses: list[RetrievedClause]
    source_clauses: list[Clause]
    all_clauses_by_section_id: dict[str, tuple[Clause, ClauseMetadata]]
    references: list[SectionReference]
    retry_count: int
    sufficiency: str
    answer: AnswerGenerationResult | None
    citation_results: list[CitationValidationResult]
    final_status: str  # "answered" | "answered_with_stripped_claims" | "flagged_for_review"
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


def _node_evaluate_sufficiency(state: CragState) -> dict:
    decision = evaluate_sufficiency(
        state["current_query_text"], state["retrieved_clauses"], llm_client=state["llm_client"]
    )
    return {"sufficiency": decision.value}


def _route_after_sufficiency(state: CragState) -> str:
    decision = state["sufficiency"]
    if decision == SufficiencyDecision.SUFFICIENT.value:
        return "generate"
    if decision == SufficiencyDecision.AMBIGUOUS.value:
        return "flag_for_review"
    if state["retry_count"] < MAX_RETRIES:  # INSUFFICIENT
        return "rewrite_and_retrieve"
    return "flag_for_review"


def _node_generate(state: CragState) -> dict:
    answer = generate_answer(
        state["current_query_text"], state["retrieved_clauses"], llm_client=state["llm_client"]
    )
    return {"answer": answer}


def _node_validate_citations(state: CragState) -> dict:
    answer = state["answer"]
    citations = [
        Citation(section_id=c.citation_section_id, claim_text=c.claim_text) for c in answer.claims
    ]
    results = validate_citations(
        citations, source_clauses=state["source_clauses"], shown_clauses=state["retrieved_clauses"]
    )
    return {"citation_results": results}


def _route_after_validation(state: CragState) -> str:
    results = state["citation_results"]
    if not results or all(r.is_valid for r in results):
        return "finalize_answered"
    if state["retry_count"] < MAX_RETRIES:
        return "rewrite_and_retrieve"
    return "strip_and_finalize"


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
    )
    reordered = reorder_for_generation(expanded)
    return {
        "current_query_text": new_query,
        "retrieved_clauses": reordered,
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
    else:
        reason = f"Insufficient context after {state['retry_count']} retry attempt(s)."
    _log(state, "flagged_for_review", {"reason": reason})
    return {"final_status": "flagged_for_review", "flagged_reason": reason}


def _node_strip_and_finalize(state: CragState) -> dict:
    answer = state["answer"]
    claims = [
        Claim(claim_text=c.claim_text, citation_section_id=c.citation_section_id) for c in answer.claims
    ]
    outcome = strip_unsupported_claims(claims, state["citation_results"])
    if outcome.stripped:
        _log(
            state,
            "citation_stripped",
            {"stripped_claim_texts": [c.claim_text for c in outcome.stripped]},
        )
    return {
        "final_status": "answered_with_stripped_claims",
        "kept_claims": outcome.kept,
        "stripped_claims": outcome.stripped,
    }


def _node_finalize_answered(state: CragState) -> dict:
    answer = state["answer"]
    return {
        "final_status": "answered",
        "kept_claims": [
            Claim(claim_text=c.claim_text, citation_section_id=c.citation_section_id)
            for c in answer.claims
        ],
        "stripped_claims": [],
    }


def build_crag_graph():
    graph = StateGraph(CragState)
    graph.add_node("evaluate_sufficiency", _node_evaluate_sufficiency)
    graph.add_node("generate", _node_generate)
    graph.add_node("validate_citations", _node_validate_citations)
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
            "rewrite_and_retrieve": "rewrite_and_retrieve",
            "strip_and_finalize": "strip_and_finalize",
        },
    )
    graph.add_edge("rewrite_and_retrieve", "evaluate_sufficiency")
    graph.add_edge("flag_for_review", END)
    graph.add_edge("strip_and_finalize", END)
    graph.add_edge("finalize_answered", END)

    return graph.compile()


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
) -> CragState:
    if query_result.jailbreak_check.is_flagged:
        # Already logged as "jailbreak_blocked" by process_query -- not
        # re-logged here, this is just the CRAG-side reflection of that.
        return {
            "original_query_text": query_result.query_text,
            "current_query_text": query_result.query_text,
            "retry_count": 0,
            "final_status": "flagged_for_review",
            "flagged_reason": f"Blocked at input: {query_result.jailbreak_check.reason}",
        }

    graph = build_crag_graph()
    initial_state: CragState = {
        "original_query_text": query_result.query_text,
        "current_query_text": query_result.query_text,
        "permission_context": permission_context,
        "query_filters": query_filters,
        "retrieved_clauses": query_result.retrieved_clauses,
        "source_clauses": source_clauses,
        "all_clauses_by_section_id": all_clauses_by_section_id,
        "references": references,
        "retry_count": 0,
        "llm_client": llm_client,
        "vector_client": vector_client,
        "bm25_index": bm25_index,
        "reranker": reranker,
        "retrieval_top_k": retrieval_top_k,
        "rerank_top_k": rerank_top_k,
        "audit_logger": audit_logger,
    }
    return graph.invoke(initial_state)

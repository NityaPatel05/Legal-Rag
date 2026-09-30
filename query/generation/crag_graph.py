"""CRAG-style evaluator + generation orchestration — CLAUDE.md #6,
architecture.md B8-B15. Uses LangGraph for the stateful branching/retry
control. Builds on top of query/pipeline.py's process_query (B1-B7); this
module picks up from its output and owns B8 onward: generate (which also
grades the evidence) -> citation validate -> (regenerate | rewrite+retrieve
| strip | flag | not found | return).

LLM calls per query. Grading the evidence used to be its own call, and an
"insufficient" grade cost another call to rewrite the query, so a /query
made 4 calls on the happy path (screen, route, grade, generate) and up to ~8
with retries. The input screen is now one call (query_screen.py) and the
generator returns the grade and, when insufficient, the next search query
(answer_generator.py): 2 calls on the happy path, at most 2 + MAX_RETRIES.

Retry budget: one SHARED counter (MAX_RETRIES) across both retry triggers,
so a query can't retry 2+2 times:

- the generator grades the evidence "insufficient" -> retrieve again with
  its suggested search query and generate again. The new evidence is ADDED
  to what was already retrieved (review finding B3: it used to replace it,
  discarding clauses already judged relevant), and the answer is always
  generated for the user's ORIGINAL question (B2: it used to answer the
  rewritten -- deliberately "more general" -- search query instead). The
  search query is model output influenced by contract text, but it only
  ever reaches the retriever, which is permission- and document-scoped
  (query/pipeline.retrieve), never a prompt as an instruction.
- citation validation fails -> REGENERATE against the same evidence, with
  the validator's failure reasons as feedback. A failed citation is almost
  always a generation error (wrong ID, paraphrase drift, misread number);
  retrieving again would fix the wrong thing.

The best attempt (most validated claims) is kept across retries, so a
regeneration that does worse can't turn a partly-answered question into a
flag.

What the user gets back is built ONLY from validated claims
(final_answer_text), plus -- for comparison / contradiction routes -- a
fixed sentence rendered from the generator's closed-set `verdict` when the
kept claims cite at least two different clauses (B10). The model's
free-text answer_text is never returned.

Terminal states:
- answered: every claim validated.
- answered_with_stripped_claims: budget exhausted, some claims failed and
  were removed, at least one validated claim remains.
- not_found: every retrieval attempt was graded insufficient and nothing
  was claimed (B11: an absence question -- "is there a non-compete?" --
  used to be indistinguishable from a failure). The answer is a fixed,
  deterministic statement about the search, not model text, and the case
  is still queued for review.
- flagged_for_review: ambiguous evidence; no verifiable claims at all (an
  uncited answer is not an answer); every claim stripped; out-of-scope
  route; or blocked at input.

Route handling: an out-of-scope query is flagged without generating. For
comparison / contradiction-check routes, "ambiguous" (clauses conflict) is
the thing being asked about, so those routes proceed to validation.
"""

from __future__ import annotations

import hashlib
from typing import TypedDict

from langgraph.graph import END, StateGraph

from ingestion.metadata.llm_client import LLMClient
from ingestion.schemas import Clause, ClauseMetadata, DefinedTerm, SectionReference
from query.filters.audit_log import AuditEvent, AuditLogger
from query.generation.answer_generator import (
    AnswerGenerationResult,
    Sufficiency,
    Verdict,
    generate_answer,
)
from query.pipeline import retrieve
from query.retrieval.bm25_search import BM25Index
from query.retrieval.reranker import Reranker
from query.retrieval.vector_search import VectorSearchClient
from query.schemas import (
    PermissionContext,
    QueryFilters,
    QueryResult,
    RetrievalTraceStep,
    RetrievedClause,
    RouteType,
)
from query.validation.citation_validator import (
    Citation,
    CitationValidationResult,
    EntailmentChecker,
    validate_citations,
)
from query.validation.strip_unsupported import Claim, strip_unsupported_claims

MAX_RETRIES = 2
# Evidence cap after a rewrite adds to the retrieved set (~1000-char passages).
MAX_EVIDENCE_CLAUSES = 40

BLOCKED_AT_INPUT_REASON = "Blocked at input: the question was flagged by the input screen."
NOT_FOUND_TEXT = (
    "No clause addressing this question was found in the searched text of this contract. "
    "This is a statement about the search, not a legal conclusion that the contract is silent."
)
_VERDICT_TEXT = {
    Verdict.CONFLICT: "the cited clauses conflict",
    Verdict.NO_CONFLICT: "the cited clauses do not conflict",
    Verdict.DIFFER: "the cited clauses differ",
    Verdict.EQUIVALENT: "the cited clauses are equivalent in substance",
}

_CONFLICT_ROUTES = {RouteType.COMPARISON.value, RouteType.CONTRADICTION_CHECK.value}


class CragState(TypedDict, total=False):
    original_query_text: str
    current_query_text: str
    tried_queries: list[str]
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
    best_answer: AnswerGenerationResult | None
    best_results: list[CitationValidationResult]
    trace: list[RetrievalTraceStep]
    final_status: str  # "answered" | "answered_with_stripped_claims" | "not_found" | "flagged_for_review"
    final_answer_text: str | None
    flagged_reason: str | None
    verdict: str | None
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


def _route_enum(state: CragState) -> RouteType:
    try:
        return RouteType(state.get("route")) if state.get("route") else RouteType.CLAUSE_LOOKUP
    except ValueError:
        return RouteType.CLAUSE_LOOKUP


def _normalized_query(text: str | None) -> str:
    return " ".join((text or "").lower().split())


# --- nodes ------------------------------------------------------------------


def _node_generate(state: CragState) -> dict:
    if not state["retrieved_clauses"]:
        # Nothing retrieved (e.g. a governing-law filter excluded every
        # clause): there is nothing to ground a claim in, so no call is made.
        answer = AnswerGenerationResult(answer_text="", claims=[], sufficiency=Sufficiency.INSUFFICIENT)
        return {"answer": answer, "sufficiency": answer.sufficiency.value}
    answer = generate_answer(
        state["original_query_text"],
        state["retrieved_clauses"],
        llm_client=state["llm_client"],
        route=_route_enum(state),
    )
    return {"answer": answer, "sufficiency": answer.sufficiency.value}


def _can_rewrite(state: CragState) -> bool:
    answer = state.get("answer")
    if answer is None or state.get("retry_count", 0) >= MAX_RETRIES:
        return False
    query = _normalized_query(answer.search_query)
    return bool(query) and query not in state.get("tried_queries", [])


def _route_after_generate(state: CragState) -> str:
    answer = state["answer"]
    if answer.sufficiency == Sufficiency.INSUFFICIENT:
        if _can_rewrite(state):
            return "rewrite_and_retrieve"
        return "validate_citations" if answer.claims else "not_found"
    if answer.sufficiency == Sufficiency.AMBIGUOUS and state.get("route") not in _CONFLICT_ROUTES:
        return "flag_for_review"
    return "validate_citations"


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
    update: dict = {"citation_results": results}
    valid = sum(r.is_valid for r in results)
    best = state.get("best_results")
    if best is None or valid > sum(r.is_valid for r in best):
        update.update(best_answer=answer, best_results=results)
    return update


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
        state["original_query_text"],
        state["retrieved_clauses"],
        llm_client=state["llm_client"],
        route=_route_enum(state),
        feedback=feedback,
    )
    return {"answer": answer, "retry_count": state["retry_count"] + 1}


def _merge_evidence(old: list[RetrievedClause], new: list[RetrievedClause]) -> list[RetrievedClause]:
    """Union by clause_id, best score wins; earlier evidence keeps its place
    and new clauses follow, capped at MAX_EVIDENCE_CLAUSES."""
    merged: dict[str, RetrievedClause] = {}
    for rc in [*old, *new]:
        current = merged.get(rc.clause.clause_id)
        if current is None:
            merged[rc.clause.clause_id] = rc
        elif rc.score > current.score:
            merged[rc.clause.clause_id] = rc.model_copy(update={"source": current.source})
    return list(merged.values())[:MAX_EVIDENCE_CLAUSES]


def _node_rewrite_and_retrieve(state: CragState) -> dict:
    new_query = state["answer"].search_query or state["current_query_text"]
    attempt = state["retry_count"] + 1
    retrieved, trace = retrieve(
        new_query,
        permission_context=state["permission_context"],
        query_filters=state["query_filters"],
        vector_client=state["vector_client"],
        bm25_index=state["bm25_index"],
        reranker=state["reranker"],
        all_clauses_by_section_id=state["all_clauses_by_section_id"],
        references=state["references"],
        defined_terms=state.get("defined_terms"),
        all_clauses_by_id=state.get("all_clauses_by_id"),
        retrieval_top_k=state.get("retrieval_top_k", 20),
        rerank_top_k=state.get("rerank_top_k", 5),
        audit_logger=state.get("audit_logger"),
        stage_prefix=f"rewrite{attempt}:",
    )
    return {
        "current_query_text": new_query,
        "tried_queries": [*state.get("tried_queries", []), _normalized_query(new_query)],
        "retrieved_clauses": _merge_evidence(state["retrieved_clauses"], retrieved),
        "trace": [*state.get("trace", []), *trace],
        "retry_count": attempt,
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
    # Order matters (B19): a comparison graded "ambiguous" that then made no
    # claims failed for lack of claims, not because of the ambiguity.
    if "citation_results" in state and not state.get("citation_results"):
        reason = "No verifiable claims: the answer cited no contract clauses."
    elif state.get("sufficiency") == Sufficiency.AMBIGUOUS.value:
        reason = "Ambiguous: retrieved context supports multiple readings or conflicting clauses."
    else:
        reason = f"Insufficient context after {state.get('retry_count', 0)} retry attempt(s)."
    _log(state, "flagged_for_review", {"reason": reason})
    return {"final_status": "flagged_for_review", "flagged_reason": reason, "final_answer_text": None}


def _node_not_found(state: CragState) -> dict:
    _log(state, "not_found", {"retry_count": state.get("retry_count", 0)})
    return {
        "final_status": "not_found",
        "final_answer_text": NOT_FOUND_TEXT,
        "flagged_reason": None,
        "kept_claims": [],
        "stripped_claims": [],
    }


def _claims_from_answer(answer: AnswerGenerationResult) -> list[Claim]:
    return [
        Claim(
            claim_text=c.claim_text,
            citation_section_id=c.citation_section_id,
            supporting_quote=c.supporting_quote,
        )
        for c in answer.claims
    ]


def compose_answer_text(claims: list[Claim], verdict_sentence: str | None = None) -> str:
    text = " ".join(c.claim_text.strip() for c in claims)
    return f"{text} {verdict_sentence}".strip() if verdict_sentence else text


def _verdict(state: CragState, answer: AnswerGenerationResult, results: list[CitationValidationResult]) -> Verdict | None:
    """The closed-set verdict, only for conflict routes and only when the
    kept claims ground at least two different clauses -- a "conflict"
    resting on one clause isn't a comparison."""
    if state.get("route") not in _CONFLICT_ROUTES or answer.verdict is None:
        return None
    cited = {r.resolved_clause_id or r.citation.section_id for r in results if r.is_valid}
    return answer.verdict if len(cited) >= 2 else None


def _verdict_sentence(verdict: Verdict | None) -> str | None:
    if verdict is None:
        return None
    return f"Assessment: {_VERDICT_TEXT[verdict]} (model assessment of the quoted clauses; verify before relying on it)."


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _node_strip_and_finalize(state: CragState) -> dict:
    answer = state.get("best_answer") or state["answer"]
    results = state.get("best_results") or state["citation_results"]
    outcome = strip_unsupported_claims(_claims_from_answer(answer), results)
    if outcome.stripped:
        # Hashes, not claim text: the audit log is not a store for contract
        # content (review finding S7).
        _log(
            state,
            "citation_stripped",
            {"stripped_count": len(outcome.stripped), "stripped_claim_hashes": [_hash(c.claim_text) for c in outcome.stripped]},
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
            "citation_results": results,
        }
    verdict = _verdict(state, answer, results)
    return {
        "final_status": "answered_with_stripped_claims",
        "final_answer_text": compose_answer_text(outcome.kept, _verdict_sentence(verdict)),
        "verdict": verdict.value if verdict else None,
        "kept_claims": outcome.kept,
        "stripped_claims": outcome.stripped,
        "citation_results": results,
    }


def _node_finalize_answered(state: CragState) -> dict:
    kept = _claims_from_answer(state["answer"])
    verdict = _verdict(state, state["answer"], state["citation_results"])
    return {
        "final_status": "answered",
        "final_answer_text": compose_answer_text(kept, _verdict_sentence(verdict)),
        "verdict": verdict.value if verdict else None,
        "kept_claims": kept,
        "stripped_claims": [],
    }


def build_crag_graph():
    graph = StateGraph(CragState)
    graph.add_node("generate", _node_generate)
    graph.add_node("validate_citations", _node_validate_citations)
    graph.add_node("regenerate_with_feedback", _node_regenerate_with_feedback)
    graph.add_node("rewrite_and_retrieve", _node_rewrite_and_retrieve)
    graph.add_node("flag_for_review", _node_flag_for_review)
    graph.add_node("not_found", _node_not_found)
    graph.add_node("strip_and_finalize", _node_strip_and_finalize)
    graph.add_node("finalize_answered", _node_finalize_answered)

    graph.set_entry_point("generate")
    graph.add_conditional_edges(
        "generate",
        _route_after_generate,
        {
            "validate_citations": "validate_citations",
            "rewrite_and_retrieve": "rewrite_and_retrieve",
            "flag_for_review": "flag_for_review",
            "not_found": "not_found",
        },
    )
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
    graph.add_edge("rewrite_and_retrieve", "generate")
    graph.add_edge("flag_for_review", END)
    graph.add_edge("not_found", END)
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
    base: CragState = {
        "original_query_text": query_result.query_text,
        "current_query_text": query_result.query_text,
        "tried_queries": [_normalized_query(query_result.query_text)],
        "route": query_result.route.value,
        "retry_count": 0,
        "final_answer_text": None,
        "trace": list(query_result.trace),
    }
    if query_result.jailbreak_check.is_flagged:
        # Already logged as "jailbreak_blocked" (with the classifier's reason)
        # by process_query. The client gets a fixed message: echoing model
        # text back would give the screen an output channel (B9).
        return {**base, "final_status": "flagged_for_review", "flagged_reason": BLOCKED_AT_INPUT_REASON}
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

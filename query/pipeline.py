"""process_query orchestrates the query-time baseline — README build-order
step 3 / architecture.md B1-B7: input sanitization -> input screen
(jailbreak classifier + router, one call) -> permission filter ->
jurisdiction/governing-law filter -> hybrid retrieval -> rerank ->
cross-reference expansion -> reorder -> sibling passages. Stops here, before
generation; query/generation/crag_graph.py's run_crag takes the QueryResult
from there. An out-of-scope route skips retrieval entirely.

If the screen flags the query, the pipeline short-circuits before the
permission filter or vector store are touched at all — architecture.md
§9.3: "runs once, at input, before anything else touches the pipeline."

retrieve() is the ONLY retrieval path: process_query and the CRAG rewrite
branch both call it (review finding B1 -- the rewrite branch used to re-run
search, rerank and expansion itself, without the permission assertion, the
document scope check, the injection-flag audit or trace steps, so a clause
leaked by a misconfigured store on the second retrieval reached the
prompt unchecked).

Structured PII redaction (CLAUDE.md #8) runs unconditionally, before
anything else — including the screen call — so an SSN or account number
embedded in a query never reaches a third-party model provider at all,
flagged or not. It's regex-only (no LLM call); unstructured (LLM-based) PII
detection is available in query/filters/pii_sanitizer.py but is an opt-in
pre-step a caller runs before process_query, not auto-wired here — see that
module's docstring for why.

audit_logger and trace_exporter are optional dependencies (default: don't
log / don't export) so adding either never breaks an existing caller."""

from __future__ import annotations

from ingestion.metadata.llm_client import LLMClient
from ingestion.schemas import Clause, ClauseMetadata, DefinedTerm, SectionReference
from query.filters.audit_log import AuditEvent, AuditLogger
from query.filters.permission import PermissionViolation, assert_permission
from query.filters.pii_sanitizer import redact_structured_pii
from query.filters.query_screen import screen_query
from query.observability.langsmith_tracer import NullTraceExporter, TraceExporter
from query.retrieval.bm25_search import BM25Index
from query.retrieval.hybrid import hybrid_retrieve
from query.retrieval.passage_context import expand_with_sibling_passages
from query.retrieval.reference_expansion import expand_with_references
from query.retrieval.reorder import reorder_for_generation
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


def _clause_ids(clauses) -> list[str]:
    return [rc.clause.clause_id for rc in clauses]


def retrieve(
    query_text: str,
    *,
    permission_context: PermissionContext,
    query_filters: QueryFilters,
    vector_client: VectorSearchClient,
    bm25_index: BM25Index,
    reranker: Reranker,
    all_clauses_by_section_id: dict[str, tuple[Clause, ClauseMetadata]],
    references: list[SectionReference],
    defined_terms: list[DefinedTerm] | None = None,
    all_clauses_by_id: dict[str, tuple[Clause, ClauseMetadata]] | None = None,
    retrieval_top_k: int = 20,
    rerank_top_k: int = 5,
    audit_logger: AuditLogger | None = None,
    stage_prefix: str = "",
) -> tuple[list[RetrievedClause], list[RetrievalTraceStep]]:
    """hybrid -> rerank -> reference expansion -> reorder -> sibling
    passages, asserting tenant / matter / document scope after every stage
    that can introduce clauses. Returns (clauses, trace)."""
    trace: list[RetrievalTraceStep] = []
    document_id = query_filters.document_id

    def step(stage: str, clauses) -> None:
        trace.append(RetrievalTraceStep(stage=f"{stage_prefix}{stage}", clause_ids=_clause_ids(clauses)))

    candidates = hybrid_retrieve(
        query_text,
        vector_client=vector_client,
        bm25_index=bm25_index,
        permission_context=permission_context,
        query_filters=query_filters,
        top_k=retrieval_top_k,
    )
    step("hybrid_retrieve", candidates)
    _assert_permission_and_audit(candidates, permission_context, audit_logger, document_id)

    reranked = reranker.rerank(query_text, candidates, top_k=rerank_top_k)
    step("rerank", reranked)

    expanded = expand_with_references(
        reranked,
        all_clauses_by_section_id=all_clauses_by_section_id,
        references=references,
        defined_terms=defined_terms,
        all_clauses_by_id=all_clauses_by_id,
    )
    step("reference_expansion", expanded)
    _assert_permission_and_audit(expanded, permission_context, audit_logger, document_id)

    reordered = reorder_for_generation(expanded)
    step("reorder", reordered)

    with_context = expand_with_sibling_passages(reordered, all_clauses_by_id=all_clauses_by_id)
    step("passage_context", with_context)
    _assert_permission_and_audit(with_context, permission_context, audit_logger, document_id)

    # Clauses flagged at ingestion as instruction-like (injection_scan.py) are
    # still shown -- they're evidence -- but the reviewer sees which ones.
    flagged = [rc.clause.clause_id for rc in with_context if rc.metadata.injection_suspected]
    if flagged:
        trace.append(RetrievalTraceStep(stage=f"{stage_prefix}injection_flagged", clause_ids=flagged))
        if audit_logger:
            audit_logger.log(
                AuditEvent(
                    event_type="injection_suspected_clause_retrieved",
                    tenant_id=permission_context.tenant_id,
                    matter_id=permission_context.matter_id,
                    details={"clause_ids": flagged},
                )
            )
    return with_context, trace


def process_query(
    query_text: str,
    *,
    permission_context: PermissionContext,
    query_filters: QueryFilters,
    llm_client: LLMClient,
    vector_client: VectorSearchClient,
    bm25_index: BM25Index,
    reranker: Reranker,
    all_clauses_by_section_id: dict[str, tuple[Clause, ClauseMetadata]],
    references: list[SectionReference],
    retrieval_top_k: int = 20,
    rerank_top_k: int = 5,
    audit_logger: AuditLogger | None = None,
    trace_exporter: TraceExporter | None = None,
    defined_terms: list[DefinedTerm] | None = None,
    all_clauses_by_id: dict[str, tuple[Clause, ClauseMetadata]] | None = None,
) -> QueryResult:
    trace_exporter = trace_exporter or NullTraceExporter()

    sanitized = redact_structured_pii(query_text)
    query_text = sanitized.sanitized_text
    if sanitized.any_pii_found and audit_logger:
        audit_logger.log(
            AuditEvent(
                event_type="pii_redacted",
                tenant_id=permission_context.tenant_id,
                matter_id=permission_context.matter_id,
                details={"finding_kinds": [f.kind for f in sanitized.findings]},
            )
        )

    jailbreak_check, route = screen_query(query_text, llm_client=llm_client)
    if jailbreak_check.is_flagged:
        if audit_logger:
            audit_logger.log(
                AuditEvent(
                    event_type="jailbreak_blocked",
                    tenant_id=permission_context.tenant_id,
                    matter_id=permission_context.matter_id,
                    details={"reason": jailbreak_check.reason},
                )
            )
        result = QueryResult(
            query_text=query_text,
            route=RouteType.AMBIGUOUS_OUT_OF_SCOPE,
            jailbreak_check=jailbreak_check,
            retrieved_clauses=[],
        )
        trace_exporter.export(result)
        return result

    if route == RouteType.AMBIGUOUS_OUT_OF_SCOPE:
        # Nothing to retrieve for; run_crag flags this route for review.
        result = QueryResult(query_text=query_text, route=route, jailbreak_check=jailbreak_check)
        trace_exporter.export(result)
        return result

    retrieved, trace = retrieve(
        query_text,
        permission_context=permission_context,
        query_filters=query_filters,
        vector_client=vector_client,
        bm25_index=bm25_index,
        reranker=reranker,
        all_clauses_by_section_id=all_clauses_by_section_id,
        references=references,
        defined_terms=defined_terms,
        all_clauses_by_id=all_clauses_by_id,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        audit_logger=audit_logger,
    )
    result = QueryResult(
        query_text=query_text,
        route=route,
        jailbreak_check=jailbreak_check,
        retrieved_clauses=retrieved,
        trace=trace,
    )
    trace_exporter.export(result)
    return result


def _assert_permission_and_audit(
    clauses,
    permission_context: PermissionContext,
    audit_logger: AuditLogger | None,
    document_id: str | None = None,
) -> None:
    try:
        assert_permission(clauses, permission_context, document_id)
    except PermissionViolation:
        if audit_logger:
            audit_logger.log(
                AuditEvent(
                    event_type="permission_violation",
                    tenant_id=permission_context.tenant_id,
                    matter_id=permission_context.matter_id,
                )
            )
        raise

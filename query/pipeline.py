"""process_query orchestrates the query-time baseline — README build-order
step 3 / architecture.md B1-B7: input sanitization -> jailbreak classifier ->
permission filter -> jurisdiction/governing-law filter -> router -> hybrid
retrieval -> rerank -> cross-reference expansion -> reorder. Stops here,
before generation — CLAUDE.md's build order adds the citation validator and
CRAG evaluator (steps 4-5) before wiring in actual answer generation.

If the jailbreak classifier flags the query, the pipeline short-circuits
before the permission filter or vector store are touched at all —
architecture.md §9.3: "runs once, at input, before anything else touches the
pipeline."

Structured PII redaction (CLAUDE.md #8) runs unconditionally, before
anything else — including the jailbreak/router LLM calls — so an SSN or
account number embedded in a query never reaches a third-party model
provider at all, flagged or not. It's regex-only (no LLM call), so it never
changes the pipeline's LLM call pattern; unstructured (LLM-based) PII
detection is available in query/filters/pii_sanitizer.py but is an opt-in
pre-step a caller runs before process_query, not auto-wired here — see that
module's docstring for why.

audit_logger and trace_exporter are optional dependencies (default: don't
log / don't export) so adding either never breaks an existing caller."""

from __future__ import annotations

from ingestion.metadata.llm_client import LLMClient
from ingestion.schemas import Clause, ClauseMetadata, SectionReference
from query.filters.audit_log import AuditEvent, AuditLogger
from query.filters.jailbreak_classifier import classify_query
from query.filters.permission import PermissionViolation, assert_permission
from query.filters.pii_sanitizer import redact_structured_pii
from query.generation.router import classify_route
from query.observability.langsmith_tracer import NullTraceExporter, TraceExporter
from query.retrieval.bm25_search import BM25Index
from query.retrieval.hybrid import hybrid_retrieve
from query.retrieval.reference_expansion import expand_with_references
from query.retrieval.reorder import reorder_for_generation
from query.retrieval.reranker import Reranker
from query.retrieval.vector_search import VectorSearchClient
from query.schemas import PermissionContext, QueryFilters, QueryResult, RetrievalTraceStep, RouteType


def _clause_ids(clauses) -> list[str]:
    return [rc.clause.clause_id for rc in clauses]


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

    jailbreak_check = classify_query(query_text, llm_client=llm_client)
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

    route = classify_route(query_text, llm_client=llm_client)
    trace: list[RetrievalTraceStep] = []

    candidates = hybrid_retrieve(
        query_text,
        vector_client=vector_client,
        bm25_index=bm25_index,
        permission_context=permission_context,
        query_filters=query_filters,
        top_k=retrieval_top_k,
    )
    trace.append(RetrievalTraceStep(stage="hybrid_retrieve", clause_ids=_clause_ids(candidates)))
    _assert_permission_and_audit(candidates, permission_context, audit_logger)

    reranked = reranker.rerank(query_text, candidates, top_k=rerank_top_k)
    trace.append(RetrievalTraceStep(stage="rerank", clause_ids=_clause_ids(reranked)))

    expanded = expand_with_references(
        reranked,
        all_clauses_by_section_id=all_clauses_by_section_id,
        references=references,
    )
    trace.append(RetrievalTraceStep(stage="reference_expansion", clause_ids=_clause_ids(expanded)))
    _assert_permission_and_audit(expanded, permission_context, audit_logger)

    reordered = reorder_for_generation(expanded)
    trace.append(RetrievalTraceStep(stage="reorder", clause_ids=_clause_ids(reordered)))

    result = QueryResult(
        query_text=query_text,
        route=route,
        jailbreak_check=jailbreak_check,
        retrieved_clauses=reordered,
        trace=trace,
    )
    trace_exporter.export(result)
    return result


def _assert_permission_and_audit(
    clauses, permission_context: PermissionContext, audit_logger: AuditLogger | None
) -> None:
    try:
        assert_permission(clauses, permission_context)
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

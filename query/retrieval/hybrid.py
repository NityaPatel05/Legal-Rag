"""Combines vector and BM25 retrieval via Reciprocal Rank Fusion (RRF) —
CLAUDE.md's hybrid-search decision. RRF over raw score blending because
vector cosine similarity and BM25 scores live on incomparable scales; RRF
only needs each list's rank order, not comparable magnitudes.

Every filter condition -- permission, jurisdiction/governing-law, document
scope -- is passed to the vector search call (pre-retrieval, the primary
control), and the SAME predicate list is applied to BM25 hits before
fusion. BM25 has no per-query filter of its own (see bm25_search.py): the
original version re-checked only tenant/matter on BM25 hits, so a
`governing_law=delaware` query still fused in keyword hits from a
California-law contract in the same matter.
"""

from __future__ import annotations

from ingestion.embedding.indexer import document_id_from_clause_id
from query.filters.jurisdiction_governing_law import (
    build_document_filter,
    build_jurisdiction_governing_law_filter,
)
from query.filters.permission import build_permission_filter
from query.retrieval.bm25_search import BM25Index
from query.retrieval.vector_search import VectorSearchClient
from query.schemas import FilterCondition, PermissionContext, QueryFilters, RetrievedClause

_RRF_K = 60


def _field_value(retrieved: RetrievedClause, field: str):
    if field == "document_id":
        return document_id_from_clause_id(retrieved.clause.clause_id)
    value = getattr(retrieved.metadata, field, None)
    return value.value if hasattr(value, "value") else value


def matches_all(retrieved: RetrievedClause, conditions: list[FilterCondition]) -> bool:
    return all(_field_value(retrieved, c.field) == c.value for c in conditions)


def _dedupe_by_clause_id(ranked: list[RetrievedClause]) -> list[RetrievedClause]:
    """Keeps each clause's first (best) rank. RRF sums one term per list a
    clause appears in; a clause listed twice in ONE list (e.g. a store
    returning the same point twice) would otherwise get two terms and
    outrank a clause both retrievers agree on. Ranks are recomputed on the
    deduped list, so the clauses below a duplicate move up one."""
    seen: set[str] = set()
    out: list[RetrievedClause] = []
    for retrieved in ranked:
        if retrieved.clause.clause_id not in seen:
            seen.add(retrieved.clause.clause_id)
            out.append(retrieved)
    return out


def build_filters(permission_context: PermissionContext, query_filters: QueryFilters) -> list[FilterCondition]:
    return (
        build_permission_filter(permission_context)
        + build_jurisdiction_governing_law_filter(query_filters)
        + build_document_filter(query_filters)
    )


def hybrid_retrieve(
    query_text: str,
    *,
    vector_client: VectorSearchClient,
    bm25_index: BM25Index,
    permission_context: PermissionContext,
    query_filters: QueryFilters,
    top_k: int,
    candidate_pool_size: int = 20,
) -> list[RetrievedClause]:
    filters = build_filters(permission_context, query_filters)

    # Vector results are NOT silently post-filtered: the store already
    # applied these predicates, so a mismatch there means the primary control
    # failed, and process_query's assert_permission must see it and fail
    # loudly (and audit it) rather than have it quietly dropped here.
    vector_results = vector_client.search(query_text, filters=filters, top_k=candidate_pool_size)
    bm25_results = [
        r for r in bm25_index.search(query_text, top_k=candidate_pool_size) if matches_all(r, filters)
    ]

    rrf_scores: dict[str, float] = {}
    by_clause_id: dict[str, RetrievedClause] = {}
    for ranked_list in (vector_results, bm25_results):
        for rank, retrieved in enumerate(_dedupe_by_clause_id(ranked_list)):
            clause_id = retrieved.clause.clause_id
            rrf_scores[clause_id] = rrf_scores.get(clause_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            by_clause_id.setdefault(clause_id, retrieved)

    fused = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    return [
        by_clause_id[clause_id].model_copy(update={"score": score, "source": "hybrid"})
        for clause_id, score in fused[:top_k]
    ]

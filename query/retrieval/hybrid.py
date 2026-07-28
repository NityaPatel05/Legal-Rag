"""Combines vector and BM25 retrieval via Reciprocal Rank Fusion (RRF) —
CLAUDE.md's hybrid-search decision. RRF over raw score blending because
vector cosine similarity and BM25 scores live on incomparable scales; RRF
only needs each list's rank order, not comparable magnitudes.

Applies the permission filter (CLAUDE.md #7, pre-retrieval, primary control)
via the vector search call, then re-checks every candidate — including BM25
results, which have no per-query filter of their own (see bm25_search.py) —
before fusion. Anything that doesn't match the requesting tenant/matter is
dropped here rather than silently fused in.
"""

from __future__ import annotations

from query.filters.jurisdiction_governing_law import build_jurisdiction_governing_law_filter
from query.filters.permission import build_permission_filter
from query.retrieval.bm25_search import BM25Index
from query.retrieval.vector_search import VectorSearchClient
from query.schemas import PermissionContext, QueryFilters, RetrievedClause

_RRF_K = 60


def _matches_permission(retrieved: RetrievedClause, context: PermissionContext) -> bool:
    return (
        retrieved.metadata.tenant_id == context.tenant_id
        and retrieved.metadata.matter_id == context.matter_id
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
    filters = build_permission_filter(permission_context) + build_jurisdiction_governing_law_filter(
        query_filters
    )

    vector_results = vector_client.search(query_text, filters=filters, top_k=candidate_pool_size)
    bm25_results = [
        r
        for r in bm25_index.search(query_text, top_k=candidate_pool_size)
        if _matches_permission(r, permission_context)
    ]

    rrf_scores: dict[str, float] = {}
    by_clause_id: dict[str, RetrievedClause] = {}
    for ranked_list in (vector_results, bm25_results):
        for rank, retrieved in enumerate(ranked_list):
            clause_id = retrieved.clause.clause_id
            rrf_scores[clause_id] = rrf_scores.get(clause_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            by_clause_id.setdefault(clause_id, retrieved)

    fused = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)
    return [
        by_clause_id[clause_id].model_copy(update={"score": score, "source": "hybrid"})
        for clause_id, score in fused[:top_k]
    ]

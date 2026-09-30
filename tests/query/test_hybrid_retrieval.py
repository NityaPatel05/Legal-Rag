from query.retrieval.hybrid import hybrid_retrieve
from query.schemas import PermissionContext, QueryFilters, RetrievedClause
from tests.query.factories import make_clause, make_metadata


class FakeVectorSearchClient:
    def __init__(self, results):
        self._results = results
        self.last_filters = None

    def search(self, query_text, *, filters, top_k):
        self.last_filters = filters
        return self._results[:top_k]


class FakeBM25Index:
    def __init__(self, results):
        self._results = results

    def search(self, query_text, *, top_k):
        return self._results[:top_k]


def test_clause_in_both_lists_ranks_above_clause_in_only_one():
    a = RetrievedClause(clause=make_clause("doc::a"), metadata=make_metadata(), score=0.9, source="vector")
    b_vector = RetrievedClause(clause=make_clause("doc::b"), metadata=make_metadata(), score=0.8, source="vector")
    b_bm25 = RetrievedClause(clause=make_clause("doc::b"), metadata=make_metadata(), score=5.0, source="bm25")
    c = RetrievedClause(clause=make_clause("doc::c"), metadata=make_metadata(), score=3.0, source="bm25")

    results = hybrid_retrieve(
        "query",
        vector_client=FakeVectorSearchClient([a, b_vector]),
        bm25_index=FakeBM25Index([b_bm25, c]),
        permission_context=PermissionContext(tenant_id="tenant-1", matter_id="matter-1"),
        query_filters=QueryFilters(),
        top_k=10,
    )

    ids = [r.clause.clause_id for r in results]
    assert ids[0] == "doc::b"  # appears in both lists -> highest RRF score
    assert set(ids) == {"doc::a", "doc::b", "doc::c"}
    assert all(r.source == "hybrid" for r in results)


def test_permission_and_jurisdiction_filters_are_passed_to_vector_search():
    vector_client = FakeVectorSearchClient([])

    hybrid_retrieve(
        "query",
        vector_client=vector_client,
        bm25_index=FakeBM25Index([]),
        permission_context=PermissionContext(tenant_id="tenant-1", matter_id="matter-1"),
        query_filters=QueryFilters(jurisdiction="New York"),
        top_k=5,
    )

    fields = {c.field: c.value for c in vector_client.last_filters}
    assert fields == {"tenant_id": "tenant-1", "matter_id": "matter-1", "jurisdiction": "new york"}


def test_mismatched_tenant_bm25_result_is_dropped_before_fusion():
    """Defense-in-depth: bm25_search.py has no per-query permission filter of
    its own, so hybrid_retrieve must not let a cross-tenant BM25 hit through
    just because it scored well."""
    matching = RetrievedClause(
        clause=make_clause("doc::ok"), metadata=make_metadata(tenant_id="tenant-1"), score=1.0, source="bm25"
    )
    leaked = RetrievedClause(
        clause=make_clause("doc::leak"), metadata=make_metadata(tenant_id="tenant-2"), score=10.0, source="bm25"
    )

    results = hybrid_retrieve(
        "query",
        vector_client=FakeVectorSearchClient([]),
        bm25_index=FakeBM25Index([leaked, matching]),
        permission_context=PermissionContext(tenant_id="tenant-1", matter_id="matter-1"),
        query_filters=QueryFilters(),
        top_k=10,
    )

    ids = [r.clause.clause_id for r in results]
    assert "doc::leak" not in ids
    assert "doc::ok" in ids


def test_top_k_limits_fused_results():
    vector_results = [
        RetrievedClause(clause=make_clause(f"doc::v{i}"), metadata=make_metadata(), score=1.0, source="vector")
        for i in range(5)
    ]

    results = hybrid_retrieve(
        "query",
        vector_client=FakeVectorSearchClient(vector_results),
        bm25_index=FakeBM25Index([]),
        permission_context=PermissionContext(tenant_id="tenant-1", matter_id="matter-1"),
        query_filters=QueryFilters(),
        top_k=2,
    )
    assert len(results) == 2

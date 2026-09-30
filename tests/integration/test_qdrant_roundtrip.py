"""Round-trip through a REAL qdrant-client (local in-memory mode) -- no
server, no network, but the actual client library and its actual API.

Every other test in this repo uses fakes, which is how two production-
breaking bugs shipped with a green suite: vector search called
`QdrantClient.search` (removed from qdrant-client), and zero-vector
reference points were returned as search hits and crashed payload parsing.
These tests exercise the real indexer -> vector search -> document reader
path so that class of bug fails here."""

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from ingestion.embedding.document_reader import QdrantDocumentReader  # noqa: E402
from ingestion.embedding.indexer import QdrantIndexer  # noqa: E402
from ingestion.schemas import (  # noqa: E402
    Clause,
    ClauseMetadata,
    DefinedTerm,
    SectionReference,
    TaggedClause,
)
from query.retrieval.vector_search import QdrantVectorSearchClient  # noqa: E402
from query.schemas import FilterCondition  # noqa: E402

DIM = 4


class _FixedEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _tagged(doc, idx, text, tenant="tenant-1", matter="matter-1", vector=(1.0, 0.0, 0.0, 0.0), section=None):
    clause = Clause(
        clause_id=f"{doc}::clause-{idx:03d}",
        section_id=section,
        text=text,
        start_offset=idx * 100,
        end_offset=idx * 100 + len(text),
    )
    return TaggedClause(
        clause=clause,
        metadata=ClauseMetadata(tenant_id=tenant, matter_id=matter),
        embedding=list(vector),
    )


@pytest.fixture
def stores():
    client = qdrant_client.QdrantClient(":memory:")
    indexer = QdrantIndexer(url="unused", collection_name="c")
    indexer._client = client
    indexer.ensure_collection(DIM)
    search = QdrantVectorSearchClient("unused", "c", _FixedEmbedder())
    search._client = client
    reader = QdrantDocumentReader("unused", "c")
    reader._client = client
    return indexer, search, reader


def _perm(tenant="tenant-1", matter="matter-1"):
    return [FilterCondition(field="tenant_id", value=tenant), FilterCondition(field="matter_id", value=matter)]


def test_vector_search_never_returns_reference_or_defined_term_points(stores):
    indexer, search, _ = stores
    indexer.upsert([_tagged("doc-1", 0, "Governing law clause.", section="1")])
    indexer.upsert_references(
        document_id="doc-1",
        tenant_id="tenant-1",
        matter_id="matter-1",
        references=[SectionReference(source_clause_id="doc-1::clause-000", referenced_section_id="2", raw_text="Section 2")],
        vector_dim=DIM,
    )
    indexer.upsert_defined_terms(
        document_id="doc-1",
        tenant_id="tenant-1",
        matter_id="matter-1",
        defined_terms=[DefinedTerm(term="Agreement", definition="this", defined_in_clause_id="doc-1::clause-000")],
        vector_dim=DIM,
    )

    results = search.search("q", filters=_perm(), top_k=10)

    assert [r.clause.clause_id for r in results] == ["doc-1::clause-000"]


def test_same_document_id_in_two_tenants_does_not_overwrite(stores):
    indexer, search, _ = stores
    indexer.upsert([_tagged("contract-1", 0, "Tenant one text.", tenant="tenant-1")])
    indexer.upsert([_tagged("contract-1", 0, "Tenant two text.", tenant="tenant-2")])

    t1 = search.search("q", filters=_perm("tenant-1"), top_k=10)
    t2 = search.search("q", filters=_perm("tenant-2"), top_k=10)

    assert [r.clause.text for r in t1] == ["Tenant one text."]
    assert [r.clause.text for r in t2] == ["Tenant two text."]


def test_reingest_removes_stale_clauses(stores):
    indexer, _, reader = stores
    indexer.upsert([_tagged("doc-1", i, f"v1 clause {i}") for i in range(3)])

    indexer.delete_document(document_id="doc-1", tenant_id="tenant-1", matter_id="matter-1")
    indexer.upsert([_tagged("doc-1", 0, "v2 clause 0")])

    context = reader.load_document("doc-1", tenant_id="tenant-1", matter_id="matter-1")
    assert [c.text for c in context.clauses] == ["v2 clause 0"]


def test_document_filter_scopes_vector_search(stores):
    indexer, search, _ = stores
    indexer.upsert([_tagged("doc-a", 0, "A text."), _tagged("doc-b", 0, "B text.")])

    results = search.search(
        "q", filters=_perm() + [FilterCondition(field="document_id", value="doc-a")], top_k=10
    )
    assert [r.clause.text for r in results] == ["A text."]


def test_reader_returns_clauses_in_document_order_with_defined_terms(stores):
    indexer, _, reader = stores
    indexer.upsert([_tagged("doc-1", i, f"clause {i}", section=str(i + 1)) for i in range(6)])
    indexer.upsert_defined_terms(
        document_id="doc-1",
        tenant_id="tenant-1",
        matter_id="matter-1",
        defined_terms=[DefinedTerm(term="Services", definition="the work", defined_in_clause_id="doc-1::clause-000")],
        vector_dim=DIM,
    )

    context = reader.load_document("doc-1", tenant_id="tenant-1", matter_id="matter-1")

    assert [c.text for c in context.clauses] == [f"clause {i}" for i in range(6)]
    assert [t.term for t in context.defined_terms] == ["Services"]

"""VectorStore is the seam pipeline.py depends on, so indexing is unit-testable
without a running Qdrant instance. QdrantIndexer is the production adapter.
Every point is written with its full ClauseMetadata payload — including
tenant_id/matter_id/privilege_level — as Qdrant payload fields, since the
permission-first filter (CLAUDE.md #7, lives in query/filters/) predicates on
them before anything else touches the vector store.

Qdrant point IDs must be an unsigned integer or a UUID — NOT an arbitrary
string. clause_id (e.g. "doc-1::clause-000") is neither, so it can't be used
as the point id directly. Point ids are derived via uuid5 instead —
deterministic, so re-ingesting the same document produces the same point ids
— with the real clause_id kept as its own payload field so callers reading
points back (query/retrieval/vector_search.py, document_reader.py) use that,
not the opaque UUID, as the application-level identifier.

The uuid5 name includes tenant_id and matter_id, not just clause_id.
document_id is caller-chosen, so two tenants can both ingest a
"contract-1"; keyed on clause_id alone, the second upsert would silently
overwrite the first tenant's points -- a cross-tenant integrity failure no
query-time filter can detect.

Re-ingestion is delete-then-write (delete_document), not a bare upsert: a
revised contract with fewer clauses would otherwise leave the old version's
trailing clauses in the index, retrievable and citable.

document_id is stored as its own field too (derived from clause_id, which
clause_parser.py always constructs as f"{document_id}::clause-{idx}") so a
document's full clause set can be scrolled back out of Qdrant at query time
— see ingestion/embedding/document_reader.py.

References and defined terms are persisted as their own points in the SAME
collection rather than a second storage technology — they're small,
structured, lookup-only data that doesn't need vector search, so they're
written with a zero-vector and a `record_type` payload field ("reference" /
"defined_term" vs "clause"). Vector search excludes them explicitly (see
query/retrieval/vector_search.py) -- a zero vector still scores 0.0 and gets
returned by an unfiltered search. document_reader.py's scroll splits on
record_type to reconstruct all three."""

from __future__ import annotations

import uuid
from typing import Protocol

from ingestion.schemas import DefinedTerm, SectionReference, TaggedClause

_UUID_NAMESPACE = uuid.UUID("6f9cc1a4-2b0e-4f0a-9c1a-4b2b0e4f0a9c")

# Payload fields every read path filters on. tenant_id is flagged is_tenant
# so Qdrant co-locates each tenant's vectors (see Qdrant's multitenancy
# guide); the rest are plain keyword indexes.
_KEYWORD_INDEXES = ("matter_id", "document_id", "record_type", "jurisdiction", "governing_law")


def document_id_from_clause_id(clause_id: str) -> str:
    return clause_id.split("::clause-")[0]


def _scoped(name: str, tenant_id: str, matter_id: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, f"{tenant_id}\x1f{matter_id}\x1f{name}"))


def point_id_for_clause(clause_id: str, *, tenant_id: str, matter_id: str) -> str:
    return _scoped(clause_id, tenant_id, matter_id)


def point_id_for_reference(document_id: str, index: int, *, tenant_id: str, matter_id: str) -> str:
    return _scoped(f"{document_id}::reference-{index}", tenant_id, matter_id)


def point_id_for_defined_term(document_id: str, index: int, *, tenant_id: str, matter_id: str) -> str:
    return _scoped(f"{document_id}::defined-term-{index}", tenant_id, matter_id)


class VectorStore(Protocol):
    def delete_document(self, *, document_id: str, tenant_id: str, matter_id: str) -> None: ...

    def upsert(self, tagged_clauses: list[TaggedClause]) -> None: ...

    def upsert_references(
        self,
        *,
        document_id: str,
        tenant_id: str,
        matter_id: str,
        references: list[SectionReference],
        vector_dim: int,
    ) -> None: ...

    def upsert_defined_terms(
        self,
        *,
        document_id: str,
        tenant_id: str,
        matter_id: str,
        defined_terms: list[DefinedTerm],
        vector_dim: int,
    ) -> None: ...


class QdrantIndexer:
    """qdrant-client is imported lazily so importing this module never requires
    a Qdrant deployment to exist."""

    def __init__(self, url: str, collection_name: str, api_key: str | None = None):
        self._url = url
        self._collection_name = collection_name
        self._api_key = api_key
        self._client = None

    def _client_instance(self):
        if self._client is None:
            from qdrant_client import QdrantClient  # lazy import

            self._client = QdrantClient(url=self._url, api_key=self._api_key)
        return self._client

    def ensure_collection(self, vector_dim: int) -> None:
        """Creates the collection and its payload indexes if missing.
        Idempotent -- safe to call on every startup."""
        from qdrant_client.models import (  # lazy import
            Distance,
            KeywordIndexParams,
            PayloadSchemaType,
            VectorParams,
        )

        client = self._client_instance()
        if not client.collection_exists(self._collection_name):
            client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
        client.create_payload_index(
            collection_name=self._collection_name,
            field_name="tenant_id",
            field_schema=KeywordIndexParams(type="keyword", is_tenant=True),
        )
        for field in _KEYWORD_INDEXES:
            client.create_payload_index(
                collection_name=self._collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def delete_document(self, *, document_id: str, tenant_id: str, matter_id: str) -> None:
        from qdrant_client.models import (  # lazy import
            FieldCondition,
            Filter,
            FilterSelector,
            MatchValue,
        )

        self._client_instance().delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                        FieldCondition(key="matter_id", match=MatchValue(value=matter_id)),
                    ]
                )
            ),
        )

    def upsert(self, tagged_clauses: list[TaggedClause]) -> None:
        from qdrant_client.models import PointStruct  # lazy import

        client = self._client_instance()
        points = [
            PointStruct(
                id=point_id_for_clause(
                    tc.clause.clause_id, tenant_id=tc.metadata.tenant_id, matter_id=tc.metadata.matter_id
                ),
                vector=tc.embedding,
                payload={
                    "record_type": "clause",
                    "clause_id": tc.clause.clause_id,
                    "parent_clause_id": tc.clause.parent_clause_id,
                    "document_id": document_id_from_clause_id(tc.clause.clause_id),
                    "section_id": tc.clause.section_id,
                    "heading": tc.clause.heading,
                    "text": tc.clause.text,
                    "start_offset": tc.clause.start_offset,
                    "end_offset": tc.clause.end_offset,
                    "context_note": tc.context_note,
                    **tc.metadata.model_dump(mode="json"),
                },
            )
            for tc in tagged_clauses
        ]
        if points:
            client.upsert(collection_name=self._collection_name, points=points)

    def _upsert_aux_points(self, point_ids: list[str], payloads: list[dict], vector_dim: int) -> None:
        if not payloads:
            return

        from qdrant_client.models import PointStruct  # lazy import

        zero_vector = [0.0] * vector_dim
        points = [PointStruct(id=pid, vector=zero_vector, payload=p) for pid, p in zip(point_ids, payloads)]
        self._client_instance().upsert(collection_name=self._collection_name, points=points)

    def upsert_references(
        self,
        *,
        document_id: str,
        tenant_id: str,
        matter_id: str,
        references: list[SectionReference],
        vector_dim: int,
    ) -> None:
        self._upsert_aux_points(
            [
                point_id_for_reference(document_id, i, tenant_id=tenant_id, matter_id=matter_id)
                for i in range(len(references))
            ],
            [
                {
                    "record_type": "reference",
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "matter_id": matter_id,
                    "source_clause_id": ref.source_clause_id,
                    "referenced_section_id": ref.referenced_section_id,
                    "raw_text": ref.raw_text,
                }
                for ref in references
            ],
            vector_dim,
        )

    def upsert_defined_terms(
        self,
        *,
        document_id: str,
        tenant_id: str,
        matter_id: str,
        defined_terms: list[DefinedTerm],
        vector_dim: int,
    ) -> None:
        self._upsert_aux_points(
            [
                point_id_for_defined_term(document_id, i, tenant_id=tenant_id, matter_id=matter_id)
                for i in range(len(defined_terms))
            ],
            [
                {
                    "record_type": "defined_term",
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "matter_id": matter_id,
                    "term": term.term,
                    "definition": term.definition,
                    "defined_in_clause_id": term.defined_in_clause_id,
                }
                for term in defined_terms
            ],
            vector_dim,
        )

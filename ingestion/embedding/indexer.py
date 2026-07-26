"""VectorStore is the seam pipeline.py depends on, so indexing is unit-testable
without a running Qdrant instance. QdrantIndexer is the production adapter.
Every point is written with its full ClauseMetadata payload — including
tenant_id/matter_id/privilege_level — as Qdrant payload fields, since the
permission-first filter (CLAUDE.md #7, lives in query/filters/) predicates on
them before anything else touches the vector store.

Qdrant point IDs must be an unsigned integer or a UUID — NOT an arbitrary
string. clause_id (e.g. "doc-1::clause-000") is neither, so it can't be used
as the point id directly; every fake/in-memory test in this codebase let
that slide silently since none of them enforce Qdrant's actual ID format.
Point ids are derived via uuid5(clause_id) instead — deterministic, so
re-ingesting the same document produces the same point ids (idempotent
upsert) — with the real clause_id kept as its own payload field so callers
reading points back (query/retrieval/vector_search.py, document_reader.py)
use that, not the opaque UUID, as the application-level identifier.
document_id is stored as its own field too (derived from clause_id, which
clause_parser.py always constructs as f"{document_id}::clause-{idx}") so a
document's full clause set can be scrolled back out of Qdrant at query time
— see ingestion/embedding/document_reader.py.

References (ingestion/parsing/references.py's section-reference graph) are
persisted as their own points in the SAME collection rather than a second
storage technology — CLAUDE.md locks in Qdrant, and a reference is small,
structured, lookup-only data that doesn't need vector search of its own, so
it's written with a zero-vector and a `record_type: "reference"` payload
field distinguishing it from `record_type: "clause"` points.
document_reader.py's scroll splits on that field to reconstruct both."""

from __future__ import annotations

import uuid
from typing import Protocol

from ingestion.schemas import SectionReference, TaggedClause

_UUID_NAMESPACE = uuid.UUID("6f9cc1a4-2b0e-4f0a-9c1a-4b2b0e4f0a9c")


def document_id_from_clause_id(clause_id: str) -> str:
    return clause_id.split("::clause-")[0]


def point_id_for_clause(clause_id: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, clause_id))


def point_id_for_reference(document_id: str, index: int) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, f"{document_id}::reference-{index}"))


class VectorStore(Protocol):
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

    def upsert(self, tagged_clauses: list[TaggedClause]) -> None:
        from qdrant_client.models import PointStruct  # lazy import

        client = self._client_instance()
        points = [
            PointStruct(
                id=point_id_for_clause(tc.clause.clause_id),
                vector=tc.embedding,
                payload={
                    "record_type": "clause",
                    "clause_id": tc.clause.clause_id,
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
        client.upsert(collection_name=self._collection_name, points=points)

    def upsert_references(
        self,
        *,
        document_id: str,
        tenant_id: str,
        matter_id: str,
        references: list[SectionReference],
        vector_dim: int,
    ) -> None:
        if not references:
            return

        from qdrant_client.models import PointStruct  # lazy import

        client = self._client_instance()
        zero_vector = [0.0] * vector_dim
        points = [
            PointStruct(
                id=point_id_for_reference(document_id, i),
                vector=zero_vector,
                payload={
                    "record_type": "reference",
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "matter_id": matter_id,
                    "source_clause_id": ref.source_clause_id,
                    "referenced_section_id": ref.referenced_section_id,
                    "raw_text": ref.raw_text,
                },
            )
            for i, ref in enumerate(references)
        ]
        client.upsert(collection_name=self._collection_name, points=points)

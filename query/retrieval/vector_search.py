"""Vector-side retrieval against Qdrant. VectorSearchClient is the seam
hybrid.py depends on, so retrieval is unit-testable without a running Qdrant
instance. QdrantVectorSearchClient reconstructs full Clause/ClauseMetadata
objects from the point payload written by ingestion/embedding/indexer.py —
the two must stay in sync on payload field names (both go through
document_reader._point_to_clause_and_metadata).

Reads clause_id back from the payload, NOT from hit.id — the point id is an
opaque uuid5 derived from clause_id (Qdrant requires int/UUID ids, clause_id
is neither), so hit.id is meaningless at the application level. See
ingestion/embedding/indexer.py's module docstring.

Two things verified against a real qdrant-client (1.18), which no fake-based
test catches:

- `QdrantClient.search` no longer exists; the Query API (`query_points`) is
  the only search entry point.
- Reference / defined-term points are stored with a zero vector in the same
  collection. They are NOT excluded by similarity: an unfiltered search
  returns them with score 0.0, and their payload has no clause_id. They're
  excluded here with a must_not on record_type (must_not rather than
  must=clause so points written before record_type existed still match).
"""

from __future__ import annotations

from typing import Protocol

from ingestion.embedding.embedder import EmbeddingClient
from query.schemas import FilterCondition, RetrievedClause

_NON_CLAUSE_RECORD_TYPES = ["reference", "defined_term"]


class VectorSearchClient(Protocol):
    def search(
        self, query_text: str, *, filters: list[FilterCondition], top_k: int
    ) -> list[RetrievedClause]: ...


class QdrantVectorSearchClient:
    def __init__(
        self,
        url: str,
        collection_name: str,
        embedding_client: EmbeddingClient,
        api_key: str | None = None,
    ):
        self._url = url
        self._collection_name = collection_name
        self._embedding_client = embedding_client
        self._api_key = api_key
        self._client = None

    def _client_instance(self):
        if self._client is None:
            from qdrant_client import QdrantClient  # lazy import

            self._client = QdrantClient(url=self._url, api_key=self._api_key)
        return self._client

    def search(
        self, query_text: str, *, filters: list[FilterCondition], top_k: int
    ) -> list[RetrievedClause]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue  # lazy import

        [query_embedding] = self._embedding_client.embed([query_text])
        qdrant_filter = Filter(
            must=[FieldCondition(key=f.field, match=MatchValue(value=f.value)) for f in filters],
            must_not=[FieldCondition(key="record_type", match=MatchAny(any=_NON_CLAUSE_RECORD_TYPES))],
        )
        client = self._client_instance()
        response = client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [_hit_to_retrieved_clause(hit) for hit in response.points]


def _hit_to_retrieved_clause(hit) -> RetrievedClause:
    from ingestion.embedding.document_reader import _point_to_clause_and_metadata

    clause, metadata = _point_to_clause_and_metadata(hit.payload)
    return RetrievedClause(clause=clause, metadata=metadata, score=hit.score, source="vector")

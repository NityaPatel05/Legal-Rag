"""Vector-side retrieval against Qdrant. VectorSearchClient is the seam
hybrid.py depends on, so retrieval is unit-testable without a running Qdrant
instance. QdrantVectorSearchClient reconstructs full Clause/ClauseMetadata
objects from the point payload written by ingestion/embedding/indexer.py —
the two must stay in sync on payload field names.

Reads clause_id back from the payload, NOT from hit.id — the point id is an
opaque uuid5 derived from clause_id (Qdrant requires int/UUID ids, clause_id
is neither), so hit.id is meaningless at the application level. See
ingestion/embedding/indexer.py's module docstring."""

from __future__ import annotations

from typing import Protocol

from ingestion.embedding.embedder import EmbeddingClient
from query.schemas import FilterCondition, RetrievedClause


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
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # lazy import

        [query_embedding] = self._embedding_client.embed([query_text])
        qdrant_filter = Filter(
            must=[FieldCondition(key=f.field, match=MatchValue(value=f.value)) for f in filters]
        )
        client = self._client_instance()
        hits = client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
        )
        return [_hit_to_retrieved_clause(hit) for hit in hits]


def _hit_to_retrieved_clause(hit) -> RetrievedClause:
    from ingestion.schemas import Clause, ClauseMetadata  # avoid import cycle at module load

    payload = hit.payload
    clause = Clause(
        clause_id=payload["clause_id"],
        section_id=payload.get("section_id"),
        heading=payload.get("heading"),
        text=payload["text"],
        start_offset=payload.get("start_offset", 0),
        end_offset=payload.get("end_offset", 0),
    )
    metadata_fields = {k: v for k, v in payload.items() if k in ClauseMetadata.model_fields}
    metadata = ClauseMetadata(**metadata_fields)
    return RetrievedClause(clause=clause, metadata=metadata, score=hit.score, source="vector")

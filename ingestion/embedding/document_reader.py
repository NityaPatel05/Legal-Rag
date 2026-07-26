"""Reconstructs the whole-document context process_query needs beyond a
single retrieval call: the citation validator's `exists` check
(query/validation/citation_validator.py) needs the FULL document's clause
list, not just what was retrieved for one query, and reference-expansion
(query/retrieval/reference_expansion.py) needs a section_id -> clause lookup
AND the section-reference graph covering the whole document too. All three
are rebuilt here from a single scroll over every point belonging to one
document, filtered by tenant_id/matter_id for the same permission-first
reason as every other read path in this codebase (CLAUDE.md #7) — clause
points and reference points share a collection (see indexer.py's
upsert_references) and are told apart by their `record_type` payload field.
"""

from __future__ import annotations

from pydantic import BaseModel

from ingestion.schemas import Clause, ClauseMetadata, SectionReference


class DocumentContext(BaseModel):
    clauses: list[Clause]
    clause_metadata_by_id: dict[str, ClauseMetadata]
    by_section_id: dict[str, tuple[Clause, ClauseMetadata]]
    references: list[SectionReference]

    def bm25_corpus(self) -> list[tuple[Clause, ClauseMetadata]]:
        return [(clause, self.clause_metadata_by_id[clause.clause_id]) for clause in self.clauses]


class QdrantDocumentReader:
    """qdrant-client is imported lazily so importing this module never
    requires a Qdrant deployment to exist."""

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

    def load_document(
        self, document_id: str, *, tenant_id: str, matter_id: str
    ) -> DocumentContext:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # lazy import

        client = self._client_instance()
        scroll_filter = Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                FieldCondition(key="matter_id", match=MatchValue(value=matter_id)),
            ]
        )

        clauses: list[Clause] = []
        clause_metadata_by_id: dict[str, ClauseMetadata] = {}
        by_section_id: dict[str, tuple[Clause, ClauseMetadata]] = {}
        references: list[SectionReference] = []

        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self._collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                # Payloads written before record_type existed have neither
                # field -- default to "clause" so old data doesn't silently
                # vanish from a re-deployed reader.
                if point.payload.get("record_type", "clause") == "reference":
                    references.append(_point_to_reference(point.payload))
                    continue
                clause, metadata = _point_to_clause_and_metadata(point.payload)
                clauses.append(clause)
                clause_metadata_by_id[clause.clause_id] = metadata
                if clause.section_id:
                    by_section_id[clause.section_id] = (clause, metadata)
            if offset is None:
                break

        return DocumentContext(
            clauses=clauses,
            clause_metadata_by_id=clause_metadata_by_id,
            by_section_id=by_section_id,
            references=references,
        )


def _point_to_clause_and_metadata(payload: dict) -> tuple[Clause, ClauseMetadata]:
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
    return clause, metadata


def _point_to_reference(payload: dict) -> SectionReference:
    return SectionReference(
        source_clause_id=payload["source_clause_id"],
        referenced_section_id=payload["referenced_section_id"],
        raw_text=payload["raw_text"],
    )

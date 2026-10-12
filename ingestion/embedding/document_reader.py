"""Reconstructs the whole-document context process_query needs beyond a
single retrieval call: the citation validator's `exists` check
(query/validation/citation_validator.py) needs the FULL document's clause
list, not just what was retrieved for one query, and reference-expansion
(query/retrieval/reference_expansion.py) needs a section_id -> clause lookup,
the section-reference graph and the defined-terms table covering the whole
document too. All are rebuilt here from a single scroll over every point
belonging to one document, filtered by tenant_id/matter_id for the same
permission-first reason as every other read path in this codebase
(CLAUDE.md #7) — clause, reference and defined-term points share a
collection (see indexer.py) and are told apart by their `record_type`
payload field.

Two details that matter for correctness:

- Scroll returns points in point-id (uuid) order, not document order, so
  clauses are re-sorted by start_offset.
- by_section_id only contains section ids that resolve to ONE clause.
  Real contracts reuse numbering (an exhibit restarting at "1."), and a
  last-write-wins dict would silently map "Section 1" to whichever clause
  happened to scroll last. Passages split from one clause share its
  section_id; those resolve to the first passage (the one with the heading),
  since that's still unambiguous. A genuinely ambiguous id is left out: an
  expansion that doesn't happen is safer than one that pulls in the wrong
  clause.
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from ingestion.schemas import Clause, ClauseMetadata, DefinedTerm, SectionReference


class DocumentContext(BaseModel):
    clauses: list[Clause]
    clause_metadata_by_id: dict[str, ClauseMetadata]
    by_section_id: dict[str, tuple[Clause, ClauseMetadata]]
    references: list[SectionReference]
    defined_terms: list[DefinedTerm] = Field(default_factory=list)

    def bm25_corpus(self) -> list[tuple[Clause, ClauseMetadata]]:
        return [(clause, self.clause_metadata_by_id[clause.clause_id]) for clause in self.clauses]

    def by_clause_id(self) -> dict[str, tuple[Clause, ClauseMetadata]]:
        return {c.clause_id: (c, self.clause_metadata_by_id[c.clause_id]) for c in self.clauses}


def build_section_lookup(
    clauses: list[Clause], metadata_by_id: dict[str, ClauseMetadata]
) -> dict[str, tuple[Clause, ClauseMetadata]]:
    groups: dict[str, list[Clause]] = defaultdict(list)
    for clause in clauses:
        if clause.section_id:
            groups[clause.section_id].append(clause)

    lookup: dict[str, tuple[Clause, ClauseMetadata]] = {}
    for section_id, group in groups.items():
        owners = {c.parent_clause_id or c.clause_id for c in group}
        if len(owners) != 1:
            continue  # ambiguous: two different clauses claim this number
        first = min(group, key=lambda c: c.start_offset)
        lookup[section_id] = (first, metadata_by_id[first.clause_id])
    return lookup


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
        references: list[SectionReference] = []
        defined_terms: list[DefinedTerm] = []

        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self._collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                # Payloads written before record_type existed have neither
                # field -- default to "clause" so old data doesn't silently
                # vanish from a re-deployed reader.
                record_type = point.payload.get("record_type", "clause")
                if record_type == "reference":
                    references.append(_point_to_reference(point.payload))
                    continue
                if record_type == "defined_term":
                    defined_terms.append(_point_to_defined_term(point.payload))
                    continue
                clause, metadata = _point_to_clause_and_metadata(point.payload)
                clauses.append(clause)
                clause_metadata_by_id[clause.clause_id] = metadata
            if offset is None:
                break

        clauses.sort(key=lambda c: (c.start_offset, c.clause_id))
        return DocumentContext(
            clauses=clauses,
            clause_metadata_by_id=clause_metadata_by_id,
            by_section_id=build_section_lookup(clauses, clause_metadata_by_id),
            references=references,
            defined_terms=defined_terms,
        )

    def list_documents(self, *, tenant_id: str, matter_id: str) -> dict[str, list[ClauseMetadata]]:
        """document_id -> the metadata of each of its clause points, for one
        tenant + matter. Scrolls metadata fields only (no text, no vectors):
        one pass over the matter's clause points, fine at matter scale
        (tens of contracts); a per-matter document registry would replace it
        if matters grow to thousands of documents.

        Reference / defined-term points are excluded with must_not rather
        than must record_type == "clause", so pre-record_type payloads are
        listed exactly as load_document reads them (as clauses)."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue  # lazy import

        client = self._client_instance()
        scroll_filter = Filter(
            must=[
                FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                FieldCondition(key="matter_id", match=MatchValue(value=matter_id)),
            ],
            must_not=[FieldCondition(key="record_type", match=MatchAny(any=["reference", "defined_term"]))],
        )
        fields = ["document_id", *ClauseMetadata.model_fields]

        documents: dict[str, list[ClauseMetadata]] = defaultdict(list)
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self._collection_name,
                scroll_filter=scroll_filter,
                limit=1024,
                offset=offset,
                with_payload=fields,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                metadata_fields = {k: v for k, v in payload.items() if k in ClauseMetadata.model_fields}
                documents[payload["document_id"]].append(ClauseMetadata(**metadata_fields))
            if offset is None:
                break
        return dict(sorted(documents.items()))


def _point_to_clause_and_metadata(payload: dict) -> tuple[Clause, ClauseMetadata]:
    clause = Clause(
        clause_id=payload["clause_id"],
        section_id=payload.get("section_id"),
        heading=payload.get("heading"),
        text=payload["text"],
        start_offset=payload.get("start_offset", 0),
        end_offset=payload.get("end_offset", 0),
        parent_clause_id=payload.get("parent_clause_id"),
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


def _point_to_defined_term(payload: dict) -> DefinedTerm:
    return DefinedTerm(
        term=payload["term"],
        definition=payload["definition"],
        defined_in_clause_id=payload["defined_in_clause_id"],
    )

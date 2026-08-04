import uuid
from unittest.mock import MagicMock

from ingestion.embedding.indexer import (
    QdrantIndexer,
    document_id_from_clause_id,
    point_id_for_clause,
    point_id_for_reference,
)
from ingestion.schemas import SectionReference


def test_document_id_from_clause_id_matches_clause_parser_convention():
    assert document_id_from_clause_id("doc-1::clause-000") == "doc-1"
    assert document_id_from_clause_id("metlife-remarketing-demo::clause-015") == "metlife-remarketing-demo"


def test_point_id_for_clause_is_a_valid_uuid():
    point_id = point_id_for_clause("doc-1::clause-000")
    assert uuid.UUID(point_id)  # raises ValueError if not a valid UUID string


def test_point_id_for_clause_is_deterministic():
    """Re-ingesting the same document must produce the same point ids, so
    upsert is idempotent rather than creating duplicate points."""
    assert point_id_for_clause("doc-1::clause-000") == point_id_for_clause("doc-1::clause-000")


def test_point_id_for_clause_differs_across_clauses():
    assert point_id_for_clause("doc-1::clause-000") != point_id_for_clause("doc-1::clause-001")


def test_point_id_for_reference_is_a_valid_deterministic_uuid():
    point_id = point_id_for_reference("doc-1", 0)
    assert uuid.UUID(point_id)
    assert point_id_for_reference("doc-1", 0) == point_id_for_reference("doc-1", 0)
    assert point_id_for_reference("doc-1", 0) != point_id_for_reference("doc-1", 1)


def _indexer_with_fake_client(fake_client) -> QdrantIndexer:
    indexer = QdrantIndexer(url="http://fake", collection_name="legal_clauses")
    indexer._client = fake_client  # bypass the lazy qdrant_client import entirely
    return indexer


def test_upsert_references_writes_one_point_per_reference_with_zero_vector():
    fake_client = MagicMock()
    references = [
        SectionReference(source_clause_id="doc-1::clause-002", referenced_section_id="1", raw_text="Section 1"),
        SectionReference(source_clause_id="doc-1::clause-003", referenced_section_id="1", raw_text="Section 1"),
    ]

    _indexer_with_fake_client(fake_client).upsert_references(
        document_id="doc-1", tenant_id="tenant-1", matter_id="matter-1", references=references, vector_dim=4
    )

    fake_client.upsert.assert_called_once()
    _, kwargs = fake_client.upsert.call_args
    points = kwargs["points"]
    assert len(points) == 2
    assert all(p.vector == [0.0, 0.0, 0.0, 0.0] for p in points)
    assert all(p.payload["record_type"] == "reference" for p in points)
    assert points[0].payload["source_clause_id"] == "doc-1::clause-002"
    assert points[0].payload["tenant_id"] == "tenant-1"


def test_upsert_references_of_empty_list_does_not_call_client():
    fake_client = MagicMock()
    _indexer_with_fake_client(fake_client).upsert_references(
        document_id="doc-1", tenant_id="tenant-1", matter_id="matter-1", references=[], vector_dim=4
    )
    fake_client.upsert.assert_not_called()


def test_upsert_writes_record_type_clause_on_every_point():
    from ingestion.schemas import Clause, ClauseMetadata, PrivilegeLevel, TaggedClause

    fake_client = MagicMock()
    clause = Clause(clause_id="doc-1::clause-000", section_id="1", heading=None, text="x", start_offset=0, end_offset=1)
    metadata = ClauseMetadata(tenant_id="tenant-1", matter_id="matter-1", privilege_level=PrivilegeLevel.CONFIDENTIAL)
    tagged = TaggedClause(clause=clause, metadata=metadata, context_note=None, embedding=[0.1])

    _indexer_with_fake_client(fake_client).upsert([tagged])

    _, kwargs = fake_client.upsert.call_args
    assert kwargs["points"][0].payload["record_type"] == "clause"

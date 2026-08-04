from types import SimpleNamespace
from unittest.mock import MagicMock

from ingestion.embedding.document_reader import QdrantDocumentReader


def _payload(clause_id, section_id, text, tenant_id="tenant-1", matter_id="matter-1"):
    return {
        "record_type": "clause",
        "clause_id": clause_id,
        "document_id": "doc-1",
        "section_id": section_id,
        "heading": None,
        "text": text,
        "start_offset": 0,
        "end_offset": len(text),
        "context_note": None,
        "contract_type": None,
        "clause_type": None,
        "parties": [],
        "effective_date": None,
        "signature_date": None,
        "jurisdiction": None,
        "governing_law": None,
        "matter_id": matter_id,
        "tenant_id": tenant_id,
        "privilege_level": "confidential",
        "contains_pii": False,
        "supersession_links": [],
    }


def _reference_payload(document_id, source_clause_id, referenced_section_id, raw_text, tenant_id="tenant-1", matter_id="matter-1"):
    return {
        "record_type": "reference",
        "document_id": document_id,
        "tenant_id": tenant_id,
        "matter_id": matter_id,
        "source_clause_id": source_clause_id,
        "referenced_section_id": referenced_section_id,
        "raw_text": raw_text,
    }


def _reader_with_fake_client(fake_client) -> QdrantDocumentReader:
    reader = QdrantDocumentReader(url="http://fake", collection_name="legal_clauses")
    reader._client = fake_client  # bypass the lazy qdrant_client import entirely
    return reader


def test_load_document_reconstructs_clauses_and_section_lookup():
    points = [
        SimpleNamespace(id="uuid-1", payload=_payload("doc-1::clause-000", None, "Preamble text.")),
        SimpleNamespace(id="uuid-2", payload=_payload("doc-1::clause-001", "4", "Governed by Delaware law.")),
    ]
    fake_client = MagicMock()
    fake_client.scroll.return_value = (points, None)

    context = _reader_with_fake_client(fake_client).load_document(
        "doc-1", tenant_id="tenant-1", matter_id="matter-1"
    )

    assert len(context.clauses) == 2
    assert {c.clause_id for c in context.clauses} == {"doc-1::clause-000", "doc-1::clause-001"}
    assert "4" in context.by_section_id
    assert context.by_section_id["4"][0].text == "Governed by Delaware law."
    # preamble clause has no section_id -- present in clauses, absent from the lookup
    assert None not in context.by_section_id
    assert context.references == []  # no reference points in this fixture


def test_load_document_reconstructs_references_and_keeps_them_separate_from_clauses():
    points = [
        SimpleNamespace(id="uuid-1", payload=_payload("doc-1::clause-000", "1", "Definitions clause.")),
        SimpleNamespace(id="uuid-2", payload=_payload("doc-1::clause-001", "2", "References Section 1.")),
        SimpleNamespace(
            id="uuid-3",
            payload=_reference_payload("doc-1", "doc-1::clause-001", "1", "Section 1"),
        ),
    ]
    fake_client = MagicMock()
    fake_client.scroll.return_value = (points, None)

    context = _reader_with_fake_client(fake_client).load_document(
        "doc-1", tenant_id="tenant-1", matter_id="matter-1"
    )

    assert len(context.clauses) == 2  # the reference point must not be counted as a clause
    assert len(context.references) == 1
    assert context.references[0].source_clause_id == "doc-1::clause-001"
    assert context.references[0].referenced_section_id == "1"


def test_load_document_treats_payloads_without_record_type_as_clauses():
    """Backward compatibility: points written before record_type existed
    have neither field and must not silently disappear from a re-deployed
    reader."""
    payload = _payload("doc-1::clause-000", "1", "Some clause.")
    del payload["record_type"]
    points = [SimpleNamespace(id="uuid-1", payload=payload)]
    fake_client = MagicMock()
    fake_client.scroll.return_value = (points, None)

    context = _reader_with_fake_client(fake_client).load_document(
        "doc-1", tenant_id="tenant-1", matter_id="matter-1"
    )

    assert len(context.clauses) == 1
    assert context.references == []


def test_load_document_paginates_through_multiple_scroll_pages():
    page1 = [SimpleNamespace(id="uuid-1", payload=_payload("doc-1::clause-000", "1", "First clause."))]
    page2 = [SimpleNamespace(id="uuid-2", payload=_payload("doc-1::clause-001", "2", "Second clause."))]

    fake_client = MagicMock()
    fake_client.scroll.side_effect = [(page1, "page-2-offset"), (page2, None)]

    context = _reader_with_fake_client(fake_client).load_document(
        "doc-1", tenant_id="tenant-1", matter_id="matter-1"
    )

    assert len(context.clauses) == 2
    assert fake_client.scroll.call_count == 2


def test_bm25_corpus_pairs_every_clause_with_its_metadata():
    points = [SimpleNamespace(id="uuid-1", payload=_payload("doc-1::clause-000", "4", "Some clause text."))]
    fake_client = MagicMock()
    fake_client.scroll.return_value = (points, None)

    context = _reader_with_fake_client(fake_client).load_document(
        "doc-1", tenant_id="tenant-1", matter_id="matter-1"
    )
    corpus = context.bm25_corpus()

    assert len(corpus) == 1
    clause, metadata = corpus[0]
    assert clause.clause_id == "doc-1::clause-000"
    assert metadata.tenant_id == "tenant-1"


def test_scroll_filters_by_document_tenant_and_matter():
    fake_client = MagicMock()
    fake_client.scroll.return_value = ([], None)

    _reader_with_fake_client(fake_client).load_document("doc-1", tenant_id="tenant-1", matter_id="matter-1")

    _, kwargs = fake_client.scroll.call_args
    assert kwargs["collection_name"] == "legal_clauses"
    must_conditions = kwargs["scroll_filter"].must
    filtered_fields = {c.key: c.match.value for c in must_conditions}
    assert filtered_fields == {"document_id": "doc-1", "tenant_id": "tenant-1", "matter_id": "matter-1"}

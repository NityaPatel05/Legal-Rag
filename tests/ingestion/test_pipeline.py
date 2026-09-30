import re
from pathlib import Path

from ingestion.pipeline import ingest_document
from ingestion.schemas import PrivilegeLevel, SectionReference, TaggedClause

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_contract.txt"


class FakeDocumentLoader:
    def load(self, source_path: str) -> str:
        return Path(source_path).read_text(encoding="utf-8")


class FakeLLMClient:
    """Returns a fixed metadata payload; for the batched tag+enrich call
    (R12, tagger.tag_and_enrich_clauses) one item per clause label with a
    fixed context note. Records system prompts so tests can count calls."""

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        # Lazily created: test_ingestion_boundary subclasses this fake with
        # its own __init__ that doesn't call super().
        self.__dict__.setdefault("system_prompts", []).append(system_prompt)
        metadata = {
            "contract_type": "Master Services Agreement",
            "clause_type": "General",
            "parties": ["Acme Corp", "Globex LLC"],
            "effective_date": "2024-01-15",
            "signature_date": None,
            "jurisdiction": "New York",
            "governing_law": "Delaware",
            "contains_pii": False,
        }
        if "context note" in system_prompt:
            labels = re.findall(r'<clause-[0-9a-f]+ id="(c\d+)">', user_prompt)
            return {
                "clauses": [
                    {"id": label, **metadata, "context_note": "This clause is part of a services agreement."}
                    for label in labels
                ]
            }
        return metadata


class FakeEmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]


class FakeVectorStore:
    def __init__(self):
        self.upserted: list[TaggedClause] = []
        self.upserted_references: list[SectionReference] = []
        self.reference_upsert_calls: list[dict] = []
        self.upserted_defined_terms = []
        self.deleted: list[dict] = []
        self.calls: list[str] = []
        self.info: dict | None = None  # what document_info reports

    def document_info(self, *, document_id: str, tenant_id: str, matter_id: str) -> dict | None:
        return self.info

    def delete_stale_versions(self, *, document_id: str, tenant_id: str, matter_id: str, keep_version: str) -> None:
        self.calls.append("delete_stale_versions")
        self.deleted.append({"document_id": document_id, "keep_version": keep_version})

    def delete_document(self, *, document_id: str, tenant_id: str, matter_id: str) -> None:
        self.calls.append("delete_document")
        self.deleted.append({"document_id": document_id, "tenant_id": tenant_id, "matter_id": matter_id})

    def upsert_defined_terms(
        self, *, document_id, tenant_id, matter_id, defined_terms, vector_dim, ingest_version=None
    ) -> None:
        self.upserted_defined_terms.extend(defined_terms)

    def upsert(self, tagged_clauses: list[TaggedClause]) -> None:
        self.calls.append("upsert")
        self.upserted.extend(tagged_clauses)

    def upsert_references(
        self,
        *,
        document_id: str,
        tenant_id: str,
        matter_id: str,
        references: list[SectionReference],
        vector_dim: int,
        ingest_version: str | None = None,
    ) -> None:
        self.upserted_references.extend(references)
        self.reference_upsert_calls.append(
            {"document_id": document_id, "tenant_id": tenant_id, "matter_id": matter_id, "vector_dim": vector_dim}
        )


def test_ingest_document_end_to_end_with_fakes():
    vector_store = FakeVectorStore()

    parsed_document = ingest_document(
        document_id="doc-1",
        source_path=str(FIXTURE_PATH),
        tenant_id="tenant-1",
        matter_id="matter-1",
        privilege_level=PrivilegeLevel.CONFIDENTIAL,
        document_loader=FakeDocumentLoader(),
        llm_client=FakeLLMClient(),
        embedding_client=FakeEmbeddingClient(),
        vector_store=vector_store,
    )

    assert len(parsed_document.clauses) == 7
    assert len(parsed_document.defined_terms) == 1
    assert len(parsed_document.references) == 2

    assert len(vector_store.upserted) == len(parsed_document.clauses)
    for tagged_clause in vector_store.upserted:
        assert tagged_clause.metadata.tenant_id == "tenant-1"
        assert tagged_clause.metadata.matter_id == "matter-1"
        assert tagged_clause.metadata.jurisdiction == "new york"
        assert tagged_clause.metadata.governing_law == "delaware"
        assert tagged_clause.context_note == "This clause is part of a services agreement."
        assert tagged_clause.embedding is not None

    assert len(vector_store.upserted_references) == 2
    assert vector_store.reference_upsert_calls == [
        {"document_id": "doc-1", "tenant_id": "tenant-1", "matter_id": "matter-1", "vector_dim": 1}
    ]


def test_ingestion_makes_one_llm_call_per_batch_plus_the_profile():
    """R12 + batching: metadata and context note come from one merged call
    per batch of up to 8 passages -- not two calls per clause (the original)
    or one per clause (R12 alone)."""
    llm = FakeLLMClient()

    parsed_document = ingest_document(
        document_id="doc-calls",
        source_path=str(FIXTURE_PATH),
        tenant_id="tenant-1",
        matter_id="matter-1",
        document_loader=FakeDocumentLoader(),
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(),
        vector_store=FakeVectorStore(),
    )

    clause_calls = [p for p in llm.system_prompts if "context note" in p]
    n = len(parsed_document.clauses)
    assert len(clause_calls) == -(-n // 8)
    assert len(llm.system_prompts) == len(clause_calls) + 1  # + document profile
    assert all("metadata extraction" in p for p in clause_calls)


def _ingest_fixture(vector_store, llm, **kw):
    return ingest_document(
        document_id="doc-1",
        source_path=str(FIXTURE_PATH),
        tenant_id="tenant-1",
        matter_id="matter-1",
        document_loader=FakeDocumentLoader(),
        llm_client=llm,
        embedding_client=FakeEmbeddingClient(),
        vector_store=vector_store,
        **kw,
    )


def test_new_version_is_written_before_stale_points_are_deleted():
    """B15: delete-then-write lost the whole document if a write failed."""
    store = FakeVectorStore()
    _ingest_fixture(store, FakeLLMClient())
    assert store.calls == ["upsert", "delete_stale_versions"]
    version = store.upserted[0].metadata.ingest_version
    assert version and all(tc.metadata.ingest_version == version for tc in store.upserted)
    assert store.deleted == [{"document_id": "doc-1", "keep_version": version}]


def test_unchanged_document_is_not_reingested():
    """Re-submitting identical content makes no LLM, embedding or index calls."""
    first = FakeVectorStore()
    _ingest_fixture(first, FakeLLMClient())
    store = FakeVectorStore()
    store.info = {"privilege_level": "confidential", "ingest_version": first.upserted[0].metadata.ingest_version}
    llm = FakeLLMClient()
    parsed = _ingest_fixture(store, llm)
    assert parsed.clauses  # still returns the parse
    assert llm.__dict__.get("system_prompts") is None
    assert store.calls == []
    _ingest_fixture(store, llm, force=True)
    assert store.calls == ["upsert", "delete_stale_versions"]


def test_reingest_may_not_downgrade_privilege():
    import pytest

    from ingestion.pipeline import PrivilegeDowngradeError

    store = FakeVectorStore()
    store.info = {"privilege_level": "privileged", "ingest_version": "old"}
    llm = FakeLLMClient()
    with pytest.raises(PrivilegeDowngradeError):
        _ingest_fixture(store, llm, privilege_level=PrivilegeLevel.CONFIDENTIAL)
    assert llm.__dict__.get("system_prompts") is None  # refused before any text left


def test_empty_document_is_refused_before_any_llm_call():
    """B13: used to spend the profile call, then 500 on embed([])."""
    import pytest

    from ingestion.pipeline import EmptyDocumentError

    class Blank:
        def load(self, source_path):
            return "  \n\t "

    llm = FakeLLMClient()
    store = FakeVectorStore()
    with pytest.raises(EmptyDocumentError):
        ingest_document(
            document_id="d", source_path="x", tenant_id="t", matter_id="m", document_loader=Blank(),
            llm_client=llm, embedding_client=FakeEmbeddingClient(), vector_store=store,
        )
    assert llm.__dict__.get("system_prompts") is None and store.calls == []

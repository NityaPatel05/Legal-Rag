from pathlib import Path

from ingestion.pipeline import ingest_document
from ingestion.schemas import PrivilegeLevel, SectionReference, TaggedClause

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_contract.txt"


class FakeDocumentLoader:
    def load(self, source_path: str) -> str:
        return Path(source_path).read_text(encoding="utf-8")


class FakeLLMClient:
    """Returns a fixed metadata payload, plus a fixed context note when the
    prompt asks for one (the merged tag+enrich call, R12). Records system
    prompts so tests can count calls."""

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
            metadata["context_note"] = "This clause is part of a services agreement."
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

    def delete_document(self, *, document_id: str, tenant_id: str, matter_id: str) -> None:
        self.calls.append("delete_document")
        self.deleted.append({"document_id": document_id, "tenant_id": tenant_id, "matter_id": matter_id})

    def upsert_defined_terms(self, *, document_id, tenant_id, matter_id, defined_terms, vector_dim) -> None:
        self.upserted_defined_terms.extend(defined_terms)

    def upsert(self, tagged_clauses: list[TaggedClause]) -> None:
        self.calls.append("upsert")
        self.upserted.extend(tagged_clauses)

    def upsert_references(
        self, *, document_id: str, tenant_id: str, matter_id: str, references: list[SectionReference], vector_dim: int
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


def test_ingestion_makes_one_llm_call_per_clause_plus_the_profile():
    """R12: metadata and context note come from one merged call per clause,
    not a tagging call and an enrichment call."""
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
    assert len(clause_calls) == len(parsed_document.clauses)
    assert len(llm.system_prompts) == len(parsed_document.clauses) + 1  # + document profile
    assert all("metadata extraction" in p for p in clause_calls)

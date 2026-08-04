from pathlib import Path

from ingestion.pipeline import ingest_document
from ingestion.schemas import PrivilegeLevel, SectionReference, TaggedClause

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_contract.txt"


class FakeDocumentLoader:
    def load(self, source_path: str) -> str:
        return Path(source_path).read_text(encoding="utf-8")


class FakeLLMClient:
    """Returns a fixed metadata payload for tagging calls and a fixed context
    note for enrichment calls, distinguished by a marker in the system prompt."""

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        if "context note" in system_prompt:
            return {"context_note": "This clause is part of a services agreement."}
        return {
            "contract_type": "Master Services Agreement",
            "clause_type": "General",
            "parties": ["Acme Corp", "Globex LLC"],
            "effective_date": "2024-01-15",
            "signature_date": None,
            "jurisdiction": "New York",
            "governing_law": "Delaware",
            "contains_pii": False,
        }


class FakeEmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]


class FakeVectorStore:
    def __init__(self):
        self.upserted: list[TaggedClause] = []
        self.upserted_references: list[SectionReference] = []
        self.reference_upsert_calls: list[dict] = []

    def upsert(self, tagged_clauses: list[TaggedClause]) -> None:
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
        assert tagged_clause.metadata.jurisdiction == "New York"
        assert tagged_clause.metadata.governing_law == "Delaware"
        assert tagged_clause.context_note == "This clause is part of a services agreement."
        assert tagged_clause.embedding is not None

    assert len(vector_store.upserted_references) == 2
    assert vector_store.reference_upsert_calls == [
        {"document_id": "doc-1", "tenant_id": "tenant-1", "matter_id": "matter-1", "vector_dim": 1}
    ]

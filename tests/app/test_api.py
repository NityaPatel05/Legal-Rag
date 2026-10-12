"""HTTP-level tests (roadmap R2): the real FastAPI app, the real Qdrant
adapters on an in-memory qdrant-client, and fakes only for the LLM and the
embedding model -- wired in through FastAPI's dependency_overrides.

TestClient is used WITHOUT a `with` block on purpose: that skips the
lifespan hook, which would try to reach a real Qdrant at QDRANT_URL."""

import re
import zlib
from pathlib import Path

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from fastapi.testclient import TestClient  # noqa: E402

from app.dependencies import get_dependencies  # noqa: E402
from app.main import app  # noqa: E402
from ingestion.config import Settings  # noqa: E402
from ingestion.embedding.document_reader import QdrantDocumentReader  # noqa: E402
from ingestion.embedding.indexer import QdrantIndexer  # noqa: E402
from query.filters.audit_log import InMemoryAuditLogger  # noqa: E402
from query.review.store import InMemoryReviewStore  # noqa: E402
from query.observability.langsmith_tracer import NullTraceExporter  # noqa: E402
from query.retrieval.reranker import PassthroughReranker  # noqa: E402
from query.retrieval.vector_search import QdrantVectorSearchClient  # noqa: E402

CONTRACT = (Path(__file__).parents[1] / "ingestion" / "fixtures" / "sample_contract.txt").read_text(
    encoding="utf-8"
)
DIM = 64
_QUOTE = "This Agreement is governed by the laws of the State of Delaware"
_CLAIM = "This Agreement is governed by the laws of the State of Delaware."
_CLAUSE_HEADER = re.compile(r"\[Clause ID: ([^ |]+)[^\]]*\][^\n]*\n([^\[]*)")


class HashingEmbedder:
    """Deterministic bag-of-words vectors, so similar text is near in space."""

    def embed(self, texts):
        vectors = []
        for text in texts:
            v = [0.0] * DIM
            for token in re.findall(r"[a-z]+", text.lower()):
                v[zlib.crc32(token.encode()) % DIM] += 1.0
            vectors.append(v)
        return vectors


class ScriptedLLM:
    """Dispatches on the stable system-prompt phrases listed in CLAUDE.md:
    one input-screen call, one generator call (which also grades the
    evidence), the document profile, and batched tag + enrich calls."""

    def __init__(self):
        self.user_prompts: list[str] = []

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.user_prompts.append(user_prompt)
        if "input screen" in system_prompt:
            return {"is_flagged": False, "route": "clause_lookup"}
        if "legal contract analysis assistant" in system_prompt:
            for clause_id, text in _CLAUSE_HEADER.findall(user_prompt):
                if _QUOTE in text:
                    return {
                        "sufficiency": "sufficient",
                        "answer_text": "free text that must never be returned",
                        "claims": [
                            {
                                "claim_text": _CLAIM,
                                "citation_section_id": clause_id,
                                "supporting_quote": _QUOTE,
                            }
                        ],
                    }
            return {"sufficiency": "insufficient", "answer_text": "Not addressed.", "claims": []}
        if "context note" in system_prompt:  # batched tag + enrich call (R12)
            labels = re.findall(r'<clause-[0-9a-f]+ id="(c\d+)">', user_prompt)
            return {
                "clauses": [
                    {"id": label, "clause_type": "Other", "context_note": "Part of the Acme / Globex services agreement."}
                    for label in labels
                ]
            }
        if "document-level metadata" in system_prompt:
            return {"summary": "Services agreement between Acme and Globex.", "governing_law": "Delaware"}
        return {"clause_type": "Other"}


class FakeDeps:
    def __init__(self, settings: Settings, *, privileged_llm=None):
        self.settings = settings
        client = qdrant_client.QdrantClient(":memory:")
        self.embedding_client = HashingEmbedder()
        self.llm_client = ScriptedLLM()
        self.privileged_llm_client = privileged_llm
        self.indexer = QdrantIndexer("unused", "c")
        self.indexer._client = client
        self.indexer.ensure_collection(DIM)
        self.vector_client = QdrantVectorSearchClient("unused", "c", self.embedding_client)
        self.vector_client._client = client
        self.document_reader = QdrantDocumentReader("unused", "c")
        self.document_reader._client = client
        self.reranker = PassthroughReranker()
        self.audit_logger = InMemoryAuditLogger()
        self.review_store = InMemoryReviewStore()
        self.trace_exporter = NullTraceExporter()
        self.entailment_checker = None


@pytest.fixture
def make_client():
    def _make(settings=None, client_host="127.0.0.1", **kwargs):
        deps = FakeDeps(settings or Settings(api_keys={}, dev_mode=True), **kwargs)
        app.dependency_overrides[get_dependencies] = lambda: deps
        # Loopback client: dev mode (A5) refuses anything else.
        return TestClient(app, client=(client_host, 50000)), deps

    yield _make
    app.dependency_overrides.clear()


def _ingest(client, *, privilege_level="confidential", headers=None, matter="m-1"):
    return client.post(
        "/documents/text",
        json={
            "document_id": "msa-1",
            "source_text": CONTRACT,
            "tenant_id": "t-1",
            "matter_id": matter,
            "privilege_level": privilege_level,
        },
        headers=headers or {},
    )


def _query(client, text="Which law governs this agreement?", *, headers=None, matter="m-1", **filters):
    return client.post(
        "/query",
        json={"document_id": "msa-1", "query_text": text, "tenant_id": "t-1", "matter_id": matter, **filters},
        headers=headers or {},
    )


def test_ingest_then_query_returns_only_validated_claims(make_client):
    client, _ = make_client()
    assert _ingest(client).status_code == 200

    response = _query(client)
    assert response.status_code == 200
    body = response.json()
    assert body["final_status"] == "answered"
    assert body["answer_text"] == _CLAIM
    [citation] = body["citations"]
    assert citation["clause_id"].startswith("msa-1::clause-")
    assert citation["supporting_quote"] == _QUOTE
    assert body["disclaimer"]


def test_governing_law_filter_is_normalized_and_document_level(make_client):
    client, _ = make_client()
    _ingest(client)
    # "DE" -> "delaware" via the gazetteer, matching the normalized
    # document-level value stored on every clause.
    assert _query(client, governing_law="DE").json()["final_status"] == "answered"
    # Nothing passes the filter: a fixed "not found" statement (B11), no LLM call.
    assert _query(client, governing_law="California").json()["final_status"] == "not_found"


def test_unknown_document_is_404(make_client):
    client, _ = make_client()
    assert _query(client).status_code == 404


def test_matter_scoped_key_is_deny_by_default(make_client):
    settings = Settings(api_keys={"k-scoped": {"tenant_id": "t-1", "matters": ["m-1"]}, "k-tenant": "t-1"})
    client, _ = make_client(settings)

    assert _ingest(client).status_code == 401  # no key
    assert _ingest(client, headers={"X-API-Key": "k-scoped"}).status_code == 200
    assert _query(client, headers={"X-API-Key": "k-scoped"}, matter="m-2").status_code == 403
    # Legacy string key stays tenant-wide.
    assert _ingest(client, headers={"X-API-Key": "k-tenant"}, matter="m-2").status_code == 200


def test_empty_api_keys_fails_closed(make_client):
    client, _ = make_client(Settings(api_keys={}, dev_mode=False))
    assert _ingest(client).status_code == 503
    assert _query(client).status_code == 503


def test_dev_mode_refuses_non_loopback_clients(make_client):
    client, _ = make_client(client_host="203.0.113.7")
    assert _ingest(client).status_code == 403


def test_privileged_document_never_reaches_default_llm(make_client):
    client, deps = make_client()
    assert _ingest(client, privilege_level="privileged").status_code == 200
    assert not any("Acme Corp" in p for p in deps.llm_client.user_prompts)

    response = _query(client)
    assert response.status_code == 403
    assert "self-hosted" in response.json()["detail"]
    assert not any("governed by" in p for p in deps.llm_client.user_prompts)


def test_privileged_document_uses_self_hosted_llm(make_client):
    private = ScriptedLLM()
    client, deps = make_client(privileged_llm=private)
    _ingest(client, privilege_level="privileged")

    assert _query(client).json()["final_status"] == "answered"
    assert deps.llm_client.user_prompts == []
    assert private.user_prompts


# --- Human review queue (architecture-review H1/H2) ---


def _reviews(client, path="", matter="m-1", **params):
    return client.get(f"/reviews{path}", params={"tenant_id": "t-1", "matter_id": matter, **params})


def test_flagged_query_is_queued_with_its_evidence_and_scoped_to_the_matter(make_client):
    client, deps = make_client()
    _ingest(client)

    answered = _query(client).json()
    assert answered["query_id"] and answered["review_id"] is None

    flagged = _query(client, governing_law="California").json()
    assert flagged["final_status"] == "not_found"  # still queued: a human confirms absence
    review_id = flagged["review_id"]
    assert review_id == flagged["query_id"]

    [summary] = _reviews(client).json()
    assert summary["review_id"] == review_id and summary["status"] == "open"
    assert _reviews(client, matter="m-2").json() == []
    assert _reviews(client, f"/{review_id}", matter="m-2").status_code == 404

    case = _reviews(client, f"/{review_id}").json()
    assert case["question"] == "Which law governs this agreement?"
    assert case["reason"] and case["first_viewed_at"]
    assert any(e.event_type == "review_queued" for e in deps.audit_logger.events)


def test_reviewer_decision_releases_only_a_revalidated_answer(make_client):
    client, deps = make_client()
    _ingest(client)
    law_clause = _query(client).json()["citations"][0]["clause_id"]
    review_id = _query(client, governing_law="California").json()["review_id"]

    def decide(**claim):
        return client.post(
            f"/reviews/{review_id}/decision",
            json={
                "tenant_id": "t-1",
                "matter_id": "m-1",
                "reviewer": "a.lawyer",
                "decisions": [{"index": None, "action": "edit", "citation": law_clause, **claim}],
            },
        )

    # A quote that isn't in the clause is refused without an override reason.
    bad = decide(claim_text="California law governs.", supporting_quote="the laws of the State of California")
    assert bad.status_code == 422

    good = decide(claim_text=_CLAIM, supporting_quote=_QUOTE)
    assert good.status_code == 200
    outcome = good.json()["outcome"]
    assert outcome["final_status"] == "answered_after_review"
    assert outcome["answer_text"] == _CLAIM
    assert outcome["claims"][0]["validator_passed"] is True

    assert decide(claim_text=_CLAIM, supporting_quote=_QUOTE).status_code == 409
    assert _reviews(client).json() == []  # no longer open
    assert [e.event_type for e in deps.audit_logger.events].count("review_decision") == 1


# --- LLM call budget (fewer calls) ---


def test_llm_call_budget_for_ingest_reingest_and_query(make_client):
    """Ingest: 1 profile call + 1 per batch of 8 passages (was 1 + one per
    passage). Re-ingesting identical text: 0 calls. An answered question:
    2 calls -- input screen + generator (was 4: classifier, router,
    sufficiency grader, generator)."""
    client, deps = make_client()
    body = _ingest(client).json()
    ingest_calls = len(deps.llm_client.user_prompts)
    assert ingest_calls == 1 + -(-body["num_clauses"] // 8)

    assert _ingest(client).status_code == 200
    assert len(deps.llm_client.user_prompts) == ingest_calls  # unchanged document: skipped

    assert _query(client).json()["final_status"] == "answered"
    assert len(deps.llm_client.user_prompts) == ingest_calls + 2


def test_empty_document_is_422_not_500(make_client):
    client, deps = make_client()
    response = client.post(
        "/documents/text",
        json={"document_id": "empty-1", "tenant_id": "t-1", "matter_id": "m-1", "source_text": "   \n  "},
    )
    assert response.status_code == 422
    assert deps.llm_client.user_prompts == []


def test_upstream_llm_outage_is_503_with_retry_after(make_client):
    from ingestion.metadata.llm_client import LLMCallError

    class DownLLM(ScriptedLLM):
        def generate_json(self, system_prompt, user_prompt):
            raise LLMCallError("provider down")

    client, deps = make_client()
    _ingest(client)
    deps.llm_client = DownLLM()
    response = _query(client)
    assert response.status_code == 503
    assert response.headers["retry-after"]


def test_crlf_contract_parses_like_lf(make_client):
    """B17: CRLF text used to disable ALL-CAPS heading detection."""
    client, _ = make_client()
    lf = _ingest(client).json()["num_clauses"]
    response = client.post(
        "/documents/text",
        json={"document_id": "msa-crlf", "tenant_id": "t-1", "matter_id": "m-1",
              "source_text": CONTRACT.replace("\n", "\r\n")},
    )
    assert response.json()["num_clauses"] == lf


def test_reviews_can_be_listed_across_all_statuses(make_client):
    """L2: the status filter defaulted to open and None was unreachable."""
    client, _ = make_client()
    _ingest(client)
    _query(client, governing_law="California")
    assert len(_reviews(client, status="all").json()) == 1
    assert _reviews(client, status="resolved").json() == []


# --- Endpoints a UI needs: document list / detail, upload, review status ---


def _documents(client, path="", matter="m-1", headers=None):
    return client.get(
        f"/documents{path}", params={"tenant_id": "t-1", "matter_id": matter}, headers=headers or {}
    )


def test_documents_are_listed_per_matter_with_document_level_metadata(make_client):
    client, _ = make_client()
    assert _documents(client).json() == []
    num_clauses = _ingest(client).json()["num_clauses"]

    [doc] = _documents(client).json()
    assert doc["document_id"] == "msa-1"
    assert doc["num_clauses"] == num_clauses
    assert doc["governing_law"] == "delaware"  # normalized, merged from the profile
    assert doc["privilege_level"] == "confidential"
    assert _documents(client, matter="m-2").json() == []


def test_document_detail_returns_stored_clauses_in_order_and_is_audited(make_client):
    client, deps = make_client()
    _ingest(client)
    citation = _query(client).json()["citations"][0]

    response = _documents(client, "/msa-1")
    assert response.status_code == 200
    clauses = response.json()["clauses"]
    offsets = [c["start_offset"] for c in clauses]
    assert offsets == sorted(offsets)
    cited = next(c for c in clauses if c["clause_id"] == citation["clause_id"])
    assert citation["supporting_quote"] in cited["text"]
    assert any(e.event_type == "document_read" for e in deps.audit_logger.events)

    assert _documents(client, "/msa-1", matter="m-2").status_code == 404
    assert _documents(client, "/nope").status_code == 404


def test_document_routes_are_matter_scoped(make_client):
    settings = Settings(api_keys={"k-scoped": {"tenant_id": "t-1", "matters": ["m-1"]}})
    client, _ = make_client(settings)
    headers = {"X-API-Key": "k-scoped"}
    _ingest(client, headers=headers)

    assert _documents(client).status_code == 401
    assert len(_documents(client, headers=headers).json()) == 1
    assert _documents(client, matter="m-2", headers=headers).status_code == 403
    assert _documents(client, "/msa-1", matter="m-2", headers=headers).status_code == 403


def _upload(client, filename, data, *, document_id="msa-up"):
    return client.post(
        "/documents/file",
        data={"document_id": document_id, "tenant_id": "t-1", "matter_id": "m-1"},
        files={"file": (filename, data)},
    )


def test_text_upload_ingests_like_the_text_endpoint(make_client):
    client, _ = make_client()
    expected = _ingest(client).json()["num_clauses"]
    response = _upload(client, "msa.txt", ("﻿" + CONTRACT).encode("utf-8"))
    assert response.status_code == 200
    assert response.json()["num_clauses"] == expected


def test_upload_rejects_bad_type_content_encoding_and_size(make_client, monkeypatch):
    import app.main as main

    client, deps = make_client()
    assert _upload(client, "msa.exe", b"MZ").status_code == 415
    assert _upload(client, "msa.pdf", b"not a pdf").status_code == 415  # magic bytes
    assert _upload(client, "msa.txt", "é".encode("latin-1")).status_code == 422
    assert _upload(client, "msa.txt", CONTRACT.encode(), document_id="bad id!").status_code == 422

    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)
    assert _upload(client, "msa.txt", CONTRACT.encode()).status_code == 413
    assert deps.llm_client.user_prompts == []


def test_pdf_upload_goes_through_the_document_loader(make_client):
    seen = {}

    class FakeDocling:
        def load(self, source_path):
            path = Path(source_path)
            seen["suffix"], seen["existed"], seen["path"] = path.suffix, path.exists(), path
            return CONTRACT

    client, deps = make_client()
    deps.document_loader = FakeDocling()
    response = _upload(client, "../../etc/msa.pdf", b"%PDF-1.7 ...")
    assert response.status_code == 200
    assert response.json()["num_clauses"] > 1
    assert seen["suffix"] == ".pdf" and seen["existed"]
    assert seen["path"].name == "upload.pdf"  # client filename never used as a path
    assert not seen["path"].exists()  # temp file cleaned up


def test_pdf_upload_without_docling_is_501_and_bad_pdf_is_422(make_client):
    class NoDocling:
        def load(self, source_path):
            raise ImportError("No module named 'docling'")

    class BrokenPdf:
        def load(self, source_path):
            raise ValueError("corrupt xref table")

    client, deps = make_client()
    deps.document_loader = NoDocling()
    assert _upload(client, "msa.pdf", b"%PDF-1.7").status_code == 501
    deps.document_loader = BrokenPdf()
    response = _upload(client, "msa.pdf", b"%PDF-1.7")
    assert response.status_code == 422
    assert "xref" not in response.json()["detail"]


def test_review_status_polling_does_not_start_the_reviewer_clock(make_client):
    client, deps = make_client()
    _ingest(client)
    law_clause = _query(client).json()["citations"][0]["clause_id"]
    review_id = _query(client, governing_law="California").json()["review_id"]

    status = _reviews(client, f"/{review_id}/status").json()
    assert status == {"review_id": review_id, "status": "open", "outcome": None}
    assert deps.review_store.get(review_id).first_viewed_at is None
    assert _reviews(client, f"/{review_id}/status", matter="m-2").status_code == 404

    client.post(
        f"/reviews/{review_id}/decision",
        json={
            "tenant_id": "t-1",
            "matter_id": "m-1",
            "reviewer": "a.lawyer",
            "decisions": [
                {"index": None, "action": "edit", "citation": law_clause, "claim_text": _CLAIM, "supporting_quote": _QUOTE}
            ],
        },
    )
    status = _reviews(client, f"/{review_id}/status").json()
    assert status["status"] == "resolved"
    assert status["outcome"]["answer_text"] == _CLAIM


def test_claim_check_reports_every_claim_without_deciding(make_client):
    client, deps = make_client()
    _ingest(client)
    law_clause = _query(client).json()["citations"][0]["clause_id"]
    review_id = _query(client, governing_law="California").json()["review_id"]

    def check(decisions):
        return client.post(
            f"/reviews/{review_id}/check",
            json={"tenant_id": "t-1", "matter_id": "m-1", "decisions": decisions},
        )

    edit = {"index": None, "action": "edit", "citation": law_clause}
    response = check([
        {**edit, "claim_text": _CLAIM, "supporting_quote": _QUOTE},
        {**edit, "claim_text": "California law governs.", "supporting_quote": "the laws of the State of California"},
    ])
    assert response.status_code == 200
    good, bad = response.json()
    assert good["validator_passed"] is True and good["resolved_clause_id"] == law_clause
    assert bad["validator_passed"] is False and bad["failure_reason"]

    case = deps.review_store.get(review_id)
    assert case.status.value == "open" and case.outcome is None and case.first_viewed_at is None
    assert not any(e.event_type == "review_decision" for e in deps.audit_logger.events)

    assert check([{"index": 99, "action": "approve"}]).status_code == 422
    assert check([{**edit, "claim_text": _CLAIM}]).status_code == 422  # incomplete edit
    assert client.post(
        f"/reviews/{review_id}/check", json={"tenant_id": "t-1", "matter_id": "m-2", "decisions": []}
    ).status_code == 404

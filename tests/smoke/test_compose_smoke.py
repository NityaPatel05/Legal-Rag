"""End-to-end smoke test against a running stack (R14): ingest the fixture
contract over HTTP, query it, and assert at least one citation survived
validation. This is the only test that exercises the real Groq, BGE-M3 and
Qdrant together -- every other test uses fakes or in-memory Qdrant, which is
how B1-B5 (see CLAUDE.md) went unnoticed.

Skipped unless SMOKE_BASE_URL is set, so `pytest tests/` stays offline:

    docker compose up --build -d
    SMOKE_BASE_URL=http://localhost:8000 pytest tests/smoke/ -v

The app fails closed without API_KEYS, and dev mode only admits loopback
clients -- a request from the host into a container arrives over the Docker
bridge, not loopback. So against the compose stack, configure a key in .env
(e.g. API_KEYS='{"smoke-key": "smoke-tenant"}') and set SMOKE_API_KEY to it
(and SMOKE_TENANT_ID if the key is bound to another tenant). If the key is
limited to specific matters, set SMOKE_MATTER_ID to one of them.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

from query.validation.citation_validator import _normalize_for_quote_match

BASE_URL = os.environ.get("SMOKE_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("SMOKE_API_KEY")
TENANT_ID = os.environ.get("SMOKE_TENANT_ID", "smoke-tenant")
MATTER_ID = os.environ.get("SMOKE_MATTER_ID")
# Ingestion makes ~2 LLM calls per clause plus a document profile; on the
# Groq free tier with backoff that can take a few minutes.
TIMEOUT_SECONDS = float(os.environ.get("SMOKE_TIMEOUT_SECONDS", "600"))

FIXTURE = Path(__file__).parent.parent / "ingestion" / "fixtures" / "sample_contract.txt"

pytestmark = pytest.mark.skipif(not BASE_URL, reason="SMOKE_BASE_URL not set (needs a running stack)")


def _call(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, {"detail": exc.read().decode("utf-8", "replace")}


@pytest.fixture(scope="module")
def ingested_document() -> dict:
    # Fresh ids per run: re-ingest is delete-then-write, but a unique matter
    # keeps parallel/leftover runs from sharing points at all.
    ids = {
        "document_id": "smoke-msa",
        "tenant_id": TENANT_ID,
        "matter_id": MATTER_ID or f"smoke-{uuid.uuid4().hex[:8]}",
    }
    status, body = _call(
        "POST", "/documents/text", {**ids, "source_text": FIXTURE.read_text(encoding="utf-8")}
    )
    assert status == 200, body
    assert body["num_clauses"] >= 5, body  # fixture has 4 numbered sections + subsections
    return ids


def test_health() -> None:
    status, body = _call("GET", "/health")
    assert (status, body) == (200, {"status": "ok"})


def test_query_returns_a_validated_citation(ingested_document: dict) -> None:
    status, body = _call(
        "POST", "/query", {**ingested_document, "query_text": "What law governs this Agreement?"}
    )
    assert status == 200, body
    assert body["final_status"] in ("answered", "answered_with_stripped_claims"), body
    assert body["answer_text"], body
    assert body["citations"], body

    # Same normalization the validator applies, so this checks the quote
    # really comes from the contract rather than re-testing whitespace rules.
    fixture_text = _normalize_for_quote_match(FIXTURE.read_text(encoding="utf-8"))
    for citation in body["citations"]:
        # Invariants 2 + 3: resolved to a clause, backed by a verbatim quote.
        assert citation["clause_id"], citation
        assert citation["supporting_quote"], citation
        quote = _normalize_for_quote_match(citation["supporting_quote"]).strip(" .\"'")
        assert quote in fixture_text, citation
    assert "delaware" in body["answer_text"].lower(), body


def test_other_matter_cannot_read_the_document(ingested_document: dict) -> None:
    """Invariant 6 end to end: same tenant and document_id, different matter
    -> 404 (filtered out), or 403 when the API key is matter-scoped."""
    status, body = _call(
        "POST",
        "/query",
        {**ingested_document, "matter_id": "smoke-other-matter", "query_text": "What law governs?"},
    )
    assert status in (403, 404), body

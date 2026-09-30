import hashlib
import json
import logging
from unittest.mock import MagicMock

from query.observability.langsmith_tracer import LangSmithTraceExporter, NullTraceExporter
from query.schemas import JailbreakClassification, QueryResult, RetrievalTraceStep, RetrievedClause, RouteType
from tests.query.factories import make_clause, make_metadata


def _query_result() -> QueryResult:
    return QueryResult(
        query_text="What law governs?",
        route=RouteType.CLAUSE_LOOKUP,
        jailbreak_check=JailbreakClassification(is_flagged=False),
        retrieved_clauses=[],
    )


def test_null_exporter_does_nothing():
    NullTraceExporter().export(_query_result())  # should not raise


def test_langsmith_exporter_no_ops_without_an_api_key():
    """Tracing is best-effort observability, never a correctness gate -- no
    API key configured must not raise or attempt any import/network call."""
    exporter = LangSmithTraceExporter(api_key=None)
    exporter.export(_query_result())  # should not raise, no client ever constructed


def test_langsmith_exporter_submits_a_run_when_configured():
    fake_client = MagicMock()
    exporter = LangSmithTraceExporter(api_key="fake-key", project="test-project")
    exporter._client = fake_client  # bypass the lazy `langsmith` import entirely

    exporter.export(_query_result())

    fake_client.create_run.assert_called_once()
    _, kwargs = fake_client.create_run.call_args
    assert kwargs["project_name"] == "test-project"
    assert kwargs["outputs"]["route"] == "clause_lookup"


# --- B12: never raise, never export raw text ---

SECRET_QUERY = "Does Acme owe John Smith the $2M earn-out?"
SECRET_CLAUSE_ID = "acme-smith-spa::clause-0007"


def _rich_result() -> QueryResult:
    return QueryResult(
        query_text=SECRET_QUERY,
        route=RouteType.CLAUSE_LOOKUP,
        jailbreak_check=JailbreakClassification(is_flagged=False),
        retrieved_clauses=[
            RetrievedClause(clause=make_clause(SECRET_CLAUSE_ID), metadata=make_metadata(), score=1.0, source="hybrid")
        ],
        trace=[
            RetrievalTraceStep(stage="hybrid_retrieve", clause_ids=[SECRET_CLAUSE_ID, "acme-smith-spa::clause-0008"]),
            RetrievalTraceStep(stage="rerank", clause_ids=[SECRET_CLAUSE_ID]),
        ],
    )


def test_export_swallows_langsmith_failures_and_logs(caplog):
    fake_client = MagicMock()
    fake_client.create_run.side_effect = ConnectionError("LangSmith is down")
    exporter = LangSmithTraceExporter(api_key="fake-key")
    exporter._client = fake_client

    with caplog.at_level(logging.WARNING, logger="query.observability.langsmith_tracer"):
        exporter.export(_rich_result())  # must not raise

    assert "trace export failed" in caplog.text


def test_export_swallows_client_construction_failures(monkeypatch):
    exporter = LangSmithTraceExporter(api_key="fake-key")

    def boom():
        raise RuntimeError("bad config")

    monkeypatch.setattr(exporter, "_client_instance", boom)
    exporter.export(_rich_result())  # must not raise


def test_export_sends_hashes_and_counts_not_text():
    fake_client = MagicMock()
    exporter = LangSmithTraceExporter(api_key="fake-key")
    exporter._client = fake_client

    exporter.export(_rich_result())

    _, kwargs = fake_client.create_run.call_args
    sent = json.dumps({"inputs": kwargs["inputs"], "outputs": kwargs["outputs"]})
    for secret in ("Acme", "Smith", "earn-out", "acme-smith-spa"):
        assert secret not in sent

    assert kwargs["inputs"]["query_sha256"] == hashlib.sha256(SECRET_QUERY.encode()).hexdigest()[:16]
    assert kwargs["inputs"]["query_chars"] == len(SECRET_QUERY)
    outputs = kwargs["outputs"]
    assert outputs["retrieved_count"] == 1
    assert [(s["stage"], s["count"]) for s in outputs["trace"]] == [("hybrid_retrieve", 2), ("rerank", 1)]
    # Same clause -> same hash across stages, so runs stay correlatable.
    clause_hash = hashlib.sha256(SECRET_CLAUSE_ID.encode()).hexdigest()[:16]
    assert outputs["retrieved_clause_hashes"] == [clause_hash]
    assert outputs["trace"][1]["clause_hashes"] == [clause_hash]

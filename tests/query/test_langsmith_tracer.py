from unittest.mock import MagicMock

from query.observability.langsmith_tracer import LangSmithTraceExporter, NullTraceExporter
from query.schemas import JailbreakClassification, QueryResult, RouteType


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
    assert kwargs["inputs"]["query_text"] == "What law governs?"
    assert kwargs["outputs"]["route"] == "clause_lookup"

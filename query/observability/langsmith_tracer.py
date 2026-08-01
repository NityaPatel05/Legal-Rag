"""Full retrieval trace per query — CLAUDE.md's Observability tech-stack
decision (LangSmith). TraceExporter is the seam query/pipeline.py depends
on; LangSmithTraceExporter is the production adapter. Exporting a trace is
best-effort observability, never a correctness gate — a query must never
fail because tracing is unconfigured or the LangSmith API is down, so this
adapter no-ops (rather than raises) whenever no API key is set."""

from __future__ import annotations

from typing import Protocol

from query.schemas import QueryResult


class TraceExporter(Protocol):
    def export(self, query_result: QueryResult) -> None: ...


class NullTraceExporter:
    def export(self, query_result: QueryResult) -> None:
        pass


class LangSmithTraceExporter:
    def __init__(self, api_key: str | None, project: str = "legal-contract-rag"):
        self._api_key = api_key
        self._project = project
        self._client = None

    def _client_instance(self):
        if self._client is None:
            from langsmith import Client  # lazy import

            self._client = Client(api_key=self._api_key)
        return self._client

    def export(self, query_result: QueryResult) -> None:
        if not self._api_key:
            return

        client = self._client_instance()
        client.create_run(
            name="query_pipeline",
            run_type="chain",
            project_name=self._project,
            inputs={"query_text": query_result.query_text},
            outputs={
                "route": query_result.route.value,
                "jailbreak_flagged": query_result.jailbreak_check.is_flagged,
                "trace": [step.model_dump() for step in query_result.trace],
                "retrieved_clause_ids": [rc.clause.clause_id for rc in query_result.retrieved_clauses],
            },
        )

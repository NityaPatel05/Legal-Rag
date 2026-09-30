"""Full retrieval trace per query — CLAUDE.md's Observability tech-stack
decision (LangSmith). TraceExporter is the seam query/pipeline.py depends
on; LangSmithTraceExporter is the production adapter.

Exporting a trace is best-effort observability, never a correctness gate:
export runs synchronously on the /query path, so it must never raise. The
original adapter had no try/except, so a LangSmith outage (or a bad key)
turned every /query into a 500. Now every failure is logged and swallowed,
the client gets a short HTTP timeout so an outage costs a few seconds rather
than the SDK default, and it no-ops whenever no API key is set.

What leaves the boundary: LangSmith is a third-party SaaS, and invariant
#12 keeps contract content and PII away from third parties. The raw
question can carry client names or facts, and clause ids embed the
caller-chosen document_id (often a contract or counterparty name). So the
export carries a truncated sha256 of the query text and of each clause id
-- enough to correlate runs and spot the same clause across stages -- plus
counts, stage names, the route and the jailbreak flag. Never the text.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

from query.schemas import QueryResult

logger = logging.getLogger(__name__)

# 16 hex chars = 64 bits: collision-free for correlation at this scale, and
# short enough that nobody mistakes it for a reversible identifier. Unsalted
# on purpose (the same clause hashes the same across runs); a guessable
# short query can be confirmed by hashing a guess, which is why the full
# text is never sent but also why this is correlation, not anonymization.
_HASH_HEX_CHARS = 16
# Seconds-scale, not the SDK default: this call sits on the request path.
_CLIENT_TIMEOUT_MS = 5_000


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:_HASH_HEX_CHARS]


class TraceExporter(Protocol):
    def export(self, query_result: QueryResult) -> None: ...


class NullTraceExporter:
    def export(self, query_result: QueryResult) -> None:
        pass


def build_trace_payload(query_result: QueryResult) -> tuple[dict, dict]:
    """(inputs, outputs) for the LangSmith run -- hashes and counts only."""
    inputs = {
        "query_sha256": _digest(query_result.query_text),
        "query_chars": len(query_result.query_text),
    }
    outputs = {
        "route": query_result.route.value,
        "jailbreak_flagged": query_result.jailbreak_check.is_flagged,
        "retrieved_count": len(query_result.retrieved_clauses),
        "retrieved_clause_hashes": [_digest(rc.clause.clause_id) for rc in query_result.retrieved_clauses],
        "trace": [
            {
                "stage": step.stage,
                "count": len(step.clause_ids),
                "clause_hashes": [_digest(cid) for cid in step.clause_ids],
            }
            for step in query_result.trace
        ],
    }
    return inputs, outputs


class LangSmithTraceExporter:
    def __init__(self, api_key: str | None, project: str = "legal-contract-rag"):
        self._api_key = api_key
        self._project = project
        self._client = None

    def _client_instance(self):
        if self._client is None:
            from langsmith import Client  # lazy import

            self._client = Client(api_key=self._api_key, timeout_ms=_CLIENT_TIMEOUT_MS)
        return self._client

    def export(self, query_result: QueryResult) -> None:
        if not self._api_key:
            return
        try:
            inputs, outputs = build_trace_payload(query_result)
            self._client_instance().create_run(
                name="query_pipeline",
                run_type="chain",
                project_name=self._project,
                inputs=inputs,
                outputs=outputs,
            )
        except Exception:  # noqa: BLE001 -- observability must never fail a query
            logger.warning("LangSmith trace export failed; continuing without it", exc_info=True)

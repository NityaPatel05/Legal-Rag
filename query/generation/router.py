"""Query router — architecture.md B3: classify into clause lookup /
comparison / contradiction-check / ambiguous-out-of-scope via prompted LLM,
before retrieval runs."""

from __future__ import annotations

from ingestion.metadata.llm_client import LLMClient
from query.schemas import RouteType

SYSTEM_PROMPT = """You are a query router for a legal contract analysis \
assistant. Classify the user's question into exactly one of: \
"clause_lookup" (asking what a specific clause/section says), "comparison" \
(comparing two clauses, documents, or versions), "contradiction_check" \
(asking whether two clauses conflict), or "ambiguous_out_of_scope" (not a \
contract-clause question, or too vague to route). Respond with JSON only: \
{"route": "..."}"""


def classify_route(query_text: str, *, llm_client: LLMClient) -> RouteType:
    result = llm_client.generate_json(SYSTEM_PROMPT, query_text)
    try:
        return RouteType(result.get("route"))
    except ValueError:
        return RouteType.AMBIGUOUS_OUT_OF_SCOPE

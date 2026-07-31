"""Query rewriting for the CRAG evaluator's "insufficient" retry branch —
architecture.md B11."""

from __future__ import annotations

from ingestion.metadata.llm_client import LLMClient

SYSTEM_PROMPT = """The previous search for contract clauses didn't return \
enough to answer the question. Rewrite the question as a different search \
query that might retrieve the missing clauses — try different phrasing, \
synonyms for legal terms, or a more general version of the question. \
Respond with JSON only: {"rewritten_query": "..."}"""


def rewrite_query(original_query_text: str, current_query_text: str, *, llm_client: LLMClient) -> str:
    user_prompt = (
        f"Original question: {original_query_text}\n"
        f"Most recent search query: {current_query_text}\n"
        "Rewrite it to try a different retrieval angle."
    )
    result = llm_client.generate_json(SYSTEM_PROMPT, user_prompt)
    return result.get("rewritten_query") or current_query_text

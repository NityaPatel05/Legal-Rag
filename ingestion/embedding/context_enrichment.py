"""Contextual retrieval enrichment — architecture.md §4: prepend a short
LLM-generated context note to each clause before embedding, so an isolated
chunk still carries enough context to be retrieved/understood correctly
(architecture.md §1.2, cross-referential language)."""

from __future__ import annotations

from ingestion.metadata.llm_client import LLMClient
from ingestion.schemas import Clause

SYSTEM_PROMPT = """You write a one-to-two sentence context note for a contract \
clause, to be prepended before the clause text when it's embedded for \
retrieval. The note should situate the clause: what section it's in, what \
topic it covers, and how it relates to the rest of the contract if that's \
inferable from the heading alone. Do not repeat the clause text verbatim. \
Respond with JSON only: {"context_note": "..."}"""


def enrich_clause(clause: Clause, *, llm_client: LLMClient) -> str:
    user_prompt = f"Clause heading: {clause.heading}\n\nClause text:\n{clause.text}"
    result = llm_client.generate_json(SYSTEM_PROMPT, user_prompt)
    return result["context_note"]

"""CRAG-style three-way sufficiency evaluator — CLAUDE.md #6:
sufficient -> generate; ambiguous -> explicit flag for attorney review (not
a guess); insufficient -> rewrite query and retry."""

from __future__ import annotations

from enum import Enum

from ingestion.metadata.llm_client import LLMClient
from query.schemas import RetrievedClause

SYSTEM_PROMPT = """You are grading whether retrieved contract clauses are \
enough to answer a question about a contract. Classify as exactly one of: \
"sufficient" (the clauses clearly answer the question), "ambiguous" (the \
clauses give conflicting or multiple reasonable answers — a human should \
decide, don't guess), or "insufficient" (the clauses don't address the \
question at all, or something is clearly missing). Respond with JSON only: \
{"sufficiency": "..."}"""


class SufficiencyDecision(str, Enum):
    SUFFICIENT = "sufficient"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT = "insufficient"


def evaluate_sufficiency(
    query_text: str, retrieved_clauses: list[RetrievedClause], *, llm_client: LLMClient
) -> SufficiencyDecision:
    if not retrieved_clauses:
        return SufficiencyDecision.INSUFFICIENT

    context = "\n\n".join(
        f"[Section {rc.clause.section_id}]\n{rc.clause.text}" for rc in retrieved_clauses
    )
    result = llm_client.generate_json(SYSTEM_PROMPT, f"Question: {query_text}\n\nClauses:\n{context}")
    try:
        return SufficiencyDecision(result.get("sufficiency"))
    except ValueError:
        # Never silently guess "sufficient" on a malformed response -- default
        # to the safe, human-review path instead.
        return SufficiencyDecision.AMBIGUOUS

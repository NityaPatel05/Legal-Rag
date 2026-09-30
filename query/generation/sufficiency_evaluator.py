"""CRAG-style three-way sufficiency evaluator — CLAUDE.md #6:
sufficient -> generate; ambiguous -> explicit flag for attorney review (not
a guess); insufficient -> rewrite query and retry.

"ambiguous" means the clauses support more than one reasonable reading. It
does NOT mean "the clauses conflict" when the user asked whether they
conflict: for comparison / contradiction-check routes, a conflict is the
answer, not a reason to refuse. The route is passed in so the grader is told
that, and crag_graph.py additionally lets those routes proceed to
generation on "ambiguous"."""

from __future__ import annotations

from enum import Enum

from ingestion.metadata.llm_client import LLMClient
from query.schemas import RetrievedClause, RouteType

SYSTEM_PROMPT = """You are grading whether retrieved contract clauses are \
enough to answer a question about a contract. Classify as exactly one of: \
"sufficient" (the clauses clearly answer the question), "ambiguous" (the \
clause language supports multiple reasonable readings — a human should \
decide, don't guess), or "insufficient" (the clauses don't address the \
question at all, or something is clearly missing). If the question itself \
asks whether clauses conflict or how they differ, clauses that conflict or \
differ are "sufficient" to answer it. Respond with JSON only: \
{"sufficiency": "..."}"""


class SufficiencyDecision(str, Enum):
    SUFFICIENT = "sufficient"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT = "insufficient"


def evaluate_sufficiency(
    query_text: str,
    retrieved_clauses: list[RetrievedClause],
    *,
    llm_client: LLMClient,
    route: RouteType | None = None,
) -> SufficiencyDecision:
    if not retrieved_clauses:
        return SufficiencyDecision.INSUFFICIENT

    context = "\n\n".join(
        f"[Clause {rc.clause.clause_id} | Section {rc.clause.section_id or 'unnumbered'}]\n{rc.clause.text}"
        for rc in retrieved_clauses
    )
    route_line = f"Question type: {route.value}\n" if route else ""
    result = llm_client.generate_json(
        SYSTEM_PROMPT, f"{route_line}Question: {query_text}\n\nClauses:\n{context}"
    )
    try:
        return SufficiencyDecision(result.get("sufficiency"))
    except ValueError:
        # Never silently guess "sufficient" on a malformed response -- default
        # to the safe, human-review path instead.
        return SufficiencyDecision.AMBIGUOUS

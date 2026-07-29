"""Long-context reorder — architecture.md B7: put the best clause first and
last, mitigating "lost in the middle" degradation on the items sandwiched
between them. Runs after reranking/expansion, right before generation."""

from __future__ import annotations

from query.schemas import RetrievedClause


def reorder_for_generation(clauses: list[RetrievedClause]) -> list[RetrievedClause]:
    if len(clauses) <= 2:
        return list(clauses)

    ranked = sorted(clauses, key=lambda c: c.score, reverse=True)
    front = ranked[0::2]
    back = ranked[1::2]
    back.reverse()
    return front + back

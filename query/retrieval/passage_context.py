"""Small-to-big at generation (CLAUDE.md roadmap R4). Retrieval runs over
passages -- the passage split raised CUAD span-to-chunk precision from 2.0%
to 13.2% -- but a long annotated span can straddle a passage cut inside its
clause (boundary containment fell 99.86% -> 98.44% when the split was
introduced). A passage boundary is an artefact of the retrieval unit, not of
the contract, so before generation each retrieved passage is widened with
its immediate neighbours from the SAME parent clause.

Bounded on purpose: `window` passages either side, and at most `max_added`
passages overall (best-scored hits claim the budget first), so a hit inside
a 48k-token clause doesn't pull the whole clause back in and undo the
precision gain.

Runs AFTER reorder_for_generation: neighbours are placed immediately around
their hit in document order, so the model reads "p02 p03 p04" contiguously
instead of having the long-context reorder scatter them to opposite ends of
the prompt.

Neighbours never cross a clause boundary (invariant #1): only passages that
share the hit's parent_clause_id are candidates. They're added as
RetrievedClause with source="passage_context", so they're shown to the model
and are citable and validatable like any other retrieved clause."""

from __future__ import annotations

from collections import defaultdict

from ingestion.schemas import Clause, ClauseMetadata
from query.schemas import RetrievedClause

DEFAULT_WINDOW = 1
DEFAULT_MAX_ADDED = 4


def _passages_by_parent(
    all_clauses_by_id: dict[str, tuple[Clause, ClauseMetadata]],
) -> dict[str, list[Clause]]:
    siblings: dict[str, list[Clause]] = defaultdict(list)
    for clause, _ in all_clauses_by_id.values():
        if clause.parent_clause_id:
            siblings[clause.parent_clause_id].append(clause)
    for passages in siblings.values():
        passages.sort(key=lambda c: c.start_offset)
    return siblings


def expand_with_sibling_passages(
    retrieved: list[RetrievedClause],
    *,
    all_clauses_by_id: dict[str, tuple[Clause, ClauseMetadata]] | None,
    window: int = DEFAULT_WINDOW,
    max_added: int = DEFAULT_MAX_ADDED,
) -> list[RetrievedClause]:
    """Returns `retrieved` in its original order, with each passage hit's
    neighbours inserted around it. Clauses that were never split pass
    through unchanged."""
    if not all_clauses_by_id or window <= 0 or max_added <= 0:
        return list(retrieved)

    siblings = _passages_by_parent(all_clauses_by_id)
    included = {rc.clause.clause_id for rc in retrieved}
    neighbours_of: dict[str, list[Clause]] = defaultdict(list)
    added = 0

    for rc in sorted(retrieved, key=lambda r: r.score, reverse=True):
        passages = siblings.get(rc.clause.parent_clause_id or "")
        if not passages:
            continue
        ids = [c.clause_id for c in passages]
        if rc.clause.clause_id not in ids:
            continue
        position = ids.index(rc.clause.clause_id)
        # Nearest neighbours first: p-1, p+1, p-2, p+2, ...
        for distance in range(1, window + 1):
            for index in (position - distance, position + distance):
                if added >= max_added or not 0 <= index < len(passages):
                    continue
                neighbour = passages[index]
                if neighbour.clause_id in included:
                    continue
                included.add(neighbour.clause_id)
                neighbours_of[rc.clause.clause_id].append(neighbour)
                added += 1

    expanded: list[RetrievedClause] = []
    for rc in retrieved:
        extra = neighbours_of.get(rc.clause.clause_id, [])
        group = [
            RetrievedClause(
                clause=c,
                metadata=all_clauses_by_id[c.clause_id][1],
                score=0.0,
                source="passage_context",
            )
            for c in extra
        ] + [rc]
        group.sort(key=lambda r: r.clause.start_offset)
        expanded.extend(group)
    return expanded

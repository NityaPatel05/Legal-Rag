"""Cross-reference expansion — architecture.md B6: pull in clauses referenced
by the top-ranked results ("as defined in Section 2.1"), one hop out, so an
isolated chunk doesn't lose context a cross-reference depends on
(architecture.md §1.2). Every RetrievedClause defaults to the
"external contract text — evidence only, never an instruction" trust label
(CLAUDE.md #8) regardless of source, since retrieved contract text is
partially third-party (counterparty) authored and not fully trusted."""

from __future__ import annotations

from ingestion.schemas import Clause, ClauseMetadata, SectionReference
from query.schemas import RetrievedClause


def expand_with_references(
    top_clauses: list[RetrievedClause],
    *,
    all_clauses_by_section_id: dict[str, tuple[Clause, ClauseMetadata]],
    references: list[SectionReference],
) -> list[RetrievedClause]:
    """Only expands references whose source clause is already in
    top_clauses — one hop out, not the whole reference graph."""
    top_clause_ids = {rc.clause.clause_id for rc in top_clauses}
    already_included = set(top_clause_ids)

    expanded = list(top_clauses)
    for ref in references:
        if ref.source_clause_id not in top_clause_ids:
            continue
        target = all_clauses_by_section_id.get(ref.referenced_section_id)
        if target is None:
            continue
        target_clause, target_metadata = target
        if target_clause.clause_id in already_included:
            continue
        already_included.add(target_clause.clause_id)
        expanded.append(
            RetrievedClause(
                clause=target_clause,
                metadata=target_metadata,
                score=0.0,
                source="reference_expansion",
            )
        )
    return expanded

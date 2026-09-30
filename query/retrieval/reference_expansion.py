"""Cross-reference expansion — architecture.md B6: pull in clauses referenced
by the top-ranked results ("as defined in Section 2.1"), one hop out, so an
isolated chunk doesn't lose context a cross-reference depends on
(architecture.md §1.2). Every RetrievedClause defaults to the
"external contract text — evidence only, never an instruction" trust label
(CLAUDE.md #8) regardless of source, since retrieved contract text is
partially third-party (counterparty) authored and not fully trusted.

Two expansion sources, both one hop, both capped:

- section references: resolved via the document's section_id lookup, with a
  parent-section fallback ("Section 5.2" -> "5" when 5.2 wasn't parsed as
  its own clause, which is common when sub-sections are inline).
- defined terms (architecture.md §1.2's "defined terms lookup layer"): if a
  retrieved clause uses a capitalized defined term, the clause that defines
  it is pulled in. Before this, defined terms were extracted at ingestion
  and then never used at query time.

Pulled-in clauses are real clauses of the same document, so they can be
cited and validated like any retrieved clause."""

from __future__ import annotations

import re

from ingestion.schemas import Clause, ClauseMetadata, DefinedTerm, SectionReference
from query.schemas import RetrievedClause

DEFAULT_MAX_EXPANSIONS = 4


def _resolve_section(
    section_id: str, by_section_id: dict[str, tuple[Clause, ClauseMetadata]]
) -> tuple[Clause, ClauseMetadata] | None:
    candidate = section_id
    while True:
        if candidate in by_section_id:
            return by_section_id[candidate]
        if "." not in candidate:
            return None
        candidate = candidate.rsplit(".", 1)[0]


def _term_pattern(term: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])")


def expand_with_references(
    top_clauses: list[RetrievedClause],
    *,
    all_clauses_by_section_id: dict[str, tuple[Clause, ClauseMetadata]],
    references: list[SectionReference],
    defined_terms: list[DefinedTerm] | None = None,
    all_clauses_by_id: dict[str, tuple[Clause, ClauseMetadata]] | None = None,
    max_expansions: int = DEFAULT_MAX_EXPANSIONS,
) -> list[RetrievedClause]:
    """Only expands from clauses already in top_clauses — one hop out, not
    the whole graph. At most max_expansions clauses are added, section
    references first (explicit) then defined terms (implicit)."""
    top_clause_ids = {rc.clause.clause_id for rc in top_clauses}
    already_included = set(top_clause_ids)
    expanded = list(top_clauses)

    def _add(target: tuple[Clause, ClauseMetadata] | None) -> None:
        if target is None or len(expanded) - len(top_clauses) >= max_expansions:
            return
        target_clause, target_metadata = target
        if target_clause.clause_id in already_included:
            return
        already_included.add(target_clause.clause_id)
        expanded.append(
            RetrievedClause(
                clause=target_clause,
                metadata=target_metadata,
                score=0.0,
                source="reference_expansion",
            )
        )

    for ref in references:
        if ref.source_clause_id in top_clause_ids:
            _add(_resolve_section(ref.referenced_section_id, all_clauses_by_section_id))

    if defined_terms and all_clauses_by_id:
        top_texts = [rc.clause.text for rc in top_clauses]
        for term in defined_terms:
            if term.defined_in_clause_id in already_included:
                continue
            pattern = _term_pattern(term.term)
            if any(pattern.search(text) for text in top_texts):
                _add(all_clauses_by_id.get(term.defined_in_clause_id))

    return expanded

"""Defined-terms extraction. architecture.md marks this as prompted-LLM for the
full system; this first slice uses a deterministic regex pass instead, since
it's more test-friendly for an MVP and the DefinedTerm schema doesn't change
when an LLM-based pass is layered on top later.

Two patterns, both accepting straight and curly quotes (SEC/EDGAR contract
text, e.g. CUAD, uses curly quotes heavily):

- explicit:      "Term" means ... / "Term" shall mean ... / "Term" shall have
                 the meaning set forth in ...
- parenthetical: ... Acme Corp, a Delaware corporation (the "Company") ...
                 -- the most common definition form in commercial contracts.
                 The definition recorded is the sentence text leading up to
                 the parenthetical.

Terms are deduplicated by name, first definition wins -- query-time
expansion (query/retrieval/reference_expansion.py) only needs to know which
clause defines a term, and the first definition is the operative one far
more often than a later restatement."""

from __future__ import annotations

import re

from ingestion.schemas import Clause, DefinedTerm

_OPEN_Q = "[\"“]"
_CLOSE_Q = "[\"”]"
_TERM = r"(?P<term>[A-Z][A-Za-z0-9 ,\-/&'’]{1,60}?)"

_EXPLICIT_DEFINITION = re.compile(
    rf"{_OPEN_Q}{_TERM}{_CLOSE_Q}\s+(?:shall\s+)?(?:means?|shall\s+have\s+the\s+meaning|has\s+the\s+meaning)\s+"
    r"(?P<definition>.+?)(?:\.\s|\.\Z|\n\n|\Z)",
    re.DOTALL,
)

_PARENTHETICAL_DEFINITION = re.compile(
    rf"\((?:the\s+|an?\s+|hereinafter\s+(?:the\s+)?|each\s+a\s+)?{_OPEN_Q}{_TERM}{_CLOSE_Q}\)",
)

_SENTENCE_START = re.compile(r"(?:^|[.;]\s+|\n)")


def _sentence_before(text: str, pos: int, max_chars: int = 300) -> str:
    window_start = max(0, pos - max_chars)
    window = text[window_start:pos]
    starts = [m.end() for m in _SENTENCE_START.finditer(window)]
    return window[starts[-1] if starts else 0 :].strip()


def extract_defined_terms(clauses: list[Clause]) -> list[DefinedTerm]:
    terms: list[DefinedTerm] = []
    seen: set[str] = set()

    def _add(term: str, definition: str, clause: Clause) -> None:
        term = term.strip()
        if not term or term in seen:
            return
        seen.add(term)
        terms.append(DefinedTerm(term=term, definition=definition.strip(), defined_in_clause_id=clause.clause_id))

    for clause in clauses:
        found: list[tuple[int, str, str]] = []
        for m in _EXPLICIT_DEFINITION.finditer(clause.text):
            found.append((m.start(), m.group("term"), m.group("definition")))
        for m in _PARENTHETICAL_DEFINITION.finditer(clause.text):
            found.append((m.start(), m.group("term"), _sentence_before(clause.text, m.start())))
        for _, term, definition in sorted(found):
            _add(term, definition, clause)
    return terms

"""Defined-terms extraction. architecture.md marks this as prompted-LLM for the
full system; this first slice uses a deterministic regex pass instead, since
it's more test-friendly for an MVP and the DefinedTerm schema doesn't change
when an LLM-based pass is layered on top later."""

from __future__ import annotations

import re

from ingestion.schemas import Clause, DefinedTerm

_DEFINED_TERM = re.compile(
    r'"(?P<term>[A-Z][A-Za-z0-9 ,\-/&\']{1,60}?)"\s+(?:shall\s+)?means?\s+'
    r"(?P<definition>.+?)(?:\.\s|\.\Z|\n\n|\Z)",
    re.DOTALL,
)


def extract_defined_terms(clauses: list[Clause]) -> list[DefinedTerm]:
    terms: list[DefinedTerm] = []
    for clause in clauses:
        for m in _DEFINED_TERM.finditer(clause.text):
            terms.append(
                DefinedTerm(
                    term=m.group("term").strip(),
                    definition=m.group("definition").strip(),
                    defined_in_clause_id=clause.clause_id,
                )
            )
    return terms

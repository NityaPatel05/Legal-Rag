"""Section cross-reference graph — architecture.md §4 "section-reference graph":
parses "as defined in Section 2.1" / "pursuant to Article IV" style references
so query-time cross-reference expansion can pull in the referenced clause."""

from __future__ import annotations

import re

from ingestion.schemas import Clause, SectionReference

_REFERENCE = re.compile(
    r"\b(?:Section|Article|Clause)\s+(?P<num>\d+(?:\.\d+)*|[IVXLCDM]+)\b",
    re.IGNORECASE,
)


def extract_references(clauses: list[Clause]) -> list[SectionReference]:
    references: list[SectionReference] = []
    for clause in clauses:
        # Skip the heading line itself so a clause doesn't "reference" its own
        # section id when the heading is captioned ("Section 5: ...").
        heading_line_end = clause.text.find("\n")
        body_start = heading_line_end + 1 if heading_line_end != -1 else 0

        for m in _REFERENCE.finditer(clause.text, pos=body_start):
            num = m.group("num")
            if clause.section_id is not None and num == clause.section_id:
                continue
            references.append(
                SectionReference(
                    source_clause_id=clause.clause_id,
                    referenced_section_id=num,
                    raw_text=m.group(0),
                )
            )
    return references

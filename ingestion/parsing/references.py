"""Section cross-reference graph — architecture.md §4 "section-reference graph":
parses "as defined in Section 2.1" / "pursuant to Article IV" style references
so query-time cross-reference expansion can pull in the referenced clause.

The whole clause text is scanned, heading line included. An earlier version
skipped the first line so a captioned heading ("Section 5: ...") wouldn't
reference itself -- but most CUAD clauses carry heading AND body on one line
("2. Termination. ... under Section 7.2 and Section 9"), and every
continuation passage starts mid-clause, so that skip threw away the first
line's references everywhere. A self-reference is already dropped by
comparing against the clause's own section_id.

Ranges ("Sections 5 through 9", "5 to 9", "5-9", "5.1–5.4") are expanded when
both ends are plain integers or share every component but the last, and the
range spans at most _MAX_RANGE sections; otherwise only the two endpoints
are kept (a long or cross-level range is more likely a typo or an
unexpandable "3.1 through 4.2" than something worth 50 lookups).

Exhibit / Schedule / Annex references are deliberately NOT extracted:
SectionReference.referenced_section_id is resolved against clause section
ids (query/retrieval/reference_expansion.py), and exhibits never get one --
their stubs are blocklisted as headings in clause_parser -- so "Exhibit A"
would be a reference nothing can resolve, and would read as a section id to
every consumer of the field."""

from __future__ import annotations

import re

from ingestion.schemas import Clause, SectionReference

_NUM = r"\d+(?:\.\d+)*|[IVXLCDM]+"
_RANGE_WORD = r"through|thru|to|[-–—]"
_SEPARATOR = rf",\s*(?:and|or)?|and/or|and|or|{_RANGE_WORD}"

# Singular or plural keyword, then a list: "Section 5", "Sections 5 and 7.2",
# "Sections 3, 4, or 9", "Sections 5 through 9". Case-insensitive on the
# keyword only -- the roman numeral branch must stay uppercase or
# "Section i" style noise creeps in.
_REFERENCE = re.compile(
    rf"\b(?i:Sections?|Articles?|Clauses?)\s+(?P<list>(?:{_NUM})(?:\s*(?:{_SEPARATOR})\s*(?:{_NUM})\b)*)\b",
)
_LIST_TOKEN = re.compile(rf"(?<![\w.])(?P<num>{_NUM})(?![\w])|(?P<range>\b(?:through|thru|to)\b|[-–—])")
_MAX_RANGE = 20


def _expand_range(first: str, last: str) -> list[str]:
    """[first, ..., last] when the ends differ only in the last numeric
    component and span at most _MAX_RANGE items; otherwise [last] (first is
    already recorded by the caller)."""
    if not (first[0].isdigit() and last[0].isdigit()):
        return [last]
    *prefix_a, a = first.split(".")
    *prefix_b, b = last.split(".")
    lo, hi = int(a), int(b)
    if prefix_a != prefix_b or not 0 < hi - lo < _MAX_RANGE:
        return [last]
    head = ".".join(prefix_a) + "." if prefix_a else ""
    return [f"{head}{n}" for n in range(lo + 1, hi + 1)]


def _section_ids(reference_list: str) -> list[str]:
    ids: list[str] = []
    pending_range = False
    for token in _LIST_TOKEN.finditer(reference_list):
        num = token.group("num")
        if num is None:
            pending_range = bool(ids)
            continue
        if pending_range:
            ids.extend(_expand_range(ids[-1], num))
        else:
            ids.append(num)
        pending_range = False
    return list(dict.fromkeys(ids))


def extract_references(clauses: list[Clause]) -> list[SectionReference]:
    references: list[SectionReference] = []
    for clause in clauses:
        for m in _REFERENCE.finditer(clause.text):
            for num in _section_ids(m.group("list")):
                if clause.section_id is not None and num == clause.section_id:
                    continue  # self-reference, e.g. a "Section 5: ..." caption
                references.append(
                    SectionReference(
                        source_clause_id=clause.clause_id,
                        referenced_section_id=num,
                        raw_text=m.group(0),
                    )
                )
    return references

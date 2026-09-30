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

Both need sentence boundaries, and a bare "period + space" rule cut
definitions at the abbreviations contracts are full of: '"Licensor" means
Acme Holdings, Inc. and its Affiliates' became "Acme Holdings, Inc", and
'between Acme Corp. (the "Company")' recorded an EMPTY definition (the
sentence "started" after "Corp. "). So a period is not a sentence end when
it closes an abbreviation (_is_abbreviation_period): e.g./i.e./U.S./No./
Sec./vs. never end a sentence; company suffixes and "etc." (Inc., Corp.,
Ltd., Co., L.L.C., LLC.) end one only when a capitalised word or the end of
the paragraph follows. A following opening quote always ends it -- that is
the next definition starting ('"Buyer" means Acme Corp. "Seller" means ...').

Terms are deduplicated by name, first definition wins -- query-time
expansion (query/retrieval/reference_expansion.py) only needs to know which
clause defines a term, and the first definition is the operative one far
more often than a later restatement."""

from __future__ import annotations

import re

from ingestion.schemas import Clause, DefinedTerm

_OPEN_QUOTES = "\"“"
_OPEN_Q = "[\"“]"
_CLOSE_Q = "[\"”]"
_TERM = r"(?P<term>[A-Z][A-Za-z0-9 ,\-/&'’]{1,60}?)"

_EXPLICIT_HEAD = re.compile(
    rf"{_OPEN_Q}{_TERM}{_CLOSE_Q}\s+(?:shall\s+)?(?:means?|shall\s+have\s+the\s+meaning|has\s+the\s+meaning)\s+"
)
# Candidate ends of an explicit definition; a period only counts if it isn't
# an abbreviation's (see _is_abbreviation_period).
_DEFINITION_END = re.compile(r"\.(?=\s|\Z)|\n\n")

_PARENTHETICAL_DEFINITION = re.compile(
    rf"\((?:the\s+|an?\s+|hereinafter\s+(?:the\s+)?|each\s+a\s+)?{_OPEN_Q}{_TERM}{_CLOSE_Q}\)",
)

_SENTENCE_START = re.compile(r"(?:^|[.;]\s+|\n)")

# Abbreviations whose period never ends a sentence, and those whose period
# ends one only before a capitalised word / end of paragraph. Matched against
# the text just before the period; the lookbehind keeps "Zinc." from
# matching "Inc".
_NEVER_TERMINAL = re.compile(r"(?<![A-Za-z.])(?:[Ee]\.g|[Ii]\.e|U\.S|U\.K|Nos?|Secs?|[Vv]s|cf|viz)\Z")
_SUFFIX = r"(?:Inc|Corp|Ltd|Co|L\.L\.C|LLC|L\.P|N\.A|P\.C|PLC|etc)"
_SUFFIX_BEFORE = re.compile(rf"(?<![A-Za-z.])(?i:{_SUFFIX})\Z")
_SUFFIX_AFTER = re.compile(rf"(?i:{_SUFFIX})\.")
_ABBREVIATION_WINDOW = 16


def _is_abbreviation_period(text: str, dot: int) -> bool:
    """True when the period at text[dot] closes an abbreviation rather than
    a sentence (module docstring)."""
    rest = text[dot + 1 :].lstrip(" \t")
    if rest and rest[0] in _OPEN_QUOTES:
        return False  # the next quoted definition starts here
    before = text[max(0, dot - _ABBREVIATION_WINDOW) : dot]
    if _NEVER_TERMINAL.search(before):
        return True
    if _SUFFIX_BEFORE.search(before):
        if _SUFFIX_AFTER.match(rest):  # "Acme Co. Ltd."
            return True
        return bool(rest) and not rest[0].isupper() and rest[0] != "\n"
    return False


def _definition_end(text: str, start: int) -> int:
    for m in _DEFINITION_END.finditer(text, start):
        if m.group(0) == "." and _is_abbreviation_period(text, m.start()):
            continue
        return m.start()
    return len(text)


def _sentence_before(text: str, pos: int, max_chars: int = 300) -> str:
    window_start = max(0, pos - max_chars)
    window = text[window_start:pos]
    start = 0
    for m in _SENTENCE_START.finditer(window):
        if m.group(0).startswith(".") and _is_abbreviation_period(text, window_start + m.start()):
            continue
        start = m.end()
    return window[start:].strip()


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
        resume = 0  # like finditer: a definition's own text isn't searched for heads
        for m in _EXPLICIT_HEAD.finditer(clause.text):
            if m.start() < resume:
                continue
            end = _definition_end(clause.text, m.end())
            if end > m.end():
                found.append((m.start(), m.group("term"), clause.text[m.end() : end]))
            resume = end
        for m in _PARENTHETICAL_DEFINITION.finditer(clause.text):
            found.append((m.start(), m.group("term"), _sentence_before(clause.text, m.start())))
        for _, term, definition in sorted(found):
            _add(term, definition, clause)
    return terms

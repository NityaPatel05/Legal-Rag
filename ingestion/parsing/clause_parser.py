"""Clause-level parsing — CLAUDE.md #1, "the single most important decision in
the whole system." Boundaries come only from structure detection; a clause is
never split or truncated by a token/char budget, and a chunk never straddles
two clauses.

Four heading patterns, in descending order of confidence: captioned
("ARTICLE IV", "Section 3:"), numbered ("1.", "1.1"), ALL-CAPS standalone
lines ("SERVICES PROVIDED TO THE CUSTOMER"), and lettered subsections
("A.", "B."). Both heading patterns below terminate on EITHER end-of-line OR
"period + space + capital letter", so an inline heading+body on one line is
recognized without swallowing the sentence that follows it into the heading
text — see the CUAD validation note below for why that mattered more than
which patterns are supported at all.

Validated against CUAD's 510 annotated contracts (build-order step 2, see
eval/cuad_preflight/). First pass found the majority of real SEC-filed
contracts put the heading and its first sentence on the same line rather than
the heading alone on its own line — a standalone-line-only regex left over
half the corpus as one giant unsegmented clause. Fixing that raised avg
clauses/document from 12.0 to 20.7 with containment unchanged (99.9%).

Second pass added ALL-CAPS and lettered patterns after checking, against the
same 510 contracts, which gaps actually mattered: ALL-CAPS-only headers
appear in 79 of the 214 still-under-segmented documents (37%), lettered
subsections in only 16 (7%) — so ALL-CAPS was worth the false-positive risk,
lettered subsections marginal but cheap to add once handled safely (see
below). Measured result: avg clauses/document 20.7 -> 23.1, median 5 -> 6,
docs at <=2 clauses 214 (42%) -> 183 (36%), containment held at 99.9%
(straddled count rose slightly, 14 -> 19, an expected tradeoff of finer
segmentation creating more boundary opportunities). Smaller gain than the
first pass, as expected -- this one targets a harder long tail. See
eval/cuad_preflight/reports/boundary_report.json for the full breakdown.
Third pass (R13) bucketed the 183 remaining docs at <=2 clauses: 64 indent
their headings ("    1. Services", anchored-at-column-0 patterns missed
them), and most of the rest are flattened text -- newlines lost in
extraction, every heading mid-line. Fixes: leading whitespace allowed on
numbered/captioned/lettered headings (183 -> 119), then an inline pass for
documents where line-based detection finds < 3 headings, guarded by
requiring an increasing section-number sequence (119 -> 53). With passage
splitting: containment 98.44% -> 98.43%, median span precision 13.2% ->
14.7%; clause-level containment 99.86% -> 99.72%. Of the remaining 53,
39 have fewer than three "N. Heading" markers anywhere (unnumbered prose,
roman-numeral or "(a)" schemes -- the next bucket to look at); 14 do but
fail the sequence/threshold guards. Report:
eval/cuad_preflight/reports/chunk_quality_r13.json.

ALL-CAPS false-positive risk is real, not hypothetical: sampling actual
matches turned up document titles, "EXHIBIT 10.6"-style stubs, and signatory
names in all caps alongside genuine headings like "DEFINITIONS". Mitigated,
not eliminated: a small blocklist for common non-heading boilerplate words,
and skipping a match at document offset 0 (almost always the title, since
there's no preamble before it). Residual false positives (e.g. an all-caps
party/trust name mid-document) are accepted as a known limitation — a
title/signature match becomes an extra, mostly-harmless clause boundary
rather than corrupting an existing one.

Lettered subsections get their own boundary but deliberately NO section_id:
sampling showed a document reuses "A."/"B."/"C." repeatedly across unrelated
enumerations (recitals, then later an unrelated list) — assigning the bare
letter as section_id would let two different clauses collide on the same ID
in query/retrieval's by_section_id lookup, silently shadowing one with the
other. The clause still exists and still doesn't straddle a boundary; it's
just not independently citable by a bare letter. (ALL-CAPS headings get no
section_id either -- only numbered/captioned headings do. Citations use
clause_id, which every clause has; see query/validation/citation_validator.py.)
"""

from __future__ import annotations

import re

from ingestion.schemas import Clause

_HEADING_TERMINATOR = r"(?:\.(?=[ \t]+[A-Z])|$)"

_NUMBERED_HEADING = re.compile(
    r"^[ \t]*(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?[ \t]+"
    rf"(?P<heading>[A-Z][^\n]{{0,80}}?){_HEADING_TERMINATOR}",
    re.MULTILINE,
)

_CAPTIONED_HEADING = re.compile(
    r"^[ \t]*(?P<kind>ARTICLE|SECTION)[ \t]+(?P<num>[IVXLCDM]+|\d+(?:\.\d+)*)"
    rf"[ \t]*[:.\-–—]?[ \t]*(?P<heading>[^\n]{{0,80}}?){_HEADING_TERMINATOR}",
    re.MULTILINE | re.IGNORECASE,
)

_ALLCAPS_HEADING = re.compile(
    r"^(?P<heading>[A-Z][A-Z0-9 ,&/\-]{3,69})$",
    re.MULTILINE,
)

_ALLCAPS_BLOCKLIST = re.compile(
    r"^(EXHIBIT|SCHEDULE|APPENDIX|ANNEX|ATTACHMENT|REDACTED|CONFIDENTIAL|PAGE|CONTINUED)\b"
)

_LETTERED_HEADING = re.compile(
    r"^[ \t]*(?P<letter>[A-Z])\.[ \t]+(?P<heading>[A-Z][^\n]{0,80}?)" + _HEADING_TERMINATOR,
    re.MULTILINE,
)


def _find_headings(raw_text: str) -> list[tuple[int, str | None, str | None]]:
    """Returns (start_offset, section_id, heading) tuples sorted by position,
    deduped so a line only ever matches once even if multiple patterns could
    fire — priority order is captioned > numbered > ALL-CAPS > lettered,
    most to least confident pattern."""
    matches: dict[int, tuple[str | None, str | None]] = {}
    for m in _CAPTIONED_HEADING.finditer(raw_text):
        matches[m.start()] = (m.group("num"), m.group("heading").strip() or None)
    for m in _NUMBERED_HEADING.finditer(raw_text):
        matches.setdefault(m.start(), (m.group("num"), m.group("heading").strip() or None))
    for m in _ALLCAPS_HEADING.finditer(raw_text):
        heading = m.group("heading").strip()
        if m.start() == 0 or _ALLCAPS_BLOCKLIST.match(heading):
            continue
        matches.setdefault(m.start(), (None, heading))
    for m in _LETTERED_HEADING.finditer(raw_text):
        # No section_id -- see module docstring on why a bare letter can't
        # safely be used as a citation-lookup key.
        matches.setdefault(m.start(), (None, m.group("heading").strip() or None))
    if len(matches) < _MIN_LINE_HEADINGS:
        for offset, sec_id, heading in _find_inline_headings(raw_text):
            matches.setdefault(offset, (sec_id, heading))
    return [(offset, sec_id, heading) for offset, (sec_id, heading) in sorted(matches.items())]


# Flattened text (newlines lost in extraction): headings sit mid-line, e.g.
# '... the "Shares"). SECTION 1. SERVICES AS UNDERWRITER Section 1.1 Integrity
# will ...'. A bare number must follow sentence-ending punctuation; a
# Section/Article caption may appear anywhere (it ends the previous heading
# line) but must be followed by a capital. Either way the section number must
# be followed by a capitalized word, and survivors must form an increasing
# numbering sequence (_in_sequence) -- that is what rejects cross-references
# ("under Section 12.3 Licensee ...") and stray numbers ("$5. 2 Parties").
_INLINE_HEADING = re.compile(
    r"(?:(?<=[.:;\"”)\]])[ \t]+|[ \t]+(?=(?:ARTICLE|SECTION|Article|Section)[ \t]))"
    r"(?P<start>(?:(?:ARTICLE|SECTION|Article|Section)[ \t]+)?)"
    r"(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?[ \t]+(?=[A-Z])"
)
_INLINE_TITLE = re.compile(r"[A-Z][^\n.]{0,80}?(?=\.[ \t]+[A-Z])")
# Line-based detection finding fewer headings than this means the layout
# probably was flattened; only then is the (noisier) inline pass consulted.
_MIN_LINE_HEADINGS = 3


def _find_inline_headings(raw_text: str) -> list[tuple[int, str | None, str | None]]:
    found: list[tuple[int, str | None, str | None]] = []
    last: tuple[int, ...] = (0,)
    for m in _INLINE_HEADING.finditer(raw_text):
        number = tuple(int(part) for part in m.group("num").split("."))
        if not _in_sequence(last, number):
            continue
        last = number
        title = _INLINE_TITLE.match(raw_text, m.end())
        found.append((m.start("start"), m.group("num"), title.group(0).strip() if title else None))
    return found


def _in_sequence(last: tuple[int, ...], number: tuple[int, ...]) -> bool:
    """Next top-level section (1 -> 2), or a subsection of the current one
    (1 -> 1.1, 1.1 -> 1.2); anything else is treated as a stray number."""
    if number[0] == last[0] + 1:
        return True
    return number[0] == last[0] and len(number) > 1 and number > last


def parse_clauses(raw_text: str, *, document_id: str) -> list[Clause]:
    """Splits raw_text into clause-level chunks. If no headings are detected,
    the whole document is returned as a single clause rather than guessing at
    boundaries."""
    headings = _find_headings(raw_text)
    boundaries = [offset for offset, _, _ in headings] + [len(raw_text)]

    clauses: list[Clause] = []
    idx = 0

    if not headings or headings[0][0] > 0:
        preamble_end = headings[0][0] if headings else len(raw_text)
        preamble_text = raw_text[0:preamble_end]
        if preamble_text.strip():
            clauses.append(
                Clause(
                    clause_id=f"{document_id}::clause-{idx:03d}",
                    section_id=None,
                    heading=None,
                    text=preamble_text,
                    start_offset=0,
                    end_offset=preamble_end,
                )
            )
            idx += 1

    for i, (start, section_id, heading) in enumerate(headings):
        end = boundaries[i + 1]
        clauses.append(
            Clause(
                clause_id=f"{document_id}::clause-{idx:03d}",
                section_id=section_id,
                heading=heading,
                text=raw_text[start:end],
                start_offset=start,
                end_offset=end,
            )
        )
        idx += 1

    return clauses


# ~500 tokens. Measured on CUAD with the parser above: median clause ~174
# tokens, but p99 ~7.3k and max ~48k tokens; 97/510 documents contain a clause
# longer than BGE-M3's 8192-token input limit (the tail is silently truncated
# at embed time), and the median annotated span is only ~2% of the clause that
# contains it. The clause stays the unit of meaning; the passage is the unit
# of retrieval.
#
# ~250 tokens since R3: compared at an EQUAL amount of retrieved text,
# 1000-char passages beat 2000 at every budget on CUAD (BM25, 510 docs:
# hit@8k chars .747 vs .710), and 1000 x k=10 beats 2000 x k=5 at about the
# same context (~7.2k vs ~6.5k chars). Goes with Settings.generation_top_k=10.
# The numbers quoted above (and R13's chunk-quality report) were measured at
# 2000 chars.
DEFAULT_MAX_PASSAGE_CHARS = 1000

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\"“])")


def _split_points(text: str, pattern: re.Pattern) -> list[int]:
    return [m.end() for m in pattern.finditer(text)]


def _pack(text: str, cut_tiers: list[list[int]], max_chars: int) -> list[tuple[int, int]]:
    """Greedily packs [start, end) windows no longer than max_chars. For each
    window, cuts at the furthest candidate from the first tier that has one in
    range (paragraph breaks before sentence ends), and only hard-cuts at
    max_chars when no tier offers a candidate."""
    windows: list[tuple[int, int]] = []
    start = 0
    while len(text) - start > max_chars:
        limit = start + max_chars
        end = limit
        for tier in cut_tiers:
            # Ignore cuts that would leave a sliver window (< 1/4 of max).
            usable = [c for c in tier if start + max_chars // 4 <= c <= limit]
            if usable:
                end = max(usable)
                break
        windows.append((start, end))
        start = end
    windows.append((start, len(text)))
    return windows


def split_oversize_clauses(
    clauses: list[Clause], *, max_chars: int = DEFAULT_MAX_PASSAGE_CHARS
) -> list[Clause]:
    """Splits any clause longer than max_chars into contiguous passages,
    preferring paragraph breaks, then sentence ends. Passages keep the
    parent's section_id and heading, carry parent_clause_id, and are
    sub-ranges of the parent -- so the no-straddle invariant still holds.
    Clauses at or under the limit pass through unchanged."""
    out: list[Clause] = []
    for clause in clauses:
        if len(clause.text) <= max_chars:
            out.append(clause)
            continue
        tiers = [
            _split_points(clause.text, _PARAGRAPH_BREAK),
            _split_points(clause.text, _SENTENCE_END),
        ]
        for p, (s, e) in enumerate(_pack(clause.text, tiers, max_chars)):
            out.append(
                Clause(
                    clause_id=f"{clause.clause_id}.p{p:02d}",
                    section_id=clause.section_id,
                    heading=clause.heading,
                    text=clause.text[s:e],
                    start_offset=clause.start_offset + s,
                    end_offset=clause.start_offset + e,
                    parent_clause_id=clause.clause_id,
                )
            )
    return out

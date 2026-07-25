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
~36% of CUAD documents still under-segment; the residual causes weren't
re-diagnosed after this pass.

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
just not independently citable by a bare letter (numbered/captioned/ALL-CAPS
headings still get real section_ids as before).
"""

from __future__ import annotations

import re

from ingestion.schemas import Clause

_HEADING_TERMINATOR = r"(?:\.(?=[ \t]+[A-Z])|$)"

_NUMBERED_HEADING = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?[ \t]+"
    rf"(?P<heading>[A-Z][^\n]{{0,80}}?){_HEADING_TERMINATOR}",
    re.MULTILINE,
)

_CAPTIONED_HEADING = re.compile(
    r"^(?P<kind>ARTICLE|SECTION)[ \t]+(?P<num>[IVXLCDM]+|\d+(?:\.\d+)*)"
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
    r"^(?P<letter>[A-Z])\.[ \t]+(?P<heading>[A-Z][^\n]{0,80}?)" + _HEADING_TERMINATOR,
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
    return [(offset, sec_id, heading) for offset, (sec_id, heading) in sorted(matches.items())]


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

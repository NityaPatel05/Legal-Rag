"""Source-text normalization, applied once before parsing so the stored clause
text -- the text citations are string-matched against (CLAUDE.md #3) -- is
the same text the parser segmented and the model later reads.

Why: invisible differences in how a document was extracted break structure
detection without changing what a human sees. Each of these silently turned
a multi-clause contract into fewer, larger clauses:

- CRLF line endings: `$` in a MULTILINE pattern matches before `\\n`, not
  before `\\r\\n`, so every ALL-CAPS heading line ("DEFINITIONS\\r") failed.
- A non-breaking space after the number ("1.\\u00a0Services"): `[ \\t]+`
  doesn't match it, so the numbered heading was missed.
- A form feed (page break) before a heading: `^` doesn't match after `\\f`.
- Zero-width characters and a BOM: invisible, but break both heading
  patterns and verbatim quote matching.

NFKC folds compatibility forms (fullwidth digits, ligatures, "…" -> "...")
so quotes the model reproduces in plain ASCII still match the stored text.

Measured on CUAD's 510 contexts (already cleaned by the dataset authors): no
CR, form feed or NBSP at all, but 53 docs contain characters NFKC rewrites
(2,815 "…", 161 "™", plus en/em/thin spaces and a few zero-width spaces),
and 39 of them change length. That is why there are two functions:

- normalize_source_text: the full normalization, for text that will be
  STORED (ingestion/pipeline.py calls it before parse_clauses). Length may
  change, so offsets must be computed on its output.
- normalize_preserving_offsets: a strictly 1:1 character map used by
  parse_clauses for heading DETECTION only. Clause text is still sliced from
  the caller's string, so clause.text == raw_text[start:end] holds for any
  caller (the CUAD evals score spans by the dataset's own offsets). On text
  that is already normalized it returns the input unchanged.
"""


from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Invisible characters removed outright: zero-width space / non-joiner /
# joiner, word joiner, BOM (zero-width no-break space), soft hyphen (renders
# as nothing mid-line, but splits "termi<SHY>nation" for string matching).
_ZERO_WIDTH = "​‌‍⁠﻿­"

# Vertical breaks that should read as a line break: form feed, vertical tab,
# NEL, Unicode line / paragraph separators.
_LINE_BREAKS = "\f\v\x85  "

# Every Unicode space separator (category Zs) other than " ": NBSP, ogham
# space, en/em/thin/hair/figure spaces (U+2000-U+200A), narrow NBSP, medium
# mathematical space, ideographic space.
_SPACES = "  " + "".join(chr(c) for c in range(0x2000, 0x200B)) + "  　"

_SOURCE_TABLE: dict[int, str | None] = {
    **{ord(ch): "\n" for ch in _LINE_BREAKS},
    **{ord(ch): " " for ch in _SPACES},
    **{ord(ch): None for ch in _ZERO_WIDTH},
}

# Anything outside printable ASCII (plus tab and LF) needs a look; plain
# ASCII text takes the fast path in both functions.
_NEEDS_WORK = re.compile(r"[^\t\n\x20-\x7e]")


def normalize_source_text(text: str) -> str:
    """CRLF / CR -> LF; form feed, vertical tab and Unicode line separators ->
    LF; NBSP, narrow NBSP and every other Unicode space separator -> " ";
    zero-width characters, soft hyphen and BOM removed; then NFKC. Pure and
    idempotent: normalize_source_text(normalize_source_text(t)) ==
    normalize_source_text(t)."""
    if not _NEEDS_WORK.search(text):
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(_SOURCE_TABLE)
    return unicodedata.normalize("NFKC", text)


@lru_cache(maxsize=4096)
def _one_to_one(ch: str) -> str:
    if ch == "\r" or ch in _LINE_BREAKS:
        return "\n"
    if ch in _SPACES or ch in _ZERO_WIDTH:
        return " "
    folded = unicodedata.normalize("NFKC", ch)
    return folded if len(folded) == 1 else ch


def normalize_preserving_offsets(text: str) -> str:
    """Same intent as normalize_source_text, restricted to a 1:1 character
    map so len(result) == len(text) and every offset is shared: CRLF ->
    " " + LF (NOT two LFs -- that would read as a blank line and defeat the
    wrapped-line guard in clause_parser), a lone CR, form feed and line
    separators -> LF; Unicode spaces and zero-width
    characters -> " "; a character whose NFKC form is one character is
    folded, anything NFKC expands ("…", "™") is left alone."""
    if not _NEEDS_WORK.search(text):
        return text
    text = text.replace("\r\n", " \n")
    return _NEEDS_WORK.sub(lambda m: _one_to_one(m.group(0)), text)

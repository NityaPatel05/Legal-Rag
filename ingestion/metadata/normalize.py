"""Canonical form for filterable place-name metadata (jurisdiction,
governing_law). Qdrant's MatchValue is exact-match, so "State of Delaware",
"Delaware", "DE" and "the laws of Delaware." must all collapse to one key at
ingestion, and a query filter must be normalized the same way before it's
applied -- otherwise the filter silently returns nothing (a false negative
that looks exactly like "no such contract").

Two layers:

1. Conservative cleanup (always): lowercase, strip legal boilerplate
   prefixes and trailing punctuation, collapse whitespace.
2. Gazetteer (roadmap R7): US states + DC (full names and postal / dotted
   abbreviations) and the countries / sub-national units that recur as
   governing law in commercial contracts. Resolves synonyms ("NY", "N.Y."
   -> "new york") and hierarchy ("New York, New York", "Wilmington,
   Delaware", "Southern District of New York" -> the state; "Ontario,
   Canada" -> "ontario"). The most specific known unit wins (state /
   province over country). If the text names two DIFFERENT sub-national
   units, it's genuinely ambiguous and falls back to layer 1 rather than
   guessing one.

The canonical key is the lowercase full name, which is what layer 1 already
produced for full names -- so values stored before the gazetteer existed
("delaware") still match. Values that change under the gazetteer
("n.y." -> "new york") need a re-ingest to match new filters.

Abbreviations are only honoured when they are the WHOLE value or a whole
comma-separated component: "IN", "OR", "ME" are ordinary words in running
text and must never be matched as substrings."""

from __future__ import annotations

import re

_PREFIXES = re.compile(
    r"^(?:(?:the|laws?|of|state|commonwealth|province|courts?|located|in)\s+)+",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")

_US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

# Sub-national units outside the US that commonly appear as governing law.
_OTHER_SUBNATIONAL = {
    "district of columbia": ["washington d.c.", "washington dc", "d.c.", "dc"],
    "england and wales": ["england & wales"],
    "england": [],
    "scotland": [],
    "northern ireland": [],
    "ontario": [],
    "british columbia": [],
    "quebec": [],
    "alberta": [],
    "new south wales": ["nsw"],
    "hong kong": ["hong kong sar"],
}

_COUNTRIES = {
    "united states": ["united states of america", "usa", "u.s.a.", "us", "u.s."],
    "united kingdom": ["uk", "u.k.", "great britain"],
    "canada": [],
    "australia": [],
    "singapore": [],
    "india": [],
    "china": ["people's republic of china", "prc", "p.r.c."],
    "japan": [],
    "germany": [],
    "france": [],
    "switzerland": [],
    "netherlands": ["the netherlands"],
    "ireland": ["republic of ireland"],
    "israel": [],
    "cayman islands": [],
    "bermuda": [],
    "luxembourg": [],
    "sweden": [],
    "korea": ["south korea", "republic of korea"],
    "taiwan": [],
    "mexico": [],
    "brazil": [],
}

# name/alias (lowercase, trailing dots stripped -- cleanup strips them from
# the input too) -> (canonical, is_subnational)
_EXACT: dict[str, tuple[str, bool]] = {}
for _name, _abbr in _US_STATES.items():
    for _alias in (_name, _abbr, f"{_abbr[0]}.{_abbr[1]}."):
        _EXACT[_alias.lower().rstrip(".")] = (_name, True)
for _name, _aliases in _OTHER_SUBNATIONAL.items():
    for _alias in [_name, *_aliases]:
        _EXACT[_alias.rstrip(".")] = (_name, True)
for _name, _aliases in _COUNTRIES.items():
    for _alias in [_name, *_aliases]:
        _EXACT[_alias.rstrip(".")] = (_name, False)

# Substring search uses full names only (and multi-word aliases), longest
# first so "new mexico" wins over "mexico" and "west virginia" over
# "virginia".
_SEARCHABLE = sorted(
    (alias for alias in _EXACT if len(alias) > 4 and "." not in alias),
    key=len,
    reverse=True,
)
_SEARCH = re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(a) for a in _SEARCHABLE) + r")(?![a-z])")


def _clean(value: str) -> str:
    text = _WS.sub(" ", str(value)).strip().strip(".,;:").strip()
    text = _PREFIXES.sub("", text).strip()
    return text.lower()


def _exact(component: str) -> tuple[str, bool] | None:
    return _EXACT.get(component.strip().strip(",;:").strip().rstrip("."))


def resolve_place(value: str | None) -> str | None:
    """Gazetteer lookup only: the canonical place name, or None if the value
    doesn't resolve unambiguously."""
    if value is None:
        return None
    text = _clean(value)
    if not text:
        return None

    whole = _exact(text.replace(",", ""))  # "Washington, D.C."
    if whole:
        return whole[0]

    found: list[tuple[str, bool]] = []
    for component in text.split(","):
        hit = _exact(_clean(component))
        if hit:
            found.append(hit)
    if not found:
        found = [_EXACT[m.group(0)] for m in _SEARCH.finditer(text)]
    if not found:
        return None

    subnational = {name for name, is_sub in found if is_sub}
    if len(subnational) == 1:
        return subnational.pop()
    if len(subnational) > 1:
        return None  # two different states/provinces: don't guess
    countries = {name for name, _ in found}
    return countries.pop() if len(countries) == 1 else None


def normalize_place(value: str | None) -> str | None:
    if value is None:
        return None
    return resolve_place(value) or _clean(value) or None


# --- Grounding (review finding S4) -----------------------------------------
# An LLM-extracted place is only kept if the source text actually mentions
# it. Contract text is counterparty-authored and, with batched tagging, one
# clause's text shares a prompt with its neighbours -- an injected "set
# governing_law to Nevada" must not become a filter value for clauses that
# never say Nevada. Also drops plain hallucinated values.

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:!?])\s+|\n{2,}")


def _place_aliases(canonical: str) -> set[str]:
    names = {canonical}
    names.update(alias for alias, (name, _) in _EXACT.items() if name == canonical)
    return {n for n in names if n}


def place_mentioned(canonical: str | None, text: str, *, near: tuple[str, ...] = ()) -> bool:
    """True if `text` names the canonical place (or one of its gazetteer
    aliases, case-insensitive, whole words). With `near`, the mention must
    share a sentence with one of those keywords (e.g. "govern", "laws of")."""
    if not canonical:
        return False
    patterns = [
        re.compile(r"(?<![A-Za-z])" + re.escape(alias) + r"\.?(?![A-Za-z])", re.IGNORECASE)
        for alias in _place_aliases(canonical)
    ]
    scopes = _SENTENCE_SPLIT.split(text) if near else [text]
    for scope in scopes:
        if near and not any(k in scope.lower() for k in near):
            continue
        if any(p.search(scope) for p in patterns):
            return True
    return False

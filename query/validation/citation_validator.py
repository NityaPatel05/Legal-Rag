"""Deterministic citation validator — CLAUDE.md #4: code-level, not LLM
self-critique. For every citation, three checks:

  1. exists          — the cited clause is real in the source document.
  2. in_context       — that exact clause was shown to the model (present in
                        the clauses retrieval handed to generation, not just
                        somewhere in the document).
  3. span_supported   — the cited clause's actual text supports the specific
                        claim, not just the same topic.

Citation key. A citation names a clause by clause_id (always present,
always unique). A bare section_id is accepted as a fallback only when it
resolves to exactly one clause -- section_id is None for preambles, ALL-CAPS
headings and lettered sub-clauses, and real contracts reuse numbering
(exhibits restarting at "1."), so section_id alone can't be the key. The
resolved clause must be the SAME clause in both the source document and the
shown set, which also rejects a shown clause from a different document that
happens to share a section number.

span_supported, strongest path first:

  a. supporting_quote given (the generator is asked for one per claim): the
     quote must appear verbatim in the cited clause (after whitespace/quote
     normalization) -- a pure string check, the strongest deterministic
     grounding available. The claim-level checks below then run against
     the QUOTE's surroundings, not the whole clause, which is what makes them
     meaningful: comparing a one-line claim's negations/numbers against a
     3,000-word clause both rejects correct claims (the clause says "not"
     somewhere else) and accepts wrong ones (the wrong number appears
     somewhere else).
     If the quote occurs more than once and the occurrences sit in different
     sentences, the claim is rejected as ambiguous: the first occurrence used
     to be taken, which let a false claim borrow the wrong sentence's facts
     (and rejected a true claim whose facts sat by the second occurrence).
  b. no quote (legacy callers): the same checks against the whole clause.

Evidence scopes. Three nested spans around the quote, each chosen for the
check that uses it:
  - sentence: bounded by ". " / "; " -- numbers, word overlap and the NLI
    premise (invariant #3);
  - negation scope: the quote's own segment(s) of that sentence, split at
    ", and" / ", but" / ", provided" (a split is only made when the next
    part has its own verb, so "shall not copy, and distribute" isn't cut
    loose from its "not"), plus any other segment whose substantive words
    the claim restates. Scoping to the whole sentence rejected "Either party
    may terminate ... for material breach" because of "..., and termination
    does not affect ..." (L19); scoping to the bare quote let a claim drop a
    "not" sitting just outside it. The re-inclusion rule closes the obvious
    hole ("Fees are refundable" quoting only the ", provided ..." half of
    "Fees are not refundable, provided ...");
  - full sentence (". " only) -- party names, because the subject of a
    "; and shall ..." tail lives before the semicolon.

Claim-level checks, most specific first so the failure reason fed back into
regeneration names the actual error:

  numbers  -- whole quantities, not digit strings. Each expression ("$1
     million", "1,000,000", "one hundred twenty (120) days", "2.5%") is
     parsed to a Decimal and paired with its unit (day / business day / week
     / month / year / hour / % / currency); every (value, unit) the claim
     asserts must be among the evidence's. The old check split on \\d+ and
     compared sets, so "$1 million" matched "$1 billion", "2.5%" matched
     "5.2%", "30 months" matched "30 days" and "twenty" matched the "twenty"
     inside "one hundred twenty" -- and "$1 million" failed against
     "$1,000,000". A unitless claim number matches the value under any unit;
     a currency claim may match a bare evidence number (the currency is often
     stated once elsewhere). Numbers after Section / Article / Exhibit /
     Schedule / § are references, not asserted quantities, and are ignored.
  negation -- canonical markers (no/not/never/neither/nor/cannot -> "not";
     "without"; "at least" = "not less than"; "at most" = "not more than")
     and exception words (except / unless / other than / excluding) must
     match between the claim and the negation scope. "no later than" / "not
     later than" are deadlines, not negations, and are neutralised. One
     structured exemption: a "not ... except X" / "not ... unless X" scope
     supports a positive (no "not") claim about the carve-out X (the claim
     must name at least half of X's content words) -- "Licensee may not
     assign ..., except to an Affiliate" supports "Licensee may assign ...
     to an Affiliate" but not "... to any third party" nor "may not assign
     ... to an Affiliate".
  modality -- the claim's modal class (obligation: shall/must/will/is
     required to; permission: may/can/is entitled to; prohibition: shall
     not/may not/no party shall; no-obligation: is not required to) must be
     one the evidence uses (negation scope first, else the sentence). May/
     shall/must were stopwords, so a may <-> must swap passed every lexical
     check (and the smoke-tested NLI model missed it too, R6). Evidence with
     no modal at all doesn't constrain the claim. They stay stopwords for
     word overlap -- modality is checked here, explicitly, instead.
  parties  -- conservative by design. (1) A capitalised word in the claim
     that is a party role (Licensee, Company, Supplier, ...) or a defined
     term of THIS clause (capitalised in it, never lowercase) must appear in
     the full sentence -- except that "either party"/"each party" there
     covers a named party that the claim uses as a modal's subject
     ("Licensee may terminate" from "Either Party may terminate"). Proper
     nouns the clause never mentions are not checked (a claim may resolve a
     defined term from elsewhere; overlap and NLI own that).
     (2) Obligor: for each "<Party> <modal>" in the claim, if the negation
     scope names who bears that modal, the claim's party must be among them ("either party"/"each Party" matches anyone; passive
     "shall be indemnified by" is skipped both sides). This rejects "The
     Customer shall indemnify the Supplier" against the reverse, which
     previously only the optional NLI layer could catch.
  overlap  -- content-word overlap >= threshold against the sentence
     (numbers excluded: the quantity check owns them).

Those checks are still lexical: a claim can reuse the quote's words to
assert something subtly different (scope broadening, a swapped object). The
optional entailment layer (roadmap R6) closes that gap: when an
EntailmentChecker is supplied, a claim that passed every string check must
ALSO be entailed by its evidence (the quote's enclosing sentence) under a
fixed NLI model and fixed threshold -- still deterministic, still
code-level, never LLM self-critique. It is off unless configured
(NLI_MODEL), because the model must be validated (ContractNLI) before it is
allowed to reject claims in production.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel

from ingestion.schemas import Clause
from query.schemas import RetrievedClause

# Modals stay stopwords for word overlap: the modality check compares them
# explicitly (and tolerates "will" vs "shall", which overlap would not).
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "shall", "must", "to", "of",
    "and", "or", "in", "on", "for", "that", "this", "it", "by", "as", "be",
    "with", "at", "any", "all", "such", "from", "which", "will", "may",
}

_MIN_QUOTE_CHARS = 12

# --- numbers ----------------------------------------------------------------

_SMALL_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"thousand": 10**3, "million": 10**6, "billion": 10**9, "trillion": 10**12}
_NUMBER_WORDS = set(_SMALL_NUMBERS) | set(_TENS) | set(_SCALES) | {"hundred"}

_CURRENCY_PREFIX = {"$": "$", "usd": "$", "€": "eur", "eur": "eur", "£": "gbp", "gbp": "gbp"}
_UNIT_AFTER = {
    "%": "%", "percent": "%",
    "day": "day", "days": "day",
    "week": "week", "weeks": "week",
    "month": "month", "months": "month",
    "year": "year", "years": "year",
    "hour": "hour", "hours": "hour",
    "dollar": "$", "dollars": "$", "usd": "$",
    "euro": "eur", "euros": "eur", "eur": "eur",
    "gbp": "gbp",
}

_REF_NUMBER = r"\d+(?:\.\d+)*(?!\.?\d)(?:\s*\([a-z0-9]{1,4}\))*"
_UNIT_LOOKAHEAD = (
    r"(?!\s*-?\s*(?:%|(?:percent|days?|business|calendar|weeks?|months?|years?|hours?|dollars?|usd)\b))"
)
# "Section 12.2", "Sections 5 and 7", "§ 3(a)" -- references, not quantities.
_REFERENCE = re.compile(
    r"(?:\b(?:sections?|articles?|exhibits?|schedules?|annex(?:es)?|appendix|appendices|clauses?|paragraphs?)\b|§+)"
    rf"\s*{_REF_NUMBER}(?:\s*(?:,|and|or|through|to|-)\s*{_REF_NUMBER}{_UNIT_LOOKAHEAD})*",
    re.IGNORECASE,
)
_NUM_TOKEN = re.compile(r"\d+(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|[a-z]+|[$€£%()]|[,;:.]")

# --- negation ---------------------------------------------------------------

_NEGATION_CANON = {
    "not": "not", "no": "not", "never": "not", "neither": "not", "nor": "not",
    "cannot": "not", "none": "not", "nothing": "not",
    "without": "without",
    "notlessthan": "notlessthan", "notmorethan": "notmorethan",
}
_EXCEPTION = re.compile(r"\b(?:except|excepting|unless|excluding|other than|save (?:as|that|for))\b")
_CONTRACTIONS = [
    (re.compile(r"\bcan['’]t\b", re.I), "cannot"),
    (re.compile(r"\bwon['’]t\b", re.I), "will not"),
    (re.compile(r"\bshan['’]t\b", re.I), "shall not"),
    (re.compile(r"n['’]t\b", re.I), " not"),
]
# Deadlines ("no later than 30 days") are not negations of the claim.
_DEADLINE = re.compile(r"\b(?:no|not)\s+(?:later|earlier|sooner)\s+than\b", re.I)
_AT_LEAST = re.compile(r"\b(?:(?:no|not)\s+(?:less|fewer)\s+than|at\s+least)\b", re.I)
_AT_MOST = re.compile(r"\b(?:(?:no|not)\s+(?:more|greater)\s+than|at\s+most)\b", re.I)

# --- modality ---------------------------------------------------------------

# Applied in order; each match is masked so later patterns can't re-read it
# ("may not" is a prohibition, not also a permission).
_MODAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("prohibition", re.compile(
        r"\b(?:shall|must|may|will|can|should)\s+not\b|\bcannot\b"
        r"|\b(?:is|are)\s+(?:not\s+(?:permitted|allowed|entitled|authori[sz]ed)\s+to|prohibited\s+from)\b"
        r"|\b(?:no|neither|nor)\s+(?:[a-z'’]+\s+){0,3}?(?:shall|must|may|will|can)\b"
    )),
    ("no_obligation", re.compile(
        r"\b(?:is|are)\s+not\s+(?:required|obligated|obliged)\s+to\b|\bno\s+obligation\s+to\b|\bunder\s+no\s+obligation\b"
    )),
    ("permission", re.compile(
        r"\b(?:shall|will)\s+(?:have\s+the\s+right|be\s+entitled|be\s+permitted|be\s+allowed)\s+to\b"
    )),
    ("obligation", re.compile(
        r"\b(?:shall|must)\b|(?<!at )\bwill\b|\b(?:is|are)\s+(?:required|obligated|obliged)\s+to\b"
        r"|\bagrees?\s+to\b|\bundertakes?\s+to\b"
    )),
    ("permission", re.compile(
        r"\bmay\b(?!\s+\d)|\bcan\b|\b(?:is|are)\s+(?:entitled|permitted|allowed|free)\s+to\b"
        r"|\b(?:has|have)\s+the\s+right\s+to\b"
    )),
]
_PASSIVE_AFTER_MODAL = re.compile(r"\s+be\s+(?:[a-z]+\s+){0,2}?[a-z]+\s+by\b")

# --- parties ----------------------------------------------------------------

_PARTY_ROLES = {
    "licensee", "licensor", "sublicensee", "company", "supplier", "customer", "vendor",
    "buyer", "seller", "purchaser", "contractor", "subcontractor", "client", "provider",
    "distributor", "manufacturer", "reseller", "employer", "employee", "executive",
    "consultant", "lessor", "lessee", "landlord", "tenant", "borrower", "lender",
    "franchisee", "franchisor", "recipient", "discloser", "owner", "agent", "investor",
    "guarantor", "developer", "publisher", "sponsor", "servicer", "operator",
}
# Capitalised words that are sentence starters or structure, never parties.
_CAPITALISED_STOPWORDS = _STOPWORDS | {
    "these", "those", "either", "neither", "each", "every", "both", "no", "none", "if",
    "upon", "under", "within", "without", "after", "before", "during", "notwithstanding",
    "subject", "except", "unless", "where", "when", "whereas", "provided", "but", "nor",
    "its", "he", "she", "they", "their", "we", "our", "you", "your", "section", "sections",
    "article", "articles", "exhibit", "exhibits", "schedule", "schedules", "annex",
    "appendix", "clause", "paragraph", "party", "parties", "not", "only", "so", "then",
    "also", "in", "here", "there", "however", "further", "following", "other",
}
_WILDCARD_PARTY = {"party", "parties"}
_GENERIC_PARTY = re.compile(r"\b(?:either|each|any|both|neither|all)\s+(?:of\s+the\s+)?(?:party|parties)\b")
_CAPITALISED_WORD = re.compile(r"\b[A-Z][A-Za-z]+\b")
_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
# Evidence for a split into independent segments: the next part has a verb.
_FINITE_VERB = re.compile(
    r"\b(?:shall|may|must|will|can|cannot|is|are|was|were|does|do|did|has|have|had|agrees?|remains?|becomes?)\b"
)
_SEGMENT_SPLIT = re.compile(r",\s+(?=(?:and|but|provided)\b)")


class EntailmentChecker(Protocol):
    def entails(self, premise: str, hypothesis: str) -> bool:
        """True if `premise` entails `hypothesis`. Must be deterministic."""
        ...


class CrossEncoderEntailmentChecker:
    """NLI cross-encoder via sentence-transformers (lazy import; the model is
    loaded on first use). The entailment label index is read from the
    model's own id2label rather than assumed, since NLI checkpoints order
    their labels differently -- and a checkpoint without an entailment label
    is a configuration error, reported as such. Loading is locked: the app
    shares one checker across request threads, and two concurrent first
    requests would otherwise each load the model."""

    def __init__(self, model_name: str, *, threshold: float = 0.5):
        self._model_name = model_name
        self._threshold = threshold
        self._model = None
        self._entailment_index: int | None = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import os

                    # transformers otherwise imports TensorFlow when it's installed,
                    # which fails under Keras 3; this service never uses TF.
                    os.environ.setdefault("USE_TF", "0")
                    from sentence_transformers import CrossEncoder  # lazy import

                    model = CrossEncoder(self._model_name)
                    id2label = model.model.config.id2label
                    index = next(
                        (int(i) for i, label in id2label.items() if str(label).lower().startswith("entail")),
                        None,
                    )
                    if index is None:
                        raise ValueError(
                            f"NLI model {self._model_name!r} has no entailment label "
                            f"(id2label={dict(id2label)!r}); pick an NLI checkpoint"
                        )
                    self._entailment_index = index
                    self._model = model  # published last: readers outside the lock see a complete checker
        return self._model

    def entails(self, premise: str, hypothesis: str) -> bool:
        model = self._load()
        [probabilities] = model.predict([(premise, hypothesis)], apply_softmax=True)
        return float(probabilities[self._entailment_index]) >= self._threshold


class Citation(BaseModel):
    # Citation key: a clause_id (preferred) or a section_id that resolves to
    # exactly one clause. Field name kept for API compatibility.
    section_id: str
    claim_text: str
    supporting_quote: str | None = None


class CitationValidationResult(BaseModel):
    citation: Citation
    exists: bool
    in_context: bool
    span_supported: bool
    resolved_clause_id: str | None = None
    failure_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.exists and self.in_context and self.span_supported


# --- tokens -----------------------------------------------------------------


def _stem(token: str) -> str:
    """Crude suffix-stripping, not a real stemmer -- exists so "governed" and
    "governs", or "law" and "laws", count as the same word instead of failing
    span-support purely because a paraphrase used a different inflection.
    Can produce false collisions on short words (e.g. "class" -> "clas"),
    which is why it only fires on tokens long enough to survive it."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {
        _stem(t)
        for t in tokens
        if t not in _STOPWORDS and len(t) > 1 and not t.isdigit() and t not in _NUMBER_WORDS
    }


# --- quote location and evidence scopes --------------------------------------


def _normalize_cased(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_for_quote_match(text: str) -> str:
    return _normalize_cased(text).lower()


def _normalized_quote(quote: str) -> str:
    return _normalize_for_quote_match(quote).strip(" .\"'")


def _locate_quote(quote: str, clause_text: str) -> tuple[str, int] | None:
    """Returns (normalized clause text, start index of the FIRST occurrence),
    or None if the quote is too short or not present verbatim. Use
    `_quote_starts` where a repeated quote matters."""
    normalized_quote = _normalized_quote(quote)
    if len(normalized_quote) < _MIN_QUOTE_CHARS:
        return None
    normalized_clause = _normalize_for_quote_match(clause_text)
    index = normalized_clause.find(normalized_quote)
    return None if index == -1 else (normalized_clause, index)


def _quote_starts(normalized_quote: str, normalized_clause: str) -> list[int]:
    starts, index = [], normalized_clause.find(normalized_quote)
    while index != -1:
        starts.append(index)
        index = normalized_clause.find(normalized_quote, index + 1)
    return starts


def check_quote_in_clause(quote: str, clause_text: str) -> bool:
    return _locate_quote(quote, clause_text) is not None


@dataclass(frozen=True)
class _Evidence:
    sentence: str  # lowercased; numbers, word overlap, NLI premise
    scope: str  # cased; negation, modality, obligor
    full_sentence: str  # cased; party-name membership


def _span_bounds(text: str, start: int, end: int, separators: tuple[str, ...]) -> tuple[int, int]:
    lefts = [text.rfind(s, 0, start) for s in separators]
    left = max(lefts)
    rights = [i for i in (text.find(s, end) for s in separators) if i != -1]
    right = min(rights) + 1 if rights else len(text)
    return (left + 2 if left != -1 else 0), right


def _segments(sentence: str) -> list[tuple[int, int]]:
    """Split a sentence into independently-negated parts at ", and" /
    ", but" / ", provided". An ", and"/", but" split is only made when the
    following part carries its own verb -- otherwise it continues the
    previous part (and its "not")."""
    cuts = [0]
    points = [m.start() for m in _SEGMENT_SPLIT.finditer(sentence)] + [len(sentence)]
    for i, point in enumerate(points[:-1]):
        following = sentence[point: points[i + 1]].lower()
        if following.lstrip(", ").startswith("provided") or _FINITE_VERB.search(following):
            cuts.append(point)
    cuts.append(len(sentence))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i] < cuts[i + 1]]


def _party_stems(text_cased: str) -> set[str]:
    # Skip the sentence's first word: capitalised only because it starts it.
    stems = {_stem(w.lower()) for w in _CAPITALISED_WORD.findall(text_cased[1:])}
    stems |= {_stem(w) for w in _PARTY_ROLES} | {"agreement", "party", "parti"}
    return stems


def _negation_scope(sentence: str, qs: int, qe: int, claim_tokens: set[str]) -> str:
    segments = _segments(sentence)
    touched = [s for s in segments if s[0] < qe and s[1] > qs] or segments
    touched_tokens = set().union(*(_content_tokens(sentence[a:b]) for a, b in touched))
    ignorable = _party_stems(sentence) | set(_NEGATION_CANON)
    chosen = []
    for a, b in segments:
        if (a, b) in touched:
            chosen.append((a, b))
            continue
        # A segment the claim restates is part of what the claim asserts, so
        # its negation is too -- whichever segment the quote happened to cover.
        novel = _content_tokens(sentence[a:b]) - touched_tokens - ignorable
        if claim_tokens & novel:
            chosen.append((a, b))
    return " ".join(sentence[a:b].strip(" ,") for a, b in chosen)


def _evidence_at(cased: str, start: int, end: int, claim_tokens: set[str]) -> _Evidence:
    s_left, s_right = _span_bounds(cased.lower(), start, end, (". ", "; "))
    f_left, f_right = _span_bounds(cased.lower(), start, end, (". ",))
    sentence = cased[s_left:s_right]
    return _Evidence(
        sentence=sentence.lower(),
        scope=_negation_scope(sentence, start - s_left, end - s_left, claim_tokens),
        full_sentence=cased[f_left:f_right],
    )


# --- numbers ----------------------------------------------------------------


def _parse_number(tokens: list[str], i: int) -> tuple[Decimal, int] | None:
    """Parse the number starting at tokens[i]: digits (optionally with a
    scale word: "1.5 million") or a number-word run ("one hundred and
    twenty", "twenty-five"). Returns (value, index after it)."""
    if i >= len(tokens):
        return None
    token = tokens[i]
    if token[0].isdigit():
        value = Decimal(token.replace(",", ""))
        i += 1
        if i < len(tokens) and tokens[i] in _SCALES:
            value *= _SCALES[tokens[i]]
            i += 1
        return value, i
    if token not in _NUMBER_WORDS:
        return None
    total, current, last = 0, 0, None
    while i < len(tokens):
        word = tokens[i]
        if word == "and" and last in ("hundred", "scale") and i + 1 < len(tokens) and (
            tokens[i + 1] in _SMALL_NUMBERS or tokens[i + 1] in _TENS
        ):
            i += 1
            continue
        if word in _SMALL_NUMBERS:
            n = _SMALL_NUMBERS[word]
            if last == "small" or (last == "tens" and not 1 <= n <= 9):
                break
            current += n
            last = "small"
        elif word in _TENS:
            if last in ("small", "tens"):
                break
            current += _TENS[word]
            last = "tens"
        elif word == "hundred":
            if last == "hundred":
                break
            current = (current or 1) * 100
            last = "hundred"
        elif word in _SCALES:
            if last == "scale" and current == 0:
                break
            total += (current or 1) * _SCALES[word]
            current = 0
            last = "scale"
        else:
            break
        i += 1
    return Decimal(total + current), i


def _parse_unit(tokens: list[str], i: int) -> tuple[str | None, int]:
    if i >= len(tokens):
        return None, i
    token = tokens[i]
    if token in ("business", "calendar") and i + 1 < len(tokens) and tokens[i + 1] in ("day", "days"):
        return ("business_day" if token == "business" else "day"), i + 2
    if token in _UNIT_AFTER:
        return _UNIT_AFTER[token], i + 1
    return None, i


def _quantities(text: str) -> set[tuple[Decimal, str | None]]:
    """(value, unit) for every numeric expression in `text`. Section-style
    references are removed first; a parenthetical restatement ("thirty (30)
    days") shares the unit, and a mismatched one ("twenty (120)") yields
    both values."""
    text = text.lower().replace("per cent", "percent")
    text = re.sub(r"\bu\.?s\.?\s*\$", "$", text)
    text = re.sub(r"\bu\.?s\.?\s+(dollars?)\b", r"\1", text)
    text = _REFERENCE.sub(" ", text)
    tokens = _NUM_TOKEN.findall(text)
    found: set[tuple[Decimal, str | None]] = set()
    i = 0
    while i < len(tokens):
        prefix_unit = None
        j = i
        if tokens[j] in _CURRENCY_PREFIX and _parse_number(tokens, j + 1) is not None:
            prefix_unit = _CURRENCY_PREFIX[tokens[j]]
            j += 1
        parsed = _parse_number(tokens, j)
        if parsed is None:
            i += 1
            continue
        value, j = parsed
        values = [value]
        unit, j = _parse_unit(tokens, j)
        if j < len(tokens) and tokens[j] == "(":
            k = j + 1
            inner_prefix = _CURRENCY_PREFIX.get(tokens[k]) if k < len(tokens) else None
            if inner_prefix:
                k += 1
            inner = _parse_number(tokens, k)
            if inner is not None:
                inner_value, k = inner
                inner_unit, k = _parse_unit(tokens, k)
                if k < len(tokens) and tokens[k] == ")":
                    values.append(inner_value)
                    unit = unit or inner_unit or inner_prefix
                    j = k + 1
                    if unit is None:
                        unit, j = _parse_unit(tokens, j)
        unit = unit or prefix_unit
        found.update((v.normalize(), unit) for v in values)
        i = max(j, i + 1)
    return found


def _missing_quantities(claim: str, evidence: str) -> list[str]:
    evidence_quantities = _quantities(evidence)
    evidence_values = {v for v, _ in evidence_quantities}
    missing = []
    for value, unit in sorted(_quantities(claim), key=lambda q: (q[0], q[1] or "")):
        if (value, unit) in evidence_quantities:
            continue
        if unit is None and value in evidence_values:
            continue
        if unit in ("$", "eur", "gbp") and (value, None) in evidence_quantities:
            continue
        missing.append(f"{value:f}" + (f" {unit}" if unit else ""))
    return missing


# --- negation and modality -----------------------------------------------------


def _prepare_polarity(text: str) -> str:
    """Contractions expanded and deadline/bound phrases canonicalised. Every
    substitution keeps case, so offsets into the result index both the cased
    and the lowered string."""
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    text = _DEADLINE.sub("by", text)
    text = _AT_LEAST.sub("notlessthan", text)
    return _AT_MOST.sub("notmorethan", text)


def _negations(text_lower: str) -> list[tuple[str, int]]:
    return [
        (_NEGATION_CANON[m.group()], m.start())
        for m in re.finditer(r"[a-z]+", text_lower)
        if m.group() in _NEGATION_CANON
    ]


def _is_carve_out(scope_lower: str, claim_tokens: set[str]) -> bool:
    """True if the scope is "not ... except/unless X" and the claim is about
    X -- the positive statement the exception makes."""
    not_positions = [p for kind, p in _negations(scope_lower) if kind == "not"]
    for m in _EXCEPTION.finditer(scope_lower):
        if not any(p < m.start() for p in not_positions):
            continue
        carved = re.split(r"[,;]", scope_lower[m.end():], maxsplit=1)[0]
        carved_tokens = _content_tokens(carved) - set(_NEGATION_CANON)
        if carved_tokens and 2 * len(claim_tokens & carved_tokens) >= len(carved_tokens):
            return True
    return False


def _negation_failure(claim: str, scope: str, claim_tokens: set[str]) -> tuple[str | None, bool]:
    """Returns (failure reason or None, whether the carve-out applied)."""
    claim_lower = _prepare_polarity(claim).lower()
    scope_lower = _prepare_polarity(scope).lower()
    claim_neg = {kind for kind, _ in _negations(claim_lower)}
    scope_neg = {kind for kind, _ in _negations(scope_lower)}
    claim_exc = bool(_EXCEPTION.search(claim_lower))
    scope_exc = bool(_EXCEPTION.search(scope_lower))
    if claim_neg == scope_neg and claim_exc == scope_exc:
        return None, False
    # The carve-out supports only the POSITIVE statement: "may not assign to
    # an Affiliate" against "may not assign, except to an Affiliate" is wrong.
    carve_out_claim = "not" not in claim_neg and not claim_exc and claim_neg <= scope_neg
    if carve_out_claim and _is_carve_out(scope_lower, claim_tokens):
        return None, True
    return "negation differs between claim and evidence", False


def _modal_hits(text: str) -> list[tuple[str, int, int]]:
    """(modal class, start, end) over `text` (already _prepare_polarity'd)."""
    masked = text.lower()
    hits = []
    for modal_class, pattern in _MODAL_PATTERNS:
        for m in pattern.finditer(masked):
            hits.append((modal_class, m.start(), m.end()))
        masked = pattern.sub(lambda m: " " * len(m.group()), masked)
    return sorted(hits, key=lambda h: h[1])


def _modal_classes(text: str) -> set[str]:
    return {modal_class for modal_class, _, _ in _modal_hits(_prepare_polarity(text))}


def _subject_pairs(text: str, is_party) -> list[tuple[str, str | None]]:
    """(modal class, party stem or None for "either party"/"each Party")
    for each active-voice modal whose subject names a party."""
    prepared = _prepare_polarity(text)
    lowered = prepared.lower()
    pairs = []
    previous_end = 0
    for modal_class, start, end in _modal_hits(prepared):
        window_start = max(previous_end, *(prepared.rfind(ch, 0, start) + 1 for ch in ",;:("))
        previous_end = end
        if _PASSIVE_AFTER_MODAL.match(lowered, end):
            continue
        words = _WORD.findall(prepared[window_start:start])[-5:]
        if any(w.lower() in _WILDCARD_PARTY for w in words):
            pairs.append((modal_class, None))
            continue
        for word in words:
            base = re.split(r"['’]", word)[0]
            if is_party(base):
                pairs.append((modal_class, _stem(base.lower())))
    return pairs


def _clause_defined_terms(clause_text: str) -> set[str]:
    """Lowercased words capitalised in the clause and never written in
    lowercase there -- its parties and defined terms."""
    lowercase_words = set(re.findall(r"\b[a-z][a-z]+\b", clause_text))
    return {
        w.lower()
        for w in _CAPITALISED_WORD.findall(clause_text)
        if w.lower() not in _CAPITALISED_STOPWORDS and w.lower() not in lowercase_words
    }


def _party_failure(claim: str, evidence: _Evidence, defined_terms: set[str], carve_out: bool) -> str | None:
    full_lower = evidence.full_sentence.lower()

    def claim_party(word: str) -> bool:
        lower = word.lower()
        return lower not in _CAPITALISED_STOPWORDS and (lower in _PARTY_ROLES or (word[:1].isupper() and lower in defined_terms))

    claim_pairs = _subject_pairs(claim, claim_party)
    # "Either party may terminate" supports "Licensee may terminate": a
    # generic party in the evidence covers the claim's named obligor.
    generic_party = _GENERIC_PARTY.search(full_lower) is not None
    claim_subjects = {party for _, party in claim_pairs}
    for word in _CAPITALISED_WORD.findall(claim):
        if not claim_party(word) or re.search(rf"\b{re.escape(word.lower())}", full_lower):
            continue
        if generic_party and _stem(word.lower()) in claim_subjects:
            continue
        return f"claim names '{word}', which the evidence sentence does not mention"

    def evidence_party(word: str) -> bool:
        lower = word.lower()
        return lower in _PARTY_ROLES or (word[:1].isupper() and lower not in _CAPITALISED_STOPWORDS)

    subjects: dict[str, set[str | None]] = {}
    for modal_class, party in _subject_pairs(evidence.scope, evidence_party):
        subjects.setdefault(modal_class, set()).add(party)
        if carve_out and modal_class == "prohibition":
            subjects.setdefault("permission", set()).add(party)
    for modal_class, party in claim_pairs:
        allowed = subjects.get(modal_class)
        if party is None or not allowed or None in allowed or party in allowed:
            continue
        named = ", ".join(sorted(p for p in allowed if p))
        return f"claim attributes the {modal_class.replace('_', ' ')} to '{party}', but the evidence attributes it to '{named}'"
    return None


# --- the check ------------------------------------------------------------------


def check_span_supported(
    claim_text: str,
    clause_text: str,
    *,
    supporting_quote: str | None = None,
    min_overlap_ratio: float = 0.6,
) -> bool:
    return _span_failure(claim_text, clause_text, supporting_quote, min_overlap_ratio) is None


def _span_failure(
    claim_text: str,
    clause_text: str,
    supporting_quote: str | None,
    min_overlap_ratio: float,
    entailment_checker: EntailmentChecker | None = None,
) -> str | None:
    claim_tokens = _content_tokens(claim_text)
    if not claim_tokens:
        return "empty claim"

    cased = _normalize_cased(clause_text)
    if len(cased.lower()) != len(cased):  # lowercasing changed offsets (rare Unicode)
        cased = cased.lower()

    if supporting_quote is not None:
        quote = _normalized_quote(supporting_quote)
        starts = _quote_starts(quote, cased.lower()) if len(quote) >= _MIN_QUOTE_CHARS else []
        if not starts:
            return "supporting quote does not appear verbatim in the cited clause"
        candidates = {_evidence_at(cased, s, s + len(quote), claim_tokens) for s in starts}
        if len(candidates) > 1:
            return (
                f"supporting_quote is ambiguous (appears {len(starts)} times in the clause, "
                "in different sentences); quote more context"
            )
        [evidence] = candidates
    else:
        evidence = _Evidence(sentence=cased.lower(), scope=cased, full_sentence=cased)

    missing = _missing_quantities(claim_text, evidence.sentence)
    if missing:
        return f"claim asserts a number the evidence does not contain ({', '.join(missing)})"

    negation_failure, carve_out = _negation_failure(claim_text, evidence.scope, claim_tokens)
    if negation_failure:
        return negation_failure

    claim_modals = _modal_classes(claim_text)
    evidence_modals = _modal_classes(evidence.scope) or _modal_classes(evidence.sentence)
    if carve_out and "prohibition" in evidence_modals:
        evidence_modals.add("permission")
    if claim_modals and evidence_modals and not claim_modals <= evidence_modals:
        return (
            f"modality differs between claim and evidence (claim: {', '.join(sorted(claim_modals))}; "
            f"evidence: {', '.join(sorted(evidence_modals))})"
        )

    party_failure = _party_failure(claim_text, evidence, _clause_defined_terms(clause_text), carve_out)
    if party_failure:
        return party_failure

    evidence_tokens = _content_tokens(evidence.sentence)
    overlap_ratio = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    if overlap_ratio < min_overlap_ratio:
        return f"claim wording not supported by the evidence (overlap {overlap_ratio:.0%})"

    if entailment_checker is not None and not entailment_checker.entails(evidence.sentence, claim_text):
        return "claim is not entailed by the quoted evidence (NLI check)"

    return None


def _resolve(key: str, clauses: list[Clause]) -> Clause | None:
    for clause in clauses:
        if clause.clause_id == key:
            return clause
    by_section = [c for c in clauses if c.section_id is not None and c.section_id == key]
    return by_section[0] if len(by_section) == 1 else None


def validate_citation(
    citation: Citation,
    *,
    source_clauses: list[Clause],
    shown_clauses: list[RetrievedClause],
    min_overlap_ratio: float = 0.6,
    entailment_checker: EntailmentChecker | None = None,
) -> CitationValidationResult:
    source_clause = _resolve(citation.section_id, source_clauses)
    shown_clause = _resolve(citation.section_id, [rc.clause for rc in shown_clauses])

    exists = source_clause is not None
    # Same clause on both sides -- not merely the same key.
    in_context = exists and shown_clause is not None and shown_clause.clause_id == source_clause.clause_id

    failure_reason: str | None = None
    span_supported = False
    if not exists:
        failure_reason = f"cited clause '{citation.section_id}' does not exist (or is ambiguous) in the document"
    elif not in_context:
        failure_reason = f"cited clause '{citation.section_id}' was not among the clauses shown"
    else:
        failure_reason = _span_failure(
            citation.claim_text,
            shown_clause.text,
            citation.supporting_quote,
            min_overlap_ratio,
            entailment_checker,
        )
        span_supported = failure_reason is None

    return CitationValidationResult(
        citation=citation,
        exists=exists,
        in_context=in_context,
        span_supported=span_supported,
        resolved_clause_id=source_clause.clause_id if source_clause else None,
        failure_reason=failure_reason,
    )


def validate_citations(
    citations: list[Citation],
    *,
    source_clauses: list[Clause],
    shown_clauses: list[RetrievedClause],
    entailment_checker: EntailmentChecker | None = None,
) -> list[CitationValidationResult]:
    return [
        validate_citation(
            c, source_clauses=source_clauses, shown_clauses=shown_clauses, entailment_checker=entailment_checker
        )
        for c in citations
    ]

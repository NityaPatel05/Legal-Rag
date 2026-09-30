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
     the QUOTE, not the whole clause, which is what makes them meaningful:
     comparing a one-line claim's negations/numbers against a 3,000-word
     clause both rejects correct claims (the clause says "not" somewhere
     else) and accepts wrong ones (the wrong number appears somewhere else).
  b. no quote (legacy callers): the same checks against the whole clause.

Claim-level checks: content-word overlap >= threshold; the claim's negation
markers equal the evidence's; every number the claim asserts appears in the
evidence. Numbers are compared after mapping number words to digits, so
"ninety days" vs "thirty (30) days" fails -- the original digit-only check
let that through.

Those checks are lexical: a claim can reuse the quote's words and numbers
to assert something subtly different (swapping which party bears an
obligation, "may" vs "shall"). The optional entailment layer (roadmap R6)
closes that gap: when an EntailmentChecker is supplied, a claim that passed
every string check must ALSO be entailed by its evidence (the quote's
enclosing sentence(s)) under a fixed NLI model and fixed threshold -- still
deterministic, still code-level, never LLM self-critique. It is off unless
configured (NLI_MODEL), because the model must be validated (ContractNLI)
before it is allowed to reject claims in production.
"""

from __future__ import annotations

import re

from typing import Protocol

from pydantic import BaseModel

from ingestion.schemas import Clause
from query.schemas import RetrievedClause

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "shall", "must", "to", "of",
    "and", "or", "in", "on", "for", "that", "this", "it", "by", "as", "be",
    "with", "at", "any", "all", "such", "from", "which", "will", "may",
}

_NEGATION_MARKERS = {
    "not", "no", "never", "without", "unless", "excluding", "except", "nor", "neither",
}

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORD = re.compile(
    rf"\b(?:(?P<tens>{'|'.join(_TENS)})(?:[\s-](?P<ones>{'|'.join(k for k in _UNITS if _UNITS[k] < 10 and _UNITS[k] > 0)}))?"
    rf"|(?P<unit>{'|'.join(_UNITS)}))\b",
    re.IGNORECASE,
)

_MIN_QUOTE_CHARS = 12


class EntailmentChecker(Protocol):
    def entails(self, premise: str, hypothesis: str) -> bool:
        """True if `premise` entails `hypothesis`. Must be deterministic."""
        ...


class CrossEncoderEntailmentChecker:
    """NLI cross-encoder via sentence-transformers (lazy import; the model is
    loaded on first use). The entailment label index is read from the
    model's own id2label rather than assumed, since NLI checkpoints order
    their labels differently."""

    def __init__(self, model_name: str, *, threshold: float = 0.5):
        self._model_name = model_name
        self._threshold = threshold
        self._model = None
        self._entailment_index: int | None = None

    def _load(self):
        if self._model is None:
            import os

            # transformers otherwise imports TensorFlow when it's installed,
            # which fails under Keras 3; this service never uses TF.
            os.environ.setdefault("USE_TF", "0")
            from sentence_transformers import CrossEncoder  # lazy import

            self._model = CrossEncoder(self._model_name)
            id2label = self._model.model.config.id2label
            self._entailment_index = next(
                int(i) for i, label in id2label.items() if str(label).lower().startswith("entail")
            )
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
    return {_stem(t) for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _negation_markers_in(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z]+", text.lower().replace("n't", " not")))
    return tokens & _NEGATION_MARKERS


def _number_words_to_digits(text: str) -> str:
    def _replace(m: re.Match) -> str:
        if m.group("tens"):
            value = _TENS[m.group("tens").lower()] + (_UNITS[m.group("ones").lower()] if m.group("ones") else 0)
        else:
            value = _UNITS[m.group("unit").lower()]
        return f" {value} "

    return _NUMBER_WORD.sub(_replace, text)


def _numeric_tokens(text: str) -> set[str]:
    normalized = _number_words_to_digits(text).replace(",", "")
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+", normalized)}


def _normalize_for_quote_match(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def _locate_quote(quote: str, clause_text: str) -> tuple[str, int] | None:
    """Returns (normalized clause text, start index of the quote in it), or
    None if the quote is too short or not present verbatim."""
    normalized_quote = _normalize_for_quote_match(quote).strip(" .\"'")
    if len(normalized_quote) < _MIN_QUOTE_CHARS:
        return None
    normalized_clause = _normalize_for_quote_match(clause_text)
    index = normalized_clause.find(normalized_quote)
    return None if index == -1 else (normalized_clause, index)


def check_quote_in_clause(quote: str, clause_text: str) -> bool:
    return _locate_quote(quote, clause_text) is not None


def _enclosing_sentences(quote: str, clause_text: str) -> str:
    """The quote widened to the sentence(s) containing it. Word overlap is
    measured against this rather than the bare quote, so a correct claim
    isn't rejected because the model quoted tightly ("terminate ... upon
    thirty (30) days") and the claim names the subject just outside it
    ("Either party ...")."""
    located = _locate_quote(quote, clause_text)
    if located is None:
        return quote
    text, start = located
    end = start + len(_normalize_for_quote_match(quote).strip(" .\"'"))
    left = max(text.rfind(". ", 0, start), text.rfind("; ", 0, start))
    right_candidates = [i for i in (text.find(". ", end), text.find("; ", end)) if i != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return text[left + 2 if left != -1 else 0 : right]


def _resolve(key: str, clauses: list[Clause]) -> Clause | None:
    for clause in clauses:
        if clause.clause_id == key:
            return clause
    by_section = [c for c in clauses if c.section_id is not None and c.section_id == key]
    return by_section[0] if len(by_section) == 1 else None


def check_exists(section_id: str, source_clauses: list[Clause]) -> bool:
    return _resolve(section_id, source_clauses) is not None


def check_in_context(section_id: str, shown_clauses: list[RetrievedClause]) -> bool:
    return _resolve(section_id, [rc.clause for rc in shown_clauses]) is not None


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

    if supporting_quote is not None:
        if not check_quote_in_clause(supporting_quote, clause_text):
            return "supporting quote does not appear verbatim in the cited clause"
        # Facts (numbers, negation) are held to the sentence(s) the quote
        # sits in; topical overlap is measured against the same span.
        evidence = _enclosing_sentences(supporting_quote, clause_text)
    else:
        evidence = clause_text

    # Most specific checks first, so the failure reason fed back into
    # regeneration names the actual error.
    claim_numbers = _numeric_tokens(claim_text)
    if claim_numbers and not claim_numbers.issubset(_numeric_tokens(evidence)):
        return "claim asserts a number the evidence does not contain"

    if _negation_markers_in(claim_text) != _negation_markers_in(evidence):
        return "negation differs between claim and evidence"

    evidence_tokens = _content_tokens(evidence)
    overlap_ratio = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    if overlap_ratio < min_overlap_ratio:
        return f"claim wording not supported by the evidence (overlap {overlap_ratio:.0%})"

    if entailment_checker is not None and not entailment_checker.entails(evidence, claim_text):
        return "claim is not entailed by the quoted evidence (NLI check)"

    return None


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

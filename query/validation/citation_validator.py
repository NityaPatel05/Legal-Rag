"""Deterministic citation validator — CLAUDE.md #4: code-level, not LLM
self-critique. For every cited section/clause ID, three checks:

  1. exists          — the cited section ID is real in the source document.
  2. in_context       — the backing chunk was actually shown to the model
                        (i.e. present in the clauses retrieval handed to
                        generation, not just somewhere in the document).
  3. span_supported   — the cited clause's actual text supports the specific
                        claim, not just the same topic. The check most RAG
                        stacks skip, and the one that catches "right clause,
                        wrong claim" citations.

span_supported is a deterministic bag-of-content-words overlap ratio, plus a
negation-marker check and a numeric-token check, not an LLM entailment call —
CLAUDE.md is explicit that this validator is code-level. The numeric check
exists because word-overlap alone misses the single most common high-stakes
error in contract review: a claim that swaps one number for another ("ninety
(90) days" cited against a clause that actually says "thirty (30) days") has
~90% word overlap with the real clause, since only the digit token differs —
that would pass a pure overlap threshold and is exactly the "right clause,
wrong claim" failure CLAUDE.md calls out. Any digit the claim asserts must
appear somewhere in the cited clause, or validation fails regardless of
overall overlap.

This is still not true semantic entailment and has real limits: a claim that
reuses the clause's own words and numbers to assert something subtly
different (e.g. swapping which party bears an obligation, when both party
names appear elsewhere in the clause) can still slip through. Known gap, not
a design oversight — flagged for a future entailment-model pass once the
rest of the pipeline (steps 4-6) is in place.
"""

from __future__ import annotations

import re

from ingestion.schemas import Clause
from pydantic import BaseModel

from query.schemas import RetrievedClause

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "shall", "must", "to", "of",
    "and", "or", "in", "on", "for", "that", "this", "it", "by", "as", "be",
    "with", "at", "any", "all", "such", "from", "which", "will", "may",
}

_NEGATION_MARKERS = {"not", "no", "never", "without", "unless", "excluding", "except", "nor"}


class Citation(BaseModel):
    section_id: str
    claim_text: str


class CitationValidationResult(BaseModel):
    citation: Citation
    exists: bool
    in_context: bool
    span_supported: bool

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
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return tokens & _NEGATION_MARKERS


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def check_exists(section_id: str, source_clauses: list[Clause]) -> bool:
    return any(c.section_id == section_id for c in source_clauses)


def check_in_context(section_id: str, shown_clauses: list[RetrievedClause]) -> bool:
    return any(rc.clause.section_id == section_id for rc in shown_clauses)


def check_span_supported(claim_text: str, clause_text: str, *, min_overlap_ratio: float = 0.6) -> bool:
    claim_tokens = _content_tokens(claim_text)
    if not claim_tokens:
        return False

    clause_tokens = _content_tokens(clause_text)
    overlap_ratio = len(claim_tokens & clause_tokens) / len(claim_tokens)
    if overlap_ratio < min_overlap_ratio:
        return False

    if _negation_markers_in(claim_text) != _negation_markers_in(clause_text):
        return False

    claim_numbers = _numeric_tokens(claim_text)
    clause_numbers = _numeric_tokens(clause_text)
    if claim_numbers and not claim_numbers.issubset(clause_numbers):
        return False

    return True


def validate_citation(
    citation: Citation,
    *,
    source_clauses: list[Clause],
    shown_clauses: list[RetrievedClause],
) -> CitationValidationResult:
    exists = check_exists(citation.section_id, source_clauses)
    in_context = check_in_context(citation.section_id, shown_clauses)

    span_supported = False
    if exists and in_context:
        cited_clause = next(rc.clause for rc in shown_clauses if rc.clause.section_id == citation.section_id)
        span_supported = check_span_supported(citation.claim_text, cited_clause.text)

    return CitationValidationResult(
        citation=citation, exists=exists, in_context=in_context, span_supported=span_supported
    )


def validate_citations(
    citations: list[Citation],
    *,
    source_clauses: list[Clause],
    shown_clauses: list[RetrievedClause],
) -> list[CitationValidationResult]:
    return [
        validate_citation(c, source_clauses=source_clauses, shown_clauses=shown_clauses) for c in citations
    ]

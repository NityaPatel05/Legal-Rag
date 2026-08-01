"""Fallback when the citation validator's retry budget is exhausted —
CLAUDE.md #4: "once retries are exhausted, strip unsupported claims rather
than fail silently." Operates on claim-level (text, citation) pairs rather
than a raw generated-answer string, since parsing/re-serializing a full
structured-citation answer belongs to the generation step (not yet wired —
build-order steps 4-5 precede it); this is the deterministic piece those
later steps call once a claim/citation pair has been validated."""

from __future__ import annotations

from pydantic import BaseModel

from query.validation.citation_validator import CitationValidationResult


class Claim(BaseModel):
    claim_text: str
    citation_section_id: str


class StripResult(BaseModel):
    kept: list[Claim]
    stripped: list[Claim]

    @property
    def any_stripped(self) -> bool:
        return len(self.stripped) > 0


def strip_unsupported_claims(
    claims: list[Claim], validation_results: list[CitationValidationResult]
) -> StripResult:
    """validation_results must be the same length and order as claims — one
    CitationValidationResult per claim's citation."""
    if len(claims) != len(validation_results):
        raise ValueError("claims and validation_results must be the same length")

    kept: list[Claim] = []
    stripped: list[Claim] = []
    for claim, result in zip(claims, validation_results):
        (kept if result.is_valid else stripped).append(claim)

    return StripResult(kept=kept, stripped=stripped)

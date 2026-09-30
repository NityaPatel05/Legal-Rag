"""Review-case data model (docs/architecture-review.md H1/H2).

A ReviewCase is the evidence packet a reviewing lawyer decides on, and it is
also the persisted state of the review: everything a decision needs (the
question, the clauses the model saw, each draft claim with its quote and the
validator's verdict) is copied in at queue time, so deciding never depends on
re-running the model."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from query.schemas import RetrievalTraceStep


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ReviewStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"


class ClaimAction(str, Enum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class ShownClause(BaseModel):
    """A clause the model was shown, as it read it -- injection flags
    included, so the reviewer sees which evidence may be adversarial."""

    clause_id: str
    section_id: str | None = None
    heading: str | None = None
    text: str
    source: str
    injection_suspected: bool = False
    injection_markers: list[str] = Field(default_factory=list)


class DraftClaim(BaseModel):
    index: int
    claim_text: str
    citation: str
    supporting_quote: str | None = None
    is_valid: bool
    resolved_clause_id: str | None = None
    failure_reason: str | None = None


class ClaimDecision(BaseModel):
    """One reviewer judgement. `index` is a DraftClaim index, or None for a
    claim the reviewer adds. Edits and additions carry the full claim."""

    index: int | None = None
    action: ClaimAction
    claim_text: str | None = None
    citation: str | None = None
    supporting_quote: str | None = None
    # Required whenever the claim, as decided, fails the citation validator.
    override_reason: str | None = None


class DecisionRequest(BaseModel):
    """Every draft claim needs its own decision -- there is deliberately no
    "approve all" (H2: bulk approval is where automation bias lives)."""

    reviewer: str = Field(min_length=1)
    decisions: list[ClaimDecision]
    note: str | None = None


class ReviewedClaim(BaseModel):
    index: int | None
    action: ClaimAction
    claim_text: str
    citation: str
    supporting_quote: str | None = None
    resolved_clause_id: str | None = None
    validator_passed: bool
    failure_reason: str | None = None
    override_reason: str | None = None


class ReviewOutcome(BaseModel):
    reviewer: str
    decided_at: datetime = Field(default_factory=_now)
    # Seconds from the reviewer first opening the case to deciding -- a
    # rubber-stamping signal (H2). None if the case was never opened first.
    seconds_to_decision: float | None = None
    claims: list[ReviewedClaim]
    final_status: str  # "answered_after_review" | "rejected_after_review"
    answer_text: str | None = None
    note: str | None = None


class ReviewCase(BaseModel):
    review_id: str
    tenant_id: str
    matter_id: str
    document_id: str
    status: ReviewStatus = ReviewStatus.OPEN
    created_at: datetime = Field(default_factory=_now)
    first_viewed_at: datetime | None = None
    # Why it was queued, e.g. the CRAG flagged_reason.
    reason: str
    original_final_status: str
    question: str
    final_query_text: str
    route: str
    trace: list[RetrievalTraceStep] = Field(default_factory=list)
    shown_clauses: list[ShownClause] = Field(default_factory=list)
    # Validator failures first (H2), then in the model's order.
    draft_claims: list[DraftClaim] = Field(default_factory=list)
    outcome: ReviewOutcome | None = None

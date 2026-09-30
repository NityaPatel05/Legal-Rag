"""Human review of CRAG outcomes (docs/architecture-review.md H1/H2).

Before this, `flagged_for_review` was a label on a response and nothing
received it. Now a flagged (or partially stripped) CRAG outcome becomes a
persisted ReviewCase, and a reviewer's decision produces a released answer.

Why not LangGraph interrupt() + a checkpointer, as the review suggested: the
CRAG state threads live clients (LLM, Qdrant, reranker) through the graph,
which no checkpointer can serialize, and none of approve / edit / reject
needs the model again -- they are deterministic checks against the
contract. The ReviewCase *is* the persisted state. "Re-retrieve with a
hint" is just a new /query with a reformulated question.

The invariant this preserves: every claim released to a user is either
validated by the citation validator or explicitly signed off. Reviewer edits
and approvals are re-run through the validator against the document as
stored *now*. A claim that still fails is released only with an
`override_reason`, and every override is audited separately
(`review_override`), so "who vouched for this ungrounded claim, and why" is
answerable from the audit log.

Against automation bias (H2, arXiv:2507.19486): every draft claim needs its
own decision (no approve-all), validator failures are listed first, and the
time from first opening the case to deciding is recorded.

Exactly one decision per case, even under concurrency: pass `store=` to
`apply_decision` and the resolved case is persisted with the store's
conditional write (only if still open) BEFORE anything is audited, so a
losing concurrent decision raises ReviewAlreadyDecided (a
ReviewConflictError, HTTP 409) and leaves no audit trail of a decision that
never took effect. `mark_viewed(case, store=...)` likewise records the first
view with a conditional update instead of a whole-case save that could
overwrite a decision."""

from __future__ import annotations

import uuid

from ingestion.schemas import Clause, ClauseMetadata
from query.filters.audit_log import AuditEvent, AuditLogger
from query.generation.crag_graph import compose_answer_text
from query.review.schemas import (  # noqa: F401 -- exceptions re-exported for app/main.py
    ClaimAction,
    ClaimDecision,
    DecisionRequest,
    DraftClaim,
    ReviewAlreadyDecided,
    ReviewCase,
    ReviewConflictError,
    ReviewDecisionError,
    ReviewedClaim,
    ReviewOutcome,
    ReviewStatus,
    ShownClause,
    _now,
)
from query.review.store import ReviewStore
from query.schemas import PermissionContext, RetrievalTraceStep, RetrievedClause
from query.validation.citation_validator import Citation, EntailmentChecker, validate_citation
from query.validation.strip_unsupported import Claim

# Outcomes a human should look at. "answered_with_stripped_claims" is
# included: the stripped claims may be correct ones the validator rejected
# (L19), and the user has only received the part that survived.
# "not_found" (crag_graph: every retrieval attempt graded insufficient) is
# included too: "the contract doesn't address this" is itself a legal
# conclusion, and a retrieval miss looks identical to it -- a human confirms
# the absence.
REVIEWABLE_STATUSES = frozenset({"flagged_for_review", "answered_with_stripped_claims", "not_found"})


def should_queue(final_state: dict) -> bool:
    """Only outcomes that went through retrieval carry evidence to review.
    Jailbreak blocks and out-of-scope refusals never reach retrieval and
    need a rephrased question, not a lawyer."""
    return final_state.get("final_status") in REVIEWABLE_STATUSES and "retrieved_clauses" in final_state


def build_review_case(
    final_state: dict,
    *,
    permission_context: PermissionContext,
    document_id: str,
    trace: list[RetrievalTraceStep],
    review_id: str | None = None,
) -> ReviewCase:
    shown = [_shown_clause(rc) for rc in final_state.get("retrieved_clauses", [])]
    drafts = [
        DraftClaim(
            index=i,
            claim_text=r.citation.claim_text,
            citation=r.citation.section_id,
            supporting_quote=r.citation.supporting_quote,
            is_valid=r.is_valid,
            resolved_clause_id=r.resolved_clause_id,
            failure_reason=r.failure_reason,
        )
        for i, r in enumerate(final_state.get("citation_results", []))
    ]
    drafts.sort(key=lambda d: (d.is_valid, d.index))  # failures first (H2)
    reason = final_state.get("flagged_reason") or (
        f"{sum(not d.is_valid for d in drafts)} claim(s) removed after failing citation validation."
    )
    return ReviewCase(
        review_id=review_id or str(uuid.uuid4()),
        tenant_id=permission_context.tenant_id,
        matter_id=permission_context.matter_id,
        document_id=document_id,
        reason=reason,
        original_final_status=final_state["final_status"],
        question=final_state.get("original_query_text", ""),
        final_query_text=final_state.get("current_query_text", ""),
        route=final_state.get("route", ""),
        trace=trace,
        shown_clauses=shown,
        draft_claims=drafts,
    )


def _shown_clause(rc: RetrievedClause) -> ShownClause:
    return ShownClause(
        clause_id=rc.clause.clause_id,
        section_id=rc.clause.section_id,
        heading=rc.clause.heading,
        text=rc.clause.text,
        source=rc.source,
        injection_suspected=rc.metadata.injection_suspected,
        injection_markers=rc.metadata.injection_markers,
    )


def mark_viewed(case: ReviewCase, *, store: ReviewStore | None = None) -> bool:
    """Records the first time a reviewer opens the case. Returns True if
    the case changed.

    With `store`, the first view is persisted by the store's conditional
    update (first_viewed_at IS NULL AND status open) and the return value
    says whether this call won; nothing else needs saving. Without it, the
    caller saves -- ReviewStore.save routes an open case to the same
    conditional update."""
    if case.first_viewed_at is not None or case.status != ReviewStatus.OPEN:
        return False
    viewed_at = _now()
    if store is not None and not store.mark_viewed(case.review_id, viewed_at):
        return False
    case.first_viewed_at = viewed_at
    return True


def apply_decision(
    case: ReviewCase,
    request: DecisionRequest,
    *,
    all_clauses_by_id: dict[str, tuple[Clause, ClauseMetadata]],
    audit_logger: AuditLogger | None = None,
    entailment_checker: EntailmentChecker | None = None,
    store: ReviewStore | None = None,
) -> ReviewCase:
    """Validates the request, re-validates every accepted claim against the
    current document, and resolves the case. Mutates and returns `case`.

    With `store`, the resolved case is persisted atomically-if-still-open
    before auditing; a concurrent decision that got there first raises
    ReviewAlreadyDecided (HTTP 409) and nothing is audited. Without it the
    caller must persist (ReviewStore.save / record_decision) -- and the
    audit events are already written by then.

    The reviewer has the whole contract, not just the clauses the model was
    shown, so a reviewer's citation may point at any clause of the document
    -- but its quote must still be verbatim in that clause."""
    if case.status != ReviewStatus.OPEN:
        raise ReviewConflictError(f"Review {case.review_id} is already {case.status.value}")
    _check_decision_coverage(case, request)

    drafts = {d.index: d for d in case.draft_claims}
    source_clauses = [clause for clause, _ in all_clauses_by_id.values()]
    shown_clauses = [
        RetrievedClause(clause=clause, metadata=metadata, score=0.0, source="review")
        for clause, metadata in all_clauses_by_id.values()
    ]

    reviewed: list[ReviewedClaim] = []
    for decision in request.decisions:
        claim = _decided_claim(decision, drafts)
        if decision.action == ClaimAction.REJECT:
            reviewed.append(
                ReviewedClaim(
                    index=decision.index,
                    action=decision.action,
                    claim_text=claim.claim_text,
                    citation=claim.citation_section_id,
                    supporting_quote=claim.supporting_quote,
                    validator_passed=False,
                    override_reason=decision.override_reason,
                )
            )
            continue
        result = validate_citation(
            Citation(
                section_id=claim.citation_section_id,
                claim_text=claim.claim_text,
                supporting_quote=claim.supporting_quote,
            ),
            source_clauses=source_clauses,
            shown_clauses=shown_clauses,
            entailment_checker=entailment_checker,
        )
        override = (decision.override_reason or "").strip() or None
        if not result.is_valid and override is None:
            raise ReviewDecisionError(
                f"Claim {_label(decision)} fails citation validation ({result.failure_reason}); "
                "fix it, reject it, or give an override_reason."
            )
        reviewed.append(
            ReviewedClaim(
                index=decision.index,
                action=decision.action,
                claim_text=claim.claim_text,
                citation=claim.citation_section_id,
                supporting_quote=claim.supporting_quote,
                resolved_clause_id=result.resolved_clause_id,
                validator_passed=result.is_valid,
                failure_reason=result.failure_reason,
                override_reason=override if not result.is_valid else None,
            )
        )

    accepted = [r for r in reviewed if r.action != ClaimAction.REJECT]
    decided_at = _now()
    case.outcome = ReviewOutcome(
        reviewer=request.reviewer,
        decided_at=decided_at,
        seconds_to_decision=(
            (decided_at - case.first_viewed_at).total_seconds() if case.first_viewed_at else None
        ),
        claims=reviewed,
        final_status="answered_after_review" if accepted else "rejected_after_review",
        answer_text=(
            compose_answer_text(
                [Claim(claim_text=r.claim_text, citation_section_id=r.citation) for r in accepted]
            )
            if accepted
            else None
        ),
        note=request.note,
    )
    case.status = ReviewStatus.RESOLVED
    if store is not None:
        store.record_decision(case)  # raises ReviewAlreadyDecided if a concurrent decision won
    _audit(case, audit_logger)
    return case


def _check_decision_coverage(case: ReviewCase, request: DecisionRequest) -> None:
    draft_indexes = {d.index for d in case.draft_claims}
    seen: set[int] = set()
    for decision in request.decisions:
        if decision.index is None:
            if decision.action != ClaimAction.EDIT:
                raise ReviewDecisionError("A reviewer-added claim (no index) must use action 'edit'")
            continue
        if decision.index not in draft_indexes:
            raise ReviewDecisionError(f"No draft claim with index {decision.index}")
        if decision.index in seen:
            raise ReviewDecisionError(f"Draft claim {decision.index} decided more than once")
        seen.add(decision.index)
    missing = sorted(draft_indexes - seen)
    if missing:
        raise ReviewDecisionError(f"Every draft claim needs its own decision; missing: {missing}")


def _decided_claim(decision: ClaimDecision, drafts: dict[int, DraftClaim]) -> Claim:
    if decision.action == ClaimAction.EDIT:
        if not (decision.claim_text and decision.citation and decision.supporting_quote):
            raise ReviewDecisionError(
                f"Claim {_label(decision)}: an edit needs claim_text, citation and supporting_quote"
            )
        return Claim(
            claim_text=decision.claim_text,
            citation_section_id=decision.citation,
            supporting_quote=decision.supporting_quote,
        )
    draft = drafts[decision.index]
    return Claim(
        claim_text=draft.claim_text,
        citation_section_id=draft.citation,
        supporting_quote=draft.supporting_quote,
    )


def _label(decision: ClaimDecision) -> str:
    return str(decision.index) if decision.index is not None else "(added)"


def _audit(case: ReviewCase, audit_logger: AuditLogger | None) -> None:
    if audit_logger is None or case.outcome is None:
        return
    outcome = case.outcome
    audit_logger.log(
        AuditEvent(
            event_type="review_decision",
            tenant_id=case.tenant_id,
            matter_id=case.matter_id,
            details={
                "review_id": case.review_id,
                "document_id": case.document_id,
                "reviewer": outcome.reviewer,
                "final_status": outcome.final_status,
                "seconds_to_decision": outcome.seconds_to_decision,
                "actions": [{"index": c.index, "action": c.action.value} for c in outcome.claims],
            },
        )
    )
    for claim in outcome.claims:
        if claim.override_reason and claim.action != ClaimAction.REJECT:
            audit_logger.log(
                AuditEvent(
                    event_type="review_override",
                    tenant_id=case.tenant_id,
                    matter_id=case.matter_id,
                    details={
                        "review_id": case.review_id,
                        "reviewer": outcome.reviewer,
                        "claim_text": claim.claim_text,
                        "citation": claim.citation,
                        "failure_reason": claim.failure_reason,
                        "override_reason": claim.override_reason,
                    },
                )
            )

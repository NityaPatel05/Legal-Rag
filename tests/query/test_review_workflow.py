"""Human review queue (architecture-review H1/H2): what gets queued, the
evidence packet, and that a reviewer can only release grounded or
explicitly-overridden claims."""

import pytest

from query.filters.audit_log import InMemoryAuditLogger
from query.review.schemas import ClaimAction, ClaimDecision, DecisionRequest, ReviewStatus
from query.review.store import InMemoryReviewStore, SQLiteReviewStore
from query.review.workflow import (
    ReviewConflictError,
    ReviewDecisionError,
    apply_decision,
    build_review_case,
    mark_viewed,
    should_queue,
)
from query.schemas import PermissionContext, RetrievalTraceStep, RetrievedClause
from query.validation.citation_validator import Citation, validate_citations
from tests.query.factories import make_clause, make_metadata

TERM = make_clause(
    "doc::clause-003",
    section_id="3",
    heading="Termination",
    text="Either party may terminate this Agreement for cause upon thirty (30) days written notice.",
)
LAW = make_clause(
    "doc::clause-004",
    section_id="4",
    heading="Governing Law",
    text="This Agreement is governed by the laws of the State of Delaware.",
)
ALL = {c.clause_id: (c, make_metadata()) for c in (TERM, LAW)}
PC = PermissionContext(tenant_id="tenant-1", matter_id="matter-1")

GOOD = Citation(
    section_id=TERM.clause_id,
    claim_text="Either party may terminate for cause on thirty days notice.",
    supporting_quote="Either party may terminate this Agreement for cause upon thirty (30) days written notice",
)
BAD = Citation(
    section_id=TERM.clause_id,
    claim_text="Either party may terminate on ninety days notice.",
    supporting_quote="terminate this Agreement upon ninety (90) days notice",
)


def _state(status="answered_with_stripped_claims", citations=(GOOD, BAD), injected=False):
    metadata = make_metadata()
    metadata.injection_suspected = injected
    shown = [RetrievedClause(clause=TERM, metadata=metadata, score=1.0, source="hybrid")]
    return {
        "final_status": status,
        "flagged_reason": None,
        "original_query_text": "How can the agreement be terminated?",
        "current_query_text": "How can the agreement be terminated?",
        "route": "clause_lookup",
        "retrieved_clauses": shown,
        "citation_results": validate_citations(list(citations), source_clauses=[TERM, LAW], shown_clauses=shown),
    }


def _case(**kwargs):
    return build_review_case(
        _state(**kwargs),
        permission_context=PC,
        document_id="doc",
        trace=[RetrievalTraceStep(stage="hybrid_retrieve", clause_ids=[TERM.clause_id])],
    )


def test_only_outcomes_with_evidence_are_queued():
    assert should_queue(_state())
    assert should_queue(_state(status="flagged_for_review"))
    assert not should_queue(_state(status="answered"))
    # Jailbreak / out-of-scope: flagged, but never retrieved anything.
    assert not should_queue({"final_status": "flagged_for_review", "flagged_reason": "Blocked at input"})


def test_evidence_packet_lists_validator_failures_first_with_injection_flags():
    case = _case(injected=True)
    assert [d.is_valid for d in case.draft_claims] == [False, True]
    assert case.draft_claims[0].failure_reason
    assert case.shown_clauses[0].injection_suspected is True
    assert case.trace[0].stage == "hybrid_retrieve"
    assert case.status == ReviewStatus.OPEN


def test_every_draft_claim_needs_its_own_decision():
    case = _case()
    with pytest.raises(ReviewDecisionError, match="missing"):
        apply_decision(
            case,
            DecisionRequest(reviewer="r", decisions=[ClaimDecision(index=1, action=ClaimAction.APPROVE)]),
            all_clauses_by_id=ALL,
        )


def test_approving_a_failing_claim_requires_an_audited_override():
    case = _case()
    bad_index = case.draft_claims[0].index
    good_index = case.draft_claims[1].index
    decisions = [
        ClaimDecision(index=bad_index, action=ClaimAction.APPROVE),
        ClaimDecision(index=good_index, action=ClaimAction.APPROVE),
    ]
    with pytest.raises(ReviewDecisionError, match="override_reason"):
        apply_decision(case, DecisionRequest(reviewer="r", decisions=decisions), all_clauses_by_id=ALL)

    decisions[0].override_reason = "Confirmed against the signed amendment on file."
    audit = InMemoryAuditLogger()
    apply_decision(case, DecisionRequest(reviewer="jane", decisions=decisions), all_clauses_by_id=ALL, audit_logger=audit)

    assert case.status == ReviewStatus.RESOLVED
    assert [e.event_type for e in audit.events] == ["review_decision", "review_override"]
    assert audit.events[1].details["override_reason"].startswith("Confirmed")
    assert audit.events[1].details["reviewer"] == "jane"


def test_edit_is_revalidated_and_only_accepted_claims_are_released():
    case = _case()
    bad_index = case.draft_claims[0].index
    good_index = case.draft_claims[1].index
    fixed = ClaimDecision(
        index=bad_index,
        action=ClaimAction.EDIT,
        claim_text="The Agreement is governed by Delaware law.",
        citation=LAW.clause_id,  # not shown to the model; the reviewer has the whole contract
        supporting_quote="governed by the laws of the State of Delaware",
    )
    apply_decision(
        case,
        DecisionRequest(
            reviewer="r", decisions=[fixed, ClaimDecision(index=good_index, action=ClaimAction.REJECT)]
        ),
        all_clauses_by_id=ALL,
    )
    outcome = case.outcome
    assert outcome.final_status == "answered_after_review"
    assert outcome.answer_text == "The Agreement is governed by Delaware law."
    assert outcome.claims[0].validator_passed and outcome.claims[0].resolved_clause_id == LAW.clause_id


def test_edit_with_a_non_verbatim_quote_is_refused():
    case = _case(citations=(GOOD,))
    edit = ClaimDecision(
        index=0,
        action=ClaimAction.EDIT,
        claim_text="Termination needs sixty days notice.",
        citation=TERM.clause_id,
        supporting_quote="terminate upon sixty (60) days notice",
    )
    with pytest.raises(ReviewDecisionError, match="fails citation validation"):
        apply_decision(case, DecisionRequest(reviewer="r", decisions=[edit]), all_clauses_by_id=ALL)


def test_rejecting_everything_releases_nothing_and_case_cannot_be_redecided():
    case = _case(status="flagged_for_review", citations=())
    apply_decision(case, DecisionRequest(reviewer="r", decisions=[]), all_clauses_by_id=ALL)
    assert case.outcome.final_status == "rejected_after_review"
    assert case.outcome.answer_text is None
    with pytest.raises(ReviewConflictError):
        apply_decision(case, DecisionRequest(reviewer="r", decisions=[]), all_clauses_by_id=ALL)


def test_time_to_decision_is_measured_from_first_view():
    case = _case(citations=(GOOD,))
    assert mark_viewed(case) is True
    assert mark_viewed(case) is False  # first view only
    apply_decision(
        case,
        DecisionRequest(reviewer="r", decisions=[ClaimDecision(index=0, action=ClaimAction.APPROVE)]),
        all_clauses_by_id=ALL,
    )
    assert case.outcome.seconds_to_decision is not None and case.outcome.seconds_to_decision >= 0


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_store_is_scoped_by_tenant_matter_and_status(store_kind, tmp_path):
    store = InMemoryReviewStore() if store_kind == "memory" else SQLiteReviewStore(tmp_path / "r.sqlite3")
    case = _case()
    store.add(case)
    other = _case()
    other.matter_id = "matter-2"
    store.add(other)

    assert [c.review_id for c in store.list(tenant_id="tenant-1", matter_id="matter-1")] == [case.review_id]
    assert store.list(tenant_id="tenant-1", matter_id="matter-1", status=ReviewStatus.RESOLVED) == []

    loaded = store.get(case.review_id)
    loaded.status = ReviewStatus.RESOLVED
    store.save(loaded)
    assert store.get(case.review_id).status == ReviewStatus.RESOLVED
    assert store.get("missing") is None

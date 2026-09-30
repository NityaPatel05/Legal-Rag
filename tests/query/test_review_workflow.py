"""Human review queue (architecture-review H1/H2): what gets queued, the
evidence packet, and that a reviewer can only release grounded or
explicitly-overridden claims."""

import sqlite3
import threading
from datetime import datetime, timezone

import pytest

from query.filters.audit_log import InMemoryAuditLogger
from query.review.schemas import ClaimAction, ClaimDecision, DecisionRequest, ReviewAlreadyDecided, ReviewStatus
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


# --- B16: decisions and first-view are conditional writes (no lost update) ---



def _store(kind, tmp_path):
    return InMemoryReviewStore() if kind == "memory" else SQLiteReviewStore(tmp_path / "r.sqlite3")


def _reject_all(case):
    return DecisionRequest(
        reviewer="r",
        decisions=[ClaimDecision(index=d.index, action=ClaimAction.REJECT) for d in case.draft_claims],
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_second_concurrent_decision_is_rejected_not_overwritten(store_kind, tmp_path):
    """Both reviewers loaded the case while it was OPEN (so the in-Python
    status check passes for both); only the first persisted decision wins."""
    store = _store(store_kind, tmp_path)
    case = _case()
    store.add(case)
    first, second = store.get(case.review_id), store.get(case.review_id)
    audit = InMemoryAuditLogger()

    apply_decision(first, _reject_all(first), all_clauses_by_id=ALL, store=store, audit_logger=audit)
    second_request = _reject_all(second)
    second_request.reviewer = "someone-else"
    with pytest.raises(ReviewAlreadyDecided):
        apply_decision(second, second_request, all_clauses_by_id=ALL, store=store, audit_logger=audit)

    stored = store.get(case.review_id)
    assert stored.status == ReviewStatus.RESOLVED
    assert stored.outcome.reviewer == "r"
    # The losing decision never took effect, so it isn't audited either.
    assert [e.details["reviewer"] for e in audit.events if e.event_type == "review_decision"] == ["r"]


def test_already_decided_is_a_conflict_error_for_the_409_mapping():
    assert issubclass(ReviewAlreadyDecided, ReviewConflictError)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_save_of_a_stale_resolved_case_raises_already_decided(store_kind, tmp_path):
    """app/main.py's current path (apply_decision without store=, then
    save) is also protected: save routes to the conditional write."""
    store = _store(store_kind, tmp_path)
    case = _case()
    store.add(case)
    a, b = store.get(case.review_id), store.get(case.review_id)
    apply_decision(a, _reject_all(a), all_clauses_by_id=ALL)
    store.save(a)
    apply_decision(b, _reject_all(b), all_clauses_by_id=ALL)
    with pytest.raises(ReviewAlreadyDecided):
        store.save(b)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_racing_decisions_from_threads_exactly_one_wins(store_kind, tmp_path):
    store = _store(store_kind, tmp_path)
    case = _case()
    store.add(case)
    loaded = [store.get(case.review_id) for _ in range(8)]
    wins, losses = [], []
    barrier = threading.Barrier(len(loaded))

    def decide(c):
        barrier.wait()
        try:
            apply_decision(c, _reject_all(c), all_clauses_by_id=ALL, store=store)
            wins.append(c)
        except ReviewAlreadyDecided:
            losses.append(c)

    threads = [threading.Thread(target=decide, args=(c,)) for c in loaded]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1 and len(losses) == len(loaded) - 1


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_marking_viewed_never_reopens_a_decided_case(store_kind, tmp_path):
    """A GET that loaded the case while OPEN and saves its first view after
    a decision committed must not write the OPEN body back."""
    store = _store(store_kind, tmp_path)
    case = _case()
    store.add(case)
    viewer_copy = store.get(case.review_id)
    decider_copy = store.get(case.review_id)
    apply_decision(decider_copy, _reject_all(decider_copy), all_clauses_by_id=ALL, store=store)

    assert mark_viewed(viewer_copy) is True  # in-memory copy still thinks OPEN
    store.save(viewer_copy)

    stored = store.get(case.review_id)
    assert stored.status == ReviewStatus.RESOLVED
    assert stored.outcome is not None
    assert stored.first_viewed_at is None


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_first_view_is_recorded_once(store_kind, tmp_path):
    store = _store(store_kind, tmp_path)
    case = _case()
    store.add(case)
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert store.mark_viewed(case.review_id, t1) is True
    assert store.mark_viewed(case.review_id, t2) is False
    assert store.get(case.review_id).first_viewed_at == t1
    assert store.mark_viewed("missing", t1) is False

    # Via the workflow helper: a second viewer's copy loses the conditional write.
    other = _case()
    store.add(other)
    a, b = store.get(other.review_id), store.get(other.review_id)
    assert mark_viewed(a, store=store) is True
    assert mark_viewed(b, store=store) is False
    assert store.get(other.review_id).first_viewed_at == a.first_viewed_at


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_record_decision_on_missing_case_is_lookup_error(store_kind, tmp_path):
    store = _store(store_kind, tmp_path)
    case = _case()
    case.status = ReviewStatus.RESOLVED
    with pytest.raises(LookupError):
        store.record_decision(case)


# --- L2: list every status ---


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_list_all_statuses(store_kind, tmp_path):
    store = _store(store_kind, tmp_path)
    open_case, decided = _case(), _case()
    store.add(open_case)
    store.add(decided)
    loaded = store.get(decided.review_id)
    apply_decision(loaded, _reject_all(loaded), all_clauses_by_id=ALL, store=store)

    def ids(status):
        return {c.review_id for c in store.list(tenant_id="tenant-1", matter_id="matter-1", status=status)}

    assert ids("all") == ids(None) == {open_case.review_id, decided.review_id}
    assert ids(ReviewStatus.OPEN) == {open_case.review_id}
    assert ids(ReviewStatus.RESOLVED) == {decided.review_id}


# --- L3: SQLite connections are closed ---


def test_sqlite_store_closes_every_connection(tmp_path, monkeypatch):
    opened: list = []
    real_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def close(self):
            self.was_closed = True
            super().close()

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, factory=TrackingConnection, **kwargs)
        conn.was_closed = False
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    store = SQLiteReviewStore(tmp_path / "r.sqlite3")
    case = _case()
    store.add(case)
    store.get(case.review_id)
    store.list(tenant_id="tenant-1", matter_id="matter-1")
    store.mark_viewed(case.review_id, datetime.now(timezone.utc))
    loaded = store.get(case.review_id)
    apply_decision(loaded, _reject_all(loaded), all_clauses_by_id=ALL, store=store)
    with pytest.raises(ReviewAlreadyDecided):
        store.record_decision(loaded)

    assert opened and all(c.was_closed for c in opened)

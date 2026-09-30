"""B23: request-body size limits and identifier patterns (app/schemas.py,
query/review/schemas.py). Pure model validation -- no app or services."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    MAX_QUERY_CHARS,
    MAX_SOURCE_TEXT_CHARS,
    IngestTextRequest,
    QueryRequest,
    ReviewDecisionBody,
)
from query.review.schemas import MAX_CLAIM_CHARS, MAX_REASON_CHARS, ClaimDecision, DecisionRequest

IDS = {"document_id": "msa-1", "tenant_id": "t-1", "matter_id": "m-1"}


def test_realistic_requests_pass():
    QueryRequest(query_text="What law governs this agreement?", **IDS)
    IngestTextRequest(source_text="1. Term. This Agreement ...", **IDS)
    QueryRequest(query_text="q" * MAX_QUERY_CHARS, document_id="Acme_MSA.v2-2024", tenant_id="t", matter_id="m")


def test_query_text_over_limit_is_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(query_text="q" * (MAX_QUERY_CHARS + 1), **IDS)


def test_source_text_over_limit_is_rejected():
    with pytest.raises(ValidationError):
        IngestTextRequest(source_text="x" * (MAX_SOURCE_TEXT_CHARS + 1), **IDS)


@pytest.mark.parametrize("field", ["document_id", "tenant_id", "matter_id"])
@pytest.mark.parametrize("bad", ["", "doc::clause-1", "a b", "../etc", "x" * 129, "doc\n1", "<script>"])
def test_identifiers_must_match_the_pattern(field, bad):
    with pytest.raises(ValidationError):
        QueryRequest(query_text="q", **{**IDS, field: bad})
    with pytest.raises(ValidationError):
        IngestTextRequest(source_text="t", **{**IDS, field: bad})


def test_filter_values_are_bounded():
    with pytest.raises(ValidationError):
        QueryRequest(query_text="q", governing_law="x" * 201, **IDS)


def test_review_decision_limits():
    ok = {"reviewer": "alice", "tenant_id": "t-1", "matter_id": "m-1"}
    ReviewDecisionBody(decisions=[ClaimDecision(index=0, action="approve")], **ok)
    with pytest.raises(ValidationError):
        ReviewDecisionBody(decisions=[], **{**ok, "matter_id": "m::1"})
    with pytest.raises(ValidationError):
        ClaimDecision(action="edit", claim_text="c" * (MAX_CLAIM_CHARS + 1))
    with pytest.raises(ValidationError):
        ClaimDecision(action="approve", override_reason="r" * (MAX_REASON_CHARS + 1))
    with pytest.raises(ValidationError):
        DecisionRequest(reviewer="r" * 201, decisions=[])

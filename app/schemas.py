from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from ingestion.schemas import PrivilegeLevel
from query.review.schemas import DecisionRequest, ReviewStatus


class IngestTextRequest(BaseModel):
    document_id: str
    source_text: str
    tenant_id: str
    matter_id: str
    privilege_level: PrivilegeLevel = PrivilegeLevel.CONFIDENTIAL


class IngestResponse(BaseModel):
    document_id: str
    num_clauses: int
    num_defined_terms: int
    num_references: int


class QueryRequest(BaseModel):
    document_id: str
    query_text: str
    tenant_id: str
    matter_id: str
    jurisdiction: str | None = None
    governing_law: str | None = None


class CitationOut(BaseModel):
    claim_text: str
    # The citation key the model used (a clause_id, or a unique section id).
    section_id: str
    # Resolved, validated clause -- the stable identifier to link to.
    clause_id: str | None = None
    supporting_quote: str | None = None


class QueryResponse(BaseModel):
    # Identifies this query in the audit log and, if queued, the review.
    query_id: str | None = None
    query_text: str
    route: str
    final_status: str
    answer_text: str | None = None
    citations: list[CitationOut] = []
    flagged_reason: str | None = None
    disclaimer: str | None = None
    # Set when the outcome was queued for human review: GET /reviews/{id}
    # returns the reviewer's released answer once the case is resolved.
    review_id: str | None = None


class ReviewSummary(BaseModel):
    review_id: str
    document_id: str
    status: ReviewStatus
    created_at: datetime
    reason: str
    question: str
    failing_claims: int
    total_claims: int
    injection_flagged_clauses: int


class ReviewDecisionBody(DecisionRequest):
    tenant_id: str
    matter_id: str

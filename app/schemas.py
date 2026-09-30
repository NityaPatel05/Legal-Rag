"""HTTP request/response models.

Request bodies carry hard size limits (validated before any work happens,
so an oversize body is a 422, not an OOM, a 10-minute ingest or a huge LLM
bill):

- `query_text` <= 4,000 chars: a question, not a document. Every query costs
  several LLM calls (router, jailbreak check, generation, CRAG retries), each
  of which re-sends it.
- `source_text` <= 2,000,000 chars: ~1,000 pages; far beyond the largest CUAD
  contract (~48k tokens, ~200k chars), with room for long credit agreements.
- identifiers (`document_id`, `tenant_id`, `matter_id`) match
  `^[A-Za-z0-9._-]{1,128}$`. Beyond size, this keeps ids from colliding with
  the "::" separator inside clause ids (`<document_id>::clause-NNNN`,
  indexer.document_id_from_clause_id) and out of log/HTML injection.

Review-decision limits live on `query.review.schemas.DecisionRequest`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

from ingestion.schemas import PrivilegeLevel
from query.review.schemas import ID_PATTERN, DecisionRequest, ReviewStatus

MAX_QUERY_CHARS = 4_000
MAX_SOURCE_TEXT_CHARS = 2_000_000
MAX_FILTER_VALUE_CHARS = 200

Identifier = Annotated[str, Field(pattern=ID_PATTERN)]


class IngestTextRequest(BaseModel):
    document_id: Identifier
    source_text: str = Field(max_length=MAX_SOURCE_TEXT_CHARS)
    tenant_id: Identifier
    matter_id: Identifier
    privilege_level: PrivilegeLevel = PrivilegeLevel.CONFIDENTIAL


class IngestResponse(BaseModel):
    document_id: str
    num_clauses: int
    num_defined_terms: int
    num_references: int


class QueryRequest(BaseModel):
    document_id: Identifier
    query_text: str = Field(max_length=MAX_QUERY_CHARS)
    tenant_id: Identifier
    matter_id: Identifier
    jurisdiction: str | None = Field(default=None, max_length=MAX_FILTER_VALUE_CHARS)
    governing_law: str | None = Field(default=None, max_length=MAX_FILTER_VALUE_CHARS)


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
    # comparison / contradiction_check only: "conflict" | "no_conflict" |
    # "differ" | "equivalent", from a closed set, and only when the kept
    # claims ground at least two different clauses (crag_graph, B10).
    verdict: str | None = None
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
    tenant_id: Identifier
    matter_id: Identifier

"""HTTP request/response models.

Request bodies carry hard size limits (validated before any work happens,
so an oversize body is a 422, not an OOM, a 10-minute ingest or a huge LLM
bill):

- `query_text` <= 4,000 chars: a question, not a document. Every query costs
  several LLM calls (router, jailbreak check, generation, CRAG retries), each
  of which re-sends it.
- `source_text` <= 2,000,000 chars: ~1,000 pages; far beyond the largest CUAD
  contract (~48k tokens, ~200k chars), with room for long credit agreements.
- uploaded files <= 25 MB (`MAX_UPLOAD_BYTES`): a larger file is a 413
  before any parsing or LLM work. Starlette has already spooled the body to
  disk by then, so a hard cap on bytes received belongs in the reverse
  proxy (`client_max_body_size`). Extracted text is held to 2,000,000 chars.
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

from ingestion.schemas import DefinedTerm, PrivilegeLevel
from query.review.schemas import (
    ID_PATTERN,
    MAX_DECISIONS,
    ClaimDecision,
    DecisionRequest,
    ReviewOutcome,
    ReviewStatus,
)

MAX_QUERY_CHARS = 4_000
MAX_SOURCE_TEXT_CHARS = 2_000_000
MAX_FILTER_VALUE_CHARS = 200
# Uploaded files (POST /documents/file). A scanned 1,000-page PDF can exceed
# this; the extracted text is still held to MAX_SOURCE_TEXT_CHARS.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

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
    # Set when the outcome was queued for human review: GET
    # /reviews/{id}/status returns the reviewer's released answer once the
    # case is resolved.
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


class ReviewCheckBody(BaseModel):
    tenant_id: Identifier
    matter_id: Identifier
    decisions: list[ClaimDecision] = Field(max_length=MAX_DECISIONS)


class ReviewStatusOut(BaseModel):
    """What the asker polls after a queued query. Deliberately not the
    ReviewCase: reading it doesn't start the reviewer's time-to-decision
    clock (mark_viewed), and the evidence packet isn't the asker's concern."""

    review_id: str
    status: ReviewStatus
    outcome: ReviewOutcome | None = None


class DocumentSummary(BaseModel):
    """Document-level fields are the value most of the document's clauses
    carry: they're merged onto every clause from the document profile, and
    a clause-level value can differ (CLAUDE.md #5)."""

    document_id: str
    num_clauses: int
    privilege_level: PrivilegeLevel | None = None
    contract_type: str | None = None
    governing_law: str | None = None
    jurisdiction: str | None = None
    parties: list[str] = []
    injection_flagged_clauses: int = 0


class ClauseOut(BaseModel):
    clause_id: str
    section_id: str | None = None
    heading: str | None = None
    text: str
    start_offset: int
    end_offset: int
    parent_clause_id: str | None = None
    clause_type: str | None = None
    injection_suspected: bool = False
    injection_markers: list[str] = []


class DocumentDetail(DocumentSummary):
    # Document order (start_offset), as stored: the text citations are
    # validated against.
    clauses: list[ClauseOut]
    defined_terms: list[DefinedTerm] = []

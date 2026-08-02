from __future__ import annotations

from pydantic import BaseModel

from ingestion.schemas import PrivilegeLevel


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
    section_id: str


class QueryResponse(BaseModel):
    query_text: str
    route: str
    final_status: str
    answer_text: str | None = None
    citations: list[CitationOut] = []
    flagged_reason: str | None = None
    disclaimer: str | None = None

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class PrivilegeLevel(str, Enum):
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    PRIVILEGED = "privileged"


class Clause(BaseModel):
    """A single clause-level chunk. Boundaries come from structure detection only
    (see ingestion/parsing/clause_parser.py) — never truncated by a token/char
    budget, per CLAUDE.md #1."""

    clause_id: str
    section_id: str | None = None
    heading: str | None = None
    text: str
    start_offset: int
    end_offset: int
    # Set only on passages produced by split_oversize_clauses: the clause this
    # passage was cut from. A passage never crosses its parent's boundaries.
    parent_clause_id: str | None = None


class DefinedTerm(BaseModel):
    term: str
    definition: str
    defined_in_clause_id: str


class SectionReference(BaseModel):
    source_clause_id: str
    referenced_section_id: str
    raw_text: str


class ClauseMetadata(BaseModel):
    """Required fields per CLAUDE.md #3. `jurisdiction` and `governing_law` are
    independent, separately-filterable fields per CLAUDE.md #2 — never merge or
    derive one from the other, even when the source clause states both in the
    same sentence.

    `tenant_id` / `matter_id` / `privilege_level` are supplied by the ingestion
    caller, not extracted by the LLM, despite being grouped with the LLM-derived
    fields in CLAUDE.md #3 — permission-first filtering is a security control
    (CLAUDE.md #7), so tenant/matter identity must come from the trusted upload
    context, never be inferred from document text an LLM could be manipulated by.
    """

    contract_type: str | None = None
    clause_type: str | None = None
    parties: list[str] = Field(default_factory=list)
    effective_date: date | None = None
    signature_date: date | None = None
    jurisdiction: str | None = None
    governing_law: str | None = None
    matter_id: str
    tenant_id: str
    privilege_level: PrivilegeLevel = PrivilegeLevel.CONFIDENTIAL
    contains_pii: bool = False
    supersession_links: list[str] = Field(default_factory=list)
    # Set deterministically at ingestion (metadata/injection_scan.py), never
    # by the LLM: clause text that reads like instructions to a model.
    injection_suspected: bool = False
    injection_markers: list[str] = Field(default_factory=list)


class TaggedClause(BaseModel):
    clause: Clause
    metadata: ClauseMetadata
    context_note: str | None = None
    embedding: list[float] | None = None


class ParsedDocument(BaseModel):
    document_id: str
    source_path: str | None = None
    raw_text: str
    clauses: list[Clause] = Field(default_factory=list)
    defined_terms: list[DefinedTerm] = Field(default_factory=list)
    references: list[SectionReference] = Field(default_factory=list)

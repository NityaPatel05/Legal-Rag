"""ingest_document orchestrates the offline ingestion pipeline — mirrors
architecture.md §3 A1-A7: load -> clause-level parse -> split oversize
clauses into passages -> defined terms + section-reference graph ->
document profile -> per-clause metadata tagging + contextual enrichment (one
LLM call per clause, R12) ->
embed -> delete old version -> index.

Every external integration is passed in as a dependency (DocumentLoader,
LLMClient, EmbeddingClient, VectorStore) rather than constructed here, so this
function is fully unit-testable with fakes — see tests/ingestion/test_pipeline.py.
Production wiring (Docling/Groq/BGE-M3/Qdrant) happens at the call site using
ingestion/config.py.

What leaves the boundary (roadmap R9, review finding S4). Tagging,
enrichment and the document profile send contract text to a third-party LLM:

- Structured PII (SSNs, account-number-shaped digit runs) is redacted from
  every prompt first (RedactingLLMClient). The stored/indexed clause text is
  NOT redacted -- citations are checked verbatim against it -- and
  contains_pii is set from the regex finding as well as the LLM's opinion,
  since the LLM never sees the redacted values.
- privilege_level=privileged text never goes to the default provider. It
  goes to `privileged_llm_client` (a self-hosted model) if one is
  configured; otherwise every LLM step is skipped and the document is
  indexed with structural metadata only (parse, defined terms, references,
  embedding from the self-hosted BGE-M3 server).

Every clause is scanned for instruction-like text (injection_scan.py, R10)
before any LLM sees it; the result is stored on the clause's metadata."""

from __future__ import annotations

from ingestion.embedding.context_enrichment import build_document_outline
from ingestion.embedding.embedder import EmbeddingClient
from ingestion.embedding.indexer import VectorStore
from ingestion.metadata.document_profile import (
    DocumentProfile,
    extract_document_profile,
    merge_document_profile,
)
from ingestion.metadata.injection_scan import scan_for_injection
from ingestion.metadata.llm_client import LLMClient
from ingestion.metadata.tagger import tag_and_enrich_clause
from ingestion.parsing.clause_parser import DEFAULT_MAX_PASSAGE_CHARS, parse_clauses, split_oversize_clauses
from ingestion.parsing.defined_terms import extract_defined_terms
from ingestion.parsing.document_loader import DocumentLoader
from ingestion.parsing.references import extract_references
from ingestion.schemas import ClauseMetadata, ParsedDocument, PrivilegeLevel, TaggedClause
from query.filters.pii_sanitizer import redact_structured_pii


class RedactingLLMClient:
    """Wraps an LLMClient so structured PII is redacted from every prompt
    before it leaves the process."""

    def __init__(self, inner: LLMClient):
        self._inner = inner

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        return self._inner.generate_json(system_prompt, redact_structured_pii(user_prompt).sanitized_text)


def select_llm_client(
    privilege_level: PrivilegeLevel,
    llm_client: LLMClient,
    privileged_llm_client: LLMClient | None,
) -> LLMClient | None:
    """The client allowed to see this document's text, or None if no LLM
    may see it (privileged, and no self-hosted model configured)."""
    if privilege_level == PrivilegeLevel.PRIVILEGED:
        return privileged_llm_client
    return llm_client


def embedding_text(tagged_clause: TaggedClause, document_summary: str | None) -> str:
    """What gets embedded: document summary + context note + heading + text.
    The summary is the Summary-Augmented Chunking prefix (see
    document_profile.py); the heading is included because a passage cut from
    the middle of a long clause otherwise loses it."""
    parts = [
        document_summary,
        tagged_clause.context_note,
        tagged_clause.clause.heading,
        tagged_clause.clause.text,
    ]
    return "\n\n".join(p for p in parts if p)


def ingest_document(
    *,
    document_id: str,
    source_path: str,
    tenant_id: str,
    matter_id: str,
    privilege_level: PrivilegeLevel = PrivilegeLevel.CONFIDENTIAL,
    document_loader: DocumentLoader,
    llm_client: LLMClient,
    embedding_client: EmbeddingClient,
    vector_store: VectorStore,
    max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
    privileged_llm_client: LLMClient | None = None,
) -> ParsedDocument:
    if "::" in document_id:
        # clause_id is f"{document_id}::clause-NNN" and document_id is
        # recovered by splitting on "::clause-" -- a "::" in the id breaks that.
        raise ValueError("document_id must not contain '::'")

    raw_text = document_loader.load(source_path)

    clauses = split_oversize_clauses(
        parse_clauses(raw_text, document_id=document_id), max_chars=max_passage_chars
    )
    defined_terms = extract_defined_terms(clauses)
    references = extract_references(clauses)

    allowed_llm = select_llm_client(privilege_level, llm_client, privileged_llm_client)
    llm = RedactingLLMClient(allowed_llm) if allowed_llm is not None else None

    profile = extract_document_profile(raw_text, llm_client=llm) if llm else DocumentProfile()
    outline = build_document_outline(clauses)

    tagged_clauses: list[TaggedClause] = []
    for clause in clauses:
        if llm:
            # One call per clause for metadata + context note (R12).
            metadata, context_note = tag_and_enrich_clause(
                clause,
                tenant_id=tenant_id,
                matter_id=matter_id,
                privilege_level=privilege_level,
                llm_client=llm,
                document_summary=profile.summary,
                document_outline=outline,
            )
        else:
            metadata = ClauseMetadata(tenant_id=tenant_id, matter_id=matter_id, privilege_level=privilege_level)
            context_note = None
        metadata = merge_document_profile(metadata, profile)
        injection_markers = scan_for_injection(clause.text)
        metadata = metadata.model_copy(
            update={
                "contains_pii": metadata.contains_pii or redact_structured_pii(clause.text).any_pii_found,
                "injection_suspected": bool(injection_markers),
                "injection_markers": injection_markers,
            }
        )
        tagged_clauses.append(
            TaggedClause(clause=clause, metadata=metadata, context_note=context_note)
        )

    texts_to_embed = [embedding_text(tc, profile.summary) for tc in tagged_clauses]
    embeddings = embedding_client.embed(texts_to_embed)
    if len(embeddings) != len(tagged_clauses):
        raise RuntimeError(
            f"embedding client returned {len(embeddings)} vectors for {len(tagged_clauses)} texts"
        )
    for tagged_clause, embedding in zip(tagged_clauses, embeddings):
        tagged_clause.embedding = embedding

    # Everything above can fail (LLM/embedding calls); only touch the index
    # once the new version is fully built, so a failed re-ingest leaves the
    # previous version in place rather than a half-deleted document.
    vector_store.delete_document(document_id=document_id, tenant_id=tenant_id, matter_id=matter_id)
    vector_store.upsert(tagged_clauses)
    if tagged_clauses:
        vector_dim = len(tagged_clauses[0].embedding)
        if references:
            vector_store.upsert_references(
                document_id=document_id,
                tenant_id=tenant_id,
                matter_id=matter_id,
                references=references,
                vector_dim=vector_dim,
            )
        if defined_terms:
            vector_store.upsert_defined_terms(
                document_id=document_id,
                tenant_id=tenant_id,
                matter_id=matter_id,
                defined_terms=defined_terms,
                vector_dim=vector_dim,
            )

    return ParsedDocument(
        document_id=document_id,
        source_path=source_path,
        raw_text=raw_text,
        clauses=clauses,
        defined_terms=defined_terms,
        references=references,
    )

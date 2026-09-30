"""ingest_document orchestrates the offline ingestion pipeline — mirrors
architecture.md §3 A1-A7: load -> clause-level parse -> split oversize
clauses into passages -> defined terms + section-reference graph ->
injection scan -> document profile -> batched metadata tagging + contextual
enrichment -> embed -> write new version -> delete stale versions.

Every external integration is passed in as a dependency (DocumentLoader,
LLMClient, EmbeddingClient, VectorStore) rather than constructed here, so this
function is fully unit-testable with fakes — see tests/ingestion/test_pipeline.py.
Production wiring (Docling/Groq/BGE-M3/Qdrant) happens at the call site using
ingestion/config.py.

LLM call budget. One document-profile call, then one call per batch of
`batch_size` passages (tagger.tag_and_enrich_clauses) -- 1 + ceil(n/8)
instead of 1 + n. An unchanged document (same content hash, privilege level
and pipeline version as the indexed one) is not re-ingested at all: zero LLM
and embedding calls. An empty document is refused before any call.

What leaves the boundary (roadmap R9, review finding S4). Tagging,
enrichment and the document profile send contract text to a third-party LLM:

- Structured PII (SSNs, account-number-shaped digit runs) is redacted from
  every prompt to the default provider first (RedactingLLMClient). The
  stored/indexed clause text is NOT redacted -- citations are checked
  verbatim against it -- and contains_pii is set from the regex finding as
  well as the LLM's opinion, since the LLM never sees the redacted values.
- privilege_level=privileged text never goes to the default provider. It
  goes to `privileged_llm_client` (a self-hosted model) if one is
  configured -- unredacted, since it never leaves the boundary and a
  redacted digit run would make every quote of it fail validation (L5);
  otherwise every LLM step is skipped and the document is indexed with
  structural metadata only.
- A re-ingest may not lower a document's privilege level (S3): that would
  send privileged text to the third-party provider.

Every clause is scanned for instruction-like text (injection_scan.py, R10)
BEFORE any LLM sees it. Flagged clauses are tagged in a batch of their own,
so injected text can't steer the metadata of the clauses batched with it."""

from __future__ import annotations

import hashlib

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
from ingestion.metadata.tagger import DEFAULT_BATCH_SIZE, tag_and_enrich_clauses
from ingestion.parsing.clause_parser import DEFAULT_MAX_PASSAGE_CHARS, parse_clauses, split_oversize_clauses
from ingestion.parsing.defined_terms import extract_defined_terms
from ingestion.parsing.document_loader import DocumentLoader
from ingestion.parsing.references import extract_references
from ingestion.parsing.text_normalize import normalize_source_text
from ingestion.schemas import Clause, ClauseMetadata, ParsedDocument, PrivilegeLevel, TaggedClause
from query.filters.pii_sanitizer import redact_structured_pii

# Bump when a prompt, parser or metadata rule changes what an ingest
# produces, so unchanged documents are re-processed instead of skipped.
PIPELINE_VERSION = "2026-09-30.1"


class IngestionRefused(ValueError):
    """The request can't be ingested as given (maps to HTTP 422/409)."""


class EmptyDocumentError(IngestionRefused):
    pass


class PrivilegeDowngradeError(IngestionRefused):
    pass


class RedactingLLMClient:
    """Wraps an LLMClient so structured PII is redacted from every prompt
    before it leaves the process."""

    def __init__(self, inner: LLMClient):
        self._inner = inner

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        return self._inner.generate_json(system_prompt, redact_structured_pii(user_prompt).sanitized_text)


def redacting(client: LLMClient) -> LLMClient:
    """RedactingLLMClient(client), without wrapping twice."""
    return client if isinstance(client, RedactingLLMClient) else RedactingLLMClient(client)


def select_llm_client(
    privilege_level: PrivilegeLevel,
    llm_client: LLMClient | None,
    privileged_llm_client: LLMClient | None,
) -> LLMClient | None:
    """The client allowed to see this document's text, or None if no LLM
    may see it (privileged, and no self-hosted model configured). The
    default provider's client is always PII-redacting; the self-hosted one
    is not (see module docstring)."""
    if privilege_level == PrivilegeLevel.PRIVILEGED:
        return privileged_llm_client
    return redacting(llm_client) if llm_client is not None else None


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


def compute_ingest_version(raw_text: str, privilege_level: PrivilegeLevel, max_passage_chars: int) -> str:
    h = hashlib.sha256()
    for part in (PIPELINE_VERSION, privilege_level.value, str(max_passage_chars), raw_text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def _privilege_rank(level) -> int:
    order = [p.value for p in PrivilegeLevel]
    value = level.value if isinstance(level, PrivilegeLevel) else level
    return order.index(value) if value in order else len(order)


def _tag(
    clauses: list[Clause],
    *,
    llm: LLMClient | None,
    tenant_id: str,
    matter_id: str,
    privilege_level: PrivilegeLevel,
    profile: DocumentProfile,
    outline: str,
    flagged_ids: set[str],
    batch_size: int,
) -> dict[str, tuple[ClauseMetadata, str | None]]:
    if llm is None:
        empty = ClauseMetadata(tenant_id=tenant_id, matter_id=matter_id, privilege_level=privilege_level)
        return {c.clause_id: (empty, None) for c in clauses}
    common = dict(
        tenant_id=tenant_id,
        matter_id=matter_id,
        privilege_level=privilege_level,
        llm_client=llm,
        document_summary=profile.summary,
        document_outline=outline,
    )
    clean = [c for c in clauses if c.clause_id not in flagged_ids]
    flagged = [c for c in clauses if c.clause_id in flagged_ids]
    out: dict[str, tuple[ClauseMetadata, str | None]] = {}
    for clause, result in zip(clean, tag_and_enrich_clauses(clean, batch_size=batch_size, **common)):
        out[clause.clause_id] = result
    for clause, result in zip(flagged, tag_and_enrich_clauses(flagged, batch_size=1, **common)):
        out[clause.clause_id] = result
    return out


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
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> ParsedDocument:
    if "::" in document_id:
        # clause_id is f"{document_id}::clause-NNN" and document_id is
        # recovered by splitting on "::clause-" -- a "::" in the id breaks that.
        raise IngestionRefused("document_id must not contain '::'")

    # CRLF, NBSP, form feeds and zero-width characters broke heading
    # detection (review finding B17). The NORMALIZED text is what gets
    # stored and cited, so quotes are always checked against the same string
    # the model was shown.
    raw_text = normalize_source_text(document_loader.load(source_path) or "")
    if not raw_text.strip():
        raise EmptyDocumentError("document has no text")

    clauses = split_oversize_clauses(
        parse_clauses(raw_text, document_id=document_id), max_chars=max_passage_chars
    )
    if not clauses:
        raise EmptyDocumentError("document parsed to zero clauses")
    defined_terms = extract_defined_terms(clauses)
    references = extract_references(clauses)
    parsed = ParsedDocument(
        document_id=document_id,
        source_path=source_path,
        raw_text=raw_text,
        clauses=clauses,
        defined_terms=defined_terms,
        references=references,
    )

    ingest_version = compute_ingest_version(raw_text, privilege_level, max_passage_chars)
    indexed = vector_store.document_info(document_id=document_id, tenant_id=tenant_id, matter_id=matter_id)
    if indexed and _privilege_rank(indexed.get("privilege_level")) > _privilege_rank(privilege_level):
        raise PrivilegeDowngradeError(
            f"document is indexed as {indexed.get('privilege_level')}; re-ingest may not lower its privilege level"
        )
    if indexed and indexed.get("ingest_version") == ingest_version and not force:
        return parsed  # unchanged: nothing to call, embed or write

    llm = select_llm_client(privilege_level, llm_client, privileged_llm_client)

    # Scan before any LLM sees the text (S4).
    injection_markers = {c.clause_id: scan_for_injection(c.text) for c in clauses}
    flagged_ids = {cid for cid, markers in injection_markers.items() if markers}

    profile = extract_document_profile(raw_text, llm_client=llm) if llm else DocumentProfile()
    outline = build_document_outline(clauses)
    tag_results = _tag(
        clauses,
        llm=llm,
        tenant_id=tenant_id,
        matter_id=matter_id,
        privilege_level=privilege_level,
        profile=profile,
        outline=outline,
        flagged_ids=flagged_ids,
        batch_size=batch_size,
    )

    tagged_clauses: list[TaggedClause] = []
    for clause in clauses:
        metadata, context_note = tag_results[clause.clause_id]
        metadata = merge_document_profile(metadata, profile)
        metadata = metadata.model_copy(
            update={
                "contains_pii": metadata.contains_pii or redact_structured_pii(clause.text).any_pii_found,
                "injection_suspected": bool(injection_markers[clause.clause_id]),
                "injection_markers": injection_markers[clause.clause_id],
                "ingest_version": ingest_version,
            }
        )
        tagged_clauses.append(TaggedClause(clause=clause, metadata=metadata, context_note=context_note))

    texts_to_embed = [embedding_text(tc, profile.summary) for tc in tagged_clauses]
    embeddings = embedding_client.embed(texts_to_embed)
    if len(embeddings) != len(tagged_clauses):
        raise RuntimeError(
            f"embedding client returned {len(embeddings)} vectors for {len(tagged_clauses)} texts"
        )
    for tagged_clause, embedding in zip(tagged_clauses, embeddings):
        tagged_clause.embedding = embedding

    # Write the new version first, then delete every point of this document
    # carrying another version (B15): a failure in between leaves the
    # document queryable (old tail + new version) instead of half-deleted.
    vector_store.upsert(tagged_clauses)
    vector_dim = len(tagged_clauses[0].embedding)
    scope = dict(document_id=document_id, tenant_id=tenant_id, matter_id=matter_id)
    if references:
        vector_store.upsert_references(
            **scope, references=references, vector_dim=vector_dim, ingest_version=ingest_version
        )
    if defined_terms:
        vector_store.upsert_defined_terms(
            **scope, defined_terms=defined_terms, vector_dim=vector_dim, ingest_version=ingest_version
        )
    vector_store.delete_stale_versions(**scope, keep_version=ingest_version)

    return parsed

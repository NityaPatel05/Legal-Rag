"""ingest_document orchestrates the offline ingestion pipeline — mirrors
architecture.md §3 A1-A7: load -> clause-level parse -> defined terms +
section-reference graph -> metadata tagging -> contextual enrichment ->
embed -> index.

Every external integration is passed in as a dependency (DocumentLoader,
LLMClient, EmbeddingClient, VectorStore) rather than constructed here, so this
function is fully unit-testable with fakes — see tests/ingestion/test_pipeline.py.
Production wiring (Docling/Groq/BGE-M3/Qdrant) happens at the call site using
ingestion/config.py."""

from __future__ import annotations

from ingestion.embedding.context_enrichment import enrich_clause
from ingestion.embedding.embedder import EmbeddingClient
from ingestion.embedding.indexer import VectorStore
from ingestion.metadata.llm_client import LLMClient
from ingestion.metadata.tagger import tag_clause
from ingestion.parsing.clause_parser import parse_clauses
from ingestion.parsing.defined_terms import extract_defined_terms
from ingestion.parsing.document_loader import DocumentLoader
from ingestion.parsing.references import extract_references
from ingestion.schemas import ParsedDocument, PrivilegeLevel, TaggedClause


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
) -> ParsedDocument:
    raw_text = document_loader.load(source_path)

    clauses = parse_clauses(raw_text, document_id=document_id)
    defined_terms = extract_defined_terms(clauses)
    references = extract_references(clauses)

    tagged_clauses: list[TaggedClause] = []
    for clause in clauses:
        metadata = tag_clause(
            clause,
            tenant_id=tenant_id,
            matter_id=matter_id,
            privilege_level=privilege_level,
            llm_client=llm_client,
        )
        context_note = enrich_clause(clause, llm_client=llm_client)
        tagged_clauses.append(
            TaggedClause(clause=clause, metadata=metadata, context_note=context_note)
        )

    texts_to_embed = [
        f"{tc.context_note}\n\n{tc.clause.text}" if tc.context_note else tc.clause.text
        for tc in tagged_clauses
    ]
    embeddings = embedding_client.embed(texts_to_embed)
    for tagged_clause, embedding in zip(tagged_clauses, embeddings):
        tagged_clause.embedding = embedding

    vector_store.upsert(tagged_clauses)
    if references and tagged_clauses:
        vector_store.upsert_references(
            document_id=document_id,
            tenant_id=tenant_id,
            matter_id=matter_id,
            references=references,
            vector_dim=len(tagged_clauses[0].embedding),
        )

    return ParsedDocument(
        document_id=document_id,
        source_path=source_path,
        raw_text=raw_text,
        clauses=clauses,
        defined_terms=defined_terms,
        references=references,
    )

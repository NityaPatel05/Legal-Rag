"""FastAPI app wiring the ingestion/ and query/ library code into an
actually runnable service — see README.md's deployment section and
docker-compose.yml for how this fits together. Deliberately thin: route
handlers translate HTTP requests into calls against
ingestion.pipeline.ingest_document and query.pipeline.process_query /
query.generation.crag_graph.run_crag, using the real adapters wired in
app/dependencies.py.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from app.dependencies import Dependencies, get_dependencies
from app.schemas import CitationOut, IngestResponse, IngestTextRequest, QueryRequest, QueryResponse
from ingestion.pipeline import ingest_document
from query.generation.crag_graph import run_crag
from query.pipeline import process_query
from query.retrieval.bm25_search import InMemoryBM25Index
from query.schemas import PermissionContext, QueryFilters

# BGE-M3 dense output is 1024-dim -- see embedding_server/main.py.
_EMBEDDING_DIM = 1024
_STARTUP_RETRY_ATTEMPTS = 10
_STARTUP_RETRY_DELAY_SECONDS = 3.0


def _ensure_qdrant_collection() -> None:
    """Qdrant requires a collection to exist before anything is upserted
    into it -- nothing else in this codebase creates one, so it happens
    once here at startup rather than being an undocumented manual step.

    Retries with a short delay: in docker-compose, the app container can
    start before Qdrant is actually accepting connections even with
    depends_on ordering, and this sidesteps depending on a compose
    healthcheck being exactly right for whatever qdrant/qdrant image
    version is in use."""
    from qdrant_client.models import Distance, VectorParams  # lazy import

    deps = get_dependencies()
    collection_name = deps.settings.qdrant_collection_name

    last_error: Exception | None = None
    for attempt in range(_STARTUP_RETRY_ATTEMPTS):
        try:
            client = deps.indexer._client_instance()
            if not client.collection_exists(collection_name):
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=_EMBEDDING_DIM, distance=Distance.COSINE),
                )
            return
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
            # connection/transport error from qdrant-client during a cold
            # start should be retried the same way, not special-cased.
            last_error = exc
            if attempt < _STARTUP_RETRY_ATTEMPTS - 1:
                time.sleep(_STARTUP_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Could not reach Qdrant at {deps.settings.qdrant_url} after "
        f"{_STARTUP_RETRY_ATTEMPTS} attempts"
    ) from last_error


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_qdrant_collection()
    yield


app = FastAPI(
    title="Legal Contract Clause Analysis RAG",
    description="See CLAUDE.md / docs/architecture.md in the repo for the full design.",
    version="0.1.0",
    lifespan=lifespan,
)


class _InMemoryTextLoader:
    """DocumentLoader that returns fixed text regardless of source_path --
    lets /documents/text skip writing a temp file for the common case of
    already-extracted contract text."""

    def __init__(self, text: str):
        self._text = text

    def load(self, source_path: str) -> str:
        return self._text


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents/text", response_model=IngestResponse)
def ingest_text(
    request: IngestTextRequest, deps: Dependencies = Depends(get_dependencies)
) -> IngestResponse:
    parsed = ingest_document(
        document_id=request.document_id,
        source_path="<inline>",
        tenant_id=request.tenant_id,
        matter_id=request.matter_id,
        privilege_level=request.privilege_level,
        document_loader=_InMemoryTextLoader(request.source_text),
        llm_client=deps.llm_client,
        embedding_client=deps.embedding_client,
        vector_store=deps.indexer,
    )
    return IngestResponse(
        document_id=parsed.document_id,
        num_clauses=len(parsed.clauses),
        num_defined_terms=len(parsed.defined_terms),
        num_references=len(parsed.references),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, deps: Dependencies = Depends(get_dependencies)) -> QueryResponse:
    permission_context = PermissionContext(tenant_id=request.tenant_id, matter_id=request.matter_id)
    query_filters = QueryFilters(jurisdiction=request.jurisdiction, governing_law=request.governing_law)

    document_context = deps.document_reader.load_document(
        request.document_id, tenant_id=request.tenant_id, matter_id=request.matter_id
    )
    if not document_context.clauses:
        raise HTTPException(status_code=404, detail="Document not found for this tenant/matter")

    bm25_index = InMemoryBM25Index(document_context.bm25_corpus())

    query_result = process_query(
        request.query_text,
        permission_context=permission_context,
        query_filters=query_filters,
        llm_client=deps.llm_client,
        vector_client=deps.vector_client,
        bm25_index=bm25_index,
        reranker=deps.reranker,
        all_clauses_by_section_id=document_context.by_section_id,
        references=document_context.references,
        audit_logger=deps.audit_logger,
        trace_exporter=deps.trace_exporter,
    )

    final_state = run_crag(
        query_result,
        source_clauses=document_context.clauses,
        all_clauses_by_section_id=document_context.by_section_id,
        references=document_context.references,
        llm_client=deps.llm_client,
        vector_client=deps.vector_client,
        bm25_index=bm25_index,
        reranker=deps.reranker,
        permission_context=permission_context,
        query_filters=query_filters,
        audit_logger=deps.audit_logger,
    )

    answer = final_state.get("answer")
    kept_claims = final_state.get("kept_claims", [])
    return QueryResponse(
        query_text=query_result.query_text,
        route=query_result.route.value,
        final_status=final_state.get("final_status", "unknown"),
        answer_text=answer.answer_text if answer else None,
        citations=[
            CitationOut(claim_text=c.claim_text, section_id=c.citation_section_id) for c in kept_claims
        ],
        flagged_reason=final_state.get("flagged_reason"),
        disclaimer=answer.disclaimer if answer else None,
    )

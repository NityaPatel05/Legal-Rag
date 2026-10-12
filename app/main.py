"""FastAPI app wiring the ingestion/ and query/ library code into an
actually runnable service — see README.md's deployment section and
docker-compose.yml for how this fits together. Deliberately thin: route
handlers translate HTTP requests into calls against
ingestion.pipeline.ingest_document and query.pipeline.process_query /
query.generation.crag_graph.run_crag, using the real adapters wired in
app/dependencies.py.

Errors (review finding B22). An upstream outage (LLM provider after its
retries, embedding server, Qdrant) is a 503 with Retry-After, not a bare
500; a document that can't be ingested as given is a 422 (409 for a
privilege downgrade); a failed permission assertion is a 403 whose detail
reveals nothing about the other tenant. Every one is logged.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import urllib.error
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.dependencies import (
    ActorAuditLogger,
    Dependencies,
    actor_id,
    authorize_request,
    check_auth_configured,
    get_api_key,
    get_client_host,
    get_dependencies,
    llm_for_privilege,
    scoped_llm_clients,
)
from app.schemas import (
    MAX_SOURCE_TEXT_CHARS,
    MAX_UPLOAD_BYTES,
    CitationOut,
    ClauseOut,
    DocumentDetail,
    DocumentSummary,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    ReviewCheckBody,
    ReviewDecisionBody,
    ReviewStatusOut,
    ReviewSummary,
)
from ingestion.metadata.llm_client import LLMCallError
from ingestion.pipeline import IngestionRefused, PrivilegeDowngradeError, ingest_document
from query.filters.permission import PermissionViolation
from query.generation.answer_generator import DISCLAIMER
from query.generation.crag_graph import run_crag
from query.pipeline import process_query
from query.retrieval.bm25_search import InMemoryBM25Index
from query.retrieval.reranker import PassthroughReranker
from ingestion.schemas import ClauseMetadata, PrivilegeLevel
from query.filters.audit_log import AuditEvent
from query.review.schemas import ID_PATTERN, ReviewCase, ReviewedClaim, ReviewStatus
from query.review.workflow import (
    ReviewConflictError,
    ReviewDecisionError,
    apply_decision,
    build_review_case,
    check_claims,
    mark_viewed,
    should_queue,
)
from query.schemas import PermissionContext, QueryFilters

logger = logging.getLogger("app")

# BGE-M3 dense output is 1024-dim -- see embedding_server/main.py.
_EMBEDDING_DIM = 1024
_STARTUP_RETRY_ATTEMPTS = 10
_STARTUP_RETRY_DELAY_SECONDS = 3.0


def _ensure_qdrant_collection() -> None:
    """Qdrant requires a collection to exist before anything is upserted
    into it; the indexer creates it (and its payload indexes) idempotently,
    once here at startup rather than as an undocumented manual step.

    Retries with a short delay: in docker-compose, the app container can
    start before Qdrant is actually accepting connections even with
    depends_on ordering, and this sidesteps depending on a compose
    healthcheck being exactly right for whatever qdrant/qdrant image
    version is in use."""
    deps = get_dependencies()

    last_error: Exception | None = None
    for attempt in range(_STARTUP_RETRY_ATTEMPTS):
        try:
            deps.indexer.ensure_collection(_EMBEDDING_DIM)
            return
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any
            # connection/transport error from qdrant-client during a cold
            # start should be retried the same way, not special-cased.
            last_error = exc
            logger.info("Qdrant not ready (attempt %d): %s", attempt + 1, exc)
            if attempt < _STARTUP_RETRY_ATTEMPTS - 1:
                time.sleep(_STARTUP_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Could not reach Qdrant at {deps.settings.qdrant_url} after "
        f"{_STARTUP_RETRY_ATTEMPTS} attempts"
    ) from last_error


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    check_auth_configured(get_dependencies().settings)
    # In a worker thread: the retry loop sleeps, and must not block the
    # event loop while it waits for Qdrant (review finding L7).
    await asyncio.to_thread(_ensure_qdrant_collection)
    yield


app = FastAPI(
    title="Legal Contract Clause Analysis RAG",
    description="See CLAUDE.md / docs/architecture.md in the repo for the full design.",
    version="0.1.0",
    lifespan=lifespan,
)


def _upstream_error_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [LLMCallError, ConnectionError, TimeoutError, urllib.error.URLError]
    try:
        import httpx

        types.append(httpx.HTTPError)
    except ImportError:  # pragma: no cover
        pass
    try:
        from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

        types.extend([ResponseHandlingException, UnexpectedResponse])
    except ImportError:  # pragma: no cover
        pass
    return tuple(types)


@app.exception_handler(PermissionViolation)
async def _permission_violation(request: Request, exc: PermissionViolation) -> JSONResponse:
    # Already audited where it was raised; the detail names the other
    # tenant's clause, so it stays in the server log.
    logger.error("permission violation on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=403, content={"detail": "Permission check failed"})


@app.exception_handler(PrivilegeDowngradeError)
async def _privilege_downgrade(request: Request, exc: PrivilegeDowngradeError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(IngestionRefused)
async def _ingestion_refused(request: Request, exc: IngestionRefused) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def _upstream_unavailable(request: Request, exc: Exception) -> JSONResponse:
    logger.error("upstream failure on %s: %r", request.url.path, exc, exc_info=exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "An upstream service (LLM, embeddings or vector store) is unavailable; retry later."},
        headers={"Retry-After": "30"},
    )


for _error_type in _upstream_error_types():
    app.add_exception_handler(_error_type, _upstream_unavailable)


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


def _ingest(
    deps: Dependencies,
    *,
    document_id: str,
    tenant_id: str,
    matter_id: str,
    privilege_level: PrivilegeLevel,
    text: str,
    source_path: str,
) -> IngestResponse:
    llm_client, privileged_llm_client = scoped_llm_clients(deps, tenant_id, matter_id)
    parsed = ingest_document(
        document_id=document_id,
        source_path=source_path,
        tenant_id=tenant_id,
        matter_id=matter_id,
        privilege_level=privilege_level,
        document_loader=_InMemoryTextLoader(text),
        llm_client=llm_client,
        embedding_client=deps.embedding_client,
        vector_store=deps.indexer,
        max_passage_chars=deps.settings.max_passage_chars,
        privileged_llm_client=privileged_llm_client,
        batch_size=deps.settings.ingest_batch_size,
    )
    return IngestResponse(
        document_id=parsed.document_id,
        num_clauses=len(parsed.clauses),
        num_defined_terms=len(parsed.defined_terms),
        num_references=len(parsed.references),
    )


@app.post("/documents/text", response_model=IngestResponse)
def ingest_text(
    request: IngestTextRequest,
    deps: Dependencies = Depends(get_dependencies),
    api_key: str | None = Depends(get_api_key),
    client_host: str | None = Depends(get_client_host),
) -> IngestResponse:
    authorize_request(request.tenant_id, request.matter_id, api_key, deps.settings, client_host)
    return _ingest(
        deps,
        document_id=request.document_id,
        tenant_id=request.tenant_id,
        matter_id=request.matter_id,
        privilege_level=request.privilege_level,
        text=request.source_text,
        source_path="<inline>",
    )


# Text formats are decoded here; binary formats go through Docling
# (deps.document_loader). Each binary format must also start with its magic
# bytes, so a renamed file never reaches the converter as something else.
_TEXT_SUFFIXES = frozenset({".txt", ".md"})
_BINARY_MAGIC = {".pdf": b"%PDF-", ".docx": b"PK\x03\x04"}


def _extract_upload_text(deps: Dependencies, filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="Text files must be UTF-8") from exc
    magic = _BINARY_MAGIC.get(suffix)
    if magic is None:
        supported = ", ".join(sorted(_TEXT_SUFFIXES | _BINARY_MAGIC.keys()))
        raise HTTPException(status_code=415, detail=f"Unsupported file type; use one of {supported}")
    if not data.startswith(magic):
        raise HTTPException(status_code=415, detail=f"File content is not a valid {suffix} file")

    # Docling reads from a path. Only the allow-listed suffix comes from the
    # client; the filename itself never touches the filesystem.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        try:
            return deps.document_loader.load(str(path))
        except ImportError as exc:
            raise HTTPException(
                status_code=501, detail="PDF/DOCX ingestion needs docling installed on the server"
            ) from exc
        except Exception as exc:  # noqa: BLE001 -- converter failures on a
            # malformed file are the file's problem (422), whatever Docling
            # raises; the detail stays generic and the cause goes to the log.
            logger.warning("could not extract text from upload %r: %r", filename, exc, exc_info=exc)
            raise HTTPException(status_code=422, detail="Could not extract text from this file") from exc


@app.post("/documents/file", response_model=IngestResponse)
def ingest_file(
    file: Annotated[UploadFile, File()],
    document_id: Annotated[str, Form(pattern=ID_PATTERN)],
    tenant_id: Annotated[str, Form(pattern=ID_PATTERN)],
    matter_id: Annotated[str, Form(pattern=ID_PATTERN)],
    privilege_level: Annotated[PrivilegeLevel, Form()] = PrivilegeLevel.CONFIDENTIAL,
    deps: Dependencies = Depends(get_dependencies),
    api_key: str | None = Depends(get_api_key),
    client_host: str | None = Depends(get_client_host),
) -> IngestResponse:
    """Contract file upload (roadmap R12): .txt/.md as UTF-8, .pdf/.docx via
    Docling. Authorized before the file is decoded or converted."""
    authorize_request(tenant_id, matter_id, api_key, deps.settings, client_host)
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    text = _extract_upload_text(deps, file.filename or "", data)
    if len(text) > MAX_SOURCE_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"Extracted text exceeds {MAX_SOURCE_TEXT_CHARS:,} characters")
    return _ingest(
        deps,
        document_id=document_id,
        tenant_id=tenant_id,
        matter_id=matter_id,
        privilege_level=privilege_level,
        text=text,
        source_path=file.filename or "<upload>",
    )


def _most_common(values) -> str | None:
    counts = Counter(v for v in values if v)
    return counts.most_common(1)[0][0] if counts else None


def _document_summary(document_id: str, metadata: list[ClauseMetadata]) -> DocumentSummary:
    parties = Counter(tuple(m.parties) for m in metadata if m.parties)
    return DocumentSummary(
        document_id=document_id,
        num_clauses=len(metadata),
        privilege_level=_most_common(m.privilege_level for m in metadata),
        contract_type=_most_common(m.contract_type for m in metadata),
        governing_law=_most_common(m.governing_law for m in metadata),
        jurisdiction=_most_common(m.jurisdiction for m in metadata),
        parties=list(parties.most_common(1)[0][0]) if parties else [],
        injection_flagged_clauses=sum(m.injection_suspected for m in metadata),
    )


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents(
    tenant_id: str,
    matter_id: str,
    deps: Dependencies = Depends(get_dependencies),
    api_key: str | None = Depends(get_api_key),
    client_host: str | None = Depends(get_client_host),
) -> list[DocumentSummary]:
    authorize_request(tenant_id, matter_id, api_key, deps.settings, client_host)
    documents = deps.document_reader.list_documents(tenant_id=tenant_id, matter_id=matter_id)
    return [_document_summary(document_id, metadata) for document_id, metadata in documents.items()]


@app.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: str,
    tenant_id: str,
    matter_id: str,
    deps: Dependencies = Depends(get_dependencies),
    api_key: str | None = Depends(get_api_key),
    client_host: str | None = Depends(get_client_host),
) -> DocumentDetail:
    """The stored clause text -- exactly what citations are validated
    against -- so a client can show each quote in its clause. Privileged
    text is returned too: privilege restricts which LLM may read it, not the
    matter's own credential holders. Reads of full text are audited."""
    authorize_request(tenant_id, matter_id, api_key, deps.settings, client_host)
    context = deps.document_reader.load_document(document_id, tenant_id=tenant_id, matter_id=matter_id)
    if not context.clauses:
        raise HTTPException(status_code=404, detail="Document not found for this tenant/matter")
    ActorAuditLogger(deps.audit_logger, actor_id(api_key, deps.settings)).log(
        AuditEvent(
            event_type="document_read",
            tenant_id=tenant_id,
            matter_id=matter_id,
            details={"document_id": document_id},
        )
    )
    metadata = context.clause_metadata_by_id
    summary = _document_summary(document_id, [metadata[c.clause_id] for c in context.clauses])
    return DocumentDetail(
        **summary.model_dump(),
        clauses=[
            ClauseOut(
                **clause.model_dump(),
                clause_type=metadata[clause.clause_id].clause_type,
                injection_suspected=metadata[clause.clause_id].injection_suspected,
                injection_markers=metadata[clause.clause_id].injection_markers,
            )
            for clause in context.clauses
        ],
        defined_terms=context.defined_terms,
    )


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    deps: Dependencies = Depends(get_dependencies),
    api_key: str | None = Depends(get_api_key),
    client_host: str | None = Depends(get_client_host),
) -> QueryResponse:
    authorize_request(request.tenant_id, request.matter_id, api_key, deps.settings, client_host)
    audit_logger = ActorAuditLogger(deps.audit_logger, actor_id(api_key, deps.settings))
    permission_context = PermissionContext(tenant_id=request.tenant_id, matter_id=request.matter_id)
    # document_id scopes vector search too -- BM25, the validator and
    # expansion are all built from this one document (see QueryFilters).
    query_filters = QueryFilters(
        jurisdiction=request.jurisdiction,
        governing_law=request.governing_law,
        document_id=request.document_id,
    )

    document_context = deps.document_reader.load_document(
        request.document_id, tenant_id=request.tenant_id, matter_id=request.matter_id
    )
    if not document_context.clauses:
        raise HTTPException(status_code=404, detail="Document not found for this tenant/matter")

    # Privileged contract text never goes to the default provider (R9).
    privileged = any(
        m.privilege_level == PrivilegeLevel.PRIVILEGED for m in document_context.clause_metadata_by_id.values()
    )
    llm_client = llm_for_privilege(
        deps,
        PrivilegeLevel.PRIVILEGED if privileged else PrivilegeLevel.CONFIDENTIAL,
        request.tenant_id,
        request.matter_id,
    )
    if llm_client is None:
        raise HTTPException(
            status_code=403,
            detail="Privileged document: querying it requires a self-hosted LLM (PRIVILEGED_LLM_BASE_URL)",
        )

    bm25_index = InMemoryBM25Index(document_context.bm25_corpus())
    clauses_by_id = document_context.by_clause_id()
    # The Cohere reranker is a third-party API: privileged text never goes
    # there either (review finding S2).
    reranker = PassthroughReranker() if privileged else deps.reranker

    query_result = process_query(
        request.query_text,
        permission_context=permission_context,
        query_filters=query_filters,
        llm_client=llm_client,
        vector_client=deps.vector_client,
        bm25_index=bm25_index,
        reranker=reranker,
        all_clauses_by_section_id=document_context.by_section_id,
        references=document_context.references,
        audit_logger=audit_logger,
        trace_exporter=deps.trace_exporter,
        defined_terms=document_context.defined_terms,
        all_clauses_by_id=clauses_by_id,
        rerank_top_k=deps.settings.generation_top_k,
    )

    final_state = run_crag(
        query_result,
        source_clauses=document_context.clauses,
        all_clauses_by_section_id=document_context.by_section_id,
        references=document_context.references,
        llm_client=llm_client,
        vector_client=deps.vector_client,
        bm25_index=bm25_index,
        reranker=reranker,
        permission_context=permission_context,
        query_filters=query_filters,
        audit_logger=audit_logger,
        defined_terms=document_context.defined_terms,
        all_clauses_by_id=clauses_by_id,
        entailment_checker=deps.entailment_checker,
        rerank_top_k=deps.settings.generation_top_k,
    )

    query_id = str(uuid.uuid4())
    review_id = None
    if should_queue(final_state):
        # A flag used to end here, received by no one (architecture-review
        # A1). The case carries the whole evidence packet; the review id is
        # the query id, so one identifier links response, case and audit.
        case = build_review_case(
            final_state,
            permission_context=permission_context,
            document_id=request.document_id,
            # CRAG's trace includes any rewrite retrievals (B1), so it matches
            # the shown clauses in the case.
            trace=final_state.get("trace", query_result.trace),
            review_id=query_id,
        )
        deps.review_store.add(case)
        audit_logger.log(
            AuditEvent(
                event_type="review_queued",
                tenant_id=request.tenant_id,
                matter_id=request.matter_id,
                details={"review_id": case.review_id, "document_id": case.document_id, "reason": case.reason},
            )
        )
        review_id = case.review_id
    return build_query_response(
        query_result.query_text, query_result.route.value, final_state, query_id=query_id, review_id=review_id
    )


def build_query_response(
    query_text: str,
    route: str,
    final_state: dict,
    *,
    query_id: str | None = None,
    review_id: str | None = None,
) -> QueryResponse:
    """Only validated content leaves the service: answer_text is composed
    from the kept (validated) claims by crag_graph, never the model's
    free-text answer, which is where stripped or uncited content lives."""
    resolved = {
        (r.citation.claim_text, r.citation.section_id): r.resolved_clause_id
        for r in final_state.get("citation_results", [])
        if r.is_valid
    }
    kept_claims = final_state.get("kept_claims", [])
    # not_found carries a fixed, deterministic statement (crag_graph.NOT_FOUND_TEXT).
    answered = final_state.get("final_status") in ("answered", "answered_with_stripped_claims", "not_found")
    return QueryResponse(
        query_id=query_id,
        review_id=review_id,
        query_text=query_text,
        route=route,
        final_status=final_state.get("final_status", "unknown"),
        answer_text=final_state.get("final_answer_text") if answered else None,
        citations=[
            CitationOut(
                claim_text=c.claim_text,
                section_id=c.citation_section_id,
                clause_id=resolved.get((c.claim_text, c.citation_section_id)),
                supporting_quote=c.supporting_quote,
            )
            for c in kept_claims
        ],
        flagged_reason=final_state.get("flagged_reason"),
        verdict=final_state.get("verdict") if answered else None,
        disclaimer=DISCLAIMER,
    )


# --- Human review queue (architecture-review H1/H2) -------------------------
# Scoped like every other route: the credential must cover the tenant and
# matter, and a case outside that scope is a 404, not a 403, so review ids
# can't be probed across matters. The reviewer's name is self-asserted until
# identity comes from an IdP (R8).


def _scoped_case(deps: Dependencies, review_id: str, tenant_id: str, matter_id: str) -> ReviewCase:
    case = deps.review_store.get(review_id)
    if case is None or case.tenant_id != tenant_id or case.matter_id != matter_id:
        raise HTTPException(status_code=404, detail="Review not found for this tenant/matter")
    return case


@app.get("/reviews", response_model=list[ReviewSummary])
def list_reviews(
    tenant_id: str,
    matter_id: str,
    # "all" lists every status (review finding L2: None was unreachable).
    status: ReviewStatus | Literal["all"] = ReviewStatus.OPEN,
    deps: Dependencies = Depends(get_dependencies),
    api_key: str | None = Depends(get_api_key),
    client_host: str | None = Depends(get_client_host),
) -> list[ReviewSummary]:
    authorize_request(tenant_id, matter_id, api_key, deps.settings, client_host)
    return [
        ReviewSummary(
            review_id=c.review_id,
            document_id=c.document_id,
            status=c.status,
            created_at=c.created_at,
            reason=c.reason,
            question=c.question,
            failing_claims=sum(not d.is_valid for d in c.draft_claims),
            total_claims=len(c.draft_claims),
            injection_flagged_clauses=sum(s.injection_suspected for s in c.shown_clauses),
        )
        for c in deps.review_store.list(tenant_id=tenant_id, matter_id=matter_id, status=status)
    ]


@app.get("/reviews/{review_id}", response_model=ReviewCase)
def get_review(
    review_id: str,
    tenant_id: str,
    matter_id: str,
    deps: Dependencies = Depends(get_dependencies),
    api_key: str | None = Depends(get_api_key),
    client_host: str | None = Depends(get_client_host),
) -> ReviewCase:
    authorize_request(tenant_id, matter_id, api_key, deps.settings, client_host)
    case = _scoped_case(deps, review_id, tenant_id, matter_id)
    # Conditional write: only sets first_viewed_at if unset and still open,
    # so a stale view can't reopen a decided case (B16).
    mark_viewed(case, store=deps.review_store)
    return case


@app.get("/reviews/{review_id}/status", response_model=ReviewStatusOut)
def get_review_status(
    review_id: str,
    tenant_id: str,
    matter_id: str,
    deps: Dependencies = Depends(get_dependencies),
    api_key: str | None = Depends(get_api_key),
    client_host: str | None = Depends(get_client_host),
) -> ReviewStatusOut:
    """For the asker polling a queued query. GET /reviews/{id} is the
    reviewer opening the case and starts the time-to-decision clock (H2);
    polling it would make every case look rubber-stamped."""
    authorize_request(tenant_id, matter_id, api_key, deps.settings, client_host)
    case = _scoped_case(deps, review_id, tenant_id, matter_id)
    return ReviewStatusOut(review_id=case.review_id, status=case.status, outcome=case.outcome)


@app.post("/reviews/{review_id}/check", response_model=list[ReviewedClaim])
def check_review_claims(
    review_id: str,
    body: ReviewCheckBody,
    deps: Dependencies = Depends(get_dependencies),
    api_key: str | None = Depends(get_api_key),
    client_host: str | None = Depends(get_client_host),
) -> list[ReviewedClaim]:
    """Validator verdict for each proposed decision, without deciding: the
    decision endpoint stops at the first failing claim, so a reviewer
    editing several would otherwise learn about them one submit at a time."""
    authorize_request(body.tenant_id, body.matter_id, api_key, deps.settings, client_host)
    case = _scoped_case(deps, review_id, body.tenant_id, body.matter_id)
    document_context = deps.document_reader.load_document(
        case.document_id, tenant_id=case.tenant_id, matter_id=case.matter_id
    )
    try:
        return check_claims(
            case,
            body.decisions,
            all_clauses_by_id=document_context.by_clause_id(),
            entailment_checker=deps.entailment_checker,
        )
    except ReviewDecisionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/reviews/{review_id}/decision", response_model=ReviewCase)
def decide_review(
    review_id: str,
    body: ReviewDecisionBody,
    deps: Dependencies = Depends(get_dependencies),
    api_key: str | None = Depends(get_api_key),
    client_host: str | None = Depends(get_client_host),
) -> ReviewCase:
    authorize_request(body.tenant_id, body.matter_id, api_key, deps.settings, client_host)
    case = _scoped_case(deps, review_id, body.tenant_id, body.matter_id)
    # Re-validate against the document as stored now, not as it was when
    # the case was queued.
    document_context = deps.document_reader.load_document(
        case.document_id, tenant_id=case.tenant_id, matter_id=case.matter_id
    )
    try:
        apply_decision(
            case,
            body,
            all_clauses_by_id=document_context.by_clause_id(),
            audit_logger=ActorAuditLogger(deps.audit_logger, actor_id(api_key, deps.settings)),
            entailment_checker=deps.entailment_checker,
            # Saved with UPDATE ... WHERE status='open' before any audit
            # event: of two concurrent decisions exactly one wins, the other
            # gets ReviewAlreadyDecided -> 409 (B16).
            store=deps.review_store,
        )
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ReviewDecisionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return case

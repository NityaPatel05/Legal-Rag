"""Wires the real production adapters (built throughout ingestion/ and
query/ as lazy-import, dependency-injected Protocols) from
ingestion/config.py Settings. Kept separate from app/main.py so the FastAPI
route handlers stay thin, and so tests can construct the same shape with
fakes (FastAPI's dependency_overrides) without needing real credentials or
a running Qdrant/Groq/Cohere.

Every optional provider degrades gracefully rather than hard-failing at
startup: no COHERE_API_KEY (or RERANK_ENABLED unset, the default) -> hybrid-retrieval
order passes through unchanged instead of being reranked; no
LANGSMITH_API_KEY -> tracing no-ops. The app should come up and answer
queries with only QDRANT_URL, GROQ_API_KEY and API_KEYS set, even though
the full CLAUDE.md tech stack wants more.

Authentication is the exception: it fails closed (check_auth_configured,
authorize_tenant). Missing credentials config never degrades into "anyone
may act as any tenant"."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Header, HTTPException, Request

from ingestion.config import Settings, get_settings
from ingestion.embedding.document_reader import QdrantDocumentReader
from ingestion.embedding.embedder import BGEM3Embedder, EmbeddingClient
from ingestion.embedding.indexer import QdrantIndexer
from ingestion.metadata.llm_client import GroqLLMClient, LLMClient, OpenAICompatibleLLMClient
from ingestion.parsing.document_loader import DoclingLoader
from ingestion.pipeline import RedactingLLMClient
from ingestion.schemas import PrivilegeLevel
from query.filters.audit_log import AuditLogger, JSONLAuditLogger
from query.observability.langsmith_tracer import LangSmithTraceExporter, NullTraceExporter, TraceExporter
from query.review.store import ReviewStore, SQLiteReviewStore
from query.retrieval.reranker import CohereReranker, PassthroughReranker, Reranker
from query.retrieval.vector_search import QdrantVectorSearchClient
from query.validation.citation_validator import CrossEncoderEntailmentChecker, EntailmentChecker


class Dependencies:
    """Plain attribute bag, not a pydantic model -- these are live client
    objects (Groq/Qdrant/Cohere clients), not serializable data."""

    def __init__(self, settings: Settings):
        self.settings = settings

        self.embedding_client: EmbeddingClient = BGEM3Embedder(settings.bge_m3_endpoint)
        # Structured PII is redacted from every prompt sent to the default
        # provider, at query time too (R9). A quote the model copies from a
        # redacted span fails verbatim validation -- the safe failure.
        self.llm_client: LLMClient = RedactingLLMClient(
            GroqLLMClient(api_key=settings.groq_api_key or "", model=settings.groq_model)
        )
        # Self-hosted model for privileged documents, or None (R9).
        self.privileged_llm_client: LLMClient | None = (
            RedactingLLMClient(
                OpenAICompatibleLLMClient(
                    settings.privileged_llm_base_url,
                    settings.privileged_llm_model or "default",
                    settings.privileged_llm_api_key,
                )
            )
            if settings.privileged_llm_base_url
            else None
        )
        self.entailment_checker: EntailmentChecker | None = (
            CrossEncoderEntailmentChecker(settings.nli_model, threshold=settings.nli_entailment_threshold)
            if settings.nli_model
            else None
        )
        self.indexer = QdrantIndexer(
            settings.qdrant_url, settings.qdrant_collection_name, settings.qdrant_api_key
        )
        self.vector_client = QdrantVectorSearchClient(
            settings.qdrant_url,
            settings.qdrant_collection_name,
            self.embedding_client,
            settings.qdrant_api_key,
        )
        self.document_reader = QdrantDocumentReader(
            settings.qdrant_url, settings.qdrant_collection_name, settings.qdrant_api_key
        )
        self.reranker: Reranker = (
            CohereReranker(api_key=settings.cohere_api_key, model=settings.cohere_rerank_model)
            if settings.cohere_api_key and settings.rerank_enabled
            else PassthroughReranker()
        )
        self.audit_logger: AuditLogger = JSONLAuditLogger(settings.audit_log_path)
        self.review_store: ReviewStore = SQLiteReviewStore(settings.review_db_path)
        self.trace_exporter: TraceExporter = (
            LangSmithTraceExporter(settings.langsmith_api_key, settings.langsmith_project)
            if settings.langsmith_api_key
            else NullTraceExporter()
        )
        self.document_loader = DoclingLoader()


@lru_cache
def get_dependencies() -> Dependencies:
    return Dependencies(get_settings())


def llm_for_privilege(deps: Dependencies, privilege_level: PrivilegeLevel) -> LLMClient | None:
    """The LLM allowed to see text at this privilege level (see
    ingestion.pipeline.select_llm_client), or None if none may."""
    if privilege_level == PrivilegeLevel.PRIVILEGED:
        return deps.privileged_llm_client
    return deps.llm_client


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def check_auth_configured(settings: Settings) -> None:
    """Startup gate (architecture-review A5). Empty API_KEYS used to mean
    "unauthenticated on whatever address uvicorn binds" -- one missing env
    var in a deployment and every tenant is readable by anyone. Now the app
    refuses to start unless dev mode is explicitly requested."""
    if not settings.api_keys and not settings.dev_mode:
        raise RuntimeError(
            "API_KEYS is empty. Set API_KEYS, or DEV_MODE=true for local "
            "unauthenticated development (loopback clients only)."
        )


def authorize_request(
    tenant_id: str,
    matter_id: str,
    api_key: str | None,
    settings: Settings,
    client_host: str | None = None,
) -> None:
    """Tenant AND matter authorization from the credential (R8). A scoped
    key ({"tenant_id": ..., "matters": [...]}) may act only on the listed
    matters -- deny by default. A legacy string key is tenant-wide.

    Still a stopgap: identity should come from an IdP token (OIDC) with the
    allowed matters as claims; see CLAUDE.md R8."""
    authorize_tenant(tenant_id, api_key, settings, client_host)
    if not settings.api_keys:
        return
    grant = settings.api_keys[api_key or ""]
    if isinstance(grant, dict) and matter_id not in (grant.get("matters") or []):
        raise HTTPException(status_code=403, detail="API key not authorized for this matter")


def _bound_tenant(grant) -> str | None:
    if isinstance(grant, str):
        return grant
    if isinstance(grant, dict):
        return grant.get("tenant_id")
    return None


def authorize_tenant(
    tenant_id: str, api_key: str | None, settings: Settings, client_host: str | None = None
) -> None:
    """Binds the request's tenant to the caller's credential. Without this,
    tenant_id/matter_id come straight from the JSON body and every
    permission-first filter downstream enforces an identity the caller
    simply asserted.

    With no API_KEYS this fails closed, per request as well as at startup
    (the startup gate doesn't run when the app is mounted without its
    lifespan): 503 unless DEV_MODE, and in dev mode 403 for any client that
    isn't loopback. The client address is checked rather than the bind
    address because uvicorn's --host isn't visible to the app, and a
    0.0.0.0 bind (the Dockerfile's) must not expose dev mode.

    Matter-level authorization is authorize_request's job; route handlers
    call that, not this."""
    if not settings.api_keys:
        if not settings.dev_mode:
            raise HTTPException(status_code=503, detail="Authentication is not configured (API_KEYS)")
        if client_host not in _LOOPBACK_HOSTS:
            raise HTTPException(status_code=403, detail="Dev mode serves loopback clients only")
        return
    if (api_key or "") not in settings.api_keys:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
    bound_tenant = _bound_tenant(settings.api_keys[api_key or ""])
    if bound_tenant is None:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
    if bound_tenant != tenant_id:
        raise HTTPException(status_code=403, detail="API key not authorized for this tenant")


def get_api_key(x_api_key: str | None = Header(default=None)) -> str | None:
    return x_api_key


def get_client_host(request: Request) -> str | None:
    return request.client.host if request.client else None

"""Wires the real production adapters (built throughout ingestion/ and
query/ as lazy-import, dependency-injected Protocols) from
ingestion/config.py Settings. Kept separate from app/main.py so the FastAPI
route handlers stay thin, and so tests can construct the same shape with
fakes (FastAPI's dependency_overrides) without needing real credentials or
a running Qdrant/Groq/Cohere.

Every optional provider degrades gracefully rather than hard-failing at
startup: no COHERE_API_KEY -> hybrid-retrieval order passes through
unchanged instead of being reranked; no LANGSMITH_API_KEY -> tracing
no-ops. The app should come up and answer queries with only QDRANT_URL and
GROQ_API_KEY set, even though the full CLAUDE.md tech stack wants more."""

from __future__ import annotations

from functools import lru_cache

from ingestion.config import Settings, get_settings
from ingestion.embedding.document_reader import QdrantDocumentReader
from ingestion.embedding.embedder import BGEM3Embedder, EmbeddingClient
from ingestion.embedding.indexer import QdrantIndexer, VectorStore
from ingestion.metadata.llm_client import GroqLLMClient, LLMClient
from ingestion.parsing.document_loader import DoclingLoader
from query.filters.audit_log import AuditLogger, JSONLAuditLogger
from query.observability.langsmith_tracer import LangSmithTraceExporter, NullTraceExporter, TraceExporter
from query.retrieval.reranker import CohereReranker, Reranker
from query.retrieval.vector_search import QdrantVectorSearchClient


class _PassthroughReranker:
    """Used when COHERE_API_KEY isn't configured -- hands candidates through
    in their existing (hybrid-retrieval RRF) order rather than hard-failing
    the whole query pipeline over a missing optional provider."""

    def rerank(self, query_text, candidates, *, top_k):
        return candidates[:top_k]


class Dependencies:
    """Plain attribute bag, not a pydantic model -- these are live client
    objects (Groq/Qdrant/Cohere clients), not serializable data."""

    def __init__(self, settings: Settings):
        self.settings = settings

        self.embedding_client: EmbeddingClient = BGEM3Embedder(settings.bge_m3_endpoint)
        self.llm_client: LLMClient = GroqLLMClient(
            api_key=settings.groq_api_key or "", model=settings.groq_model
        )
        self.indexer: VectorStore = QdrantIndexer(
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
            CohereReranker(api_key=settings.cohere_api_key)
            if settings.cohere_api_key
            else _PassthroughReranker()
        )
        self.audit_logger: AuditLogger = JSONLAuditLogger(settings.audit_log_path)
        self.trace_exporter: TraceExporter = (
            LangSmithTraceExporter(settings.langsmith_api_key, settings.langsmith_project)
            if settings.langsmith_api_key
            else NullTraceExporter()
        )
        self.document_loader = DoclingLoader()


@lru_cache
def get_dependencies() -> Dependencies:
    return Dependencies(get_settings())

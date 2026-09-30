from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Loads .env per .env.example. See CLAUDE.md 'Tech stack' for the rationale
    behind each provider choice."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Vector DB: Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "legal_clauses"

    # --- Embeddings ---
    bge_m3_endpoint: str = "http://localhost:8080"
    voyage_api_key: str | None = None

    # --- Reranker ---
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"
    # Off by default (docs/architecture-review.md A6): LegalBench-RAG found
    # Cohere reranking hurt legal retrieval, and Topollai et al. 2025 saw a
    # pretrained BGE reranker cut top-1 from 56% to 12%. Turn it on only
    # when the R3 ablation shows a gain on this corpus.
    rerank_enabled: bool = False
    voyage_rerank_api_key: str | None = None

    # --- Chunking ---
    # Clauses longer than this are split into passages (never across a
    # clause boundary). See clause_parser.split_oversize_clauses.
    # 1000 chars x generation_top_k=10 (~7.2k chars of context) beat the old
    # 2000 x 5 (~6.5k chars) on CUAD: hit .741 vs .690, span recall .681 vs
    # .637 (BM25, 510 docs; CLAUDE.md R3). Change the two together -- at a
    # fixed k, passage size mostly trades how much text the model gets.
    max_passage_chars: int = 1000
    # Passages handed to generation after (optional) reranking, before
    # reference / sibling-passage expansion.
    generation_top_k: int = 10

    # --- Document parsing: Docling ---
    docling_endpoint: str | None = None

    # --- LLM provider: Groq ---
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"

    # --- Observability: LangSmith ---
    langsmith_api_key: str | None = None
    langsmith_project: str = "legal-contract-rag"
    langchain_tracing_v2: bool = False

    # --- Eval: CUAD ---
    cuad_data_dir: str = "./eval/cuad_preflight/data"

    # --- Multi-tenancy ---
    default_tenant_id: str = "dev-tenant"
    # API key -> grant, as JSON. Two forms:
    #   API_KEYS='{"key-abc": {"tenant_id": "tenant-1", "matters": ["m-1", "m-2"]}}'
    #     -- the key may act only on those matters of that tenant (deny by
    #     default: a matter not listed is refused).
    #   API_KEYS='{"key-abc": "tenant-1"}'
    #     -- legacy tenant-wide key: every matter of the tenant. Kept for
    #     compatibility; prefer the scoped form.
    # When non-empty, every request must send X-API-Key. Stopgap until an
    # IdP (R8).
    api_keys: dict[str, str | dict] = {}
    # Fail closed (docs/architecture-review.md A5): with API_KEYS empty the
    # app refuses to start unless DEV_MODE=true, and even then serves only
    # loopback clients. Dev mode trusts the tenant/matter in the request
    # body as-is -- never deploy like that.
    dev_mode: bool = False

    # --- Privileged documents (roadmap R9) ---
    # privilege_level=privileged text must never reach the default (Groq)
    # provider. If set, it goes to this OpenAI-compatible, self-hosted
    # endpoint (vLLM, Ollama, llama.cpp server, ...) instead; if unset,
    # privileged documents are ingested without LLM metadata and refused
    # at query time.
    privileged_llm_base_url: str | None = None
    privileged_llm_model: str | None = None
    privileged_llm_api_key: str | None = None

    # --- Citation validator: entailment layer (roadmap R6) ---
    # Empty = off. Otherwise a sentence-transformers CrossEncoder NLI model,
    # e.g. "cross-encoder/nli-deberta-v3-base". Deterministic: fixed model,
    # fixed threshold on the entailment probability.
    nli_model: str | None = None
    nli_entailment_threshold: float = 0.5

    # --- Guardrails: audit log (CLAUDE.md #8) ---
    audit_log_path: str = "./audit/audit.jsonl"

    # --- Human review queue (architecture-review H1) ---
    # SQLite file holding review cases; must persist across restarts.
    review_db_path: str = "./reviews/reviews.sqlite3"


def get_settings() -> Settings:
    return Settings()

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
    voyage_rerank_api_key: str | None = None

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

    # --- Guardrails: audit log (CLAUDE.md #8) ---
    audit_log_path: str = "./audit/audit.jsonl"


def get_settings() -> Settings:
    return Settings()

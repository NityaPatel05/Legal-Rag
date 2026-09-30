"""Cross-encoder reranking — CLAUDE.md's retrieval-pipeline decision: rerank
hybrid candidates down to a tight top-k (3-5) before generation. Reranker is
the seam pipeline.py depends on; CohereReranker is the production adapter.

Caution, from the literature: LegalBench-RAG (Pipitone & Alami 2024,
arXiv:2408.10343) found Cohere's general-purpose reranker made retrieval
WORSE than no reranker on CUAD/MAUD/ContractNLI. Reranking is therefore a
setting (RERANK_ENABLED), and whether to keep it on must be decided by the
retrieval ablation on this project's own eval set, not assumed."""

from __future__ import annotations

from typing import Protocol

from query.schemas import RetrievedClause


class Reranker(Protocol):
    def rerank(
        self, query_text: str, candidates: list[RetrievedClause], *, top_k: int
    ) -> list[RetrievedClause]: ...


class PassthroughReranker:
    """Keeps the hybrid-retrieval (RRF) order and truncates to top_k. Used
    when reranking is disabled or unconfigured, and as the ablation
    baseline."""

    def rerank(
        self, query_text: str, candidates: list[RetrievedClause], *, top_k: int
    ) -> list[RetrievedClause]:
        return list(candidates[:top_k])


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        self._api_key = api_key
        self._model = model
        self._client = None

    def _client_instance(self):
        if self._client is None:
            import cohere  # lazy import

            self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    def rerank(
        self, query_text: str, candidates: list[RetrievedClause], *, top_k: int
    ) -> list[RetrievedClause]:
        if not candidates:
            return []
        client = self._client_instance()
        # Heading included: a passage cut from mid-clause lacks its own.
        documents = [f"{c.clause.heading or ''}\n{c.clause.text}".strip() for c in candidates]
        response = client.rerank(model=self._model, query=query_text, documents=documents, top_n=top_k)
        return [
            candidates[result.index].model_copy(
                update={"score": result.relevance_score, "source": "hybrid+rerank"}
            )
            for result in response.results
        ]

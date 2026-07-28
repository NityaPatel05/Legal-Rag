"""Cross-encoder reranking — CLAUDE.md's retrieval-pipeline decision: rerank
hybrid candidates down to a tight top-k (3-5) before generation. Reranker is
the seam pipeline.py depends on; CohereReranker is the production adapter."""

from __future__ import annotations

from typing import Protocol

from query.schemas import RetrievedClause


class Reranker(Protocol):
    def rerank(
        self, query_text: str, candidates: list[RetrievedClause], *, top_k: int
    ) -> list[RetrievedClause]: ...


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        self._api_key = api_key
        self._model = model
        self._client = None

    def _client_instance(self):
        if self._client is None:
            import cohere  # lazy import

            self._client = cohere.Client(self._api_key)
        return self._client

    def rerank(
        self, query_text: str, candidates: list[RetrievedClause], *, top_k: int
    ) -> list[RetrievedClause]:
        if not candidates:
            return []
        client = self._client_instance()
        documents = [c.clause.text for c in candidates]
        response = client.rerank(model=self._model, query=query_text, documents=documents, top_n=top_k)
        return [
            candidates[result.index].model_copy(
                update={"score": result.relevance_score, "source": "hybrid+rerank"}
            )
            for result in response.results
        ]
